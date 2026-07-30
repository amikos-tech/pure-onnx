package ort

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// DefaultOnnxRuntimeVersion is the default ONNX Runtime version used by bootstrap.
	// This should track the runtime version validated by CI and examples.
	DefaultOnnxRuntimeVersion = "1.24.1"

	defaultBootstrapBaseURL            = "https://github.com/microsoft/onnxruntime/releases/download"
	defaultBootstrapReleaseMetadataURL = "https://api.github.com/repos/microsoft/onnxruntime/releases/tags"
	bootstrapGitHubAPIVersion          = "2022-11-28"
	bootstrapDownloaderUserAgent       = "onnx-purego-bootstrap-downloader"
	defaultBootstrapDownloadRetryCount = 3
	bootstrapManifestFilename          = ".onnx-purego-manifest.json"
	bootstrapManifestVersion           = 1

	secureDirectoryPermission = 0o750
	secureLockFilePermission  = 0o600

	maxExtractedFileBytes  int64 = 1 << 30 // 1 GiB
	maxExtractedTotalBytes int64 = 4 << 30 // 4 GiB
	maxDownloadBytes       int64 = 1 << 30 // 1 GiB
	maxMetadataBytes       int64 = 5 << 20 // 5 MiB
	maxManifestBytes       int64 = 4 << 20 // 4 MiB
)

func safeArchiveFileMode(mode os.FileMode) os.FileMode {
	return mode.Perm() &^ 0o022
}

var errBootstrapRedirectPolicy = errors.New("bootstrap redirect policy rejection")

// ErrUnsupportedPlatform is returned when resolveRuntimeArtifact cannot resolve a prebuilt ONNX Runtime artifact for the host GOOS/GOARCH combination.
var ErrUnsupportedPlatform = errors.New("unsupported platform for ONNX Runtime bootstrap")

// IsUnsupportedPlatformError reports whether err wraps ErrUnsupportedPlatform.
func IsUnsupportedPlatformError(err error) bool { return errors.Is(err, ErrUnsupportedPlatform) }

var bootstrapCacheFallbackWarnOnce sync.Once
var bootstrapInitMu sync.Mutex

// bootstrapValidatedInstalls memoizes installs that already passed full manifest
// verification in this process, so repeat calls skip re-hashing the whole tree
// (a cached install is dominated by the shared library itself). A memo hit still
// requires an unchanged metadata fingerprint, so tampering between calls is
// caught and re-verified rather than served from the memo.
var bootstrapValidatedInstalls sync.Map // bootstrapValidationKey -> bootstrapValidatedInstall

type bootstrapValidationKey struct {
	installDir       string
	runtimeVersion   string
	platform         string
	expectedSHA256   string
	allowSharedCache bool
}

type bootstrapValidatedInstall struct {
	libraryPath string
	fingerprint string
}

var (
	bootstrapLockAcquireTimeout           = 2 * time.Minute
	bootstrapLockRetryInterval            = 200 * time.Millisecond
	bootstrapLockLogInterval              = 5 * time.Second
	bootstrapRemove                       = os.Remove
	bootstrapRemoveAll                    = os.RemoveAll
	bootstrapValidateCachedRuntimeInstall = validateCachedRuntimeInstall
	bootstrapUserCacheDir                 = os.UserCacheDir
)

type cacheValidationDisposition uint8

const (
	cacheValidationOperational cacheValidationDisposition = iota
	cacheValidationMissing
	cacheValidationConfirmedInvalid
)

func (d cacheValidationDisposition) String() string {
	switch d {
	case cacheValidationMissing:
		return "missing"
	case cacheValidationConfirmedInvalid:
		return "confirmed invalid"
	default:
		return "operational"
	}
}

type cacheValidationError struct {
	disposition cacheValidationDisposition
	cause       error
}

func (e *cacheValidationError) Error() string {
	if e == nil || e.cause == nil {
		return ""
	}
	return e.cause.Error()
}

func (e *cacheValidationError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

func markCacheValidationError(disposition cacheValidationDisposition, err error) error {
	if err == nil {
		return nil
	}
	return &cacheValidationError{
		disposition: disposition,
		cause:       err,
	}
}

func cacheValidationDispositionForError(err error) cacheValidationDisposition {
	var validationErr *cacheValidationError
	if errors.As(err, &validationErr) {
		return validationErr.disposition
	}
	return cacheValidationOperational
}

// permanentBootstrapError marks errors that should abort retry loops immediately.
type permanentBootstrapError struct {
	cause error
}

func (e *permanentBootstrapError) Error() string {
	if e == nil || e.cause == nil {
		return ""
	}
	return e.cause.Error()
}

func (e *permanentBootstrapError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

func markPermanentBootstrapError(err error) error {
	if err == nil {
		return nil
	}
	return &permanentBootstrapError{cause: err}
}

func isPermanentBootstrapError(err error) bool {
	var target *permanentBootstrapError
	return errors.As(err, &target)
}

func isRetryableBootstrapHTTPStatus(statusCode int) bool {
	if statusCode == http.StatusRequestTimeout || statusCode == http.StatusTooManyRequests {
		return true
	}
	return statusCode >= 500
}

func bootstrapRetryAttempts(attempts int) int {
	if attempts < 1 {
		return 1
	}
	return attempts
}

// BootstrapOption configures EnsureOnnxRuntimeSharedLibrary.
type BootstrapOption func(*bootstrapConfig) error

type bootstrapConfig struct {
	libraryPath        string
	cacheDir           string
	version            string
	disableDownload    bool
	allowSharedCache   bool
	expectedSHA256     string
	baseURL            string
	releaseMetadataURL string
	httpClient         *http.Client
	maxDownloadSize    int64
	retryAttempts      int
	goos               string
	goarch             string
}

type runtimeArtifact struct {
	platform         string
	archiveExtension string
	primaryLibrary   string
	libraryGlob      string
}

type archiveExtractionReport struct {
	skippedLinkEntries         int
	skippedLibraryLinkEntries  int
	skippedLibraryLinkExamples []string
}

type bootstrapInstallManifest struct {
	Version          int                     `json:"version"`
	RuntimeVersion   string                  `json:"runtime_version"`
	Platform         string                  `json:"platform"`
	ArchiveSHA256    string                  `json:"archive_sha256"`
	ChecksumVerified bool                    `json:"checksum_verified"`
	Files            []bootstrapManifestFile `json:"files"`
}

type bootstrapManifestFile struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

// WithBootstrapLibraryPath sets an explicit ONNX Runtime shared library path.
// When set, bootstrap download and cache resolution are bypassed.
func WithBootstrapLibraryPath(path string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		path = strings.TrimSpace(path)
		if path == "" {
			return fmt.Errorf("bootstrap library path cannot be empty: %w", ErrInvalidArgument)
		}
		cfg.libraryPath = path
		return nil
	}
}

// WithBootstrapCacheDir sets the cache directory used by bootstrap downloads and extraction.
func WithBootstrapCacheDir(dir string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		dir = strings.TrimSpace(dir)
		if dir == "" {
			return fmt.Errorf("bootstrap cache directory cannot be empty: %w", ErrInvalidArgument)
		}
		cfg.cacheDir = dir
		return nil
	}
}

// WithBootstrapVersion sets the ONNX Runtime version to download (for example: 1.24.1).
func WithBootstrapVersion(version string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		version = strings.TrimSpace(version)
		if version == "" {
			return fmt.Errorf("bootstrap version cannot be empty: %w", ErrInvalidArgument)
		}
		cfg.version = version
		return nil
	}
}

// WithBootstrapDisableDownload enables or disables network download in bootstrap mode.
func WithBootstrapDisableDownload(disable bool) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		cfg.disableDownload = disable
		return nil
	}
}

// WithBootstrapAllowSharedCache explicitly trusts a controlled shared cache.
// Shared mode permits non-current owners and group-writable Unix paths, but
// still rejects world-writable state and performs every integrity check.
func WithBootstrapAllowSharedCache(allow bool) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		cfg.allowSharedCache = allow
		return nil
	}
}

// WithBootstrapExpectedSHA256 enforces an expected SHA256 checksum for the downloaded archive.
// For the official ONNX Runtime source, this value is cross-validated against release metadata.
func WithBootstrapExpectedSHA256(checksum string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		checksum = strings.TrimSpace(strings.ToLower(checksum))
		if checksum == "" {
			return fmt.Errorf("expected SHA256 checksum cannot be empty: %w", ErrInvalidArgument)
		}
		if !looksLikeSHA256(checksum) {
			return fmt.Errorf("expected SHA256 checksum must be 64 hex characters (0-9, a-f): %w", ErrInvalidArgument)
		}
		cfg.expectedSHA256 = checksum
		return nil
	}
}

func withBootstrapBaseURL(baseURL string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		baseURL = strings.TrimSpace(baseURL)
		if baseURL == "" {
			return fmt.Errorf("bootstrap base URL cannot be empty: %w", ErrInvalidArgument)
		}
		if err := validateBootstrapBaseURL(baseURL); err != nil {
			return err
		}
		cfg.baseURL = baseURL
		return nil
	}
}

func withBootstrapHTTPClient(client *http.Client) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if client == nil {
			return fmt.Errorf("bootstrap HTTP client cannot be nil: %w", ErrInvalidArgument)
		}
		cfg.httpClient = client
		return nil
	}
}

// EnsureOnnxRuntimeSharedLibrary ensures an ONNX Runtime shared library is available
// and returns a resolved absolute path to it.
//
// Caller-selected explicit paths may resolve ordinary shared-library symlinks
// before validating the target. Cache-managed paths reject every symlink and
// are returned only after their manifest metadata and content hashes validate.
func EnsureOnnxRuntimeSharedLibrary(opts ...BootstrapOption) (string, error) {
	cfg, err := resolveBootstrapConfig(opts...)
	if err != nil {
		return "", err
	}

	if cfg.libraryPath != "" {
		return validateExplicitLibraryFile(cfg.libraryPath)
	}

	artifact, err := resolveRuntimeArtifact(cfg.goos, cfg.goarch)
	if err != nil {
		return "", err
	}

	installDir := filepath.Join(cfg.cacheDir, artifact.archiveName(cfg.version))
	path, cacheErr := validateBootstrapCacheEntry(cfg, artifact, installDir)
	switch cacheValidationDispositionForError(cacheErr) {
	case cacheValidationOperational:
		if cacheErr == nil {
			return path, nil
		}
		return "", cacheErr
	case cacheValidationMissing:
		if cfg.disableDownload {
			return "", bootstrapCacheMissWithDownloadDisabled(installDir)
		}
	case cacheValidationConfirmedInvalid:
		// Revalidate and repair confirmed corruption under the process lock.
	}

	if err := os.MkdirAll(cfg.cacheDir, secureDirectoryPermission); err != nil {
		return "", fmt.Errorf("failed to create bootstrap cache directory %q: %w", cfg.cacheDir, err)
	}
	if err := validateBootstrapDirectoryTrust(cfg.cacheDir, cfg.allowSharedCache); err != nil {
		return "", fmt.Errorf("bootstrap cache directory is not trusted: %w", err)
	}

	lockPath := filepath.Join(cfg.cacheDir, ".locks", fmt.Sprintf("%s-%s.lock", artifact.platform, cfg.version))
	var resolvedPath string
	if err := withProcessFileLock(lockPath, cfg.allowSharedCache, func() error {
		path, cacheErr := validateBootstrapCacheEntry(cfg, artifact, installDir)
		if cacheErr == nil {
			resolvedPath = path
			return nil
		}

		switch cacheValidationDispositionForError(cacheErr) {
		case cacheValidationOperational:
			return cacheErr
		case cacheValidationConfirmedInvalid:
			if err := bootstrapRemoveAll(installDir); err != nil {
				return fmt.Errorf("failed to remove invalid cached ONNX Runtime install at %q: %w", installDir, errors.Join(cacheErr, err))
			}
			if cfg.disableDownload {
				return fmt.Errorf("cached ONNX Runtime install at %q failed integrity validation and download is disabled: %w", installDir, cacheErr)
			}
		case cacheValidationMissing:
			if cfg.disableDownload {
				return bootstrapCacheMissWithDownloadDisabled(installDir)
			}
		}

		if err := downloadAndInstallRuntime(cfg, artifact, installDir); err != nil {
			return err
		}

		path, cacheErr = validateBootstrapCacheEntry(cfg, artifact, installDir)
		if cacheErr != nil {
			return fmt.Errorf("bootstrap completed but installed runtime failed integrity validation: %w", cacheErr)
		}
		resolvedPath = path
		return nil
	}); err != nil {
		return "", err
	}

	return resolvedPath, nil
}

func validateBootstrapCacheEntry(
	cfg bootstrapConfig,
	artifact runtimeArtifact,
	installDir string,
) (string, error) {
	if err := validateBootstrapDirectoryTrust(cfg.cacheDir, cfg.allowSharedCache); err != nil {
		return "", fmt.Errorf("bootstrap cache directory is not trusted: %w", err)
	}
	return bootstrapValidateCachedRuntimeInstall(cfg, artifact, installDir)
}

func bootstrapCacheMissWithDownloadDisabled(installDir string) error {
	return fmt.Errorf(
		"ONNX Runtime library not found in cache and download is disabled at %q: %w",
		installDir,
		ErrSharedLibraryNotFound,
	)
}

// InitializeEnvironmentWithBootstrap resolves a shared library path via bootstrap,
// sets it on the runtime, and initializes the ONNX Runtime environment.
func InitializeEnvironmentWithBootstrap(opts ...BootstrapOption) error {
	path, err := EnsureOnnxRuntimeSharedLibrary(opts...)
	if err != nil {
		return err
	}

	// Serialize only the lifecycle transition. Diagnostics are emitted after
	// bootstrapInitMu is released so a handler may re-enter this function without
	// deadlocking on a non-reentrant mutex.
	runtimeVersion, newlyInitialized, initErr := func() (string, bool, error) {
		bootstrapInitMu.Lock()
		defer bootstrapInitMu.Unlock()
		return initializeEnvironmentAtLocked(path)
	}()

	if err := completeEnvironmentInitialization(runtimeVersion, newlyInitialized, initErr); err != nil {
		return fmt.Errorf("initialize environment with bootstrap library %q: %w", path, err)
	}
	return nil
}

func resolveBootstrapConfig(opts ...BootstrapOption) (bootstrapConfig, error) {
	disableDownload, err := parseBootstrapBoolEnv("ONNXRUNTIME_DISABLE_DOWNLOAD")
	if err != nil {
		return bootstrapConfig{}, err
	}
	allowSharedCache, err := parseBootstrapBoolEnv("ONNXRUNTIME_ALLOW_SHARED_CACHE")
	if err != nil {
		return bootstrapConfig{}, err
	}

	cfg := bootstrapConfig{
		libraryPath:        strings.TrimSpace(os.Getenv("ONNXRUNTIME_LIB_PATH")),
		cacheDir:           strings.TrimSpace(os.Getenv("ONNXRUNTIME_CACHE_DIR")),
		version:            strings.TrimSpace(os.Getenv("ONNXRUNTIME_VERSION")),
		disableDownload:    disableDownload,
		allowSharedCache:   allowSharedCache,
		baseURL:            defaultBootstrapBaseURL,
		releaseMetadataURL: defaultBootstrapReleaseMetadataURL,
		httpClient:         newBootstrapHTTPClient(),
		maxDownloadSize:    maxDownloadBytes,
		retryAttempts:      defaultBootstrapDownloadRetryCount,
		goos:               runtime.GOOS,
		goarch:             runtime.GOARCH,
	}

	if cfg.version == "" {
		cfg.version = DefaultOnnxRuntimeVersion
	}
	if cfg.cacheDir == "" {
		cfg.cacheDir = defaultBootstrapCacheDir()
	}

	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt(&cfg); err != nil {
			return bootstrapConfig{}, err
		}
	}

	version, err := normalizeRuntimeVersion(cfg.version)
	if err != nil {
		return bootstrapConfig{}, err
	}
	cfg.version = version

	if cfg.cacheDir == "" {
		return bootstrapConfig{}, fmt.Errorf("bootstrap cache directory is empty: %w", ErrInvalidArgument)
	}
	cfg.cacheDir = filepath.Clean(cfg.cacheDir)

	if strings.TrimSpace(cfg.baseURL) == "" {
		return bootstrapConfig{}, fmt.Errorf("bootstrap base URL is empty: %w", ErrInvalidArgument)
	}
	cfg.baseURL = strings.TrimRight(strings.TrimSpace(cfg.baseURL), "/")
	if err := validateBootstrapBaseURL(cfg.baseURL); err != nil {
		return bootstrapConfig{}, err
	}
	cfg.releaseMetadataURL = strings.TrimRight(strings.TrimSpace(cfg.releaseMetadataURL), "/")
	if cfg.releaseMetadataURL != "" {
		if err := validateBootstrapBaseURL(cfg.releaseMetadataURL); err != nil {
			return bootstrapConfig{}, err
		}
	}

	if cfg.httpClient == nil {
		return bootstrapConfig{}, fmt.Errorf("bootstrap HTTP client cannot be nil: %w", ErrInvalidArgument)
	}
	if cfg.maxDownloadSize <= 0 {
		return bootstrapConfig{}, fmt.Errorf(
			"bootstrap max download bytes must be > 0, got %d: %w",
			cfg.maxDownloadSize,
			ErrInvalidArgument,
		)
	}

	return cfg, nil
}

func validateBootstrapBaseURL(baseURL string) error {
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return fmt.Errorf("invalid bootstrap base URL %q: %w: %w", baseURL, ErrInvalidArgument, err)
	}
	if parsed.Scheme == "" {
		return fmt.Errorf("bootstrap base URL %q must include a scheme: %w", baseURL, ErrInvalidArgument)
	}
	if parsed.Host == "" {
		return fmt.Errorf("bootstrap base URL %q must include a host: %w", baseURL, ErrInvalidArgument)
	}

	scheme := strings.ToLower(parsed.Scheme)
	if scheme == "https" {
		return nil
	}
	if scheme != "http" {
		return fmt.Errorf(
			"bootstrap base URL %q uses unsupported scheme %q: %w",
			baseURL,
			parsed.Scheme,
			ErrInvalidArgument,
		)
	}
	if !isLoopbackBootstrapHost(parsed.Hostname()) {
		return fmt.Errorf(
			"bootstrap base URL %q must use https (http is allowed only for loopback hosts): %w",
			baseURL,
			ErrInvalidArgument,
		)
	}
	return nil
}

func isLoopbackBootstrapHost(host string) bool {
	host = strings.TrimSpace(strings.ToLower(host))
	if host == "" {
		return false
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func newBootstrapHTTPClient() *http.Client {
	var transport *http.Transport
	if base, ok := http.DefaultTransport.(*http.Transport); ok && base != nil {
		clone := base.Clone()
		clone.Proxy = http.ProxyFromEnvironment
		clone.DialContext = (&net.Dialer{
			Timeout: 30 * time.Second,
		}).DialContext
		clone.TLSHandshakeTimeout = 10 * time.Second
		clone.ResponseHeaderTimeout = 30 * time.Second
		clone.IdleConnTimeout = 90 * time.Second
		transport = clone
	} else {
		transport = &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout: 30 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   10 * time.Second,
			ResponseHeaderTimeout: 30 * time.Second,
			IdleConnTimeout:       90 * time.Second,
		}
	}
	return &http.Client{
		Timeout:       2 * time.Minute,
		Transport:     transport,
		CheckRedirect: rejectHTTPSDowngradeRedirect,
	}
}

func rejectHTTPSDowngradeRedirect(req *http.Request, via []*http.Request) error {
	if len(via) >= 10 {
		return fmt.Errorf("%w: stopped after 10 redirects", errBootstrapRedirectPolicy)
	}
	if len(via) == 0 {
		return nil
	}
	prev := via[len(via)-1]
	if prev.URL == nil || req.URL == nil {
		return fmt.Errorf("%w: nil URL in redirect chain", errBootstrapRedirectPolicy)
	}
	if strings.EqualFold(prev.URL.Scheme, "https") &&
		strings.EqualFold(req.URL.Scheme, "http") {
		return fmt.Errorf("%w: redirect from HTTPS to HTTP is not allowed: %s -> %s", errBootstrapRedirectPolicy, prev.URL.Redacted(), req.URL.Redacted())
	}
	return nil
}

func isBootstrapRedirectPolicyError(err error) bool {
	return errors.Is(err, errBootstrapRedirectPolicy)
}

func resolveRuntimeArtifact(goos, goarch string) (runtimeArtifact, error) {
	switch goos {
	case "darwin":
		switch goarch {
		case "arm64":
			return runtimeArtifact{
				platform:         "osx-arm64",
				archiveExtension: "tgz",
				primaryLibrary:   "libonnxruntime.dylib",
				libraryGlob:      "libonnxruntime*.dylib",
			}, nil
		case "amd64":
			return runtimeArtifact{
				platform:         "osx-x86_64",
				archiveExtension: "tgz",
				primaryLibrary:   "libonnxruntime.dylib",
				libraryGlob:      "libonnxruntime*.dylib",
			}, nil
		}
	case "linux":
		switch goarch {
		case "arm64":
			return runtimeArtifact{
				platform:         "linux-aarch64",
				archiveExtension: "tgz",
				primaryLibrary:   "libonnxruntime.so",
				libraryGlob:      "libonnxruntime.so*",
			}, nil
		case "amd64":
			return runtimeArtifact{
				platform:         "linux-x64",
				archiveExtension: "tgz",
				primaryLibrary:   "libonnxruntime.so",
				libraryGlob:      "libonnxruntime.so*",
			}, nil
		}
	case "windows":
		switch goarch {
		case "amd64":
			return runtimeArtifact{
				platform:         "win-x64",
				archiveExtension: "zip",
				primaryLibrary:   "onnxruntime.dll",
				libraryGlob:      "onnxruntime*.dll",
			}, nil
		case "arm64":
			return runtimeArtifact{
				platform:         "win-arm64",
				archiveExtension: "zip",
				primaryLibrary:   "onnxruntime.dll",
				libraryGlob:      "onnxruntime*.dll",
			}, nil
		}
	}

	return runtimeArtifact{}, fmt.Errorf("%w: GOOS=%s GOARCH=%s", ErrUnsupportedPlatform, goos, goarch)
}

func (a runtimeArtifact) archiveName(version string) string {
	return fmt.Sprintf("onnxruntime-%s-%s", a.platform, version)
}

func (a runtimeArtifact) archiveFilename(version string) string {
	return fmt.Sprintf("%s.%s", a.archiveName(version), a.archiveExtension)
}

func (a runtimeArtifact) downloadURL(baseURL, version string) string {
	return fmt.Sprintf("%s/v%s/%s", strings.TrimRight(baseURL, "/"), version, a.archiveFilename(version))
}

func downloadAndInstallRuntime(cfg bootstrapConfig, artifact runtimeArtifact, installDir string) error {
	downloadURL := artifact.downloadURL(cfg.baseURL, cfg.version)
	expectedChecksum, err := resolveRuntimeArchiveChecksum(cfg, artifact)
	if err != nil {
		return err
	}
	archivePath, checksum, err := downloadRuntimeArchive(cfg, downloadURL)
	if err != nil {
		return err
	}
	defer func() {
		if removeErr := bootstrapRemove(archivePath); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			emitDiagnostic(
				context.Background(),
				slog.LevelWarn,
				"temporary bootstrap archive cleanup failed",
				slog.String("operation", "remove temporary archive"),
				slog.String("path", archivePath),
				slog.Any("error", removeErr),
			)
		}
	}()

	if expectedChecksum != "" && checksum != expectedChecksum {
		return fmt.Errorf("download checksum mismatch: expected %s, got %s", expectedChecksum, checksum)
	}
	if expectedChecksum == "" {
		emitDiagnostic(
			context.Background(),
			slog.LevelWarn,
			"bootstrap download continued without checksum verification",
			slog.String("url", redactedBootstrapURL(downloadURL)),
			slog.Bool("checksum_verified", false),
			slog.String("observed_sha256", checksum),
		)
	}

	stagingRoot := installDir + fmt.Sprintf(".staging-%d", time.Now().UnixNano())
	if err := bootstrapRemoveAll(stagingRoot); err != nil {
		return fmt.Errorf("failed to clean bootstrap staging directory %q: %w", stagingRoot, err)
	}
	if err := os.MkdirAll(stagingRoot, secureDirectoryPermission); err != nil {
		return fmt.Errorf("failed to create bootstrap staging directory %q: %w", stagingRoot, err)
	}
	defer func() {
		if removeErr := bootstrapRemoveAll(stagingRoot); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			emitDiagnostic(
				context.Background(),
				slog.LevelWarn,
				"bootstrap staging cleanup failed",
				slog.String("operation", "remove staging directory"),
				slog.String("path", stagingRoot),
				slog.Any("error", removeErr),
			)
		}
	}()

	extractReport, err := extractArchiveFile(archivePath, stagingRoot, artifact.archiveExtension, artifact.libraryGlob)
	if err != nil {
		return err
	}

	extractedInstallDir := filepath.Join(stagingRoot, artifact.archiveName(cfg.version))
	info, statErr := os.Stat(extractedInstallDir)
	if statErr != nil {
		if !errors.Is(statErr, os.ErrNotExist) {
			return fmt.Errorf("failed to inspect extracted install directory %q: %w", extractedInstallDir, statErr)
		}
		extractedInstallDir = stagingRoot
	} else if !info.IsDir() {
		return fmt.Errorf("extracted install path is not a directory: %q", extractedInstallDir)
	}

	if _, err := resolveExtractedLibraryPath(extractedInstallDir, artifact); err != nil {
		if errors.Is(err, ErrSharedLibraryNotFound) {
			errMessage := fmt.Sprintf("downloaded archive did not contain expected shared library in %q", filepath.Join(extractedInstallDir, "lib"))
			switch {
			case extractReport.skippedLibraryLinkEntries > 0:
				if len(extractReport.skippedLibraryLinkExamples) > 0 {
					errMessage = fmt.Sprintf(
						"%s (extraction skipped %d link entries matching %q; examples: %q)",
						errMessage,
						extractReport.skippedLibraryLinkEntries,
						artifact.libraryGlob,
						strings.Join(extractReport.skippedLibraryLinkExamples, "\", \""),
					)
				} else {
					errMessage = fmt.Sprintf(
						"%s (extraction skipped %d link entries matching %q)",
						errMessage,
						extractReport.skippedLibraryLinkEntries,
						artifact.libraryGlob,
					)
				}
			case extractReport.skippedLinkEntries > 0:
				errMessage = fmt.Sprintf("%s (extraction skipped %d link entries)", errMessage, extractReport.skippedLinkEntries)
			}
			return fmt.Errorf("%s: %w", errMessage, err)
		}
		return err
	}
	if err := writeBootstrapInstallManifest(
		extractedInstallDir,
		cfg,
		artifact,
		checksum,
		expectedChecksum != "",
	); err != nil {
		return err
	}

	if _, err := os.Lstat(installDir); err == nil {
		return fmt.Errorf("refusing to replace ONNX Runtime install that appeared during bootstrap: %q", installDir)
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("failed to inspect ONNX Runtime install destination %q: %w", installDir, err)
	}

	if extractedInstallDir == stagingRoot {
		if err := os.Rename(stagingRoot, installDir); err != nil {
			return fmt.Errorf("failed to install ONNX Runtime to %q: %w", installDir, err)
		}
		return nil
	}

	if err := os.Rename(extractedInstallDir, installDir); err != nil {
		return fmt.Errorf("failed to install ONNX Runtime to %q: %w", installDir, err)
	}
	return nil
}

type onnxRuntimeReleaseMetadata struct {
	Assets []onnxRuntimeReleaseAsset `json:"assets"`
}

type onnxRuntimeReleaseAsset struct {
	Name   string `json:"name"`
	Digest string `json:"digest"`
}

func resolveRuntimeArchiveChecksum(cfg bootstrapConfig, artifact runtimeArtifact) (string, error) {
	pinnedChecksum := cfg.expectedSHA256
	officialChecksum := ""
	if shouldResolveChecksumFromReleaseMetadata(cfg.baseURL, cfg.releaseMetadataURL) {
		checksum, err := resolveRuntimeArchiveChecksumFromReleaseMetadata(cfg, artifact)
		if err != nil {
			if pinnedChecksum == "" {
				return "", err
			}
			emitDiagnostic(
				context.Background(),
				slog.LevelWarn,
				"bootstrap checksum metadata lookup failed; using pinned checksum",
				slog.String("operation", "resolve release metadata checksum"),
				slog.Any("error", err),
			)
			return pinnedChecksum, nil
		}
		officialChecksum = checksum
	}

	if pinnedChecksum != "" && officialChecksum != "" && pinnedChecksum != officialChecksum {
		return "", fmt.Errorf(
			"configured expected checksum %s does not match ONNX Runtime release metadata checksum %s: %w",
			pinnedChecksum,
			officialChecksum,
			ErrInvalidArgument,
		)
	}
	if officialChecksum != "" {
		return officialChecksum, nil
	}
	return pinnedChecksum, nil
}

func shouldResolveChecksumFromReleaseMetadata(baseURL, metadataURL string) bool {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	metadataURL = strings.TrimRight(strings.TrimSpace(metadataURL), "/")
	return strings.EqualFold(baseURL, defaultBootstrapBaseURL) && metadataURL != ""
}

func resolveRuntimeArchiveChecksumFromReleaseMetadata(cfg bootstrapConfig, artifact runtimeArtifact) (string, error) {
	metadataBaseURL := strings.TrimRight(strings.TrimSpace(cfg.releaseMetadataURL), "/")
	if metadataBaseURL == "" {
		return "", fmt.Errorf("bootstrap release metadata URL is empty: %w", ErrInvalidArgument)
	}
	metadataURL := fmt.Sprintf("%s/v%s", metadataBaseURL, cfg.version)
	archiveName := artifact.archiveFilename(cfg.version)

	attempts := bootstrapRetryAttempts(cfg.retryAttempts)
	attemptErrs := make([]error, 0, attempts)
	for attempt := 1; attempt <= attempts; attempt++ {
		checksum, err := fetchRuntimeArchiveChecksumFromReleaseMetadataURL(cfg, metadataURL, archiveName)
		if err == nil {
			return checksum, nil
		}
		attemptErrs = append(attemptErrs, fmt.Errorf("attempt %d/%d: %w", attempt, attempts, err))
		if isPermanentBootstrapError(err) {
			break
		}
		if attempt < attempts {
			time.Sleep(time.Duration(attempt) * time.Second)
		}
	}
	return "", fmt.Errorf("failed to resolve ONNX Runtime checksum for %q from %q: %w", archiveName, metadataURL, errors.Join(attemptErrs...))
}

func isRetryableGitHubMetadataStatus(statusCode int, headers http.Header, snippet string) bool {
	if isRetryableBootstrapHTTPStatus(statusCode) {
		return true
	}
	if statusCode != http.StatusForbidden {
		return false
	}
	if headers != nil {
		if strings.TrimSpace(headers.Get("Retry-After")) != "" {
			return true
		}
		if strings.TrimSpace(headers.Get("X-RateLimit-Remaining")) == "0" {
			return true
		}
	}

	lowerSnippet := strings.ToLower(strings.TrimSpace(snippet))
	if lowerSnippet == "" {
		return false
	}
	if strings.Contains(lowerSnippet, "rate limit exceeded") || strings.Contains(lowerSnippet, "secondary rate limit") {
		return true
	}
	return false
}

func fetchRuntimeArchiveChecksumFromReleaseMetadataURL(cfg bootstrapConfig, metadataURL, archiveName string) (checksum string, err error) {
	req, err := http.NewRequest(http.MethodGet, metadataURL, nil)
	if err != nil {
		return "", markPermanentBootstrapError(fmt.Errorf("failed to create release metadata request for %q: %w", metadataURL, err))
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("User-Agent", bootstrapDownloaderUserAgent)
	req.Header.Set("X-GitHub-Api-Version", bootstrapGitHubAPIVersion)
	usedGitHubToken := false
	if token := resolveGitHubToken(); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
		usedGitHubToken = true
	}

	resp, err := cfg.httpClient.Do(req)
	if err != nil {
		requestErr := fmt.Errorf("failed to fetch ONNX Runtime release metadata from %q: %w", metadataURL, err)
		if isBootstrapRedirectPolicyError(err) {
			return "", markPermanentBootstrapError(requestErr)
		}
		return "", requestErr
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			closeErr = fmt.Errorf("failed to close release metadata response body for %q: %w", metadataURL, closeErr)
			if err == nil {
				err = closeErr
			} else {
				err = errors.Join(err, closeErr)
			}
		}
	}()

	if resp.StatusCode != http.StatusOK {
		snippetBytes, snippetErr := io.ReadAll(io.LimitReader(resp.Body, 512))
		snippet := strings.TrimSpace(string(snippetBytes))
		envHint := ""
		if usedGitHubToken && (resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden) {
			envHint = " (request used GitHub settings from environment; verify setup and scopes)"
		}
		var statusErr error
		if snippet != "" {
			statusErr = fmt.Errorf("failed to fetch ONNX Runtime release metadata from %q: HTTP %d: %s%s", metadataURL, resp.StatusCode, snippet, envHint)
		} else {
			statusErr = fmt.Errorf("failed to fetch ONNX Runtime release metadata from %q: HTTP %d%s", metadataURL, resp.StatusCode, envHint)
		}
		if snippetErr != nil {
			statusErr = errors.Join(statusErr, fmt.Errorf("failed to read ONNX Runtime release metadata error response body snippet: %w", snippetErr))
		}
		if !isRetryableGitHubMetadataStatus(resp.StatusCode, resp.Header, snippet) {
			return "", markPermanentBootstrapError(statusErr)
		}
		return "", statusErr
	}

	payload, err := io.ReadAll(io.LimitReader(resp.Body, maxMetadataBytes+1))
	if err != nil {
		return "", fmt.Errorf("failed to read ONNX Runtime release metadata from %q: %w", metadataURL, err)
	}
	if int64(len(payload)) > maxMetadataBytes {
		return "", markPermanentBootstrapError(fmt.Errorf("ONNX Runtime release metadata response is too large: %d bytes exceeds limit %d", len(payload), maxMetadataBytes))
	}

	var metadata onnxRuntimeReleaseMetadata
	if err := json.Unmarshal(payload, &metadata); err != nil {
		return "", markPermanentBootstrapError(fmt.Errorf("failed to decode ONNX Runtime release metadata from %q: %w", metadataURL, err))
	}

	for _, asset := range metadata.Assets {
		if strings.TrimSpace(asset.Name) != archiveName {
			continue
		}
		checksum, err = parseSHA256Digest(asset.Digest)
		if err != nil {
			return "", markPermanentBootstrapError(fmt.Errorf("invalid digest for ONNX Runtime asset %q: %w", archiveName, err))
		}
		return checksum, nil
	}

	return "", markPermanentBootstrapError(fmt.Errorf("release metadata at %q does not contain asset %q", metadataURL, archiveName))
}

func parseSHA256Digest(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", fmt.Errorf("digest is empty")
	}
	const prefix = "sha256:"
	if !strings.HasPrefix(strings.ToLower(raw), prefix) {
		return "", fmt.Errorf("unsupported digest format %q", raw)
	}
	checksum := strings.TrimSpace(raw[len(prefix):])
	if !looksLikeSHA256(checksum) {
		return "", fmt.Errorf("invalid SHA256 digest %q", raw)
	}
	return strings.ToLower(checksum), nil
}

func looksLikeSHA256(checksum string) bool {
	if len(checksum) != 64 {
		return false
	}
	for _, r := range checksum {
		switch {
		case r >= '0' && r <= '9':
		case r >= 'a' && r <= 'f':
		case r >= 'A' && r <= 'F':
		default:
			return false
		}
	}
	return true
}

func resolveGitHubToken() string {
	if token := strings.TrimSpace(os.Getenv("GITHUB_TOKEN")); token != "" {
		return token
	}
	if token := strings.TrimSpace(os.Getenv("GH_TOKEN")); token != "" {
		return token
	}
	return ""
}

func downloadRuntimeArchive(cfg bootstrapConfig, url string) (archivePath string, checksum string, err error) {
	attempts := bootstrapRetryAttempts(cfg.retryAttempts)
	attemptErrs := make([]error, 0, attempts)
	for attempt := 1; attempt <= attempts; attempt++ {
		archivePath, checksum, err = downloadRuntimeArchiveOnce(cfg, url)
		if err == nil {
			return archivePath, checksum, nil
		}
		attemptErrs = append(attemptErrs, fmt.Errorf("attempt %d/%d: %w", attempt, attempts, err))
		if isPermanentBootstrapError(err) {
			break
		}
		if attempt < attempts {
			time.Sleep(time.Duration(attempt) * time.Second)
		}
	}
	return "", "", fmt.Errorf("failed to download ONNX Runtime archive from %q after %d attempts: %w", url, attempts, errors.Join(attemptErrs...))
}

func downloadRuntimeArchiveOnce(cfg bootstrapConfig, url string) (archivePath string, checksum string, err error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return "", "", markPermanentBootstrapError(fmt.Errorf("failed to create download request for %q: %w", url, err))
	}
	req.Header.Set("Accept", "*/*")
	req.Header.Set("User-Agent", bootstrapDownloaderUserAgent)

	resp, err := cfg.httpClient.Do(req)
	if err != nil {
		requestErr := fmt.Errorf("failed to download ONNX Runtime archive from %q: %w", url, err)
		if isBootstrapRedirectPolicyError(err) {
			return "", "", markPermanentBootstrapError(requestErr)
		}
		return "", "", requestErr
	}
	responseClosed := false
	defer func() {
		if responseClosed {
			return
		}
		if closeErr := resp.Body.Close(); closeErr != nil {
			closeErr = fmt.Errorf("failed to close download response body for %q: %w", url, closeErr)
			if err == nil {
				err = closeErr
			} else {
				err = errors.Join(err, closeErr)
			}
		}
	}()

	if resp.StatusCode != http.StatusOK {
		snippetBytes, snippetErr := io.ReadAll(io.LimitReader(resp.Body, 512))
		snippet := strings.TrimSpace(string(snippetBytes))
		var statusErr error
		if snippet != "" {
			statusErr = fmt.Errorf("failed to download ONNX Runtime archive from %q: HTTP %d: %s", url, resp.StatusCode, snippet)
		} else {
			statusErr = fmt.Errorf("failed to download ONNX Runtime archive from %q: HTTP %d", url, resp.StatusCode)
		}
		if snippetErr != nil {
			statusErr = errors.Join(statusErr, fmt.Errorf("failed to read ONNX Runtime archive error response body snippet: %w", snippetErr))
		}
		if !isRetryableBootstrapHTTPStatus(resp.StatusCode) {
			return "", "", markPermanentBootstrapError(statusErr)
		}
		return "", "", statusErr
	}

	if err := os.MkdirAll(cfg.cacheDir, secureDirectoryPermission); err != nil {
		return "", "", markPermanentBootstrapError(fmt.Errorf("failed to create cache directory %q: %w", cfg.cacheDir, err))
	}

	tmpFile, err := os.CreateTemp(cfg.cacheDir, "onnxruntime-*.archive")
	if err != nil {
		return "", "", markPermanentBootstrapError(fmt.Errorf("failed to create temporary archive file: %w", err))
	}
	tmpPath := tmpFile.Name()
	archivePath = tmpPath
	tmpFileClosed := false
	defer func() {
		if !tmpFileClosed {
			if closeErr := tmpFile.Close(); closeErr != nil {
				err = errors.Join(err, fmt.Errorf("failed to close temporary archive file %q: %w", tmpPath, closeErr))
			}
		}
		if err != nil {
			if removeErr := os.Remove(tmpPath); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
				err = errors.Join(err, fmt.Errorf("failed to remove temporary archive %q: %w", tmpPath, removeErr))
			}
		}
	}()

	downloadLimit := cfg.maxDownloadSize
	if downloadLimit <= 0 {
		downloadLimit = maxDownloadBytes
	}

	if resp.ContentLength > downloadLimit {
		err = markPermanentBootstrapError(fmt.Errorf("downloaded ONNX Runtime archive exceeds maximum size limit: content-length=%d limit=%d", resp.ContentLength, downloadLimit))
		return "", "", err
	}

	hasher := sha256.New()
	limitedBody := io.LimitReader(resp.Body, downloadLimit+1)
	written, copyErr := io.Copy(io.MultiWriter(tmpFile, hasher), limitedBody)
	if copyErr != nil {
		err = fmt.Errorf("failed to write ONNX Runtime archive to %q: %w", archivePath, copyErr)
		return "", "", err
	}
	if written > downloadLimit {
		err = fmt.Errorf("downloaded ONNX Runtime archive exceeds maximum size limit: bytes=%d limit=%d", written, downloadLimit)
		return "", "", err
	}
	if written == 0 {
		err = fmt.Errorf("downloaded ONNX Runtime archive is empty")
		return "", "", err
	}

	checksum = hex.EncodeToString(hasher.Sum(nil))
	closeErr := tmpFile.Close()
	tmpFileClosed = true
	if closeErr != nil {
		err = fmt.Errorf("failed to close temporary archive file %q: %w", tmpPath, closeErr)
		return "", "", err
	}
	closeErr = resp.Body.Close()
	responseClosed = true
	if closeErr != nil {
		err = fmt.Errorf("failed to close download response body for %q: %w", url, closeErr)
		return "", "", err
	}
	return archivePath, checksum, nil
}

func extractArchiveFile(archivePath, destinationDir, extension, libraryGlob string) (archiveExtractionReport, error) {
	switch extension {
	case "tgz":
		return extractTGZArchive(archivePath, destinationDir, libraryGlob)
	case "zip":
		return extractZIPArchive(archivePath, destinationDir, libraryGlob)
	default:
		return archiveExtractionReport{}, fmt.Errorf("unsupported archive extension %q", extension)
	}
}

func extractTGZArchive(archivePath, destinationDir, libraryGlob string) (report archiveExtractionReport, err error) {
	// #nosec G304 -- archivePath is generated internally (downloadRuntimeArchive) and not user-controlled input.
	archiveFile, err := os.Open(archivePath)
	if err != nil {
		return archiveExtractionReport{}, fmt.Errorf("failed to open archive %q: %w", archivePath, err)
	}
	defer func() {
		if closeErr := archiveFile.Close(); closeErr != nil {
			closeErr = fmt.Errorf("failed to close archive %q: %w", archivePath, closeErr)
			if err == nil {
				err = closeErr
			} else {
				err = errors.Join(err, closeErr)
			}
		}
	}()

	gzipReader, err := gzip.NewReader(archiveFile)
	if err != nil {
		return archiveExtractionReport{}, fmt.Errorf("failed to read gzip archive %q: %w", archivePath, err)
	}
	defer func() {
		if closeErr := gzipReader.Close(); closeErr != nil {
			closeErr = fmt.Errorf("gzip integrity check failed for %q: %w", archivePath, closeErr)
			if err == nil {
				err = closeErr
			} else {
				err = errors.Join(err, closeErr)
			}
		}
	}()

	tarReader := tar.NewReader(gzipReader)
	regularFiles := 0
	var totalExtracted int64
	report = archiveExtractionReport{}

	for {
		header, err := tarReader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return archiveExtractionReport{}, fmt.Errorf("failed to read tar entry from %q: %w", archivePath, err)
		}

		targetPath, err := secureArchiveJoin(destinationDir, header.Name)
		if err != nil {
			return archiveExtractionReport{}, err
		}

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(targetPath, secureDirectoryPermission); err != nil {
				return archiveExtractionReport{}, fmt.Errorf("failed to create directory %q: %w", targetPath, err)
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(targetPath), secureDirectoryPermission); err != nil {
				return archiveExtractionReport{}, fmt.Errorf("failed to create parent directory for %q: %w", targetPath, err)
			}

			if header.Size < 0 {
				return archiveExtractionReport{}, fmt.Errorf("invalid negative tar entry size for %q", header.Name)
			}

			mode := safeArchiveFileMode(header.FileInfo().Mode())
			if mode == 0 {
				mode = 0o644
			}
			// #nosec G304 -- targetPath is constrained by secureArchiveJoin to stay under destinationDir.
			outFile, err := os.OpenFile(targetPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, mode)
			if err != nil {
				return archiveExtractionReport{}, fmt.Errorf("failed to create extracted file %q: %w", targetPath, err)
			}

			if copyErr := copyExtractedFile(outFile, tarReader, header.Size, &totalExtracted, targetPath); copyErr != nil {
				if closeErr := outFile.Close(); closeErr != nil {
					return archiveExtractionReport{}, errors.Join(copyErr, fmt.Errorf("failed to close extracted file %q: %w", targetPath, closeErr))
				}
				return archiveExtractionReport{}, copyErr
			}
			if err := outFile.Close(); err != nil {
				return archiveExtractionReport{}, fmt.Errorf("failed to close extracted file %q: %w", targetPath, err)
			}
			regularFiles++
		case tar.TypeXHeader, tar.TypeXGlobalHeader:
			continue
		case tar.TypeSymlink, tar.TypeLink:
			report.skippedLinkEntries++
			if libraryGlob != "" {
				baseName := path.Base(header.Name)
				matched, matchErr := path.Match(libraryGlob, baseName)
				if matchErr != nil {
					emitDiagnostic(
						context.Background(),
						slog.LevelWarn,
						"bootstrap tar library glob match failed",
						slog.String("archive_entry", header.Name),
						slog.String("library_glob", libraryGlob),
						slog.Any("error", matchErr),
					)
				} else if matched {
					report.skippedLibraryLinkEntries++
					if len(report.skippedLibraryLinkExamples) < 3 {
						report.skippedLibraryLinkExamples = append(report.skippedLibraryLinkExamples, header.Name)
					}
				}
			}
			emitDiagnostic(
				context.Background(),
				slog.LevelWarn,
				"bootstrap tar link entry skipped",
				slog.String("archive_entry", header.Name),
				slog.Int("entry_type", int(header.Typeflag)),
			)
			continue
		default:
			// Skip non-regular archive entries (device files, FIFOs, etc.) for safety.
			emitDiagnostic(
				context.Background(),
				slog.LevelWarn,
				"bootstrap tar archive entry skipped",
				slog.String("archive_entry", header.Name),
				slog.Int("entry_type", int(header.Typeflag)),
			)
			continue
		}
	}

	if regularFiles == 0 {
		return archiveExtractionReport{}, fmt.Errorf("archive %q did not contain regular files", archivePath)
	}

	return report, nil
}

func extractZIPArchive(archivePath, destinationDir, libraryGlob string) (report archiveExtractionReport, err error) {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return archiveExtractionReport{}, fmt.Errorf("failed to open ZIP archive %q: %w", archivePath, err)
	}
	defer func() {
		if closeErr := reader.Close(); closeErr != nil {
			closeErr = fmt.Errorf("failed to close ZIP archive %q: %w", archivePath, closeErr)
			if err == nil {
				err = closeErr
			} else {
				err = errors.Join(err, closeErr)
			}
		}
	}()

	regularFiles := 0
	var totalExtracted int64
	report = archiveExtractionReport{}
	for _, entry := range reader.File {
		targetPath, err := secureArchiveJoin(destinationDir, entry.Name)
		if err != nil {
			return archiveExtractionReport{}, err
		}

		mode := entry.Mode()
		if mode.IsDir() {
			if err := os.MkdirAll(targetPath, secureDirectoryPermission); err != nil {
				return archiveExtractionReport{}, fmt.Errorf("failed to create directory %q: %w", targetPath, err)
			}
			continue
		}
		if mode&os.ModeSymlink != 0 {
			report.skippedLinkEntries++
			if libraryGlob != "" {
				baseName := path.Base(entry.Name)
				matched, matchErr := path.Match(libraryGlob, baseName)
				if matchErr != nil {
					emitDiagnostic(
						context.Background(),
						slog.LevelWarn,
						"bootstrap ZIP library glob match failed",
						slog.String("archive_entry", entry.Name),
						slog.String("library_glob", libraryGlob),
						slog.Any("error", matchErr),
					)
				} else if matched {
					report.skippedLibraryLinkEntries++
					if len(report.skippedLibraryLinkExamples) < 3 {
						report.skippedLibraryLinkExamples = append(report.skippedLibraryLinkExamples, entry.Name)
					}
				}
			}
			emitDiagnostic(
				context.Background(),
				slog.LevelWarn,
				"bootstrap ZIP symlink entry skipped",
				slog.String("archive_entry", entry.Name),
				slog.String("entry_type", "symlink"),
			)
			continue
		}

		if err := os.MkdirAll(filepath.Dir(targetPath), secureDirectoryPermission); err != nil {
			return archiveExtractionReport{}, fmt.Errorf("failed to create parent directory for %q: %w", targetPath, err)
		}

		rc, err := entry.Open()
		if err != nil {
			return archiveExtractionReport{}, fmt.Errorf("failed to open ZIP entry %q: %w", entry.Name, err)
		}

		filePerm := safeArchiveFileMode(mode)
		if filePerm == 0 {
			filePerm = 0o644
		}
		// #nosec G304 -- targetPath is constrained by secureArchiveJoin to stay under destinationDir.
		outFile, err := os.OpenFile(targetPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, filePerm)
		if err != nil {
			createErr := fmt.Errorf("failed to create extracted file %q: %w", targetPath, err)
			if closeErr := rc.Close(); closeErr != nil {
				return archiveExtractionReport{}, errors.Join(createErr, fmt.Errorf("failed to close ZIP entry %q: %w", entry.Name, closeErr))
			}
			return archiveExtractionReport{}, createErr
		}

		if entry.UncompressedSize64 > math.MaxInt64 {
			sizeErr := fmt.Errorf("ZIP entry %q size exceeds supported range", entry.Name)
			if closeErr := outFile.Close(); closeErr != nil {
				sizeErr = errors.Join(sizeErr, fmt.Errorf("failed to close extracted file %q: %w", targetPath, closeErr))
			}
			if closeErr := rc.Close(); closeErr != nil {
				sizeErr = errors.Join(sizeErr, fmt.Errorf("failed to close ZIP entry %q: %w", entry.Name, closeErr))
			}
			return archiveExtractionReport{}, sizeErr
		}
		// #nosec G115 -- upper-bound checked against math.MaxInt64 immediately above.
		entrySize := int64(entry.UncompressedSize64)
		if copyErr := copyExtractedFile(outFile, rc, entrySize, &totalExtracted, targetPath); copyErr != nil {
			if closeErr := outFile.Close(); closeErr != nil {
				copyErr = errors.Join(copyErr, fmt.Errorf("failed to close extracted file %q: %w", targetPath, closeErr))
			}
			if closeErr := rc.Close(); closeErr != nil {
				copyErr = errors.Join(copyErr, fmt.Errorf("failed to close ZIP entry %q: %w", entry.Name, closeErr))
			}
			return archiveExtractionReport{}, copyErr
		}

		if err := outFile.Close(); err != nil {
			closeErr := fmt.Errorf("failed to close extracted file %q: %w", targetPath, err)
			if rcCloseErr := rc.Close(); rcCloseErr != nil {
				closeErr = errors.Join(closeErr, fmt.Errorf("failed to close ZIP entry %q: %w", entry.Name, rcCloseErr))
			}
			return archiveExtractionReport{}, closeErr
		}
		if err := rc.Close(); err != nil {
			return archiveExtractionReport{}, fmt.Errorf("failed to close ZIP entry %q: %w", entry.Name, err)
		}

		regularFiles++
	}

	if regularFiles == 0 {
		return archiveExtractionReport{}, fmt.Errorf("archive %q did not contain regular files", archivePath)
	}

	return report, nil
}

func writeBootstrapInstallManifest(
	installDir string,
	cfg bootstrapConfig,
	artifact runtimeArtifact,
	archiveChecksum string,
	checksumVerified bool,
) error {
	if !looksLikeSHA256(archiveChecksum) {
		return fmt.Errorf("cannot write bootstrap manifest with invalid archive checksum %q", archiveChecksum)
	}

	files, err := collectBootstrapInstallFiles(installDir, cfg.allowSharedCache)
	if err != nil {
		return fmt.Errorf("failed to hash extracted ONNX Runtime files: %w", err)
	}
	if len(files) == 0 {
		return fmt.Errorf("cannot write bootstrap manifest for empty install directory %q", installDir)
	}

	manifest := bootstrapInstallManifest{
		Version:          bootstrapManifestVersion,
		RuntimeVersion:   cfg.version,
		Platform:         artifact.platform,
		ArchiveSHA256:    archiveChecksum,
		ChecksumVerified: checksumVerified,
		Files:            files,
	}
	encoded, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode bootstrap manifest: %w", err)
	}
	encoded = append(encoded, '\n')

	manifestPath := filepath.Join(installDir, bootstrapManifestFilename)
	// #nosec G304 -- manifestPath is rooted in the internal staging directory.
	file, err := os.OpenFile(manifestPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, secureLockFilePermission)
	if err != nil {
		return fmt.Errorf("failed to create bootstrap manifest %q: %w", manifestPath, err)
	}
	if _, err := file.Write(encoded); err != nil {
		writeErr := fmt.Errorf("failed to write bootstrap manifest %q: %w", manifestPath, err)
		if closeErr := file.Close(); closeErr != nil {
			return errors.Join(writeErr, fmt.Errorf("failed to close bootstrap manifest %q: %w", manifestPath, closeErr))
		}
		return writeErr
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("failed to close bootstrap manifest %q: %w", manifestPath, err)
	}
	return nil
}

func validateCachedRuntimeInstall(
	cfg bootstrapConfig,
	artifact runtimeArtifact,
	installDir string,
) (string, error) {
	info, err := os.Lstat(installDir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", markCacheValidationError(
				cacheValidationMissing,
				fmt.Errorf("cached ONNX Runtime install %q is missing: %w: %w", installDir, ErrSharedLibraryNotFound, err),
			)
		}
		return "", fmt.Errorf("failed to inspect cached ONNX Runtime install %q: %w", installDir, err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return "", markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("cached ONNX Runtime install must not be a symbolic link: %q", installDir),
		)
	}
	if !info.IsDir() {
		return "", markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("cached ONNX Runtime install is not a directory: %q", installDir),
		)
	}
	if err := validateBootstrapPathOwnershipAndMode(installDir, info, cfg.allowSharedCache); err != nil {
		return "", markCacheValidationError(cacheValidationConfirmedInvalid, err)
	}

	memoKey := bootstrapValidationKey{
		installDir:       installDir,
		runtimeVersion:   cfg.version,
		platform:         artifact.platform,
		expectedSHA256:   cfg.expectedSHA256,
		allowSharedCache: cfg.allowSharedCache,
	}
	fingerprint := bootstrapInstallFingerprint(installDir)
	if fingerprint != "" {
		if cached, ok := bootstrapValidatedInstalls.Load(memoKey); ok {
			if memo, ok := cached.(bootstrapValidatedInstall); ok && memo.fingerprint == fingerprint {
				return memo.libraryPath, nil
			}
		}
	}

	manifest, err := readBootstrapInstallManifest(installDir, cfg.allowSharedCache)
	if err != nil {
		return "", err
	}
	if manifest.Version != bootstrapManifestVersion {
		return "", markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("unsupported bootstrap manifest version %d in %q", manifest.Version, installDir),
		)
	}
	if manifest.RuntimeVersion != cfg.version {
		return "", markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("bootstrap manifest runtime version = %q, want %q", manifest.RuntimeVersion, cfg.version),
		)
	}
	if manifest.Platform != artifact.platform {
		return "", markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("bootstrap manifest platform = %q, want %q", manifest.Platform, artifact.platform),
		)
	}
	if !looksLikeSHA256(manifest.ArchiveSHA256) {
		return "", markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("bootstrap manifest contains invalid archive checksum %q", manifest.ArchiveSHA256),
		)
	}
	if cfg.expectedSHA256 != "" {
		if !manifest.ChecksumVerified {
			return "", markCacheValidationError(
				cacheValidationConfirmedInvalid,
				fmt.Errorf("cached runtime was installed without a verified archive checksum"),
			)
		}
		if manifest.ArchiveSHA256 != cfg.expectedSHA256 {
			return "", markCacheValidationError(
				cacheValidationConfirmedInvalid,
				fmt.Errorf(
					"cached runtime archive checksum %s does not match expected checksum %s",
					manifest.ArchiveSHA256,
					cfg.expectedSHA256,
				),
			)
		}
	}

	actualFiles, err := collectBootstrapInstallFiles(installDir, cfg.allowSharedCache)
	if err != nil {
		return "", fmt.Errorf("failed to verify cached ONNX Runtime files: %w", err)
	}
	if err := compareBootstrapManifestFiles(manifest.Files, actualFiles); err != nil {
		return "", err
	}

	path, err := resolveExtractedLibraryPath(installDir, artifact)
	if err != nil {
		if cacheValidationDispositionForError(err) == cacheValidationOperational &&
			!errors.Is(err, ErrSharedLibraryNotFound) &&
			!errors.Is(err, os.ErrNotExist) {
			return "", err
		}
		return "", markCacheValidationError(cacheValidationConfirmedInvalid, err)
	}
	if fingerprint != "" {
		bootstrapValidatedInstalls.Store(memoKey, bootstrapValidatedInstall{
			libraryPath: path,
			fingerprint: fingerprint,
		})
	}
	return path, nil
}

// bootstrapInstallFingerprint summarizes an install tree from directory metadata
// alone. It never reads file contents, so it costs a few stats instead of a full
// SHA-256 pass. An empty result means "no usable fingerprint" and forces callers
// back onto full manifest verification.
func bootstrapInstallFingerprint(installDir string) string {
	hash := sha256.New()
	err := filepath.WalkDir(installDir, func(filePath string, _ os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		info, err := os.Lstat(filePath)
		if err != nil {
			return err
		}
		relativePath, err := filepath.Rel(installDir, filePath)
		if err != nil {
			return err
		}
		_, _ = fmt.Fprintf(
			hash,
			"%s\x00%d\x00%d\x00%d\n",
			filepath.ToSlash(relativePath),
			info.Mode(),
			info.Size(),
			info.ModTime().UnixNano(),
		)
		return nil
	})
	if err != nil {
		return ""
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func readBootstrapInstallManifest(
	installDir string,
	allowSharedCache bool,
) (bootstrapInstallManifest, error) {
	manifestPath := filepath.Join(installDir, bootstrapManifestFilename)
	info, err := os.Lstat(manifestPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return bootstrapInstallManifest{}, markCacheValidationError(
				cacheValidationConfirmedInvalid,
				fmt.Errorf("required bootstrap manifest %q is missing: %w", manifestPath, err),
			)
		}
		return bootstrapInstallManifest{}, fmt.Errorf("failed to inspect bootstrap manifest %q: %w", manifestPath, err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return bootstrapInstallManifest{}, markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("bootstrap manifest must not be a symbolic link: %q", manifestPath),
		)
	}
	if !info.Mode().IsRegular() {
		return bootstrapInstallManifest{}, markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("bootstrap manifest is not a regular file: %q", manifestPath),
		)
	}
	if info.Size() <= 0 || info.Size() > maxManifestBytes {
		return bootstrapInstallManifest{}, markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("bootstrap manifest size %d is outside the allowed range", info.Size()),
		)
	}
	if err := validateBootstrapPathOwnershipAndMode(manifestPath, info, allowSharedCache); err != nil {
		return bootstrapInstallManifest{}, markCacheValidationError(cacheValidationConfirmedInvalid, err)
	}

	// #nosec G304 -- manifestPath is rooted in the configured cache install directory.
	encoded, err := os.ReadFile(manifestPath)
	if err != nil {
		return bootstrapInstallManifest{}, fmt.Errorf("failed to read bootstrap manifest %q: %w", manifestPath, err)
	}
	var manifest bootstrapInstallManifest
	if err := json.Unmarshal(encoded, &manifest); err != nil {
		return bootstrapInstallManifest{}, markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("failed to parse bootstrap manifest %q: %w", manifestPath, err),
		)
	}
	return manifest, nil
}

func collectBootstrapInstallFiles(
	installDir string,
	allowSharedCache bool,
) ([]bootstrapManifestFile, error) {
	manifestPath := filepath.Join(installDir, bootstrapManifestFilename)
	files := make([]bootstrapManifestFile, 0)

	err := filepath.WalkDir(installDir, func(filePath string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		info, err := os.Lstat(filePath)
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return markCacheValidationError(
				cacheValidationConfirmedInvalid,
				fmt.Errorf("cached install contains symbolic link %q", filePath),
			)
		}
		if info.IsDir() {
			if err := validateBootstrapPathOwnershipAndMode(filePath, info, allowSharedCache); err != nil {
				return markCacheValidationError(cacheValidationConfirmedInvalid, err)
			}
			return nil
		}
		if filePath == manifestPath {
			return nil
		}
		if !info.Mode().IsRegular() {
			return markCacheValidationError(
				cacheValidationConfirmedInvalid,
				fmt.Errorf("cached install contains non-regular file %q", filePath),
			)
		}
		if err := validateBootstrapPathOwnershipAndMode(filePath, info, allowSharedCache); err != nil {
			return markCacheValidationError(cacheValidationConfirmedInvalid, err)
		}

		digest, err := sha256File(filePath)
		if err != nil {
			return err
		}
		relativePath, err := filepath.Rel(installDir, filePath)
		if err != nil {
			return fmt.Errorf("failed to resolve manifest path for %q: %w", filePath, err)
		}
		files = append(files, bootstrapManifestFile{
			Path:   filepath.ToSlash(relativePath),
			SHA256: digest,
			Size:   info.Size(),
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(files, func(i, j int) bool {
		return files[i].Path < files[j].Path
	})
	return files, nil
}

func sha256File(filePath string) (digest string, err error) {
	// #nosec G304 -- callers constrain filePath to the install directory.
	file, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to open cached file %q: %w", filePath, err)
	}
	defer func() {
		if closeErr := file.Close(); closeErr != nil {
			closeErr = fmt.Errorf("failed to close cached file %q: %w", filePath, closeErr)
			if err == nil {
				err = closeErr
			} else {
				err = errors.Join(err, closeErr)
			}
		}
	}()

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", fmt.Errorf("failed to hash cached file %q: %w", filePath, err)
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func compareBootstrapManifestFiles(expected, actual []bootstrapManifestFile) error {
	if len(expected) != len(actual) {
		return markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("bootstrap manifest file count = %d, cached install file count = %d", len(expected), len(actual)),
		)
	}
	for i := range expected {
		if expected[i] != actual[i] {
			return markCacheValidationError(
				cacheValidationConfirmedInvalid,
				fmt.Errorf(
					"bootstrap manifest mismatch for cached file %q: expected sha256=%s size=%d, got path=%q sha256=%s size=%d",
					expected[i].Path,
					expected[i].SHA256,
					expected[i].Size,
					actual[i].Path,
					actual[i].SHA256,
					actual[i].Size,
				),
			)
		}
	}
	return nil
}

func resolveExtractedLibraryPath(installDir string, artifact runtimeArtifact) (string, error) {
	libDir := filepath.Join(installDir, "lib")

	var invalidCandidates []error
	var operationalCandidates []error
	trackCandidateError := func(path string, validationErr error) {
		if validationErr == nil {
			return
		}
		if errors.Is(validationErr, os.ErrNotExist) {
			return
		}
		candidateErr := fmt.Errorf("%s: %w", path, validationErr)
		if cacheValidationDispositionForError(validationErr) == cacheValidationConfirmedInvalid {
			invalidCandidates = append(invalidCandidates, candidateErr)
			return
		}
		operationalCandidates = append(operationalCandidates, candidateErr)
	}

	primaryPath := filepath.Join(libDir, artifact.primaryLibrary)
	if path, err := validateLibraryFile(primaryPath); err == nil {
		return path, nil
	} else {
		trackCandidateError(primaryPath, err)
	}

	matches, err := filepath.Glob(filepath.Join(libDir, artifact.libraryGlob))
	if err != nil {
		return "", fmt.Errorf("failed to resolve ONNX Runtime library path: %w", err)
	}
	sort.Strings(matches)
	for _, match := range matches {
		path, err := validateLibraryFile(match)
		if err == nil {
			return path, nil
		}
		trackCandidateError(match, err)
	}

	if len(operationalCandidates) > 0 {
		return "", fmt.Errorf(
			"failed to validate ONNX Runtime shared library candidates in %q: %w",
			libDir,
			errors.Join(operationalCandidates...),
		)
	}
	if len(invalidCandidates) > 0 {
		return "", markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("found ONNX Runtime shared library candidates in %q but none are valid: %w", libDir, errors.Join(invalidCandidates...)),
		)
	}

	return "", markCacheValidationError(cacheValidationConfirmedInvalid, ErrSharedLibraryNotFound)
}

func validateBootstrapDirectoryTrust(path string, allowSharedCache bool) error {
	info, err := os.Lstat(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return markCacheValidationError(
				cacheValidationMissing,
				fmt.Errorf("bootstrap directory %q is missing: %w", path, err),
			)
		}
		return fmt.Errorf("failed to inspect bootstrap directory %q: %w", path, err)
	}
	// A trust problem with the shared cache directory says nothing about
	// whether any specific install underneath it is corrupt, and installDir
	// is computed by joining onto this path — so these failures stay
	// unmarked (operational): the caller propagates the error and leaves
	// installDir untouched, rather than treating an untrusted parent as
	// grounds to delete a child install it hasn't actually inspected.
	if info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("bootstrap directory must not be a symbolic link: %q", path)
	}
	if !info.IsDir() {
		return fmt.Errorf("bootstrap path is not a directory: %q", path)
	}
	if err := validateBootstrapPathOwnershipAndMode(path, info, allowSharedCache); err != nil {
		return err
	}
	return nil
}

func validateExplicitLibraryFile(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", fmt.Errorf("library path is empty: %w", ErrInvalidArgument)
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("failed to resolve absolute path for %q: %w", path, err)
	}

	resolvedPath, err := filepath.EvalSymlinks(absPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf(
				"failed to resolve explicit library path %q: %w: %w",
				absPath,
				ErrSharedLibraryNotFound,
				err,
			)
		}
		return "", fmt.Errorf("failed to resolve explicit library path %q: %w", absPath, err)
	}

	return validateLibraryFile(resolvedPath)
}

func validateLibraryFile(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", fmt.Errorf("library path is empty: %w", ErrInvalidArgument)
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("failed to resolve absolute path for %q: %w", path, err)
	}

	info, err := os.Lstat(absPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf(
				"failed to inspect library file %q: %w: %w",
				absPath,
				ErrSharedLibraryNotFound,
				err,
			)
		}
		return "", fmt.Errorf("failed to inspect library file %q: %w", absPath, err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return "", markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("library path must not be a symbolic link: %q", absPath),
		)
	}
	if info.IsDir() {
		return "", markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("library path points to a directory: %q", absPath),
		)
	}
	if !info.Mode().IsRegular() {
		return "", markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("library path is not a regular file: %q", absPath),
		)
	}
	if info.Size() == 0 {
		return "", markCacheValidationError(
			cacheValidationConfirmedInvalid,
			fmt.Errorf("library file is empty: %q", absPath),
		)
	}

	return absPath, nil
}

func withProcessFileLock(
	lockPath string,
	allowSharedCache bool,
	fn func() error,
) (err error) {
	if fn == nil {
		return fmt.Errorf("bootstrap lock callback is nil: %w", ErrInvalidArgument)
	}

	lockDir := filepath.Dir(lockPath)
	if err := os.MkdirAll(lockDir, secureDirectoryPermission); err != nil {
		return fmt.Errorf("failed to create lock directory for %q: %w", lockPath, err)
	}
	if err := validateBootstrapDirectoryTrust(lockDir, allowSharedCache); err != nil {
		return fmt.Errorf("bootstrap lock directory is not trusted: %w", err)
	}
	if info, err := os.Lstat(lockPath); err == nil {
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("bootstrap lock file must not be a symbolic link: %q", lockPath)
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("bootstrap lock path is not a regular file: %q", lockPath)
		}
		if err := validateBootstrapPathOwnershipAndMode(lockPath, info, allowSharedCache); err != nil {
			return err
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("failed to inspect bootstrap lock file %q: %w", lockPath, err)
	}

	// #nosec G304 -- lockPath is constructed from configured cache directory and fixed internal suffix.
	file, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, secureLockFilePermission)
	if err != nil {
		return fmt.Errorf("failed to open lock file %q: %w", lockPath, err)
	}
	if info, err := os.Lstat(lockPath); err != nil {
		_ = file.Close()
		return fmt.Errorf("failed to inspect bootstrap lock file %q after opening: %w", lockPath, err)
	} else if err := validateBootstrapPathOwnershipAndMode(lockPath, info, allowSharedCache); err != nil {
		_ = file.Close()
		return err
	}

	start := time.Now()
	nextLogAt := start.Add(bootstrapLockLogInterval)
	for {
		lockErr := lockFile(file)
		if lockErr == nil {
			break
		}
		if !isLockWouldBlock(lockErr) {
			acquireErr := fmt.Errorf("failed to acquire lock %q: %w", lockPath, lockErr)
			if closeErr := file.Close(); closeErr != nil {
				return errors.Join(acquireErr, fmt.Errorf("failed to close lock file %q: %w", lockPath, closeErr))
			}
			return acquireErr
		}
		waited := time.Since(start)
		if waited >= bootstrapLockAcquireTimeout {
			timeoutErr := fmt.Errorf("timed out acquiring lock %q after %s", lockPath, bootstrapLockAcquireTimeout)
			if closeErr := file.Close(); closeErr != nil {
				return errors.Join(timeoutErr, fmt.Errorf("failed to close lock file %q: %w", lockPath, closeErr))
			}
			return timeoutErr
		}
		if time.Now().After(nextLogAt) {
			emitDiagnostic(
				context.Background(),
				slog.LevelWarn,
				"waiting for bootstrap lock",
				slog.String("path", lockPath),
				slog.Duration("wait_duration", waited),
			)
			nextLogAt = time.Now().Add(bootstrapLockLogInterval)
		}
		time.Sleep(bootstrapLockRetryInterval)
	}

	defer func() {
		unlockErr := unlockFile(file)
		if unlockErr != nil {
			unlockErr = fmt.Errorf("failed to release lock %q: %w", lockPath, unlockErr)
		}
		closeErr := file.Close()
		if closeErr != nil {
			closeErr = fmt.Errorf("failed to close lock file %q: %w", lockPath, closeErr)
		}
		err = errors.Join(err, unlockErr, closeErr)
	}()

	return fn()
}

func secureArchiveJoin(baseDir, archivePath string) (string, error) {
	archivePath = strings.TrimSpace(archivePath)
	if archivePath == "" {
		return "", fmt.Errorf("invalid empty archive entry path")
	}

	normalized := strings.ReplaceAll(archivePath, "\\", "/")
	if strings.HasPrefix(normalized, "/") {
		return "", fmt.Errorf("invalid absolute archive entry path %q", archivePath)
	}
	if len(normalized) >= 2 && ((normalized[0] >= 'A' && normalized[0] <= 'Z') || (normalized[0] >= 'a' && normalized[0] <= 'z')) && normalized[1] == ':' {
		return "", fmt.Errorf("invalid archive entry path with drive letter %q", archivePath)
	}

	cleaned := filepath.Clean(normalized)
	if cleaned == "." {
		return "", fmt.Errorf("invalid archive entry path %q", archivePath)
	}
	if cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("unsafe archive entry path %q", archivePath)
	}

	targetPath := filepath.Join(baseDir, cleaned)
	relPath, err := filepath.Rel(baseDir, targetPath)
	if err != nil {
		return "", fmt.Errorf("failed to resolve archive path %q: %w", archivePath, err)
	}
	if relPath == ".." || strings.HasPrefix(relPath, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("unsafe archive entry path %q", archivePath)
	}

	return targetPath, nil
}

func defaultBootstrapCacheDir() string {
	cacheDir, err := bootstrapUserCacheDir()
	if err == nil && cacheDir != "" {
		return filepath.Join(cacheDir, "onnx-purego", "onnxruntime")
	}

	fallback := filepath.Join(os.TempDir(), "onnx-purego", "onnxruntime")
	// Decide under the Once but emit after it returns. sync.Once holds an internal
	// mutex for the whole callback, so a handler that re-enters bootstrap from
	// inside Do would deadlock against itself on the same goroutine.
	var emitFallbackWarning func()
	bootstrapCacheFallbackWarnOnce.Do(func() {
		if err != nil {
			emitFallbackWarning = func() {
				emitDiagnostic(
					context.Background(),
					slog.LevelWarn,
					"bootstrap user cache lookup failed; using temporary cache",
					slog.String("path", fallback),
					slog.Any("error", err),
				)
			}
			return
		}
		emitFallbackWarning = func() {
			emitDiagnostic(
				context.Background(),
				slog.LevelWarn,
				"bootstrap user cache path empty; using temporary cache",
				slog.String("path", fallback),
			)
		}
	})
	if emitFallbackWarning != nil {
		emitFallbackWarning()
	}
	return fallback
}

func redactedBootstrapURL(rawURL string) string {
	parsedURL, err := url.Parse(rawURL)
	if err != nil {
		return "<invalid URL>"
	}
	return parsedURL.Redacted()
}

func normalizeRuntimeVersion(version string) (string, error) {
	version = strings.TrimSpace(version)
	version = strings.TrimPrefix(version, "v")
	if version == "" {
		return "", fmt.Errorf("ONNX Runtime version is empty: %w", ErrInvalidArgument)
	}

	parts := strings.Split(version, ".")
	if len(parts) != 3 {
		return "", fmt.Errorf("ONNX Runtime version must have format x.y.z, got %q: %w", version, ErrInvalidArgument)
	}

	canonicalParts := make([]string, len(parts))
	for i, part := range parts {
		if part == "" {
			return "", fmt.Errorf("ONNX Runtime version must have format x.y.z, got %q: %w", version, ErrInvalidArgument)
		}
		value, err := strconv.Atoi(part)
		if err != nil {
			return "", fmt.Errorf("ONNX Runtime version must have numeric segments, got %q: %w", version, ErrInvalidArgument)
		}
		if value < 0 {
			return "", fmt.Errorf("ONNX Runtime version segments must be nonnegative, got %q: %w", version, ErrInvalidArgument)
		}
		canonicalParts[i] = strconv.Itoa(value)
	}

	return strings.Join(canonicalParts, "."), nil
}

func parseBootstrapBoolEnv(name string) (bool, error) {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return false, nil
	}

	parsed, err := strconv.ParseBool(value)
	if err == nil {
		return parsed, nil
	}

	switch strings.ToLower(value) {
	case "yes", "y", "on":
		return true, nil
	case "no", "n", "off":
		return false, nil
	default:
		return false, fmt.Errorf(
			"invalid boolean value for %s: %q (expected true/false, 1/0, yes/no, y/n, on/off): %w",
			name,
			value,
			ErrInvalidArgument,
		)
	}
}

func copyExtractedFile(dst io.Writer, src io.Reader, expectedSize int64, totalExtracted *int64, targetPath string) error {
	if expectedSize < 0 {
		return fmt.Errorf("invalid negative size while extracting %q", targetPath)
	}
	if expectedSize > maxExtractedFileBytes {
		return fmt.Errorf("refusing to extract %q: entry size %d exceeds limit %d", targetPath, expectedSize, maxExtractedFileBytes)
	}
	if totalExtracted != nil && *totalExtracted+expectedSize > maxExtractedTotalBytes {
		return fmt.Errorf("refusing to extract %q: total extracted size would exceed limit %d", targetPath, maxExtractedTotalBytes)
	}

	limitedSrc := io.LimitReader(src, expectedSize+1)
	// #nosec G110 -- extraction is bounded by per-file and cumulative size checks above.
	written, err := io.Copy(dst, limitedSrc)
	if err != nil {
		return fmt.Errorf("failed to extract file %q: %w", targetPath, err)
	}
	if written != expectedSize {
		return fmt.Errorf("unexpected extracted size for %q: expected %d bytes, got %d", targetPath, expectedSize, written)
	}

	if totalExtracted != nil {
		*totalExtracted += written
	}
	return nil
}
