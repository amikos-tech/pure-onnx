package ort

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log"
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
	DefaultOnnxRuntimeVersion = "1.23.1"

	defaultBootstrapBaseURL = "https://github.com/microsoft/onnxruntime/releases/download"

	secureDirectoryPermission = 0o750
	secureLockFilePermission  = 0o600

	maxExtractedFileBytes  int64 = 1 << 30 // 1 GiB
	maxExtractedTotalBytes int64 = 4 << 30 // 4 GiB
	maxDownloadBytes       int64 = 1 << 30 // 1 GiB
)

var errSharedLibraryNotFound = errors.New("ONNX Runtime shared library not found")
var bootstrapCacheFallbackWarnOnce sync.Once
var bootstrapInitMu sync.Mutex

var (
	bootstrapLockAcquireTimeout = 2 * time.Minute
	bootstrapLockRetryInterval  = 200 * time.Millisecond
	bootstrapLockLogInterval    = 5 * time.Second
)

// BootstrapOption configures EnsureOnnxRuntimeSharedLibrary.
type BootstrapOption func(*bootstrapConfig) error

type bootstrapConfig struct {
	libraryPath     string
	cacheDir        string
	version         string
	disableDownload bool
	expectedSHA256  string
	baseURL         string
	httpClient      *http.Client
	maxDownloadSize int64
	goos            string
	goarch          string
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

// WithBootstrapLibraryPath sets an explicit ONNX Runtime shared library path.
// When set, bootstrap download and cache resolution are bypassed.
func WithBootstrapLibraryPath(path string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		path = strings.TrimSpace(path)
		if path == "" {
			return fmt.Errorf("bootstrap library path cannot be empty")
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
			return fmt.Errorf("bootstrap cache directory cannot be empty")
		}
		cfg.cacheDir = dir
		return nil
	}
}

// WithBootstrapVersion sets the ONNX Runtime version to download (for example: 1.23.1).
func WithBootstrapVersion(version string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		version = strings.TrimSpace(version)
		if version == "" {
			return fmt.Errorf("bootstrap version cannot be empty")
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

// WithBootstrapExpectedSHA256 enforces an expected SHA256 checksum for the downloaded archive.
func WithBootstrapExpectedSHA256(checksum string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		checksum = strings.TrimSpace(strings.ToLower(checksum))
		if checksum == "" {
			return fmt.Errorf("expected SHA256 checksum cannot be empty")
		}
		if len(checksum) != 64 {
			return fmt.Errorf("expected SHA256 checksum must be 64 hex characters")
		}
		for _, r := range checksum {
			if (r < '0' || r > '9') && (r < 'a' || r > 'f') {
				return fmt.Errorf("expected SHA256 checksum must be hex characters (0-9, a-f)")
			}
		}
		cfg.expectedSHA256 = checksum
		return nil
	}
}

func withBootstrapBaseURL(baseURL string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		baseURL = strings.TrimSpace(baseURL)
		if baseURL == "" {
			return fmt.Errorf("bootstrap base URL cannot be empty")
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
			return fmt.Errorf("bootstrap HTTP client cannot be nil")
		}
		cfg.httpClient = client
		return nil
	}
}

// EnsureOnnxRuntimeSharedLibrary ensures an ONNX Runtime shared library is available
// and returns a resolved absolute path to it.
//
// This function is opt-in and does not change existing explicit-path behavior.
func EnsureOnnxRuntimeSharedLibrary(opts ...BootstrapOption) (string, error) {
	cfg, err := resolveBootstrapConfig(opts...)
	if err != nil {
		return "", err
	}

	if cfg.libraryPath != "" {
		return validateLibraryFile(cfg.libraryPath)
	}

	artifact, err := resolveRuntimeArtifact(cfg.goos, cfg.goarch)
	if err != nil {
		return "", err
	}

	installDir := filepath.Join(cfg.cacheDir, artifact.archiveName(cfg.version))
	if path, resolveErr := resolveExtractedLibraryPath(installDir, artifact); resolveErr == nil {
		return path, nil
	} else if !errors.Is(resolveErr, errSharedLibraryNotFound) {
		return "", resolveErr
	}

	if cfg.disableDownload {
		return "", fmt.Errorf("ONNX Runtime library not found in cache and download is disabled: %s", installDir)
	}

	if err := os.MkdirAll(cfg.cacheDir, secureDirectoryPermission); err != nil {
		return "", fmt.Errorf("failed to create bootstrap cache directory %q: %w", cfg.cacheDir, err)
	}

	lockPath := filepath.Join(cfg.cacheDir, ".locks", fmt.Sprintf("%s-%s.lock", artifact.platform, cfg.version))
	var resolvedPath string
	if err := withProcessFileLock(lockPath, func() error {
		if path, resolveErr := resolveExtractedLibraryPath(installDir, artifact); resolveErr == nil {
			resolvedPath = path
			return nil
		} else if !errors.Is(resolveErr, errSharedLibraryNotFound) {
			return resolveErr
		}

		if err := downloadAndInstallRuntime(cfg, artifact, installDir); err != nil {
			return err
		}

		path, resolveErr := resolveExtractedLibraryPath(installDir, artifact)
		if resolveErr != nil {
			return fmt.Errorf("bootstrap completed but shared library could not be resolved: %w", resolveErr)
		}
		resolvedPath = path
		return nil
	}); err != nil {
		return "", err
	}

	return resolvedPath, nil
}

// InitializeEnvironmentWithBootstrap resolves a shared library path via bootstrap,
// sets it on the runtime, and initializes the ONNX Runtime environment.
func InitializeEnvironmentWithBootstrap(opts ...BootstrapOption) error {
	path, err := EnsureOnnxRuntimeSharedLibrary(opts...)
	if err != nil {
		return err
	}

	bootstrapInitMu.Lock()
	defer bootstrapInitMu.Unlock()

	if err := SetSharedLibraryPath(path); err != nil {
		// Another goroutine may have initialized after path resolution.
		mu.Lock()
		alreadyInitialized := refCount > 0
		currentPath := libPath
		mu.Unlock()
		if !alreadyInitialized || currentPath != path {
			return err
		}
	}

	return InitializeEnvironment()
}

func resolveBootstrapConfig(opts ...BootstrapOption) (bootstrapConfig, error) {
	disableDownload, err := parseBootstrapBoolEnv("ONNXRUNTIME_DISABLE_DOWNLOAD")
	if err != nil {
		return bootstrapConfig{}, err
	}

	cfg := bootstrapConfig{
		libraryPath:     strings.TrimSpace(os.Getenv("ONNXRUNTIME_LIB_PATH")),
		cacheDir:        strings.TrimSpace(os.Getenv("ONNXRUNTIME_CACHE_DIR")),
		version:         strings.TrimSpace(os.Getenv("ONNXRUNTIME_VERSION")),
		disableDownload: disableDownload,
		baseURL:         defaultBootstrapBaseURL,
		httpClient: &http.Client{
			Timeout: 2 * time.Minute,
		},
		maxDownloadSize: maxDownloadBytes,
		goos:            runtime.GOOS,
		goarch:          runtime.GOARCH,
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
		return bootstrapConfig{}, fmt.Errorf("bootstrap cache directory is empty")
	}
	cfg.cacheDir = filepath.Clean(cfg.cacheDir)

	if strings.TrimSpace(cfg.baseURL) == "" {
		return bootstrapConfig{}, fmt.Errorf("bootstrap base URL is empty")
	}
	cfg.baseURL = strings.TrimRight(strings.TrimSpace(cfg.baseURL), "/")
	if err := validateBootstrapBaseURL(cfg.baseURL); err != nil {
		return bootstrapConfig{}, err
	}

	if cfg.httpClient == nil {
		return bootstrapConfig{}, fmt.Errorf("bootstrap HTTP client cannot be nil")
	}
	if cfg.maxDownloadSize <= 0 {
		return bootstrapConfig{}, fmt.Errorf("bootstrap max download bytes must be > 0, got %d", cfg.maxDownloadSize)
	}

	return cfg, nil
}

func validateBootstrapBaseURL(baseURL string) error {
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return fmt.Errorf("invalid bootstrap base URL %q: %w", baseURL, err)
	}
	if parsed.Scheme == "" {
		return fmt.Errorf("bootstrap base URL %q must include a scheme", baseURL)
	}
	if parsed.Host == "" {
		return fmt.Errorf("bootstrap base URL %q must include a host", baseURL)
	}

	scheme := strings.ToLower(parsed.Scheme)
	if scheme == "https" {
		return nil
	}
	if scheme != "http" {
		return fmt.Errorf("bootstrap base URL %q uses unsupported scheme %q", baseURL, parsed.Scheme)
	}
	if !isLoopbackBootstrapHost(parsed.Hostname()) {
		return fmt.Errorf("bootstrap base URL %q must use https (http is allowed only for loopback hosts)", baseURL)
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

	return runtimeArtifact{}, fmt.Errorf("unsupported platform for ONNX Runtime bootstrap: GOOS=%s GOARCH=%s", goos, goarch)
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
	url := artifact.downloadURL(cfg.baseURL, cfg.version)
	archivePath, checksum, err := downloadRuntimeArchive(cfg, url)
	if err != nil {
		return err
	}
	defer func() {
		if removeErr := os.Remove(archivePath); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			log.Printf("WARNING: failed to remove temporary ONNX Runtime archive %q: %v", archivePath, removeErr)
		}
	}()

	if cfg.expectedSHA256 != "" && checksum != cfg.expectedSHA256 {
		return fmt.Errorf("download checksum mismatch: expected %s, got %s", cfg.expectedSHA256, checksum)
	}

	stagingRoot := installDir + fmt.Sprintf(".staging-%d", time.Now().UnixNano())
	if err := os.RemoveAll(stagingRoot); err != nil {
		return fmt.Errorf("failed to clean bootstrap staging directory %q: %w", stagingRoot, err)
	}
	if err := os.MkdirAll(stagingRoot, secureDirectoryPermission); err != nil {
		return fmt.Errorf("failed to create bootstrap staging directory %q: %w", stagingRoot, err)
	}
	defer func() {
		if removeErr := os.RemoveAll(stagingRoot); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			log.Printf("WARNING: failed to remove bootstrap staging directory %q: %v", stagingRoot, removeErr)
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
		if errors.Is(err, errSharedLibraryNotFound) {
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
			return errors.New(errMessage)
		}
		return err
	}

	if err := os.RemoveAll(installDir); err != nil {
		return fmt.Errorf("failed to remove previous ONNX Runtime install at %q: %w", installDir, err)
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

func downloadRuntimeArchive(cfg bootstrapConfig, url string) (archivePath string, checksum string, err error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return "", "", fmt.Errorf("failed to create download request for %q: %w", url, err)
	}

	resp, err := cfg.httpClient.Do(req)
	if err != nil {
		return "", "", fmt.Errorf("failed to download ONNX Runtime archive from %q: %w", url, err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		snippet = []byte(strings.TrimSpace(string(snippet)))
		if len(snippet) > 0 {
			return "", "", fmt.Errorf("failed to download ONNX Runtime archive from %q: HTTP %d: %s", url, resp.StatusCode, string(snippet))
		}
		return "", "", fmt.Errorf("failed to download ONNX Runtime archive from %q: HTTP %d", url, resp.StatusCode)
	}

	if err := os.MkdirAll(cfg.cacheDir, secureDirectoryPermission); err != nil {
		return "", "", fmt.Errorf("failed to create cache directory %q: %w", cfg.cacheDir, err)
	}

	tmpFile, err := os.CreateTemp(cfg.cacheDir, "onnxruntime-*.archive")
	if err != nil {
		return "", "", fmt.Errorf("failed to create temporary archive file: %w", err)
	}
	tmpPath := tmpFile.Name()
	archivePath = tmpPath
	success := false
	defer func() {
		closeErr := tmpFile.Close()
		if err == nil && closeErr != nil {
			err = closeErr
		}
		if !success {
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
		err = fmt.Errorf("downloaded ONNX Runtime archive exceeds maximum size limit: content-length=%d limit=%d", resp.ContentLength, downloadLimit)
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
	success = true
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

func extractTGZArchive(archivePath, destinationDir, libraryGlob string) (archiveExtractionReport, error) {
	// #nosec G304 -- archivePath is generated internally (downloadRuntimeArchive) and not user-controlled input.
	archiveFile, err := os.Open(archivePath)
	if err != nil {
		return archiveExtractionReport{}, fmt.Errorf("failed to open archive %q: %w", archivePath, err)
	}
	defer func() {
		_ = archiveFile.Close()
	}()

	gzipReader, err := gzip.NewReader(archiveFile)
	if err != nil {
		return archiveExtractionReport{}, fmt.Errorf("failed to read gzip archive %q: %w", archivePath, err)
	}
	defer func() {
		_ = gzipReader.Close()
	}()

	tarReader := tar.NewReader(gzipReader)
	regularFiles := 0
	var totalExtracted int64
	report := archiveExtractionReport{}

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

			mode := header.FileInfo().Mode().Perm()
			if mode == 0 {
				mode = 0o644
			}
			// #nosec G304 -- targetPath is constrained by secureArchiveJoin to stay under destinationDir.
			outFile, err := os.OpenFile(targetPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, mode)
			if err != nil {
				return archiveExtractionReport{}, fmt.Errorf("failed to create extracted file %q: %w", targetPath, err)
			}

			if err := copyExtractedFile(outFile, tarReader, header.Size, &totalExtracted, targetPath); err != nil {
				_ = outFile.Close()
				return archiveExtractionReport{}, err
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
					log.Printf("WARNING: failed to match library glob %q against tar entry %q: %v", libraryGlob, baseName, matchErr)
				} else if matched {
					report.skippedLibraryLinkEntries++
					if len(report.skippedLibraryLinkExamples) < 3 {
						report.skippedLibraryLinkExamples = append(report.skippedLibraryLinkExamples, header.Name)
					}
				}
			}
			log.Printf("WARNING: skipping link archive entry %q (type=%d) during ONNX Runtime bootstrap extraction", header.Name, header.Typeflag)
			continue
		default:
			// Skip non-regular archive entries (device files, FIFOs, etc.) for safety.
			log.Printf("WARNING: skipping unsupported archive entry %q (type=%d) during ONNX Runtime bootstrap extraction", header.Name, header.Typeflag)
			continue
		}
	}

	if regularFiles == 0 {
		return archiveExtractionReport{}, fmt.Errorf("archive %q did not contain regular files", archivePath)
	}

	return report, nil
}

func extractZIPArchive(archivePath, destinationDir, libraryGlob string) (archiveExtractionReport, error) {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return archiveExtractionReport{}, fmt.Errorf("failed to open ZIP archive %q: %w", archivePath, err)
	}
	defer func() {
		_ = reader.Close()
	}()

	regularFiles := 0
	var totalExtracted int64
	report := archiveExtractionReport{}
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
					log.Printf("WARNING: failed to match library glob %q against zip entry %q: %v", libraryGlob, baseName, matchErr)
				} else if matched {
					report.skippedLibraryLinkEntries++
					if len(report.skippedLibraryLinkExamples) < 3 {
						report.skippedLibraryLinkExamples = append(report.skippedLibraryLinkExamples, entry.Name)
					}
				}
			}
			log.Printf("WARNING: skipping symlink ZIP entry %q during ONNX Runtime bootstrap extraction", entry.Name)
			continue
		}

		if err := os.MkdirAll(filepath.Dir(targetPath), secureDirectoryPermission); err != nil {
			return archiveExtractionReport{}, fmt.Errorf("failed to create parent directory for %q: %w", targetPath, err)
		}

		rc, err := entry.Open()
		if err != nil {
			return archiveExtractionReport{}, fmt.Errorf("failed to open ZIP entry %q: %w", entry.Name, err)
		}

		filePerm := mode.Perm()
		if filePerm == 0 {
			filePerm = 0o644
		}
		// #nosec G304 -- targetPath is constrained by secureArchiveJoin to stay under destinationDir.
		outFile, err := os.OpenFile(targetPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, filePerm)
		if err != nil {
			_ = rc.Close()
			return archiveExtractionReport{}, fmt.Errorf("failed to create extracted file %q: %w", targetPath, err)
		}

		if entry.UncompressedSize64 > math.MaxInt64 {
			_ = outFile.Close()
			_ = rc.Close()
			return archiveExtractionReport{}, fmt.Errorf("ZIP entry %q size exceeds supported range", entry.Name)
		}
		// #nosec G115 -- upper-bound checked against math.MaxInt64 immediately above.
		entrySize := int64(entry.UncompressedSize64)
		if err := copyExtractedFile(outFile, rc, entrySize, &totalExtracted, targetPath); err != nil {
			_ = outFile.Close()
			_ = rc.Close()
			return archiveExtractionReport{}, err
		}

		if err := outFile.Close(); err != nil {
			_ = rc.Close()
			return archiveExtractionReport{}, fmt.Errorf("failed to close extracted file %q: %w", targetPath, err)
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

func resolveExtractedLibraryPath(installDir string, artifact runtimeArtifact) (string, error) {
	libDir := filepath.Join(installDir, "lib")

	var invalidCandidates []error
	trackCandidateError := func(path string, validationErr error) {
		if validationErr == nil {
			return
		}
		if errors.Is(validationErr, os.ErrNotExist) {
			return
		}
		invalidCandidates = append(invalidCandidates, fmt.Errorf("%s: %w", path, validationErr))
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

	if len(invalidCandidates) > 0 {
		return "", fmt.Errorf("found ONNX Runtime shared library candidates in %q but none are valid: %w", libDir, errors.Join(invalidCandidates...))
	}

	return "", errSharedLibraryNotFound
}

func validateLibraryFile(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", fmt.Errorf("library path is empty")
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("failed to resolve absolute path for %q: %w", path, err)
	}

	info, err := os.Stat(absPath)
	if err != nil {
		return "", fmt.Errorf("failed to stat library file %q: %w", absPath, err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("library path points to a directory: %q", absPath)
	}
	if info.Size() == 0 {
		return "", fmt.Errorf("library file is empty: %q", absPath)
	}

	return absPath, nil
}

func withProcessFileLock(lockPath string, fn func() error) (err error) {
	if fn == nil {
		return fmt.Errorf("lock callback is nil")
	}

	if err := os.MkdirAll(filepath.Dir(lockPath), secureDirectoryPermission); err != nil {
		return fmt.Errorf("failed to create lock directory for %q: %w", lockPath, err)
	}

	// #nosec G304 -- lockPath is constructed from configured cache directory and fixed internal suffix.
	file, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, secureLockFilePermission)
	if err != nil {
		return fmt.Errorf("failed to open lock file %q: %w", lockPath, err)
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
			log.Printf("INFO: waiting for ONNX Runtime bootstrap lock %q (%s elapsed)", lockPath, waited.Round(time.Second))
			nextLogAt = time.Now().Add(bootstrapLockLogInterval)
		}
		time.Sleep(bootstrapLockRetryInterval)
	}

	defer func() {
		unlockErr := unlockFile(file)
		closeErr := file.Close()
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
	cacheDir, err := os.UserCacheDir()
	if err == nil && cacheDir != "" {
		return filepath.Join(cacheDir, "onnx-purego", "onnxruntime")
	}

	fallback := filepath.Join(os.TempDir(), "onnx-purego", "onnxruntime")
	bootstrapCacheFallbackWarnOnce.Do(func() {
		if err != nil {
			log.Printf("WARNING: failed to resolve user cache directory (%v); using temporary ONNX Runtime cache at %q. Set ONNXRUNTIME_CACHE_DIR for a persistent cache.", err, fallback)
			return
		}
		log.Printf("WARNING: user cache directory is empty; using temporary ONNX Runtime cache at %q. Set ONNXRUNTIME_CACHE_DIR for a persistent cache.", fallback)
	})
	return fallback
}

func normalizeRuntimeVersion(version string) (string, error) {
	version = strings.TrimSpace(version)
	version = strings.TrimPrefix(version, "v")
	if version == "" {
		return "", fmt.Errorf("ONNX Runtime version is empty")
	}

	parts := strings.Split(version, ".")
	if len(parts) != 3 {
		return "", fmt.Errorf("ONNX Runtime version must have format x.y.z, got %q", version)
	}

	for _, part := range parts {
		if part == "" {
			return "", fmt.Errorf("ONNX Runtime version must have format x.y.z, got %q", version)
		}
		if _, err := strconv.Atoi(part); err != nil {
			return "", fmt.Errorf("ONNX Runtime version must have numeric segments, got %q", version)
		}
	}

	return version, nil
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
		return false, fmt.Errorf("invalid boolean value for %s: %q (expected true/false, 1/0, yes/no, y/n, on/off)", name, value)
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
