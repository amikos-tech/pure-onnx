package openclip

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode"
)

// These pins are duplicated in .github/workflows/ci.yml (ONNXRUNTIME_TEST_OPENCLIP_*)
// so CI can preload and checksum-verify assets before the Go bootstrapper runs.
// Nothing enforces the two copies stay in sync — update both when bumping the revision.
const (
	// DefaultBootstrapRepoID is the default Hugging Face repository for OpenCLIP ONNX artifacts.
	DefaultBootstrapRepoID = "amikos/openclip-vit-b-32-laion2b-s34b-b79k-onnx"
	// DefaultBootstrapRevision is the default repository revision used by the bootstrapper.
	DefaultBootstrapRevision = "248a2ed76a7189fc080e654e36930171331ef085"
	// DefaultBootstrapBaseURL is the default Hugging Face host used by the bootstrapper.
	DefaultBootstrapBaseURL = "https://huggingface.co"
)

const (
	// #nosec G101 -- deterministic artifact checksum, not credential material.
	defaultTextModelSHA256 = "252b86e0ef1fc95b22cfd52fbf647142727fdbecc152556ffe0fba0b10a80370"
	// #nosec G101 -- deterministic artifact checksum, not credential material.
	defaultVisionSHA256 = "7e14f76233d0c840c0621b1ef68f5877efe9357850782b1bbaf0c01693f73b43"
	// #nosec G101 -- deterministic artifact checksum, not credential material.
	defaultTokenizerSHA256 = "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842"
	// #nosec G101 -- deterministic artifact checksum, not credential material.
	defaultPreprocSHA256 = "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd"
)

const (
	defaultTextModelSize    int64 = 253973818
	defaultVisionModelSize  int64 = 351618799
	defaultTokenizerSize    int64 = 2224041
	defaultPreprocessorSize int64 = 316
)

const (
	textModelFileName    = "text_model.onnx"
	visionModelFileName  = "vision_model.onnx"
	tokenizerFileName    = "tokenizer.json" // #nosec G101 -- Public tokenizer asset filename, not credential material.
	preprocessorFileName = "preprocessor_config.json"
)

const (
	defaultMaxDownloadBytes int64         = 1 << 30 // 1 GiB cap for each downloaded file.
	defaultLockWaitTimeout  time.Duration = 30 * time.Second
	defaultLockStaleAfter   time.Duration = 10 * time.Minute
	defaultHTTPTimeout      time.Duration = 60 * time.Second
)

var bootstrapAssetSpecs = []bootstrapAssetSpec{
	{fileName: textModelFileName},
	{fileName: visionModelFileName},
	{fileName: tokenizerFileName},
	{fileName: preprocessorFileName},
}

// ModelAssets describes a local OpenCLIP artifact bundle.
type ModelAssets struct {
	TextModelPath          string
	VisionModelPath        string
	TokenizerPath          string
	PreprocessorConfigPath string
}

// BootstrapOption customizes EnsureDefaultAssets behavior.
type BootstrapOption func(*bootstrapConfig) error

type bootstrapConfig struct {
	repoID             string
	revision           string
	baseURL            string
	cacheDir           string
	hfToken            string
	verifySHA          bool
	shaByFile          map[string]string
	expectedSizeByFile map[string]int64
	maxDownloadBytes   int64
	httpClient         *http.Client
}

type bootstrapAssetSpec struct {
	fileName string
}

type downloadStatusError struct {
	StatusCode int
	URL        string
	Body       string
}

func (e *downloadStatusError) Error() string {
	if e == nil {
		return "download status error: <nil>"
	}
	if e.Body == "" {
		return fmt.Sprintf("HTTP %d for %s", e.StatusCode, e.URL)
	}
	return fmt.Sprintf("HTTP %d for %s: %s", e.StatusCode, e.URL, e.Body)
}

type fileLock struct {
	path string
}

func (l *fileLock) Release() error {
	if l == nil {
		return nil
	}
	if err := os.Remove(l.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

// WithBootstrapCacheDir sets the local cache directory used for downloaded assets.
// The path is rejected if it is empty or carries leading/trailing whitespace.
func WithBootstrapCacheDir(path string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if err := validateBootstrapField(path, "bootstrap cache directory"); err != nil {
			return err
		}
		cfg.cacheDir = path
		return nil
	}
}

// WithBootstrapRepoID sets the Hugging Face repo ID to fetch assets from.
// The repo ID is rejected if it is empty or contains any whitespace: unlike
// revisions, Hugging Face repo IDs never legitimately contain spaces.
func WithBootstrapRepoID(repoID string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if err := validateBootstrapField(repoID, "bootstrap repo ID"); err != nil {
			return err
		}
		if strings.IndexFunc(repoID, unicode.IsSpace) != -1 {
			return fmt.Errorf("invalid bootstrap repo ID %q: whitespace is not allowed", repoID)
		}
		cfg.repoID = repoID
		return nil
	}
}

// WithBootstrapRevision sets the Hugging Face revision to fetch assets from.
// The revision is rejected if it is empty or carries leading/trailing whitespace.
// Interior whitespace is permitted, since branch names may contain spaces.
func WithBootstrapRevision(revision string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if err := validateBootstrapField(revision, "bootstrap revision"); err != nil {
			return err
		}
		cfg.revision = revision
		return nil
	}
}

// WithBootstrapToken sets an optional Hugging Face access token for downloads.
// The token is rejected if it carries leading/trailing whitespace; an empty
// token is valid and means "no token".
func WithBootstrapToken(token string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if err := validateOptionalBootstrapField(token, "bootstrap token"); err != nil {
			return err
		}
		cfg.hfToken = token
		return nil
	}
}

// WithBootstrapChecksum pins a SHA256 checksum for a required artifact.
// The checksum is rejected if it is empty or carries leading/trailing whitespace.
// All rejection messages name fileName, since callers typically pin several assets.
func WithBootstrapChecksum(fileName string, checksum string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if err := validateAssetFileName(fileName); err != nil {
			return err
		}
		if err := validateBootstrapField(checksum, fmt.Sprintf("checksum for %s", fileName)); err != nil {
			return err
		}
		normalized := strings.ToLower(checksum)
		if err := validateSHA256(normalized); err != nil {
			return fmt.Errorf("invalid checksum for %s: %w", fileName, err)
		}
		cfg.shaByFile[fileName] = normalized
		return nil
	}
}

// WithBootstrapExpectedSize pins an exact byte size for a required artifact.
func WithBootstrapExpectedSize(fileName string, sizeBytes int64) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if err := validateAssetFileName(fileName); err != nil {
			return err
		}
		if sizeBytes <= 0 {
			return fmt.Errorf("expected size for %s must be > 0, got %d", fileName, sizeBytes)
		}
		cfg.expectedSizeByFile[fileName] = sizeBytes
		return nil
	}
}

// WithBootstrapMaxDownloadBytes sets a hard per-file cap to protect disk usage.
func WithBootstrapMaxDownloadBytes(limit int64) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if limit <= 0 {
			return fmt.Errorf("max download bytes must be > 0, got %d", limit)
		}
		cfg.maxDownloadBytes = limit
		return nil
	}
}

// WithoutBootstrapChecksumVerification disables checksum verification for downloaded assets.
func WithoutBootstrapChecksumVerification() BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		cfg.verifySHA = false
		return nil
	}
}

// EnsureDefaultAssets ensures OpenCLIP assets are present locally and returns their paths.
//
// By default this fetches:
//   - repo: amikos/openclip-vit-b-32-laion2b-s34b-b79k-onnx
//   - revision: 248a2ed76a7189fc080e654e36930171331ef085
//   - files: text_model.onnx, vision_model.onnx, tokenizer.json, preprocessor_config.json
func EnsureDefaultAssets(opts ...BootstrapOption) (ModelAssets, error) {
	cfg, err := defaultBootstrapConfig()
	if err != nil {
		return ModelAssets{}, err
	}
	for _, opt := range opts {
		if err := opt(&cfg); err != nil {
			return ModelAssets{}, err
		}
	}
	applyDefaultBundleIntegrity(&cfg)
	if err := validateIntegrityConfig(cfg); err != nil {
		return ModelAssets{}, err
	}
	if err := ensureHTTPClientSecurity(&cfg); err != nil {
		return ModelAssets{}, err
	}
	return ensureModelAssets(cfg)
}

func defaultBootstrapConfig() (bootstrapConfig, error) {
	cacheDir := strings.TrimSpace(os.Getenv("ONNXRUNTIME_OPENCLIP_CACHE_DIR"))
	if cacheDir == "" {
		userCacheDir, err := os.UserCacheDir()
		if err != nil {
			return bootstrapConfig{}, fmt.Errorf("cannot determine user cache directory: %w", err)
		}
		cacheDir = filepath.Join(userCacheDir, "onnx-purego", "openclip")
	}

	// Trimmed rather than rejected: env padding is outside the caller's control,
	// unlike a value passed to WithBootstrapToken.
	hfToken := strings.TrimSpace(os.Getenv("HF_TOKEN"))

	return bootstrapConfig{
		repoID:             DefaultBootstrapRepoID,
		revision:           DefaultBootstrapRevision,
		baseURL:            DefaultBootstrapBaseURL,
		cacheDir:           cacheDir,
		hfToken:            hfToken,
		verifySHA:          true,
		shaByFile:          map[string]string{},
		expectedSizeByFile: map[string]int64{},
		maxDownloadBytes:   defaultMaxDownloadBytes,
		httpClient: &http.Client{
			Timeout: defaultHTTPTimeout,
		},
	}, nil
}

func applyDefaultBundleIntegrity(cfg *bootstrapConfig) {
	if cfg == nil || !cfg.verifySHA {
		return
	}
	if cfg.repoID != DefaultBootstrapRepoID || cfg.revision != DefaultBootstrapRevision {
		return
	}
	if _, ok := cfg.shaByFile[textModelFileName]; !ok {
		cfg.shaByFile[textModelFileName] = defaultTextModelSHA256
	}
	if _, ok := cfg.shaByFile[visionModelFileName]; !ok {
		cfg.shaByFile[visionModelFileName] = defaultVisionSHA256
	}
	if _, ok := cfg.shaByFile[tokenizerFileName]; !ok {
		cfg.shaByFile[tokenizerFileName] = defaultTokenizerSHA256
	}
	if _, ok := cfg.shaByFile[preprocessorFileName]; !ok {
		cfg.shaByFile[preprocessorFileName] = defaultPreprocSHA256
	}

	if _, ok := cfg.expectedSizeByFile[textModelFileName]; !ok {
		cfg.expectedSizeByFile[textModelFileName] = defaultTextModelSize
	}
	if _, ok := cfg.expectedSizeByFile[visionModelFileName]; !ok {
		cfg.expectedSizeByFile[visionModelFileName] = defaultVisionModelSize
	}
	if _, ok := cfg.expectedSizeByFile[tokenizerFileName]; !ok {
		cfg.expectedSizeByFile[tokenizerFileName] = defaultTokenizerSize
	}
	if _, ok := cfg.expectedSizeByFile[preprocessorFileName]; !ok {
		cfg.expectedSizeByFile[preprocessorFileName] = defaultPreprocessorSize
	}
}

func validateIntegrityConfig(cfg bootstrapConfig) error {
	if !cfg.verifySHA {
		return nil
	}
	for _, spec := range bootstrapAssetSpecs {
		checksum := strings.TrimSpace(cfg.shaByFile[spec.fileName])
		if checksum == "" {
			return fmt.Errorf(
				"missing checksum for required asset %q. "+
					"Provide WithBootstrapChecksum(...) for every required file, or explicitly disable verification with WithoutBootstrapChecksumVerification()",
				spec.fileName,
			)
		}
		if err := validateSHA256(checksum); err != nil {
			return fmt.Errorf("invalid checksum for %s: %w", spec.fileName, err)
		}
	}
	return nil
}

func ensureHTTPClientSecurity(cfg *bootstrapConfig) error {
	if cfg == nil {
		return nil
	}

	requireHTTPS := cfg.hfToken != ""
	baseHost, err := parseBootstrapBaseHost(cfg.baseURL, requireHTTPS)
	if err != nil {
		return err
	}
	if cfg.httpClient == nil {
		cfg.httpClient = &http.Client{
			Timeout:       defaultHTTPTimeout,
			CheckRedirect: makeRedirectPolicy(baseHost),
		}
		return nil
	}
	if cfg.httpClient.Timeout <= 0 {
		cfg.httpClient.Timeout = defaultHTTPTimeout
	}
	if cfg.httpClient.CheckRedirect == nil {
		cfg.httpClient.CheckRedirect = makeRedirectPolicy(baseHost)
	}
	return nil
}

func parseBootstrapBaseHost(baseURL string, requireHTTPS bool) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return "", fmt.Errorf("invalid bootstrap base URL %q: %w", baseURL, err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("bootstrap base URL must include scheme and host, got %q", baseURL)
	}
	if requireHTTPS && !strings.EqualFold(parsed.Scheme, "https") {
		return "", fmt.Errorf("bootstrap base URL must use HTTPS, got %q", baseURL)
	}
	return strings.ToLower(parsed.Hostname()), nil
}

func makeRedirectPolicy(baseHost string) func(req *http.Request, via []*http.Request) error {
	base := strings.ToLower(strings.TrimSpace(baseHost))
	if base == "" {
		return func(_ *http.Request, _ []*http.Request) error {
			return fmt.Errorf("bootstrap redirect policy unavailable: missing allowed base host")
		}
	}
	return func(req *http.Request, via []*http.Request) error {
		if len(via) == 0 {
			return nil
		}
		if len(via) >= 10 {
			return fmt.Errorf("redirect blocked: too many redirects (%d)", len(via))
		}
		previous := via[len(via)-1].URL
		if previous != nil && req != nil && req.URL != nil {
			if strings.EqualFold(previous.Scheme, "https") && strings.EqualFold(req.URL.Scheme, "http") {
				return fmt.Errorf("redirect blocked: HTTPS downgrade from %q to %q", previous.String(), req.URL.String())
			}
			if !isAllowedRedirectHost(baseHost, req.URL.Hostname()) {
				return fmt.Errorf("redirect blocked: host %q is not allowed for base host %q", req.URL.Hostname(), baseHost)
			}
		}
		return nil
	}
}

func isAllowedRedirectHost(baseHost string, candidateHost string) bool {
	candidate := strings.ToLower(strings.TrimSpace(candidateHost))
	base := strings.ToLower(strings.TrimSpace(baseHost))
	if candidate == "" || base == "" {
		return false
	}
	if base == "huggingface.co" {
		if candidate == "huggingface.co" || strings.HasSuffix(candidate, ".huggingface.co") {
			return true
		}
		if candidate == "hf.co" || strings.HasSuffix(candidate, ".hf.co") {
			return true
		}
		return false
	}
	return candidate == base || strings.HasSuffix(candidate, "."+base)
}

func ensureModelAssets(cfg bootstrapConfig) (ModelAssets, error) {
	if strings.TrimSpace(cfg.repoID) == "" {
		return ModelAssets{}, fmt.Errorf("bootstrap repo ID cannot be empty")
	}
	if strings.TrimSpace(cfg.revision) == "" {
		return ModelAssets{}, fmt.Errorf("bootstrap revision cannot be empty")
	}
	if strings.TrimSpace(cfg.cacheDir) == "" {
		return ModelAssets{}, fmt.Errorf("bootstrap cache directory cannot be empty")
	}
	if strings.TrimSpace(cfg.baseURL) == "" {
		return ModelAssets{}, fmt.Errorf("bootstrap base URL cannot be empty")
	}
	if cfg.maxDownloadBytes <= 0 {
		return ModelAssets{}, fmt.Errorf("max download bytes must be > 0, got %d", cfg.maxDownloadBytes)
	}
	requireHTTPS := cfg.hfToken != ""
	baseHost, err := parseBootstrapBaseHost(cfg.baseURL, requireHTTPS)
	if err != nil {
		return ModelAssets{}, err
	}
	if cfg.httpClient == nil {
		cfg.httpClient = &http.Client{
			Timeout:       defaultHTTPTimeout,
			CheckRedirect: makeRedirectPolicy(baseHost),
		}
	}
	if cfg.httpClient.CheckRedirect == nil {
		cfg.httpClient.CheckRedirect = makeRedirectPolicy(baseHost)
	}

	repoSlug, err := sanitizeBootstrapPathComponent(cfg.repoID, "repo ID")
	if err != nil {
		return ModelAssets{}, err
	}
	revisionSlug, err := sanitizeBootstrapPathComponent(cfg.revision, "revision")
	if err != nil {
		return ModelAssets{}, err
	}
	baseDir, err := resolveBootstrapBaseDir(cfg.cacheDir, repoSlug, revisionSlug)
	if err != nil {
		return ModelAssets{}, err
	}
	if err := os.MkdirAll(baseDir, 0o700); err != nil {
		return ModelAssets{}, fmt.Errorf("failed to create bootstrap cache directory %q: %w", baseDir, err)
	}

	paths := map[string]string{}
	for _, spec := range bootstrapAssetSpecs {
		targetPath := filepath.Join(baseDir, spec.fileName)
		expectedChecksum := strings.ToLower(strings.TrimSpace(cfg.shaByFile[spec.fileName]))
		expectedSize := cfg.expectedSizeByFile[spec.fileName]

		if expectedSize < 0 {
			return ModelAssets{}, fmt.Errorf("invalid expected size for %s: %d", spec.fileName, expectedSize)
		}
		if err := ensureAssetFile(cfg, targetPath, spec.fileName, expectedChecksum, expectedSize); err != nil {
			return ModelAssets{}, err
		}
		paths[spec.fileName] = targetPath
	}

	return ModelAssets{
		TextModelPath:          paths[textModelFileName],
		VisionModelPath:        paths[visionModelFileName],
		TokenizerPath:          paths[tokenizerFileName],
		PreprocessorConfigPath: paths[preprocessorFileName],
	}, nil
}

func ensureAssetFile(cfg bootstrapConfig, destinationPath string, fileName string, expectedSHA256 string, expectedSize int64) (retErr error) {
	lockPath := destinationPath + ".lock"
	lock, err := acquireFileLock(lockPath, defaultLockWaitTimeout, defaultLockStaleAfter)
	if err != nil {
		return fmt.Errorf("failed to acquire bootstrap lock for %s: %w", fileName, err)
	}
	defer func() {
		if releaseErr := lock.Release(); releaseErr != nil && retErr == nil {
			retErr = fmt.Errorf("failed to release bootstrap lock for %s: %w", fileName, releaseErr)
		}
	}()

	staleReason := ""
	if info, statErr := os.Stat(destinationPath); statErr == nil {
		if !info.Mode().IsRegular() {
			return fmt.Errorf("cached %s is not a regular file: %s", fileName, destinationPath)
		}
		switch {
		case !cfg.verifySHA && expectedSize == 0 && info.Size() == 0:
			staleReason = fmt.Sprintf("cached file %s is empty", fileName)
		case expectedSize > 0 && info.Size() != expectedSize:
			staleReason = fmt.Sprintf("cached file %s has unexpected size: got %d bytes, want %d bytes", fileName, info.Size(), expectedSize)
		case cfg.verifySHA && expectedSHA256 != "":
			if verifyErr := verifyFileSHA256(destinationPath, expectedSHA256); verifyErr != nil {
				staleReason = fmt.Sprintf("cached file %s failed checksum check: %v", fileName, verifyErr)
				break
			}
			return nil
		default:
			return nil
		}
		if removeErr := os.Remove(destinationPath); removeErr != nil {
			if staleReason != "" {
				return fmt.Errorf("failed to replace stale %s: %w", fileName, removeErr)
			}
			return fmt.Errorf("failed to remove stale %s: %w", fileName, removeErr)
		}
	} else if !errors.Is(statErr, os.ErrNotExist) {
		return fmt.Errorf("failed to inspect cached %s at %s: %w", fileName, destinationPath, statErr)
	}

	maxBytes := cfg.maxDownloadBytes
	if expectedSize > maxBytes {
		return fmt.Errorf(
			"expected size for %s exceeds configured max download bytes: got %d, max %d",
			fileName,
			expectedSize,
			maxBytes,
		)
	}
	if expectedSize > 0 && expectedSize < maxBytes {
		maxBytes = expectedSize
	}
	if maxBytes <= 0 {
		return fmt.Errorf("invalid max download limit for %s: %d", fileName, maxBytes)
	}

	escapedRepoID := escapeRepoIDForURL(cfg.repoID)
	escapedRevision := url.PathEscape(cfg.revision)
	assetURL := fmt.Sprintf(
		"%s/%s/resolve/%s/%s",
		strings.TrimRight(cfg.baseURL, "/"),
		escapedRepoID,
		escapedRevision,
		fileName,
	)

	if err := downloadFileWithRetry(cfg.httpClient, assetURL, destinationPath, cfg.hfToken, maxBytes, expectedSize); err != nil {
		if staleReason != "" {
			return fmt.Errorf("%s; download failed: %w", staleReason, err)
		}
		return fmt.Errorf("failed to download %s: %w", fileName, err)
	}
	if cfg.verifySHA && expectedSHA256 != "" {
		if err := verifyFileSHA256(destinationPath, expectedSHA256); err != nil {
			if removeErr := os.Remove(destinationPath); removeErr != nil {
				return fmt.Errorf("downloaded %s failed checksum verification: %w", fileName, errors.Join(err, removeErr))
			}
			return fmt.Errorf("downloaded %s failed checksum verification: %w", fileName, err)
		}
	}
	return nil
}

func escapeRepoIDForURL(repoID string) string {
	repoSegments := strings.Split(repoID, "/")
	for i, segment := range repoSegments {
		repoSegments[i] = url.PathEscape(segment)
	}
	return strings.Join(repoSegments, "/")
}

// validateBootstrapField rejects option values that are empty or that carry
// leading/trailing whitespace, so a value that passes validation is stored verbatim.
// Interior whitespace is allowed: revisions may legitimately contain spaces.
func validateBootstrapField(value string, field string) error {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return fmt.Errorf("%s cannot be empty", field)
	}
	if trimmed != value {
		return fmt.Errorf("invalid %s %q: leading or trailing whitespace is not allowed", field, value)
	}
	return nil
}

// validateOptionalBootstrapField is validateBootstrapField for fields where an
// empty value is meaningful ("unset") rather than an error. A whitespace-only
// value is still rejected, since it is ambiguous rather than a deliberate unset.
// The rejected value is never echoed: this guards credentials, and errors reach
// logs and error trackers.
func validateOptionalBootstrapField(value string, field string) error {
	if value == "" {
		return nil
	}
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return fmt.Errorf("invalid %s: whitespace-only value is not allowed", field)
	}
	if trimmed != value {
		return fmt.Errorf("invalid %s: leading or trailing whitespace is not allowed", field)
	}
	return nil
}

func sanitizeBootstrapPathComponent(value string, field string) (string, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return "", fmt.Errorf("%s cannot be empty", field)
	}

	for _, segment := range strings.FieldsFunc(trimmed, func(r rune) bool {
		return r == '/' || r == '\\'
	}) {
		if segment == "." || segment == ".." {
			return "", fmt.Errorf("invalid %s %q: path traversal components are not allowed", field, value)
		}
	}

	replaced := strings.ReplaceAll(trimmed, "/", "--")
	replaced = strings.ReplaceAll(replaced, "\\", "--")
	replaced = strings.ReplaceAll(replaced, string(filepath.Separator), "--")
	return replaced, nil
}

func resolveBootstrapBaseDir(cacheDir string, repoSlug string, revisionSlug string) (string, error) {
	absCacheDir, err := filepath.Abs(cacheDir)
	if err != nil {
		return "", fmt.Errorf("failed to resolve absolute path for cache directory %q: %w", cacheDir, err)
	}
	baseDir := filepath.Join(absCacheDir, repoSlug, revisionSlug)
	absBaseDir, err := filepath.Abs(baseDir)
	if err != nil {
		return "", fmt.Errorf("failed to resolve absolute path for bootstrap cache directory %q: %w", baseDir, err)
	}

	rel, err := filepath.Rel(absCacheDir, absBaseDir)
	if err != nil {
		return "", fmt.Errorf("failed to validate bootstrap cache path %q: %w", absBaseDir, err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("bootstrap cache directory %q escapes cache root %q", absBaseDir, absCacheDir)
	}
	return absBaseDir, nil
}

func downloadFileWithRetry(client *http.Client, assetURL string, destinationPath string, hfToken string, maxBytes int64, expectedSize int64) error {
	var lastErr error
	var seenErrs []error
	const maxAttempts = 3
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if err := downloadFileOnce(client, assetURL, destinationPath, hfToken, maxBytes, expectedSize); err != nil {
			lastErr = err
			seenErrs = append(seenErrs, fmt.Errorf("attempt %d: %w", attempt, err))
			if !isRetryableDownloadError(err) || attempt == maxAttempts {
				break
			}
			time.Sleep(time.Duration(attempt) * 500 * time.Millisecond)
			continue
		}
		return nil
	}
	if len(seenErrs) > 1 {
		return fmt.Errorf("failed to download %s after %d attempts: %w", assetURL, maxAttempts, errors.Join(seenErrs...))
	}
	return fmt.Errorf("failed to download %s after %d attempts: %w", assetURL, maxAttempts, lastErr)
}

func downloadFileOnce(client *http.Client, assetURL string, destinationPath string, hfToken string, maxBytes int64, expectedSize int64) (err error) {
	if maxBytes <= 0 {
		return fmt.Errorf("max download bytes must be > 0, got %d", maxBytes)
	}

	req, err := http.NewRequest(http.MethodGet, assetURL, nil)
	if err != nil {
		return err
	}
	// hfToken carries no surrounding whitespace by construction: WithBootstrapToken
	// rejects padded values and the HF_TOKEN fallback is trimmed at config time.
	if hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return &downloadStatusError{
			StatusCode: resp.StatusCode,
			URL:        assetURL,
			Body:       strings.TrimSpace(string(body)),
		}
	}
	if resp.ContentLength > 0 && resp.ContentLength > maxBytes {
		return fmt.Errorf("response content-length %d exceeds max allowed %d bytes for %s", resp.ContentLength, maxBytes, assetURL)
	}

	tempPath := destinationPath + ".tmp"
	if err := os.MkdirAll(filepath.Dir(destinationPath), 0o700); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}
	// #nosec G304 -- destination path is controlled by ensureAssetFile under the cache root.
	file, err := os.OpenFile(tempPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	// Keep explicit close for the file because Windows rename on an open
	// handle is fragile; this closure does not perform a second close.
	var closed bool
	defer func() {
		if !closed {
			_ = file.Close()
		}
		if err != nil {
			_ = os.Remove(tempPath)
		}
	}()

	limitedReader := io.LimitReader(resp.Body, maxBytes+1)
	written, copyErr := io.Copy(file, limitedReader)
	if copyErr != nil {
		return fmt.Errorf("failed to write response body: %w", copyErr)
	}
	if written > maxBytes {
		return fmt.Errorf("download exceeded max allowed size (%d > %d) for %s", written, maxBytes, assetURL)
	}
	if expectedSize > 0 && written != expectedSize {
		return fmt.Errorf("downloaded size mismatch for %s: got %d bytes, want %d bytes", assetURL, written, expectedSize)
	}

	if err = file.Sync(); err != nil {
		return fmt.Errorf("failed to flush temp file: %w", err)
	}
	if err = file.Close(); err != nil {
		return fmt.Errorf("failed to close temp file: %w", err)
	}
	closed = true
	if err = os.Rename(tempPath, destinationPath); err != nil {
		return fmt.Errorf("failed to move temp file into place: %w", err)
	}
	return nil
}

func isRetryableDownloadError(err error) bool {
	statusErr := (*downloadStatusError)(nil)
	if !errors.As(err, &statusErr) {
		urlErr := (*url.Error)(nil)
		if errors.As(err, &urlErr) {
			var netErr net.Error
			if errors.As(urlErr.Err, &netErr) && netErr.Timeout() {
				return true
			}
			if errors.Is(urlErr.Err, io.EOF) {
				return true
			}
			if isTransientURLSubError(urlErr.Err) {
				return true
			}
			return false
		}
		return false
	}
	if statusErr.StatusCode == http.StatusRequestTimeout || statusErr.StatusCode == http.StatusTooManyRequests {
		return true
	}
	return statusErr.StatusCode >= 500 && statusErr.StatusCode <= 599
}

func isTransientURLSubError(err error) bool {
	if err == nil {
		return false
	}
	lower := strings.ToLower(err.Error())
	switch {
	case strings.Contains(lower, "connection refused"):
		return true
	case strings.Contains(lower, "connection reset by peer"):
		return true
	case strings.Contains(lower, "i/o timeout"):
		return true
	case strings.Contains(lower, "timed out"):
		return true
	default:
		return false
	}
}

func acquireFileLock(lockPath string, waitTimeout time.Duration, staleAfter time.Duration) (*fileLock, error) {
	if strings.TrimSpace(lockPath) == "" {
		return nil, fmt.Errorf("lock path cannot be empty")
	}
	deadline := time.Now().Add(waitTimeout)
	for {
		// #nosec G304 -- lock path is derived from destination cache path in ensureAssetFile.
		lockFile, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			_, _ = io.WriteString(lockFile, fmt.Sprintf("pid=%d\n", os.Getpid()))
			_ = lockFile.Close()
			return &fileLock{path: lockPath}, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return nil, err
		}

		// Best-effort stale lock cleanup for crashed writers.
		if staleAfter > 0 {
			if info, statErr := os.Stat(lockPath); statErr == nil {
				if time.Since(info.ModTime()) > staleAfter {
					if err := os.Remove(lockPath); err != nil && !errors.Is(err, os.ErrNotExist) {
						return nil, fmt.Errorf("failed to clean stale lock %q: %w", lockPath, err)
					}
					continue
				}
			}
		}

		if time.Now().After(deadline) {
			return nil, fmt.Errorf("timed out waiting for lock %q", lockPath)
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func verifyFileSHA256(path string, expectedSHA256 string) error {
	expected, err := parseSHA256(expectedSHA256)
	if err != nil {
		return err
	}
	// #nosec G304 -- path is controlled by ensureAssetFile under the cache root.
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open file %q: %w", path, err)
	}
	defer func() {
		_ = file.Close()
	}()

	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return fmt.Errorf("failed to hash file %q: %w", path, err)
	}
	actual := hasher.Sum(nil)
	if !equalBytes(expected, actual) {
		return fmt.Errorf("checksum mismatch: got %s want %s", hex.EncodeToString(actual), strings.ToLower(expectedSHA256))
	}
	return nil
}

func validateSHA256(value string) error {
	_, err := parseSHA256(value)
	return err
}

func parseSHA256(value string) ([]byte, error) {
	trimmed := strings.TrimSpace(value)
	if len(trimmed) != 64 {
		return nil, fmt.Errorf("checksum must be 64 hex chars, got %d", len(trimmed))
	}
	decoded, err := hex.DecodeString(trimmed)
	if err != nil {
		return nil, fmt.Errorf("checksum is not valid hex: %w", err)
	}
	return decoded, nil
}

func validateAssetFileName(fileName string) error {
	switch strings.TrimSpace(fileName) {
	case textModelFileName, visionModelFileName, tokenizerFileName, preprocessorFileName:
		return nil
	default:
		return fmt.Errorf("unsupported bootstrap asset file %q", fileName)
	}
}

func equalBytes(a []byte, b []byte) bool {
	return subtle.ConstantTimeCompare(a, b) == 1
}
