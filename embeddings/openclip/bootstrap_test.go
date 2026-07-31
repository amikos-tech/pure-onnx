package openclip

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestEnsureModelAssetsDownloadsAndCaches(t *testing.T) {
	repoID := "unit/test-repo"
	revision := "main"
	assetData := map[string][]byte{
		textModelFileName:    []byte("text-onnx"),
		visionModelFileName:  []byte("vision-onnx"),
		tokenizerFileName:    []byte(`{"type":"tokenizer"}`),
		preprocessorFileName: []byte(`{"size":224,"crop_size":224}`),
	}

	var mu sync.Mutex
	requestCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requestCount++
		mu.Unlock()

		prefix := "/" + repoID + "/resolve/" + revision + "/"
		if len(r.URL.Path) <= len(prefix) || r.URL.Path[:len(prefix)] != prefix {
			http.NotFound(w, r)
			return
		}
		fileName := r.URL.Path[len(prefix):]
		payload, ok := assetData[fileName]
		if !ok {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(payload)
	}))
	defer server.Close()

	tempDir := t.TempDir()
	shaByFile := map[string]string{}
	sizeByFile := map[string]int64{}
	for fileName, data := range assetData {
		shaByFile[fileName] = sha256Hex(data)
		sizeByFile[fileName] = int64(len(data))
	}

	cfg := bootstrapConfig{
		repoID:             repoID,
		revision:           revision,
		baseURL:            server.URL,
		cacheDir:           tempDir,
		verifySHA:          true,
		shaByFile:          shaByFile,
		expectedSizeByFile: sizeByFile,
		maxDownloadBytes:   1024 * 1024,
		httpClient: &http.Client{
			Transport: http.DefaultTransport,
		},
	}

	assets, err := ensureModelAssets(cfg)
	if err != nil {
		t.Fatalf("ensureModelAssets failed: %v", err)
	}
	assertFileContains(t, assets.TextModelPath, assetData[textModelFileName])
	assertFileContains(t, assets.VisionModelPath, assetData[visionModelFileName])
	assertFileContains(t, assets.TokenizerPath, assetData[tokenizerFileName])
	assertFileContains(t, assets.PreprocessorConfigPath, assetData[preprocessorFileName])

	mu.Lock()
	firstRequestCount := requestCount
	mu.Unlock()
	if firstRequestCount != 4 {
		t.Fatalf("unexpected request count after first download: got %d, want 4", firstRequestCount)
	}

	_, err = ensureModelAssets(cfg)
	if err != nil {
		t.Fatalf("second ensureModelAssets failed: %v", err)
	}
	mu.Lock()
	secondRequestCount := requestCount
	mu.Unlock()
	if secondRequestCount != firstRequestCount {
		t.Fatalf("expected cached second call to avoid network requests, got first=%d second=%d", firstRequestCount, secondRequestCount)
	}
}

func TestDefaultBootstrapConfigSkipsTestEnvVar(t *testing.T) {
	t.Setenv("ONNXRUNTIME_OPENCLIP_CACHE_DIR", t.TempDir())
	t.Setenv("ONNXRUNTIME_TEST_MODEL_CACHE_DIR", t.TempDir())

	cfg, err := defaultBootstrapConfig()
	if err != nil {
		t.Fatalf("defaultBootstrapConfig failed: %v", err)
	}
	if cfg.cacheDir != os.Getenv("ONNXRUNTIME_OPENCLIP_CACHE_DIR") {
		t.Fatalf("expected production cache env var to win, got %q", cfg.cacheDir)
	}
}

func TestEnsureModelAssetsEscapesRepoIDAndRevisionInURL(t *testing.T) {
	repoID := "owner/repo"
	revision := "branch/with space"
	assetData := map[string][]byte{
		textModelFileName:    []byte("text-onnx"),
		visionModelFileName:  []byte("vision-onnx"),
		tokenizerFileName:    []byte(`{"type":"tokenizer"}`),
		preprocessorFileName: []byte(`{"size":224,"crop_size":224}`),
	}

	escapedRepoID := escapeRepoIDForURL(repoID)
	escapedRevision := url.PathEscape(revision)
	expectedPrefix := "/" + escapedRepoID + "/resolve/" + escapedRevision + "/"

	var mu sync.Mutex
	requestCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requestCount++
		mu.Unlock()

		if r.URL.EscapedPath() == "" || !strings.HasPrefix(r.URL.EscapedPath(), expectedPrefix) {
			http.NotFound(w, r)
			return
		}
		if r.URL.RawQuery != "" {
			http.Error(w, "unexpected query", http.StatusBadRequest)
			return
		}
		fileName := r.URL.EscapedPath()[len(expectedPrefix):]
		payload, ok := assetData[fileName]
		if !ok {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(payload)
	}))
	defer server.Close()

	tempDir := t.TempDir()
	shaByFile := map[string]string{}
	sizeByFile := map[string]int64{}
	for fileName, data := range assetData {
		shaByFile[fileName] = sha256Hex(data)
		sizeByFile[fileName] = int64(len(data))
	}

	cfg := bootstrapConfig{
		repoID:             repoID,
		revision:           revision,
		baseURL:            server.URL,
		cacheDir:           tempDir,
		verifySHA:          true,
		shaByFile:          shaByFile,
		expectedSizeByFile: sizeByFile,
		maxDownloadBytes:   1024 * 1024,
		httpClient: &http.Client{
			Transport: http.DefaultTransport,
		},
	}

	if _, err := ensureModelAssets(cfg); err != nil {
		t.Fatalf("ensureModelAssets failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if requestCount != len(bootstrapAssetSpecs) {
		t.Fatalf("expected %d requests, got %d", len(bootstrapAssetSpecs), requestCount)
	}
}

func TestEscapeRepoIDForURL(t *testing.T) {
	repoID := "owner/repo?token=abc&raw=value"
	escaped := escapeRepoIDForURL(repoID)
	parts := strings.Split(escaped, "/")
	if len(parts) != 2 {
		t.Fatalf("expected 2 path segments, got %d: %q", len(parts), escaped)
	}
	if parts[1] == "repo?token=abc&raw=value" {
		t.Fatalf("expected encoded repo segment, got %q", parts[1])
	}
	if expected := url.PathEscape("repo?token=abc&raw=value"); parts[1] != expected {
		t.Fatalf("expected query-sensitive encoding in repo segment, got %q", parts[1])
	}
}

func TestEnsureAssetFileReplacesCorruptFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("good-content"))
	}))
	defer server.Close()

	tempDir := t.TempDir()
	destination := filepath.Join(tempDir, textModelFileName)
	if err := os.WriteFile(destination, []byte("corrupt"), 0o644); err != nil {
		t.Fatalf("failed to write corrupt fixture: %v", err)
	}

	expected := sha256Hex([]byte("good-content"))
	cfg := bootstrapConfig{
		repoID:             "repo/test",
		revision:           "main",
		baseURL:            server.URL,
		cacheDir:           tempDir,
		verifySHA:          true,
		shaByFile:          map[string]string{textModelFileName: expected},
		expectedSizeByFile: map[string]int64{textModelFileName: int64(len("good-content"))},
		maxDownloadBytes:   1024 * 1024,
		httpClient: &http.Client{
			Transport: http.DefaultTransport,
		},
	}

	if err := ensureAssetFile(cfg, destination, textModelFileName, expected, int64(len("good-content"))); err != nil {
		t.Fatalf("ensureAssetFile failed: %v", err)
	}
	assertFileContains(t, destination, []byte("good-content"))
}

func TestEnsureAssetFileRejectsDownloadedChecksumMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("bad-content"))
	}))
	defer server.Close()

	tempDir := t.TempDir()
	destination := filepath.Join(tempDir, textModelFileName)
	cfg := bootstrapConfig{
		repoID:             "repo/test",
		revision:           "main",
		baseURL:            server.URL,
		cacheDir:           tempDir,
		verifySHA:          true,
		shaByFile:          map[string]string{textModelFileName: "0000000000000000000000000000000000000000000000000000000000000000"},
		expectedSizeByFile: map[string]int64{textModelFileName: int64(len("bad-content"))},
		maxDownloadBytes:   defaultMaxDownloadBytes,
		httpClient:         &http.Client{Transport: http.DefaultTransport},
	}

	err := ensureAssetFile(cfg, destination, textModelFileName, cfg.shaByFile[textModelFileName], cfg.expectedSizeByFile[textModelFileName])
	if err == nil || !strings.Contains(err.Error(), "downloaded text_model.onnx failed checksum verification") {
		t.Fatalf("expected checksum verification failure, got: %v", err)
	}
	if _, err := os.Stat(destination); err == nil {
		t.Fatalf("expected file to be removed after checksum mismatch")
	}
}

func TestEnsureAssetFileRejectsEmptyCachedFileWithoutVerification(t *testing.T) {
	tempDir := t.TempDir()
	destination := filepath.Join(tempDir, textModelFileName)
	if err := os.WriteFile(destination, []byte{}, 0o600); err != nil {
		t.Fatalf("failed to seed empty cached file: %v", err)
	}

	cfg := bootstrapConfig{
		repoID:           "repo/test",
		revision:         "main",
		baseURL:          "http://example.invalid",
		cacheDir:         tempDir,
		verifySHA:        false,
		maxDownloadBytes: defaultMaxDownloadBytes,
		httpClient: &http.Client{
			Transport: http.DefaultTransport,
		},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("downloaded"))
	}))
	defer server.Close()
	cfg.baseURL = server.URL

	if err := ensureAssetFile(cfg, destination, textModelFileName, "", 0); err != nil {
		t.Fatalf("ensureAssetFile failed: %v", err)
	}
	assertFileContains(t, destination, []byte("downloaded"))
}

func TestEnsureAssetFileRejectsNonFileCacheEntry(t *testing.T) {
	tempDir := t.TempDir()
	destination := filepath.Join(tempDir, textModelFileName)
	if err := os.Mkdir(destination, 0o700); err != nil {
		t.Fatalf("failed to seed directory cache entry: %v", err)
	}

	cfg := bootstrapConfig{
		repoID:           "repo/test",
		revision:         "main",
		baseURL:          "http://example.invalid",
		cacheDir:         tempDir,
		verifySHA:        false,
		httpClient:       &http.Client{},
		maxDownloadBytes: defaultMaxDownloadBytes,
	}
	if err := ensureAssetFile(cfg, destination, textModelFileName, "", 0); err == nil || !strings.Contains(err.Error(), "not a regular file") {
		t.Fatalf("expected non-regular cache path error, got: %v", err)
	}
}

func TestEnsureAssetFileReturnsStatError(t *testing.T) {
	tempDir := t.TempDir()
	cfg := bootstrapConfig{
		repoID:           "repo/test",
		revision:         "main",
		baseURL:          "http://example.invalid",
		cacheDir:         tempDir,
		verifySHA:        false,
		httpClient:       &http.Client{},
		maxDownloadBytes: defaultMaxDownloadBytes,
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("should not reach network for stat error path")
	}))
	defer server.Close()
	cfg.baseURL = server.URL

	err := ensureAssetFile(cfg, "/\x00", textModelFileName, "", 0)
	if err == nil {
		t.Fatalf("expected stat error")
	}
	if !strings.Contains(err.Error(), "failed to inspect cached") && !strings.Contains(err.Error(), "acquire bootstrap lock") {
		t.Fatalf("unexpected error for invalid destination path: %v", err)
	}
}

func TestDownloadFileOnceRejectsOversizeByContentLength(t *testing.T) {
	payload := strings.Repeat("x", 64)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "64")
		_, _ = w.Write([]byte(payload))
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "asset.bin")
	err := downloadFileOnce(&http.Client{}, server.URL+"/asset.bin", destination, "", 32, 0)
	if err == nil || !strings.Contains(err.Error(), "content-length") {
		t.Fatalf("expected content-length oversize error, got: %v", err)
	}
}

func TestDownloadFileOnceRejectsOversizeWithoutContentLength(t *testing.T) {
	payload := strings.Repeat("x", 64)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Transfer-Encoding", "chunked")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("response writer must support flush")
		}
		for i := 0; i < len(payload); i++ {
			_, _ = w.Write([]byte{payload[i]})
			flusher.Flush()
		}
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "asset.bin")
	err := downloadFileOnce(&http.Client{}, server.URL+"/asset.bin", destination, "", 32, 0)
	if err == nil || !strings.Contains(err.Error(), "download exceeded max allowed size") {
		t.Fatalf("expected streamed max-bytes error, got: %v", err)
	}
}

func TestDownloadFileOnceRejectsExpectedSizeMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("0123456789")) // 10 bytes
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "asset.bin")
	err := downloadFileOnce(&http.Client{}, server.URL+"/asset.bin", destination, "", 16, 11)
	if err == nil || !strings.Contains(err.Error(), "downloaded size mismatch") {
		t.Fatalf("expected size mismatch error, got: %v", err)
	}
}

func TestDownloadFileWithRetryRetryableHTTPStatus(t *testing.T) {
	var attempts int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			http.Error(w, "temporary error", http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte("ok"))
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "asset.bin")
	err := downloadFileWithRetry(
		&http.Client{},
		server.URL+"/asset.bin",
		destination,
		"",
		16,
		2,
	)
	if err != nil {
		t.Fatalf("downloadFileWithRetry failed: %v", err)
	}
	if attempts != 2 {
		t.Fatalf("expected 2 attempts for retryable status, got %d", attempts)
	}
	assertFileContains(t, destination, []byte("ok"))
}

func TestDownloadFileWithRetryIncludesIntermediateErrors(t *testing.T) {
	var attempts int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		switch attempts {
		case 1:
			http.Error(w, "temporary error", http.StatusInternalServerError)
		case 2:
			http.Error(w, "still temporary", http.StatusServiceUnavailable)
		case 3:
			http.Error(w, "not retryable", http.StatusBadRequest)
		}
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "asset.bin")
	err := downloadFileWithRetry(
		&http.Client{},
		server.URL+"/asset.bin",
		destination,
		"",
		16,
		2,
	)
	if err == nil {
		t.Fatalf("expected download failure")
	}
	if attempts != 3 {
		t.Fatalf("expected 3 attempts, got %d", attempts)
	}
	if !strings.Contains(err.Error(), "attempt 1") || !strings.Contains(err.Error(), "attempt 2") || !strings.Contains(err.Error(), "attempt 3") {
		t.Fatalf("expected error to include all attempts, got: %v", err)
	}
}

func TestDownloadFileWithRetryDoesNotRetryClientError(t *testing.T) {
	var attempts int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "asset.bin")
	err := downloadFileWithRetry(
		&http.Client{},
		server.URL+"/asset.bin",
		destination,
		"",
		16,
		0,
	)
	if err == nil {
		t.Fatalf("expected download to fail for 400")
	}
	if attempts != 1 {
		t.Fatalf("expected no retry for non-retryable status, got %d", attempts)
	}
}

func TestAcquireFileLockTimeout(t *testing.T) {
	lockPath := filepath.Join(t.TempDir(), "asset.lock")
	first, err := acquireFileLock(lockPath, time.Second, time.Hour)
	if err != nil {
		t.Fatalf("failed to acquire first lock: %v", err)
	}
	defer func() {
		_ = first.Release()
	}()

	_, err = acquireFileLock(lockPath, 200*time.Millisecond, time.Hour)
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("expected timed out lock acquisition, got: %v", err)
	}
}

func TestAcquireFileLockStaleCleanup(t *testing.T) {
	tempDir := t.TempDir()
	lockPath := filepath.Join(tempDir, "asset.lock")
	if err := os.WriteFile(lockPath, []byte("stale"), 0o600); err != nil {
		t.Fatalf("failed to seed stale lock: %v", err)
	}
	oldTime := time.Now().Add(-time.Hour)
	if err := os.Chtimes(lockPath, oldTime, oldTime); err != nil {
		t.Fatalf("failed to set stale lock mtime: %v", err)
	}

	lock, err := acquireFileLock(lockPath, time.Second, time.Millisecond*10)
	if err != nil {
		t.Fatalf("acquireFileLock stale cleanup failed: %v", err)
	}
	defer func() {
		_ = lock.Release()
	}()
}

func TestIsRetryableDownloadError(t *testing.T) {
	if !isRetryableDownloadError(&downloadStatusError{StatusCode: http.StatusRequestTimeout}) {
		t.Fatalf("expected HTTP 408 to be retryable")
	}
	if !isRetryableDownloadError(&downloadStatusError{StatusCode: http.StatusTooManyRequests}) {
		t.Fatalf("expected HTTP 429 to be retryable")
	}
	if !isRetryableDownloadError(&downloadStatusError{StatusCode: http.StatusInternalServerError}) {
		t.Fatalf("expected HTTP 500 to be retryable")
	}
	if isRetryableDownloadError(&downloadStatusError{StatusCode: http.StatusBadRequest}) {
		t.Fatalf("expected HTTP 400 to be non-retryable")
	}
}

func TestIsRetryableDownloadErrorNetworkTransient(t *testing.T) {
	if !isRetryableDownloadError(&url.Error{
		Op:  "Get",
		URL: "http://example.com",
		Err: errors.New("connection refused"),
	}) {
		t.Fatalf("expected transient connection error to be retryable")
	}

	if isRetryableDownloadError(&url.Error{
		Op:  "Get",
		URL: "http://example.com",
		Err: errors.New("no such host"),
	}) {
		t.Fatalf("expected DNS resolution error to be non-retryable")
	}

	if isRetryableDownloadError(&url.Error{
		Op:  "Get",
		URL: "http://example.com",
		Err: errors.New("permission denied"),
	}) {
		t.Fatalf("expected unrelated URL error to be non-retryable")
	}
}

func TestRedirectPolicyRejectsHTTPSDowngrade(t *testing.T) {
	policy := makeRedirectPolicy("huggingface.co")
	req := &http.Request{URL: mustParseURL(t, "http://example.com/file")}
	via := []*http.Request{
		{URL: mustParseURL(t, "https://huggingface.co/repo/resolve/main/file")},
	}
	err := policy(req, via)
	if err == nil || !strings.Contains(err.Error(), "downgrade") {
		t.Fatalf("expected HTTPS downgrade rejection, got: %v", err)
	}
}

func TestIsAllowedRedirectHost(t *testing.T) {
	if !isAllowedRedirectHost("huggingface.co", "cas-bridge.xethub.hf.co") {
		t.Fatalf("expected hf.co subdomain to be allowed")
	}
	if isAllowedRedirectHost("huggingface.co", "example.com") {
		t.Fatalf("expected unrelated host to be rejected")
	}
	if !isAllowedRedirectHost("my.mirror.local", "cdn.my.mirror.local") {
		t.Fatalf("expected subdomain of custom host to be allowed")
	}
}

func TestEnsureDefaultAssetsValidation(t *testing.T) {
	_, err := EnsureDefaultAssets(WithBootstrapCacheDir(""))
	if err == nil {
		t.Fatalf("expected validation error for empty cache dir")
	}
}

func TestEnsureDefaultAssetsCustomRepoRequiresChecksums(t *testing.T) {
	_, err := EnsureDefaultAssets(
		WithBootstrapRepoID("custom/repo"),
		WithBootstrapRevision("main"),
	)
	if err == nil || !strings.Contains(err.Error(), "missing checksum") {
		t.Fatalf("expected missing checksum error for custom repo without explicit checksums, got: %v", err)
	}
}

func TestSanitizeBootstrapPathComponentRejectsTraversal(t *testing.T) {
	if _, err := sanitizeBootstrapPathComponent("..", "repo ID"); err == nil {
		t.Fatalf("expected traversal rejection for dot-segment repo ID")
	}
	if _, err := sanitizeBootstrapPathComponent("repo\\..\\owner", "repo ID"); err == nil {
		t.Fatalf("expected traversal rejection for backslash segment repo ID")
	}
	if _, err := sanitizeBootstrapPathComponent("owner/repo", "repo ID"); err != nil {
		t.Fatalf("expected slash replacement to be accepted, got: %v", err)
	}
	if _, err := sanitizeBootstrapPathComponent("owner\\repo", "repo ID"); err != nil {
		t.Fatalf("expected backslash replacement to be accepted, got: %v", err)
	}
}

func TestEnsureModelAssetsRejectsPathTraversalRepoID(t *testing.T) {
	_, err := ensureModelAssets(bootstrapConfig{
		repoID:           "..",
		revision:         "main",
		baseURL:          DefaultBootstrapBaseURL,
		cacheDir:         t.TempDir(),
		verifySHA:        false,
		maxDownloadBytes: defaultMaxDownloadBytes,
		httpClient:       &http.Client{},
	})
	if err == nil || !strings.Contains(err.Error(), "invalid repo ID") {
		t.Fatalf("expected invalid repo ID error, got: %v", err)
	}
}

func TestEnsureModelAssetsRejectsPathTraversalRevision(t *testing.T) {
	_, err := ensureModelAssets(bootstrapConfig{
		repoID:           "repo/owner",
		revision:         "..",
		baseURL:          DefaultBootstrapBaseURL,
		cacheDir:         t.TempDir(),
		verifySHA:        false,
		maxDownloadBytes: defaultMaxDownloadBytes,
		httpClient:       &http.Client{},
	})
	if err == nil || !strings.Contains(err.Error(), "invalid revision") {
		t.Fatalf("expected invalid revision error, got: %v", err)
	}
}

func TestEnsureAssetFileRejectsExpectedSizeOverMaxDownloadBytes(t *testing.T) {
	cfg := bootstrapConfig{
		verifySHA:        false,
		maxDownloadBytes: 16,
		httpClient:       &http.Client{},
	}
	destination := filepath.Join(t.TempDir(), textModelFileName)
	err := ensureAssetFile(cfg, destination, textModelFileName, "", 64)
	if err == nil || !strings.Contains(err.Error(), "expected size for text_model.onnx exceeds configured max download bytes") {
		t.Fatalf("expected expected-size cap mismatch error, got: %v", err)
	}
	if _, statErr := os.Stat(destination); statErr == nil {
		t.Fatalf("expected no download artifact for failed size-cap validation")
	}
}

func assertFileContains(t *testing.T, path string, expected []byte) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read %q: %v", path, err)
	}
	if string(got) != string(expected) {
		t.Fatalf("unexpected file content at %q: got %q, want %q", path, string(got), string(expected))
	}
}

func mustParseURL(t *testing.T, raw string) *url.URL {
	t.Helper()
	parsed, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("failed to parse URL %q: %v", raw, err)
	}
	return parsed
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func TestBootstrapOptionsRejectSurroundingWhitespace(t *testing.T) {
	cases := []struct {
		name    string
		value   string
		wantErr string
	}{
		{name: "empty", value: "", wantErr: "cannot be empty"},
		{name: "only whitespace", value: "   ", wantErr: "cannot be empty"},
		{name: "leading space", value: " value", wantErr: "whitespace"},
		{name: "trailing space", value: "value ", wantErr: "whitespace"},
		{name: "trailing tab", value: "value\t", wantErr: "whitespace"},
		{name: "leading newline", value: "\nvalue", wantErr: "whitespace"},
	}

	options := map[string]func(string) BootstrapOption{
		"WithBootstrapCacheDir": WithBootstrapCacheDir,
		"WithBootstrapRepoID":   WithBootstrapRepoID,
		"WithBootstrapRevision": WithBootstrapRevision,
	}

	for optName, optFn := range options {
		for _, tc := range cases {
			t.Run(optName+"/"+tc.name, func(t *testing.T) {
				var cfg bootstrapConfig
				err := optFn(tc.value)(&cfg)
				if err == nil {
					t.Fatalf("expected rejection for %q", tc.value)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
				}
			})
		}
	}
}

// Values that pass validation must be stored verbatim, and interior whitespace
// stays legal because Hugging Face branch names may contain spaces.
func TestBootstrapOptionsStoreAcceptedValuesVerbatim(t *testing.T) {
	var cfg bootstrapConfig

	if err := WithBootstrapCacheDir("/tmp/openclip cache")(&cfg); err != nil {
		t.Fatalf("expected interior space to be accepted, got: %v", err)
	}
	if cfg.cacheDir != "/tmp/openclip cache" {
		t.Fatalf("expected cache dir stored verbatim, got %q", cfg.cacheDir)
	}

	if err := WithBootstrapRepoID("owner/repo")(&cfg); err != nil {
		t.Fatalf("expected repo ID to be accepted, got: %v", err)
	}
	if cfg.repoID != "owner/repo" {
		t.Fatalf("expected repo ID stored verbatim, got %q", cfg.repoID)
	}

	if err := WithBootstrapRevision("branch/with space")(&cfg); err != nil {
		t.Fatalf("expected interior space in revision to be accepted, got: %v", err)
	}
	if cfg.revision != "branch/with space" {
		t.Fatalf("expected revision stored verbatim, got %q", cfg.revision)
	}
}
