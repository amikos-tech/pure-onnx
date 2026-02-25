package ort

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestResolveRuntimeArtifact(t *testing.T) {
	tests := []struct {
		name    string
		goos    string
		goarch  string
		want    runtimeArtifact
		wantErr bool
	}{
		{
			name:   "darwin arm64",
			goos:   "darwin",
			goarch: "arm64",
			want: runtimeArtifact{
				platform:         "osx-arm64",
				archiveExtension: "tgz",
				primaryLibrary:   "libonnxruntime.dylib",
				libraryGlob:      "libonnxruntime*.dylib",
			},
		},
		{
			name:   "darwin amd64",
			goos:   "darwin",
			goarch: "amd64",
			want: runtimeArtifact{
				platform:         "osx-x86_64",
				archiveExtension: "tgz",
				primaryLibrary:   "libonnxruntime.dylib",
				libraryGlob:      "libonnxruntime*.dylib",
			},
		},
		{
			name:   "linux amd64",
			goos:   "linux",
			goarch: "amd64",
			want: runtimeArtifact{
				platform:         "linux-x64",
				archiveExtension: "tgz",
				primaryLibrary:   "libonnxruntime.so",
				libraryGlob:      "libonnxruntime.so*",
			},
		},
		{
			name:   "linux arm64",
			goos:   "linux",
			goarch: "arm64",
			want: runtimeArtifact{
				platform:         "linux-aarch64",
				archiveExtension: "tgz",
				primaryLibrary:   "libonnxruntime.so",
				libraryGlob:      "libonnxruntime.so*",
			},
		},
		{
			name:   "windows amd64",
			goos:   "windows",
			goarch: "amd64",
			want: runtimeArtifact{
				platform:         "win-x64",
				archiveExtension: "zip",
				primaryLibrary:   "onnxruntime.dll",
				libraryGlob:      "onnxruntime*.dll",
			},
		},
		{
			name:   "windows arm64",
			goos:   "windows",
			goarch: "arm64",
			want: runtimeArtifact{
				platform:         "win-arm64",
				archiveExtension: "zip",
				primaryLibrary:   "onnxruntime.dll",
				libraryGlob:      "onnxruntime*.dll",
			},
		},
		{
			name:    "unsupported",
			goos:    "linux",
			goarch:  "386",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolveRuntimeArtifact(tc.goos, tc.goarch)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("unexpected artifact resolution: got %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestEnsureOnnxRuntimeSharedLibraryWithExplicitPath(t *testing.T) {
	clearBootstrapEnv(t)

	tmpDir := t.TempDir()
	libPath := filepath.Join(tmpDir, "libonnxruntime.so")
	if err := os.WriteFile(libPath, []byte("dummy"), 0o644); err != nil {
		t.Fatalf("failed to write test library: %v", err)
	}

	resolved, err := EnsureOnnxRuntimeSharedLibrary(WithBootstrapLibraryPath(libPath))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	want, _ := filepath.Abs(libPath)
	if resolved != want {
		t.Fatalf("unexpected resolved path: got %q, want %q", resolved, want)
	}
}

func TestEnsureOnnxRuntimeSharedLibraryDownloadAndCache(t *testing.T) {
	clearBootstrapEnv(t)

	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}

	cacheDir := t.TempDir()
	version := "1.99.1"
	archiveBytes := buildORTArchive(t, artifact, version, true)
	server, hits := newArchiveServer(t, artifact, version, archiveBytes)

	opts := []BootstrapOption{
		WithBootstrapCacheDir(cacheDir),
		WithBootstrapVersion(version),
		withBootstrapBaseURL(server.URL),
		withBootstrapHTTPClient(server.Client()),
	}

	firstPath, err := EnsureOnnxRuntimeSharedLibrary(opts...)
	if err != nil {
		t.Fatalf("unexpected bootstrap error: %v", err)
	}
	if _, statErr := os.Stat(firstPath); statErr != nil {
		t.Fatalf("resolved library path does not exist: %v", statErr)
	}

	secondPath, err := EnsureOnnxRuntimeSharedLibrary(opts...)
	if err != nil {
		t.Fatalf("unexpected bootstrap error on second call: %v", err)
	}
	if firstPath != secondPath {
		t.Fatalf("expected stable resolved path, got %q and %q", firstPath, secondPath)
	}

	if got := hits.Load(); got != 1 {
		t.Fatalf("expected exactly one archive download, got %d", got)
	}
}

func TestEnsureOnnxRuntimeSharedLibraryConcurrentLockSingleDownload(t *testing.T) {
	clearBootstrapEnv(t)

	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}

	cacheDir := t.TempDir()
	version := "1.99.2"
	archiveBytes := buildORTArchive(t, artifact, version, true)
	server, hits := newArchiveServer(t, artifact, version, archiveBytes)

	opts := []BootstrapOption{
		WithBootstrapCacheDir(cacheDir),
		WithBootstrapVersion(version),
		withBootstrapBaseURL(server.URL),
		withBootstrapHTTPClient(server.Client()),
	}

	const workers = 8
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	pathCh := make(chan string, workers)

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			path, err := EnsureOnnxRuntimeSharedLibrary(opts...)
			if err != nil {
				errCh <- err
				return
			}
			pathCh <- path
		}()
	}

	wg.Wait()
	close(errCh)
	close(pathCh)

	for err := range errCh {
		t.Fatalf("unexpected bootstrap error in concurrent call: %v", err)
	}

	var expectedPath string
	for path := range pathCh {
		if expectedPath == "" {
			expectedPath = path
			continue
		}
		if path != expectedPath {
			t.Fatalf("expected same resolved path across workers, got %q and %q", expectedPath, path)
		}
	}

	if got := hits.Load(); got != 1 {
		t.Fatalf("expected exactly one download under concurrent access, got %d", got)
	}
}

func TestEnsureOnnxRuntimeSharedLibraryChecksumMismatch(t *testing.T) {
	clearBootstrapEnv(t)

	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}

	cacheDir := t.TempDir()
	version := "1.99.3"
	archiveBytes := buildORTArchive(t, artifact, version, true)
	server, _ := newArchiveServer(t, artifact, version, archiveBytes)

	_, err = EnsureOnnxRuntimeSharedLibrary(
		WithBootstrapCacheDir(cacheDir),
		WithBootstrapVersion(version),
		WithBootstrapExpectedSHA256(strings.Repeat("0", 64)),
		withBootstrapBaseURL(server.URL),
		withBootstrapHTTPClient(server.Client()),
	)
	if err == nil {
		t.Fatalf("expected checksum mismatch error")
	}
	if !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("expected checksum mismatch error, got: %v", err)
	}
}

func TestEnsureOnnxRuntimeSharedLibraryDisableDownload(t *testing.T) {
	clearBootstrapEnv(t)

	_, err := EnsureOnnxRuntimeSharedLibrary(
		WithBootstrapCacheDir(t.TempDir()),
		WithBootstrapVersion("1.99.4"),
		WithBootstrapDisableDownload(true),
	)
	if err == nil {
		t.Fatalf("expected error when download is disabled and cache is empty")
	}
	if !strings.Contains(err.Error(), "download is disabled") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEnsureOnnxRuntimeSharedLibraryInvalidArchive(t *testing.T) {
	clearBootstrapEnv(t)

	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}

	cacheDir := t.TempDir()
	version := "1.99.5"
	archiveBytes := buildORTArchive(t, artifact, version, false)
	server, _ := newArchiveServer(t, artifact, version, archiveBytes)

	_, err = EnsureOnnxRuntimeSharedLibrary(
		WithBootstrapCacheDir(cacheDir),
		WithBootstrapVersion(version),
		withBootstrapBaseURL(server.URL),
		withBootstrapHTTPClient(server.Client()),
	)
	if err == nil {
		t.Fatalf("expected invalid archive error")
	}
	if !strings.Contains(err.Error(), "did not contain expected shared library") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEnsureOnnxRuntimeSharedLibraryInvalidArchiveMentionsSkippedLibraryLinks(t *testing.T) {
	clearBootstrapEnv(t)

	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}
	if artifact.archiveExtension != "tgz" {
		t.Skipf("symlink extraction behavior only applies to tgz archives, got %q", artifact.archiveExtension)
	}

	cacheDir := t.TempDir()
	version := "1.99.51"
	archiveBytes := buildORTArchiveWithLibrarySymlinkOnly(t, artifact, version)
	server, _ := newArchiveServer(t, artifact, version, archiveBytes)

	_, err = EnsureOnnxRuntimeSharedLibrary(
		WithBootstrapCacheDir(cacheDir),
		WithBootstrapVersion(version),
		withBootstrapBaseURL(server.URL),
		withBootstrapHTTPClient(server.Client()),
	)
	if err == nil {
		t.Fatalf("expected invalid archive error")
	}
	if !strings.Contains(err.Error(), "did not contain expected shared library") {
		t.Fatalf("expected shared library missing error, got: %v", err)
	}
	if !strings.Contains(err.Error(), "skipped") || !strings.Contains(err.Error(), artifact.libraryGlob) {
		t.Fatalf("expected skipped-library-link context in error, got: %v", err)
	}
}

func TestEnsureOnnxRuntimeSharedLibraryChecksumMatch(t *testing.T) {
	clearBootstrapEnv(t)

	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}

	cacheDir := t.TempDir()
	version := "1.99.6"
	archiveBytes := buildORTArchive(t, artifact, version, true)
	hash := sha256.Sum256(archiveBytes)
	checksum := hex.EncodeToString(hash[:])
	server, _ := newArchiveServer(t, artifact, version, archiveBytes)

	path, err := EnsureOnnxRuntimeSharedLibrary(
		WithBootstrapCacheDir(cacheDir),
		WithBootstrapVersion(version),
		WithBootstrapExpectedSHA256(checksum),
		withBootstrapBaseURL(server.URL),
		withBootstrapHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("unexpected error with valid checksum: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected resolved library path to exist: %v", err)
	}
}

func TestDownloadRuntimeArchiveCleansTempFileOnError(t *testing.T) {
	clearBootstrapEnv(t)

	cacheDir := t.TempDir()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		cacheDir:   cacheDir,
		httpClient: server.Client(),
	}

	_, _, err := downloadRuntimeArchive(cfg, server.URL+"/archive")
	if err == nil {
		t.Fatalf("expected error for empty archive response")
	}

	matches, globErr := filepath.Glob(filepath.Join(cacheDir, "onnxruntime-*.archive"))
	if globErr != nil {
		t.Fatalf("unexpected glob error: %v", globErr)
	}
	if len(matches) != 0 {
		t.Fatalf("expected no temp archives after failed download, found %v", matches)
	}
}

func TestDownloadRuntimeArchiveHTTPStatusError(t *testing.T) {
	clearBootstrapEnv(t)

	cacheDir := t.TempDir()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("service unavailable"))
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		cacheDir:        cacheDir,
		httpClient:      server.Client(),
		maxDownloadSize: 1024,
	}

	_, _, err := downloadRuntimeArchive(cfg, server.URL+"/archive")
	if err == nil {
		t.Fatalf("expected HTTP status download error")
	}
	if !strings.Contains(err.Error(), "HTTP 503") {
		t.Fatalf("expected HTTP status in error, got: %v", err)
	}
}

func TestDownloadRuntimeArchiveRejectsOversize(t *testing.T) {
	clearBootstrapEnv(t)

	cacheDir := t.TempDir()
	payload := bytes.Repeat([]byte("a"), 64)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		cacheDir:        cacheDir,
		httpClient:      server.Client(),
		maxDownloadSize: 16,
	}

	_, _, err := downloadRuntimeArchive(cfg, server.URL+"/archive")
	if err == nil {
		t.Fatalf("expected oversize archive error")
	}
	if !strings.Contains(err.Error(), "exceeds maximum size limit") {
		t.Fatalf("unexpected oversize error: %v", err)
	}

	matches, globErr := filepath.Glob(filepath.Join(cacheDir, "onnxruntime-*.archive"))
	if globErr != nil {
		t.Fatalf("unexpected glob error: %v", globErr)
	}
	if len(matches) != 0 {
		t.Fatalf("expected no temp archives after oversize rejection, found %v", matches)
	}
}

func TestDownloadRuntimeArchiveRejectsOversizeByContentLengthHeader(t *testing.T) {
	clearBootstrapEnv(t)

	cacheDir := t.TempDir()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "64")
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		cacheDir:        cacheDir,
		httpClient:      server.Client(),
		maxDownloadSize: 16,
	}

	_, _, err := downloadRuntimeArchive(cfg, server.URL+"/archive")
	if err == nil {
		t.Fatalf("expected oversize archive error")
	}
	if !strings.Contains(err.Error(), "content-length=64") {
		t.Fatalf("expected content-length oversize error, got: %v", err)
	}

	matches, globErr := filepath.Glob(filepath.Join(cacheDir, "onnxruntime-*.archive"))
	if globErr != nil {
		t.Fatalf("unexpected glob error: %v", globErr)
	}
	if len(matches) != 0 {
		t.Fatalf("expected no temp archives after content-length rejection, found %v", matches)
	}
}

func TestResolveExtractedLibraryPathDistinguishesInvalidCandidates(t *testing.T) {
	installDir := t.TempDir()
	libDir := filepath.Join(installDir, "lib")
	if err := os.MkdirAll(libDir, 0o755); err != nil {
		t.Fatalf("failed to create lib directory: %v", err)
	}

	primary := filepath.Join(libDir, "libonnxruntime.so")
	if err := os.WriteFile(primary, nil, 0o644); err != nil {
		t.Fatalf("failed to create invalid primary library: %v", err)
	}
	alt := filepath.Join(libDir, "libonnxruntime.so.1")
	if err := os.WriteFile(alt, nil, 0o644); err != nil {
		t.Fatalf("failed to create invalid alternative library: %v", err)
	}

	_, err := resolveExtractedLibraryPath(installDir, runtimeArtifact{
		primaryLibrary: "libonnxruntime.so",
		libraryGlob:    "libonnxruntime.so*",
	})
	if err == nil {
		t.Fatalf("expected invalid-candidate error")
	}
	if errors.Is(err, errSharedLibraryNotFound) {
		t.Fatalf("expected invalid-candidate error, got not-found: %v", err)
	}
	if !strings.Contains(err.Error(), "none are valid") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestResolveExtractedLibraryPathReturnsNotFoundWhenMissing(t *testing.T) {
	installDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(installDir, "lib"), 0o755); err != nil {
		t.Fatalf("failed to create lib directory: %v", err)
	}

	_, err := resolveExtractedLibraryPath(installDir, runtimeArtifact{
		primaryLibrary: "libonnxruntime.so",
		libraryGlob:    "libonnxruntime.so*",
	})
	if !errors.Is(err, errSharedLibraryNotFound) {
		t.Fatalf("expected not-found error, got: %v", err)
	}
}

func TestWithBootstrapVersionRejectsEmpty(t *testing.T) {
	var cfg bootstrapConfig
	if err := WithBootstrapVersion("   ")(&cfg); err == nil {
		t.Fatalf("expected empty version validation error")
	}
}

func TestWithBootstrapLibraryPathAndCacheDirRejectEmpty(t *testing.T) {
	var cfg bootstrapConfig

	if err := WithBootstrapLibraryPath("   ")(&cfg); err == nil {
		t.Fatalf("expected empty library path validation error")
	}
	if err := WithBootstrapCacheDir("   ")(&cfg); err == nil {
		t.Fatalf("expected empty cache directory validation error")
	}
}

func TestWithBootstrapExpectedSHA256Validation(t *testing.T) {
	tests := []struct {
		name     string
		checksum string
		wantErr  bool
		want     string
	}{
		{name: "empty", checksum: "", wantErr: true},
		{name: "short", checksum: strings.Repeat("a", 63), wantErr: true},
		{name: "long", checksum: strings.Repeat("a", 65), wantErr: true},
		{name: "uppercase", checksum: strings.Repeat("A", 64), wantErr: false, want: strings.Repeat("a", 64)},
		{name: "non-hex", checksum: strings.Repeat("g", 64), wantErr: true},
		{name: "valid", checksum: strings.Repeat("a", 64), wantErr: false, want: strings.Repeat("a", 64)},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var cfg bootstrapConfig
			err := WithBootstrapExpectedSHA256(tc.checksum)(&cfg)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected validation error for checksum %q", tc.checksum)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected checksum validation error: %v", err)
			}
			if cfg.expectedSHA256 != tc.want {
				t.Fatalf("unexpected stored checksum: got %q, want %q", cfg.expectedSHA256, tc.want)
			}
		})
	}
}

func TestWithBootstrapBaseURLValidation(t *testing.T) {
	var cfg bootstrapConfig

	tests := []struct {
		name    string
		baseURL string
		wantErr bool
	}{
		{name: "reject non-loopback http", baseURL: "http://example.com", wantErr: true},
		{name: "accept https", baseURL: "https://example.com", wantErr: false},
		{name: "accept loopback ipv4 http", baseURL: "http://127.0.0.1:8080", wantErr: false},
		{name: "accept localhost http", baseURL: "http://localhost:8080", wantErr: false},
		{name: "accept loopback ipv6 http", baseURL: "http://[::1]:8080", wantErr: false},
		{name: "reject ftp", baseURL: "ftp://example.com", wantErr: true},
		{name: "reject schemeless URL", baseURL: "example.com/path", wantErr: true},
		{name: "reject hostless https", baseURL: "https://", wantErr: true},
		{name: "reject bare path", baseURL: "/tmp/archive-root", wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := withBootstrapBaseURL(tc.baseURL)(&cfg)
			if tc.wantErr && err == nil {
				t.Fatalf("expected validation error for %q", tc.baseURL)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected validation error for %q: %v", tc.baseURL, err)
			}
		})
	}
}

func TestResolveBootstrapConfigRespectsEnvOverrides(t *testing.T) {
	clearBootstrapEnv(t)
	t.Setenv("ONNXRUNTIME_LIB_PATH", " ./libonnxruntime.so ")
	t.Setenv("ONNXRUNTIME_CACHE_DIR", " ./cache-dir ")
	t.Setenv("ONNXRUNTIME_VERSION", " v1.2.3 ")

	cfg, err := resolveBootstrapConfig()
	if err != nil {
		t.Fatalf("unexpected resolveBootstrapConfig error: %v", err)
	}
	if cfg.libraryPath != "./libonnxruntime.so" {
		t.Fatalf("unexpected library path: got %q", cfg.libraryPath)
	}
	if cfg.cacheDir != filepath.Clean("./cache-dir") {
		t.Fatalf("unexpected cache dir: got %q, want %q", cfg.cacheDir, filepath.Clean("./cache-dir"))
	}
	if cfg.version != "1.2.3" {
		t.Fatalf("unexpected normalized version: got %q, want 1.2.3", cfg.version)
	}
}

func TestParseBootstrapBoolEnv(t *testing.T) {
	t.Setenv("ONNXRUNTIME_DISABLE_DOWNLOAD", "")
	parsed, err := parseBootstrapBoolEnv("ONNXRUNTIME_DISABLE_DOWNLOAD")
	if err != nil || parsed {
		t.Fatalf("expected default false with no error, got parsed=%v err=%v", parsed, err)
	}

	tests := []struct {
		value     string
		want      bool
		expectErr bool
	}{
		{value: "true", want: true},
		{value: "false", want: false},
		{value: "1", want: true},
		{value: "0", want: false},
		{value: "yes", want: true},
		{value: "no", want: false},
		{value: "on", want: true},
		{value: "off", want: false},
		{value: "disabled", expectErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.value, func(t *testing.T) {
			t.Setenv("ONNXRUNTIME_DISABLE_DOWNLOAD", tc.value)
			got, err := parseBootstrapBoolEnv("ONNXRUNTIME_DISABLE_DOWNLOAD")
			if tc.expectErr {
				if err == nil {
					t.Fatalf("expected parse error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected parse error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("unexpected parsed value: got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestResolveBootstrapConfigRejectsInvalidDisableDownloadEnv(t *testing.T) {
	clearBootstrapEnv(t)
	t.Setenv("ONNXRUNTIME_DISABLE_DOWNLOAD", "disabled")

	_, err := resolveBootstrapConfig()
	if err == nil {
		t.Fatalf("expected invalid env parse error")
	}
	if !strings.Contains(err.Error(), "ONNXRUNTIME_DISABLE_DOWNLOAD") {
		t.Fatalf("expected variable name in error, got: %v", err)
	}
}

func TestValidateLibraryFile(t *testing.T) {
	if _, err := validateLibraryFile("   "); err == nil {
		t.Fatalf("expected empty library path error")
	}

	dir := t.TempDir()
	if _, err := validateLibraryFile(dir); err == nil {
		t.Fatalf("expected directory library path error")
	}

	zeroPath := filepath.Join(dir, "libonnxruntime-empty.so")
	if err := os.WriteFile(zeroPath, nil, 0o644); err != nil {
		t.Fatalf("failed to create zero-size library file: %v", err)
	}
	if _, err := validateLibraryFile(zeroPath); err == nil {
		t.Fatalf("expected zero-size library file error")
	}

	validPath := filepath.Join(dir, "libonnxruntime.so")
	if err := os.WriteFile(validPath, []byte("onnxruntime"), 0o644); err != nil {
		t.Fatalf("failed to create valid library file: %v", err)
	}
	resolved, err := validateLibraryFile(validPath)
	if err != nil {
		t.Fatalf("unexpected valid library file error: %v", err)
	}
	want, _ := filepath.Abs(validPath)
	if resolved != want {
		t.Fatalf("unexpected resolved path: got %q, want %q", resolved, want)
	}
}

func TestCopyExtractedFileLimits(t *testing.T) {
	if err := copyExtractedFile(io.Discard, strings.NewReader(""), maxExtractedFileBytes+1, nil, "big.bin"); err == nil {
		t.Fatalf("expected per-file extraction limit error")
	}

	total := maxExtractedTotalBytes - 2
	if err := copyExtractedFile(io.Discard, strings.NewReader("1234"), 4, &total, "cumulative.bin"); err == nil {
		t.Fatalf("expected cumulative extraction limit error")
	}

	var totalWritten int64
	if err := copyExtractedFile(io.Discard, strings.NewReader("abc"), 5, &totalWritten, "short.bin"); err == nil {
		t.Fatalf("expected size mismatch extraction error")
	}

	var okTotal int64
	if err := copyExtractedFile(io.Discard, strings.NewReader("hello"), 5, &okTotal, "ok.bin"); err != nil {
		t.Fatalf("unexpected extraction error for valid sizes: %v", err)
	}
	if okTotal != 5 {
		t.Fatalf("unexpected total extracted bytes: got %d, want 5", okTotal)
	}
}

func TestWithProcessFileLockTimesOut(t *testing.T) {
	lockPath := filepath.Join(t.TempDir(), "bootstrap.lock")

	oldTimeout := bootstrapLockAcquireTimeout
	oldRetry := bootstrapLockRetryInterval
	oldLogInterval := bootstrapLockLogInterval
	bootstrapLockAcquireTimeout = 80 * time.Millisecond
	bootstrapLockRetryInterval = 5 * time.Millisecond
	bootstrapLockLogInterval = 15 * time.Millisecond
	t.Cleanup(func() {
		bootstrapLockAcquireTimeout = oldTimeout
		bootstrapLockRetryInterval = oldRetry
		bootstrapLockLogInterval = oldLogInterval
	})

	locked := make(chan struct{})
	release := make(chan struct{})
	holderErrCh := make(chan error, 1)
	go func() {
		holderErrCh <- withProcessFileLock(lockPath, func() error {
			close(locked)
			<-release
			return nil
		})
	}()

	select {
	case <-locked:
	case <-time.After(time.Second):
		t.Fatalf("timed out waiting for lock holder to acquire lock")
	}

	err := withProcessFileLock(lockPath, func() error { return nil })
	if err == nil {
		t.Fatalf("expected timeout while waiting for lock")
	}
	if !strings.Contains(err.Error(), "timed out acquiring lock") {
		t.Fatalf("unexpected lock timeout error: %v", err)
	}

	close(release)
	if holderErr := <-holderErrCh; holderErr != nil {
		t.Fatalf("unexpected lock holder error: %v", holderErr)
	}
}

func TestWithProcessFileLockRejectsNilCallback(t *testing.T) {
	lockPath := filepath.Join(t.TempDir(), "bootstrap.lock")
	err := withProcessFileLock(lockPath, nil)
	if err == nil || !strings.Contains(err.Error(), "lock callback is nil") {
		t.Fatalf("expected nil callback error, got: %v", err)
	}
}

func TestSecureArchiveJoin(t *testing.T) {
	baseDir := t.TempDir()

	path, err := secureArchiveJoin(baseDir, "onnxruntime/lib/libonnxruntime.so")
	if err != nil {
		t.Fatalf("expected valid path, got error: %v", err)
	}
	if !strings.HasPrefix(path, baseDir+string(os.PathSeparator)) {
		t.Fatalf("expected path to stay in base dir, got %q", path)
	}

	tests := []string{
		"",
		"/etc/passwd",
		"../evil",
		"..\\evil",
		"a/../../evil",
		"C:\\windows\\system32\\kernel32.dll",
	}

	for _, candidate := range tests {
		t.Run(candidate, func(t *testing.T) {
			_, err := secureArchiveJoin(baseDir, candidate)
			if err == nil {
				t.Fatalf("expected secureArchiveJoin to reject %q", candidate)
			}
		})
	}
}

func TestNormalizeRuntimeVersion(t *testing.T) {
	tests := []struct {
		name      string
		in        string
		want      string
		expectErr bool
	}{
		{name: "plain", in: "1.23.1", want: "1.23.1"},
		{name: "prefixed", in: "v1.23.1", want: "1.23.1"},
		{name: "trimmed", in: " 1.2.3 ", want: "1.2.3"},
		{name: "empty", in: "", expectErr: true},
		{name: "too few segments", in: "1.2", expectErr: true},
		{name: "too many segments", in: "1.2.3.4", expectErr: true},
		{name: "empty segment", in: "1..3", expectErr: true},
		{name: "non-numeric", in: "1.a.3", expectErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := normalizeRuntimeVersion(tc.in)
			if tc.expectErr {
				if err == nil {
					t.Fatalf("expected error for %q", tc.in)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("unexpected normalized version: got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestExtractArchiveFileCrossFormat(t *testing.T) {
	files := map[string]string{
		"onnxruntime-sample/lib/libonnxruntime.so": "library-bytes",
		"onnxruntime-sample/include/header.h":      "header",
	}

	testCases := []struct {
		name      string
		extension string
		data      []byte
	}{
		{name: "tgz", extension: "tgz", data: buildTGZArchive(t, files)},
		{name: "zip", extension: "zip", data: buildZIPArchive(t, files)},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			archivePath := filepath.Join(t.TempDir(), "archive."+tc.extension)
			if err := os.WriteFile(archivePath, tc.data, 0o644); err != nil {
				t.Fatalf("failed to write archive: %v", err)
			}

			destDir := t.TempDir()
			if _, err := extractArchiveFile(archivePath, destDir, tc.extension, ""); err != nil {
				t.Fatalf("unexpected extraction error: %v", err)
			}

			extractedLib := filepath.Join(destDir, "onnxruntime-sample", "lib", "libonnxruntime.so")
			if _, err := os.Stat(extractedLib); err != nil {
				t.Fatalf("expected extracted library file at %q: %v", extractedLib, err)
			}
		})
	}
}

func TestExtractTGZArchiveSkipsSymlinkEntries(t *testing.T) {
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)

	const regularPath = "onnxruntime-sample/lib/libonnxruntime-real.so"
	regularContent := []byte("regular-library")
	if err := tw.WriteHeader(&tar.Header{
		Name: regularPath,
		Mode: 0o644,
		Size: int64(len(regularContent)),
	}); err != nil {
		t.Fatalf("failed to write regular tar header: %v", err)
	}
	if _, err := tw.Write(regularContent); err != nil {
		t.Fatalf("failed to write regular tar payload: %v", err)
	}

	const symlinkPath = "onnxruntime-sample/lib/libonnxruntime.so"
	if err := tw.WriteHeader(&tar.Header{
		Name:     symlinkPath,
		Mode:     0o777,
		Typeflag: tar.TypeSymlink,
		Linkname: "libonnxruntime-real.so",
	}); err != nil {
		t.Fatalf("failed to write symlink tar header: %v", err)
	}

	if err := tw.Close(); err != nil {
		t.Fatalf("failed to close tar writer: %v", err)
	}
	if err := gz.Close(); err != nil {
		t.Fatalf("failed to close gzip writer: %v", err)
	}

	archivePath := filepath.Join(t.TempDir(), "archive.tgz")
	if err := os.WriteFile(archivePath, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("failed to write tgz archive: %v", err)
	}

	destDir := t.TempDir()
	report, err := extractArchiveFile(archivePath, destDir, "tgz", "libonnxruntime*.so")
	if err != nil {
		t.Fatalf("unexpected extraction error: %v", err)
	}

	extractedRegular := filepath.Join(destDir, filepath.FromSlash(regularPath))
	if _, err := os.Stat(extractedRegular); err != nil {
		t.Fatalf("expected regular file to be extracted: %v", err)
	}

	extractedSymlink := filepath.Join(destDir, filepath.FromSlash(symlinkPath))
	if _, err := os.Lstat(extractedSymlink); err == nil {
		t.Fatalf("expected symlink entry to be skipped, but found %q", extractedSymlink)
	} else if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("unexpected symlink lstat error: %v", err)
	}
	if report.skippedLinkEntries == 0 {
		t.Fatalf("expected skipped link entries in extraction report")
	}
	if report.skippedLibraryLinkEntries == 0 {
		t.Fatalf("expected skipped library link entries in extraction report")
	}
}

func TestExtractZIPArchiveSkipsSymlinkEntries(t *testing.T) {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)

	const regularPath = "onnxruntime-sample/lib/onnxruntime-real.dll"
	regularEntry, err := zw.Create(regularPath)
	if err != nil {
		t.Fatalf("failed to create regular zip entry: %v", err)
	}
	if _, err := regularEntry.Write([]byte("regular-library")); err != nil {
		t.Fatalf("failed to write regular zip entry: %v", err)
	}

	const symlinkPath = "onnxruntime-sample/lib/onnxruntime.dll"
	symlinkHeader := &zip.FileHeader{Name: symlinkPath, Method: zip.Deflate}
	symlinkHeader.SetMode(os.ModeSymlink | 0o777)
	symlinkEntry, err := zw.CreateHeader(symlinkHeader)
	if err != nil {
		t.Fatalf("failed to create symlink zip entry: %v", err)
	}
	if _, err := symlinkEntry.Write([]byte("onnxruntime-real.dll")); err != nil {
		t.Fatalf("failed to write symlink zip payload: %v", err)
	}

	if err := zw.Close(); err != nil {
		t.Fatalf("failed to close zip writer: %v", err)
	}

	archivePath := filepath.Join(t.TempDir(), "archive.zip")
	if err := os.WriteFile(archivePath, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("failed to write zip archive: %v", err)
	}

	destDir := t.TempDir()
	report, err := extractArchiveFile(archivePath, destDir, "zip", "onnxruntime*.dll")
	if err != nil {
		t.Fatalf("unexpected extraction error: %v", err)
	}

	extractedRegular := filepath.Join(destDir, filepath.FromSlash(regularPath))
	if _, err := os.Stat(extractedRegular); err != nil {
		t.Fatalf("expected regular file to be extracted: %v", err)
	}

	extractedSymlink := filepath.Join(destDir, filepath.FromSlash(symlinkPath))
	if _, err := os.Lstat(extractedSymlink); err == nil {
		t.Fatalf("expected symlink entry to be skipped, but found %q", extractedSymlink)
	} else if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("unexpected symlink lstat error: %v", err)
	}
	if report.skippedLinkEntries == 0 {
		t.Fatalf("expected skipped link entries in extraction report")
	}
	if report.skippedLibraryLinkEntries == 0 {
		t.Fatalf("expected skipped library link entries in extraction report")
	}
}

func TestInitializeEnvironmentWithBootstrapInitializedDifferentPath(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	dir := t.TempDir()
	currentLib := filepath.Join(dir, "lib-current.so")
	if err := os.WriteFile(currentLib, []byte("current"), 0o644); err != nil {
		t.Fatalf("failed to write current lib: %v", err)
	}
	otherLib := filepath.Join(dir, "lib-other.so")
	if err := os.WriteFile(otherLib, []byte("other"), 0o644); err != nil {
		t.Fatalf("failed to write other lib: %v", err)
	}

	absCurrent, _ := filepath.Abs(currentLib)
	mu.Lock()
	refCount = 1
	libPath = absCurrent
	mu.Unlock()

	err := InitializeEnvironmentWithBootstrap(WithBootstrapLibraryPath(otherLib))
	if err == nil {
		t.Fatalf("expected error for initialized environment with different path")
	}
	if !strings.Contains(err.Error(), "cannot change library path") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func clearBootstrapEnv(t *testing.T) {
	t.Helper()
	t.Setenv("ONNXRUNTIME_LIB_PATH", "")
	t.Setenv("ONNXRUNTIME_CACHE_DIR", "")
	t.Setenv("ONNXRUNTIME_VERSION", "")
	t.Setenv("ONNXRUNTIME_DISABLE_DOWNLOAD", "")
}

func newArchiveServer(t *testing.T, artifact runtimeArtifact, version string, archive []byte) (*httptest.Server, *atomic.Int32) {
	t.Helper()

	hits := &atomic.Int32{}
	archivePath := "/v" + version + "/" + artifact.archiveFilename(version)

	mux := http.NewServeMux()
	mux.HandleFunc(archivePath, func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		// Small delay makes concurrent lock behavior easier to observe.
		time.Sleep(40 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(archive)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	})

	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return server, hits
}

func buildORTArchive(t *testing.T, artifact runtimeArtifact, version string, includeLibrary bool) []byte {
	t.Helper()

	archiveRoot := artifact.archiveName(version)
	files := map[string]string{
		fmt.Sprintf("%s/include/onnxruntime_c_api.h", archiveRoot): "header",
	}
	if includeLibrary {
		files[fmt.Sprintf("%s/lib/%s", archiveRoot, artifact.primaryLibrary)] = "fake-onnxruntime-library-bytes"
	} else {
		files[fmt.Sprintf("%s/lib/not-onnxruntime.txt", archiveRoot)] = "not-a-library"
	}

	switch artifact.archiveExtension {
	case "tgz":
		return buildTGZArchive(t, files)
	case "zip":
		return buildZIPArchive(t, files)
	default:
		t.Fatalf("unsupported archive extension in test: %s", artifact.archiveExtension)
		return nil
	}
}

func buildORTArchiveWithLibrarySymlinkOnly(t *testing.T, artifact runtimeArtifact, version string) []byte {
	t.Helper()

	archiveRoot := artifact.archiveName(version)
	if artifact.archiveExtension != "tgz" {
		t.Fatalf("symlink-only archive helper only supports tgz, got %q", artifact.archiveExtension)
	}

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)

	headerPath := filepath.ToSlash(fmt.Sprintf("%s/include/onnxruntime_c_api.h", archiveRoot))
	headerContent := []byte("header")
	if err := tw.WriteHeader(&tar.Header{Name: headerPath, Mode: 0o644, Size: int64(len(headerContent))}); err != nil {
		t.Fatalf("failed to write header file entry: %v", err)
	}
	if _, err := tw.Write(headerContent); err != nil {
		t.Fatalf("failed to write header file payload: %v", err)
	}

	linkPath := filepath.ToSlash(fmt.Sprintf("%s/lib/%s", archiveRoot, artifact.primaryLibrary))
	if err := tw.WriteHeader(&tar.Header{
		Name:     linkPath,
		Mode:     0o777,
		Typeflag: tar.TypeSymlink,
		Linkname: "libonnxruntime-real.so",
	}); err != nil {
		t.Fatalf("failed to write library symlink entry: %v", err)
	}

	if err := tw.Close(); err != nil {
		t.Fatalf("failed to close tar writer: %v", err)
	}
	if err := gz.Close(); err != nil {
		t.Fatalf("failed to close gzip writer: %v", err)
	}

	return buf.Bytes()
}

func buildTGZArchive(t *testing.T, files map[string]string) []byte {
	t.Helper()

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)

	for name, content := range files {
		hdr := &tar.Header{
			Name: filepath.ToSlash(name),
			Mode: 0o644,
			Size: int64(len(content)),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("failed to write tar header %q: %v", name, err)
		}
		if _, err := tw.Write([]byte(content)); err != nil {
			t.Fatalf("failed to write tar entry %q: %v", name, err)
		}
	}

	if err := tw.Close(); err != nil {
		t.Fatalf("failed to close tar writer: %v", err)
	}
	if err := gz.Close(); err != nil {
		t.Fatalf("failed to close gzip writer: %v", err)
	}

	return buf.Bytes()
}

func buildZIPArchive(t *testing.T, files map[string]string) []byte {
	t.Helper()

	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)

	for name, content := range files {
		entry, err := zw.Create(filepath.ToSlash(name))
		if err != nil {
			t.Fatalf("failed to create zip entry %q: %v", name, err)
		}
		if _, err := entry.Write([]byte(content)); err != nil {
			t.Fatalf("failed to write zip entry %q: %v", name, err)
		}
	}

	if err := zw.Close(); err != nil {
		t.Fatalf("failed to close zip writer: %v", err)
	}

	return buf.Bytes()
}
