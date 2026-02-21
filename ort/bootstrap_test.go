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
			if err := extractArchiveFile(archivePath, destDir, tc.extension); err != nil {
				t.Fatalf("unexpected extraction error: %v", err)
			}

			extractedLib := filepath.Join(destDir, "onnxruntime-sample", "lib", "libonnxruntime.so")
			if _, err := os.Stat(extractedLib); err != nil {
				t.Fatalf("expected extracted library file at %q: %v", extractedLib, err)
			}
		})
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
