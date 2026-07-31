package ort

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"
)

func TestResolveRuntimeArtifact(t *testing.T) {
	tests := []struct {
		name                    string
		goos                    string
		goarch                  string
		want                    runtimeArtifact
		wantErr                 bool
		wantUnsupportedPlatform bool
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
			name:                    "unsupported",
			goos:                    "linux",
			goarch:                  "386",
			wantErr:                 true,
			wantUnsupportedPlatform: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolveRuntimeArtifact(tc.goos, tc.goarch)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				if tc.wantUnsupportedPlatform {
					if !errors.Is(err, ErrUnsupportedPlatform) {
						t.Fatalf("expected ErrUnsupportedPlatform, got: %v", err)
					}
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

	t.Run("supported platform library absence is distinct", func(t *testing.T) {
		artifact, err := resolveRuntimeArtifact("linux", "amd64")
		if err != nil {
			t.Fatalf("unexpected supported-platform resolution error: %v", err)
		}

		_, err = resolveExtractedLibraryPath(t.TempDir(), artifact)
		if !errors.Is(err, ErrSharedLibraryNotFound) {
			t.Fatalf("expected ErrSharedLibraryNotFound, got: %v", err)
		}
		if errors.Is(err, ErrUnsupportedPlatform) {
			t.Fatalf("supported-platform absence unexpectedly matched ErrUnsupportedPlatform: %v", err)
		}
	})
}

func TestBootstrapErrorChains(t *testing.T) {
	t.Run("filesystem cause and library category", func(t *testing.T) {
		missingPath := filepath.Join(t.TempDir(), "missing", "libonnxruntime.so")

		_, err := validateLibraryFile(missingPath)
		if !errors.Is(err, ErrSharedLibraryNotFound) {
			t.Fatalf("expected ErrSharedLibraryNotFound, got: %v", err)
		}
		if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("expected os.ErrNotExist in filesystem chain, got: %v", err)
		}
		if !strings.Contains(err.Error(), missingPath) {
			t.Fatalf("expected missing library path in error, got: %v", err)
		}
	})

	t.Run("network cause", func(t *testing.T) {
		networkCause := errors.New("synthetic network failure")
		const archiveURL = "https://example.invalid/onnxruntime.tgz"
		cfg := bootstrapConfig{
			cacheDir: t.TempDir(),
			httpClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, networkCause
			})},
			retryAttempts: 1,
		}

		_, _, err := downloadRuntimeArchive(cfg, archiveURL)
		if !errors.Is(err, networkCause) {
			t.Fatalf("expected network cause in error chain, got: %v", err)
		}
		if !strings.Contains(err.Error(), archiveURL) {
			t.Fatalf("expected download URL in error, got: %v", err)
		}
	})

	t.Run("checksum metadata parse cause", func(t *testing.T) {
		artifact, err := resolveRuntimeArtifact("linux", "amd64")
		if err != nil {
			t.Fatalf("unexpected artifact resolution error: %v", err)
		}

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"assets":`))
		}))
		t.Cleanup(server.Close)

		cfg := bootstrapConfig{
			version:            "1.2.3",
			releaseMetadataURL: server.URL,
			httpClient:         server.Client(),
			retryAttempts:      1,
		}
		_, err = resolveRuntimeArchiveChecksumFromReleaseMetadata(cfg, artifact)
		var syntaxErr *json.SyntaxError
		if !errors.As(err, &syntaxErr) {
			t.Fatalf("expected JSON syntax cause in checksum error chain, got: %v", err)
		}
		if !strings.Contains(err.Error(), server.URL) {
			t.Fatalf("expected metadata URL in error, got: %v", err)
		}
	})

	t.Run("archive cause", func(t *testing.T) {
		archivePath := filepath.Join(t.TempDir(), "invalid.tgz")
		if err := os.WriteFile(archivePath, []byte("not a gzip archive"), 0o600); err != nil {
			t.Fatalf("failed to write invalid archive: %v", err)
		}

		_, err := extractTGZArchive(archivePath, t.TempDir(), "")
		if !errors.Is(err, gzip.ErrHeader) {
			t.Fatalf("expected gzip.ErrHeader in archive error chain, got: %v", err)
		}
		// The error quotes the path with %q, which escapes Windows separators.
		if !strings.Contains(err.Error(), fmt.Sprintf("%q", archivePath)) {
			t.Fatalf("expected archive path in error, got: %v", err)
		}
	})

	t.Run("dynamic library cause", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		libPath := filepath.Join(t.TempDir(), "libonnxruntime.so")
		if err := os.WriteFile(libPath, []byte("synthetic library"), 0o600); err != nil {
			t.Fatalf("failed to write synthetic library: %v", err)
		}
		resolvedLibPath, resolveErr := filepath.EvalSymlinks(libPath)
		if resolveErr != nil {
			t.Fatalf("resolve synthetic library path: %v", resolveErr)
		}
		loadCause := errors.New("synthetic dynamic loader failure")
		mu.Lock()
		environmentLoadLibrary = func(path string) (uintptr, error) {
			return 0, &os.PathError{Op: "dlopen", Path: path, Err: loadCause}
		}
		mu.Unlock()

		err := InitializeEnvironmentWithBootstrap(WithBootstrapLibraryPath(libPath))
		if !errors.Is(err, loadCause) {
			t.Fatalf("expected dynamic loader cause in error chain, got: %v", err)
		}
		var pathErr *os.PathError
		if !errors.As(err, &pathErr) {
			t.Fatalf("expected *os.PathError in dynamic loader chain, got: %v", err)
		}
		if pathErr.Path != resolvedLibPath {
			t.Fatalf("loader path = %q, want resolved path %q", pathErr.Path, resolvedLibPath)
		}
	})

	t.Run("primary and cleanup causes", func(t *testing.T) {
		primaryCause := errors.New("synthetic response read failure")
		cleanupCause := errors.New("synthetic response close failure")
		const archiveURL = "https://example.invalid/onnxruntime.tgz"
		cfg := bootstrapConfig{
			cacheDir: t.TempDir(),
			httpClient: &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body: &failingReadCloser{
						readErr:  primaryCause,
						closeErr: cleanupCause,
					},
					Request: req,
				}, nil
			})},
			maxDownloadSize: 1024,
			retryAttempts:   1,
		}

		_, _, err := downloadRuntimeArchive(cfg, archiveURL)
		if !errors.Is(err, primaryCause) {
			t.Fatalf("expected primary cause in joined error, got: %v", err)
		}
		if !errors.Is(err, cleanupCause) {
			t.Fatalf("expected cleanup cause in joined error, got: %v", err)
		}
		if !strings.Contains(err.Error(), archiveURL) {
			t.Fatalf("expected download URL in joined error, got: %v", err)
		}
	})
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

	want, err := filepath.EvalSymlinks(libPath)
	if err != nil {
		t.Fatalf("resolve expected library path: %v", err)
	}
	if resolved != want {
		t.Fatalf("unexpected resolved path: got %q, want %q", resolved, want)
	}
}

func TestEnsureOnnxRuntimeSharedLibraryExplicitSymlink(t *testing.T) {
	sources := []struct {
		name      string
		configure func(*testing.T, string) []BootstrapOption
	}{
		{
			name: "option",
			configure: func(_ *testing.T, path string) []BootstrapOption {
				return []BootstrapOption{WithBootstrapLibraryPath(path)}
			},
		},
		{
			name: "environment",
			configure: func(t *testing.T, path string) []BootstrapOption {
				t.Setenv("ONNXRUNTIME_LIB_PATH", path)
				return nil
			},
		},
	}

	for _, source := range sources {
		t.Run(source.name, func(t *testing.T) {
			clearBootstrapEnv(t)

			dir := t.TempDir()
			target := filepath.Join(dir, "libonnxruntime.so.1.24.1")
			if err := os.WriteFile(target, []byte("onnxruntime"), 0o600); err != nil {
				t.Fatalf("write explicit library target: %v", err)
			}
			link := filepath.Join(dir, "libonnxruntime.so")
			if err := os.Symlink(target, link); err != nil {
				t.Skipf("cannot create symlink on this platform: %v", err)
			}

			resolved, err := EnsureOnnxRuntimeSharedLibrary(source.configure(t, link)...)
			if err != nil {
				t.Fatalf("resolve explicit library symlink: %v", err)
			}
			want, err := filepath.EvalSymlinks(target)
			if err != nil {
				t.Fatalf("resolve expected target path: %v", err)
			}
			if resolved != want {
				t.Fatalf("resolved path = %q, want symlink target %q", resolved, want)
			}
		})
	}

	t.Run("dangling target", func(t *testing.T) {
		clearBootstrapEnv(t)

		dir := t.TempDir()
		link := filepath.Join(dir, "libonnxruntime.so")
		if err := os.Symlink(filepath.Join(dir, "missing.so"), link); err != nil {
			t.Skipf("cannot create symlink on this platform: %v", err)
		}

		_, err := EnsureOnnxRuntimeSharedLibrary(WithBootstrapLibraryPath(link))
		if !errors.Is(err, ErrSharedLibraryNotFound) {
			t.Fatalf("dangling explicit symlink error = %v, want ErrSharedLibraryNotFound", err)
		}
	})

	t.Run("directory target", func(t *testing.T) {
		clearBootstrapEnv(t)

		dir := t.TempDir()
		target := filepath.Join(dir, "runtime")
		if err := os.Mkdir(target, 0o700); err != nil {
			t.Fatalf("create directory target: %v", err)
		}
		link := filepath.Join(dir, "libonnxruntime.so")
		if err := os.Symlink(target, link); err != nil {
			t.Skipf("cannot create symlink on this platform: %v", err)
		}

		if _, err := EnsureOnnxRuntimeSharedLibrary(WithBootstrapLibraryPath(link)); err == nil {
			t.Fatal("expected explicit symlink to a directory to be rejected")
		}
	})

	t.Run("empty file target", func(t *testing.T) {
		clearBootstrapEnv(t)

		dir := t.TempDir()
		target := filepath.Join(dir, "libonnxruntime.so.1.24.1")
		if err := os.WriteFile(target, nil, 0o600); err != nil {
			t.Fatalf("write empty target: %v", err)
		}
		link := filepath.Join(dir, "libonnxruntime.so")
		if err := os.Symlink(target, link); err != nil {
			t.Skipf("cannot create symlink on this platform: %v", err)
		}

		if _, err := EnsureOnnxRuntimeSharedLibrary(WithBootstrapLibraryPath(link)); err == nil {
			t.Fatal("expected explicit symlink to an empty file to be rejected")
		}
	})
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

func TestEnsureOnnxRuntimeSharedLibraryReplacesUntrustedCacheEntry(t *testing.T) {
	clearBootstrapEnv(t)

	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}

	cacheDir := t.TempDir()
	version := "1.99.61"
	installDir := filepath.Join(cacheDir, artifact.archiveName(version))
	libDir := filepath.Join(installDir, "lib")
	if err := os.MkdirAll(libDir, secureDirectoryPermission); err != nil {
		t.Fatalf("create planted cache: %v", err)
	}
	plantedPath := filepath.Join(libDir, artifact.primaryLibrary)
	if err := os.WriteFile(plantedPath, []byte("planted-library"), 0o600); err != nil {
		t.Fatalf("write planted library: %v", err)
	}

	archiveBytes := buildORTArchive(t, artifact, version, true)
	sum := sha256.Sum256(archiveBytes)
	server, hits := newArchiveServer(t, artifact, version, archiveBytes)

	resolved, err := EnsureOnnxRuntimeSharedLibrary(
		WithBootstrapCacheDir(cacheDir),
		WithBootstrapVersion(version),
		WithBootstrapExpectedSHA256(hex.EncodeToString(sum[:])),
		withBootstrapBaseURL(server.URL),
		withBootstrapHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("replace planted cache entry: %v", err)
	}
	contents, err := os.ReadFile(resolved)
	if err != nil {
		t.Fatalf("read resolved library: %v", err)
	}
	if string(contents) == "planted-library" {
		t.Fatal("bootstrap returned the planted cache library")
	}
	if got := hits.Load(); got != 1 {
		t.Fatalf("archive download count = %d, want 1", got)
	}
}

func TestEnsureOnnxRuntimeSharedLibraryRedownloadsTamperedManifestFile(t *testing.T) {
	clearBootstrapEnv(t)

	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}

	cacheDir := t.TempDir()
	version := "1.99.62"
	archiveBytes := buildORTArchive(t, artifact, version, true)
	sum := sha256.Sum256(archiveBytes)
	server, hits := newArchiveServer(t, artifact, version, archiveBytes)
	opts := []BootstrapOption{
		WithBootstrapCacheDir(cacheDir),
		WithBootstrapVersion(version),
		WithBootstrapExpectedSHA256(hex.EncodeToString(sum[:])),
		withBootstrapBaseURL(server.URL),
		withBootstrapHTTPClient(server.Client()),
	}

	resolved, err := EnsureOnnxRuntimeSharedLibrary(opts...)
	if err != nil {
		t.Fatalf("initial bootstrap: %v", err)
	}
	if err := os.WriteFile(resolved, []byte("tampered-library"), 0o600); err != nil {
		t.Fatalf("tamper cached library: %v", err)
	}

	resolved, err = EnsureOnnxRuntimeSharedLibrary(opts...)
	if err != nil {
		t.Fatalf("bootstrap after tamper: %v", err)
	}
	contents, err := os.ReadFile(resolved)
	if err != nil {
		t.Fatalf("read restored library: %v", err)
	}
	if string(contents) == "tampered-library" {
		t.Fatal("bootstrap returned the tampered cached library")
	}
	if got := hits.Load(); got != 2 {
		t.Fatalf("archive download count = %d, want 2 after manifest mismatch", got)
	}
}

func TestEnsureOnnxRuntimeSharedLibraryRejectsCachedSymlink(t *testing.T) {
	clearBootstrapEnv(t)

	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}

	cacheDir := t.TempDir()
	version := "1.99.63"
	installDir := filepath.Join(cacheDir, artifact.archiveName(version))
	libDir := filepath.Join(installDir, "lib")
	if err := os.MkdirAll(libDir, secureDirectoryPermission); err != nil {
		t.Fatalf("create planted cache: %v", err)
	}
	target := filepath.Join(cacheDir, "planted-library")
	if err := os.WriteFile(target, []byte("planted"), 0o600); err != nil {
		t.Fatalf("write symlink target: %v", err)
	}
	symlink := filepath.Join(libDir, artifact.primaryLibrary)
	if err := os.Symlink(target, symlink); err != nil {
		t.Skipf("cannot create symlink on this platform: %v", err)
	}

	resolved, err := EnsureOnnxRuntimeSharedLibrary(
		WithBootstrapCacheDir(cacheDir),
		WithBootstrapVersion(version),
		WithBootstrapExpectedSHA256(strings.Repeat("a", 64)),
		WithBootstrapDisableDownload(true),
	)
	if err == nil {
		t.Fatalf("bootstrap returned cached symlink %q", resolved)
	}
	if resolved != "" {
		t.Fatalf("resolved path = %q, want empty on cached symlink rejection", resolved)
	}
}

func TestEnsureOnnxRuntimeSharedLibraryPreservesCacheOnOperationalValidationError(t *testing.T) {
	clearBootstrapEnv(t)

	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}

	causes := []struct {
		name  string
		cause error
	}{
		{name: "permission", cause: os.ErrPermission},
		{name: "io", cause: syscall.EIO},
	}
	for _, cause := range causes {
		for _, disableDownload := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/disable-download-%t", cause.name, disableDownload), func(t *testing.T) {
				cacheDir := t.TempDir()
				const version = "1.99.64"
				installDir := filepath.Join(cacheDir, artifact.archiveName(version))
				if err := os.MkdirAll(installDir, secureDirectoryPermission); err != nil {
					t.Fatalf("create install directory: %v", err)
				}
				sentinel := filepath.Join(installDir, "preserve-me")
				if err := os.WriteFile(sentinel, []byte("sentinel"), 0o600); err != nil {
					t.Fatalf("write cache sentinel: %v", err)
				}

				previousValidator := bootstrapValidateCachedRuntimeInstall
				previousRemoveAll := bootstrapRemoveAll
				removeCount := 0
				bootstrapValidateCachedRuntimeInstall = func(
					bootstrapConfig,
					runtimeArtifact,
					string,
				) (string, error) {
					return "", fmt.Errorf("injected cache validation failure: %w", cause.cause)
				}
				bootstrapRemoveAll = func(path string) error {
					removeCount++
					return os.RemoveAll(path)
				}
				t.Cleanup(func() {
					bootstrapValidateCachedRuntimeInstall = previousValidator
					bootstrapRemoveAll = previousRemoveAll
				})

				_, err := EnsureOnnxRuntimeSharedLibrary(
					WithBootstrapCacheDir(cacheDir),
					WithBootstrapVersion(version),
					WithBootstrapDisableDownload(disableDownload),
				)
				if !errors.Is(err, cause.cause) {
					t.Fatalf("bootstrap error = %v, want original cause %v", err, cause.cause)
				}
				if removeCount != 0 {
					t.Fatalf("bootstrapRemoveAll calls = %d, want 0", removeCount)
				}
				contents, readErr := os.ReadFile(sentinel)
				if readErr != nil {
					t.Fatalf("read preserved sentinel: %v", readErr)
				}
				if string(contents) != "sentinel" {
					t.Fatalf("sentinel contents = %q, want sentinel", contents)
				}
			})
		}
	}
}

func TestBootstrapDirectoryTrustFailureIsOperational(t *testing.T) {
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("Unix-specific cache directory trust checks")
	}
	cacheDir := t.TempDir()
	if err := os.Chmod(cacheDir, 0o770); err != nil {
		t.Fatalf("make cache directory group-writable: %v", err)
	}
	err := validateBootstrapDirectoryTrust(cacheDir, false)
	if err == nil {
		t.Fatal("expected untrusted cache directory to be rejected")
	}
	if got := cacheValidationDispositionForError(err); got != cacheValidationOperational {
		t.Fatalf("cache directory trust failure disposition = %v, want operational", got)
	}
}

func TestEnsureOnnxRuntimeSharedLibraryPreservesCacheOnDirectoryTrustFailure(t *testing.T) {
	clearBootstrapEnv(t)
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("Unix-specific cache directory trust checks")
	}

	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}

	for _, disableDownload := range []bool{false, true} {
		t.Run(fmt.Sprintf("disable-download-%t", disableDownload), func(t *testing.T) {
			cacheDir := t.TempDir()
			const version = "1.99.66"
			installDir := filepath.Join(cacheDir, artifact.archiveName(version))
			libDir := filepath.Join(installDir, "lib")
			if err := os.MkdirAll(libDir, secureDirectoryPermission); err != nil {
				t.Fatalf("create install directory: %v", err)
			}
			libraryPath := filepath.Join(libDir, artifact.primaryLibrary)
			if err := os.WriteFile(libraryPath, []byte("cached-runtime"), 0o600); err != nil {
				t.Fatalf("write cached runtime: %v", err)
			}
			cfg := bootstrapConfig{version: version}
			if err := writeBootstrapInstallManifest(installDir, cfg, artifact, strings.Repeat("a", 64), true); err != nil {
				t.Fatalf("write valid manifest: %v", err)
			}

			// A trust problem with the parent cache directory, planted after
			// the install itself was already valid, must not be treated as
			// evidence that this install is corrupt.
			if err := os.Chmod(cacheDir, 0o770); err != nil {
				t.Fatalf("make cache directory group-writable: %v", err)
			}

			previousRemoveAll := bootstrapRemoveAll
			removeCount := 0
			bootstrapRemoveAll = func(path string) error {
				removeCount++
				return os.RemoveAll(path)
			}
			t.Cleanup(func() { bootstrapRemoveAll = previousRemoveAll })

			_, err := EnsureOnnxRuntimeSharedLibrary(
				WithBootstrapCacheDir(cacheDir),
				WithBootstrapVersion(version),
				WithBootstrapDisableDownload(disableDownload),
			)
			if err == nil || !strings.Contains(err.Error(), "not trusted") {
				t.Fatalf("bootstrap error = %v, want cache directory trust failure", err)
			}
			if removeCount != 0 {
				t.Fatalf("bootstrapRemoveAll calls = %d, want 0", removeCount)
			}
			contents, readErr := os.ReadFile(libraryPath)
			if readErr != nil {
				t.Fatalf("read preserved cached library: %v", readErr)
			}
			if string(contents) != "cached-runtime" {
				t.Fatalf("cached library contents = %q, want cached-runtime", contents)
			}
		})
	}
}

func TestBootstrapCacheValidationDisposition(t *testing.T) {
	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}
	cfg := bootstrapConfig{version: "1.99.65"}

	missingInstall := filepath.Join(t.TempDir(), "missing-install")
	_, err = validateCachedRuntimeInstall(cfg, artifact, missingInstall)
	if got := cacheValidationDispositionForError(err); got != cacheValidationMissing {
		t.Fatalf("missing install disposition = %v, want missing", got)
	}

	installDir := filepath.Join(t.TempDir(), artifact.archiveName(cfg.version))
	if err := os.MkdirAll(installDir, secureDirectoryPermission); err != nil {
		t.Fatalf("create install directory: %v", err)
	}
	_, err = validateCachedRuntimeInstall(cfg, artifact, installDir)
	if got := cacheValidationDispositionForError(err); got != cacheValidationConfirmedInvalid {
		t.Fatalf("missing manifest disposition = %v, want confirmed invalid", got)
	}

	manifestPath := filepath.Join(installDir, bootstrapManifestFilename)
	if err := os.WriteFile(manifestPath, []byte("{invalid"), 0o600); err != nil {
		t.Fatalf("write invalid manifest: %v", err)
	}
	_, err = validateCachedRuntimeInstall(cfg, artifact, installDir)
	if got := cacheValidationDispositionForError(err); got != cacheValidationConfirmedInvalid {
		t.Fatalf("malformed manifest disposition = %v, want confirmed invalid", got)
	}

	seedValidInstall := func() string {
		t.Helper()
		if err := os.RemoveAll(installDir); err != nil {
			t.Fatalf("reset install directory: %v", err)
		}
		libDir := filepath.Join(installDir, "lib")
		if err := os.MkdirAll(libDir, secureDirectoryPermission); err != nil {
			t.Fatalf("create library directory: %v", err)
		}
		libraryPath := filepath.Join(libDir, artifact.primaryLibrary)
		if err := os.WriteFile(libraryPath, []byte("cached-runtime"), 0o600); err != nil {
			t.Fatalf("write cached runtime: %v", err)
		}
		if err := writeBootstrapInstallManifest(
			installDir,
			cfg,
			artifact,
			strings.Repeat("a", 64),
			true,
		); err != nil {
			t.Fatalf("write valid manifest: %v", err)
		}
		return libraryPath
	}

	seedValidInstall()
	encoded, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read valid manifest: %v", err)
	}
	var manifest bootstrapInstallManifest
	if err := json.Unmarshal(encoded, &manifest); err != nil {
		t.Fatalf("decode valid manifest: %v", err)
	}
	manifest.Platform = "wrong-platform"
	encoded, err = json.Marshal(manifest)
	if err != nil {
		t.Fatalf("encode wrong-platform manifest: %v", err)
	}
	if err := os.WriteFile(manifestPath, encoded, 0o600); err != nil {
		t.Fatalf("write wrong-platform manifest: %v", err)
	}
	_, err = validateCachedRuntimeInstall(cfg, artifact, installDir)
	if got := cacheValidationDispositionForError(err); got != cacheValidationConfirmedInvalid {
		t.Fatalf("wrong metadata disposition = %v, want confirmed invalid", got)
	}

	seedValidInstall()
	checksumCfg := cfg
	checksumCfg.expectedSHA256 = strings.Repeat("b", 64)
	_, err = validateCachedRuntimeInstall(checksumCfg, artifact, installDir)
	if got := cacheValidationDispositionForError(err); got != cacheValidationConfirmedInvalid {
		t.Fatalf("checksum mismatch disposition = %v, want confirmed invalid", got)
	}

	seedValidInstall()
	if err := os.WriteFile(filepath.Join(installDir, "unexpected-file"), []byte("extra"), 0o600); err != nil {
		t.Fatalf("write unexpected cached file: %v", err)
	}
	_, err = validateCachedRuntimeInstall(cfg, artifact, installDir)
	if got := cacheValidationDispositionForError(err); got != cacheValidationConfirmedInvalid {
		t.Fatalf("file-list mismatch disposition = %v, want confirmed invalid", got)
	}

	operationalErr := fmt.Errorf("wrapped inspection failure: %w", syscall.EIO)
	if got := cacheValidationDispositionForError(operationalErr); got != cacheValidationOperational {
		t.Fatalf("operational error disposition = %v, want operational", got)
	}
}

func TestBootstrapReadOnlyCacheHit(t *testing.T) {
	clearBootstrapEnv(t)

	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}

	cacheDir := t.TempDir()
	const version = "1.99.66"
	installDir := filepath.Join(cacheDir, artifact.archiveName(version))
	libDir := filepath.Join(installDir, "lib")
	if err := os.MkdirAll(libDir, secureDirectoryPermission); err != nil {
		t.Fatalf("create cached library directory: %v", err)
	}
	libraryPath := filepath.Join(libDir, artifact.primaryLibrary)
	if err := os.WriteFile(libraryPath, []byte("read-only-runtime"), 0o600); err != nil {
		t.Fatalf("write cached library: %v", err)
	}
	cfg := bootstrapConfig{
		cacheDir: cacheDir,
		version:  version,
		goos:     runtime.GOOS,
		goarch:   runtime.GOARCH,
	}
	if err := writeBootstrapInstallManifest(
		installDir,
		cfg,
		artifact,
		strings.Repeat("a", 64),
		true,
	); err != nil {
		t.Fatalf("write bootstrap manifest: %v", err)
	}

	makeBootstrapTreeReadOnly(t, cacheDir)
	previousRemoveAll := bootstrapRemoveAll
	removeCount := 0
	bootstrapRemoveAll = func(path string) error {
		removeCount++
		return os.RemoveAll(path)
	}
	t.Cleanup(func() { bootstrapRemoveAll = previousRemoveAll })

	resolved, err := EnsureOnnxRuntimeSharedLibrary(
		WithBootstrapCacheDir(cacheDir),
		WithBootstrapVersion(version),
		WithBootstrapDisableDownload(true),
	)
	if err != nil {
		t.Fatalf("resolve read-only cache hit: %v", err)
	}
	want, err := filepath.Abs(libraryPath)
	if err != nil {
		t.Fatalf("resolve expected absolute path: %v", err)
	}
	if resolved != want {
		t.Fatalf("resolved path = %q, want %q", resolved, want)
	}
	if removeCount != 0 {
		t.Fatalf("bootstrapRemoveAll calls = %d, want 0", removeCount)
	}
	if _, err := os.Lstat(filepath.Join(cacheDir, ".locks")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("lock directory inspection error = %v, want not-exist", err)
	}
}

func TestDownloadAndInstallRuntimeDoesNotReplaceExistingDestination(t *testing.T) {
	cfg, artifact, installDir, checksum := newBootstrapDiagnosticDownloadFixture(t, false)
	cfg.expectedSHA256 = checksum
	if err := os.MkdirAll(installDir, secureDirectoryPermission); err != nil {
		t.Fatalf("create competing install directory: %v", err)
	}
	sentinel := filepath.Join(installDir, "preserve-me")
	if err := os.WriteFile(sentinel, []byte("sentinel"), 0o600); err != nil {
		t.Fatalf("write competing install sentinel: %v", err)
	}

	previousRemoveAll := bootstrapRemoveAll
	removedDestination := false
	bootstrapRemoveAll = func(path string) error {
		if path == installDir {
			removedDestination = true
		}
		return os.RemoveAll(path)
	}
	t.Cleanup(func() { bootstrapRemoveAll = previousRemoveAll })

	err := downloadAndInstallRuntime(cfg, artifact, installDir)
	if err == nil || !strings.Contains(err.Error(), "refusing to replace") {
		t.Fatalf("download error = %v, want destination collision rejection", err)
	}
	if removedDestination {
		t.Fatal("download removed an existing destination without confirmed-invalid validation")
	}
	if contents, readErr := os.ReadFile(sentinel); readErr != nil || string(contents) != "sentinel" {
		t.Fatalf("competing destination changed: contents=%q error=%v", contents, readErr)
	}
}

func TestBootstrapPlatformTrustPolicy(t *testing.T) {
	path := t.TempDir()
	info, err := os.Lstat(path)
	if err != nil {
		t.Fatalf("inspect test directory: %v", err)
	}

	switch runtime.GOOS {
	case "darwin", "linux":
		if err := validateBootstrapPathOwnershipAndMode(path, info, false); err != nil {
			t.Fatalf("strict policy rejected current owner: %v", err)
		}

		stat := reflect.ValueOf(info.Sys())
		if stat.Kind() != reflect.Pointer || stat.IsNil() {
			t.Fatalf("unexpected Unix stat value %T", info.Sys())
		}
		uid := stat.Elem().FieldByName("Uid")
		if !uid.IsValid() {
			t.Fatalf("Unix stat value %T has no settable Uid", info.Sys())
		}
		withUID := func(value uint64) bootstrapTestFileInfo {
			clonedStat := reflect.New(stat.Elem().Type())
			clonedStat.Elem().Set(stat.Elem())
			clonedUID := clonedStat.Elem().FieldByName("Uid")
			if !clonedUID.CanSet() {
				t.Fatalf("Unix stat value %T has no settable Uid", info.Sys())
			}
			clonedUID.SetUint(value)
			return bootstrapTestFileInfo{
				FileInfo: info,
				mode:     info.Mode(),
				system:   clonedStat.Interface(),
			}
		}
		if effectiveUID := uint32(os.Geteuid()); effectiveUID != 0 { // #nosec G115 -- Unix effective UIDs are non-negative uid_t values.
			rootInfo := withUID(0)
			if err := validateBootstrapPathOwnershipAndMode(path, rootInfo, false); err == nil {
				t.Fatal("strict policy accepted root owner")
			}
			if err := validateBootstrapPathOwnershipAndMode(path, rootInfo, true); err != nil {
				t.Fatalf("shared policy rejected root owner: %v", err)
			}
		}
		otherUID := uid.Uint() + 1
		if otherUID == 0 {
			otherUID = 1
		}
		otherOwnerInfo := withUID(otherUID)
		if err := validateBootstrapPathOwnershipAndMode(path, otherOwnerInfo, false); err == nil {
			t.Fatal("strict policy accepted non-current owner")
		}
		if err := validateBootstrapPathOwnershipAndMode(path, otherOwnerInfo, true); err != nil {
			t.Fatalf("shared policy rejected non-current owner: %v", err)
		}

		if err := os.Chmod(path, 0o770); err != nil {
			t.Fatalf("make test directory group-writable: %v", err)
		}
		info, err = os.Lstat(path)
		if err != nil {
			t.Fatalf("inspect group-writable directory: %v", err)
		}
		if err := validateBootstrapPathOwnershipAndMode(path, info, false); err == nil {
			t.Fatal("strict policy accepted group-writable directory")
		}
		if err := validateBootstrapPathOwnershipAndMode(path, info, true); err != nil {
			t.Fatalf("shared policy rejected group-writable directory: %v", err)
		}

		if err := os.Chmod(path, 0o777); err != nil {
			t.Fatalf("make test directory world-writable: %v", err)
		}
		info, err = os.Lstat(path)
		if err != nil {
			t.Fatalf("inspect world-writable directory: %v", err)
		}
		for _, allowShared := range []bool{false, true} {
			if err := validateBootstrapPathOwnershipAndMode(path, info, allowShared); err == nil {
				t.Fatalf("allowShared=%t accepted world-writable directory", allowShared)
			}
		}
	default:
		unixLikeInfo := bootstrapTestFileInfo{
			FileInfo: info,
			mode:     os.ModeDir | 0o777,
			system:   nil,
		}
		for _, allowShared := range []bool{false, true} {
			if err := validateBootstrapPathOwnershipAndMode(path, unixLikeInfo, allowShared); err != nil {
				t.Fatalf("allowShared=%t claimed Unix trust validation on %s: %v", allowShared, runtime.GOOS, err)
			}
		}
	}
}

type bootstrapTestFileInfo struct {
	os.FileInfo
	mode   os.FileMode
	system any
}

func (i bootstrapTestFileInfo) Mode() os.FileMode { return i.mode }
func (i bootstrapTestFileInfo) Sys() any          { return i.system }

func TestResolveRuntimeArchiveChecksumFromReleaseMetadata(t *testing.T) {
	artifact, err := resolveRuntimeArtifact("linux", "amd64")
	if err != nil {
		t.Fatalf("unexpected artifact resolution error: %v", err)
	}

	const version = "1.99.8"
	expectedChecksum := strings.Repeat("a", 64)
	archiveName := artifact.archiveFilename(version)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v"+version {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"assets":[{"name":"%s","digest":"sha256:%s"}]}`, archiveName, expectedChecksum)
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		version:            version,
		releaseMetadataURL: server.URL,
		httpClient:         server.Client(),
	}

	checksum, err := resolveRuntimeArchiveChecksumFromReleaseMetadata(cfg, artifact)
	if err != nil {
		t.Fatalf("unexpected checksum resolution error: %v", err)
	}
	if checksum != expectedChecksum {
		t.Fatalf("unexpected checksum from release metadata: got %q, want %q", checksum, expectedChecksum)
	}
}

func TestResolveRuntimeArchiveChecksumFromReleaseMetadataMissingAsset(t *testing.T) {
	artifact, err := resolveRuntimeArtifact("linux", "amd64")
	if err != nil {
		t.Fatalf("unexpected artifact resolution error: %v", err)
	}

	const version = "1.99.801"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v"+version {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"assets":[{"name":"some-other-asset.tgz","digest":"sha256:` + strings.Repeat("a", 64) + `"}]}`))
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		version:            version,
		releaseMetadataURL: server.URL,
		httpClient:         server.Client(),
		retryAttempts:      1,
	}

	_, err = resolveRuntimeArchiveChecksumFromReleaseMetadata(cfg, artifact)
	if err == nil {
		t.Fatalf("expected metadata missing-asset error")
	}
	if !strings.Contains(err.Error(), "does not contain asset") {
		t.Fatalf("unexpected missing-asset error: %v", err)
	}
}

func TestResolveRuntimeArchiveChecksumFallsBackToPinnedChecksumWhenMetadataFails(t *testing.T) {
	artifact, err := resolveRuntimeArtifact("linux", "amd64")
	if err != nil {
		t.Fatalf("unexpected artifact resolution error: %v", err)
	}

	const version = "1.99.9"
	pinnedChecksum := strings.Repeat("b", 64)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte("rate limited"))
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		version:            version,
		baseURL:            defaultBootstrapBaseURL,
		releaseMetadataURL: server.URL,
		expectedSHA256:     pinnedChecksum,
		httpClient:         server.Client(),
		retryAttempts:      1,
	}

	checksum, err := resolveRuntimeArchiveChecksum(cfg, artifact)
	if err != nil {
		t.Fatalf("unexpected fallback error: %v", err)
	}
	if checksum != pinnedChecksum {
		t.Fatalf("unexpected fallback checksum: got %q, want %q", checksum, pinnedChecksum)
	}
}

func TestResolveRuntimeArchiveChecksumFailsWhenMetadataUnavailableAndNoPinnedChecksum(t *testing.T) {
	artifact, err := resolveRuntimeArtifact("linux", "amd64")
	if err != nil {
		t.Fatalf("unexpected artifact resolution error: %v", err)
	}

	const version = "1.99.91"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("metadata unavailable"))
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		version:            version,
		baseURL:            defaultBootstrapBaseURL,
		releaseMetadataURL: server.URL,
		httpClient:         server.Client(),
		retryAttempts:      1,
	}

	_, err = resolveRuntimeArchiveChecksum(cfg, artifact)
	if err == nil {
		t.Fatalf("expected metadata resolution error")
	}
	if !strings.Contains(err.Error(), "failed to resolve ONNX Runtime checksum") {
		t.Fatalf("unexpected metadata resolution error: %v", err)
	}
}

func TestResolveRuntimeArchiveChecksumRejectsPinnedMismatch(t *testing.T) {
	artifact, err := resolveRuntimeArtifact("linux", "amd64")
	if err != nil {
		t.Fatalf("unexpected artifact resolution error: %v", err)
	}

	const version = "1.99.10"
	officialChecksum := strings.Repeat("c", 64)
	pinnedChecksum := strings.Repeat("d", 64)
	archiveName := artifact.archiveFilename(version)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v"+version {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"assets":[{"name":"%s","digest":"sha256:%s"}]}`, archiveName, officialChecksum)
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		version:            version,
		baseURL:            defaultBootstrapBaseURL,
		releaseMetadataURL: server.URL,
		expectedSHA256:     pinnedChecksum,
		httpClient:         server.Client(),
	}

	_, err = resolveRuntimeArchiveChecksum(cfg, artifact)
	if err == nil {
		t.Fatalf("expected mismatch error")
	}
	if !strings.Contains(err.Error(), "does not match ONNX Runtime release metadata checksum") {
		t.Fatalf("unexpected mismatch error: %v", err)
	}
}

func TestResolveRuntimeArchiveChecksumOfficialSourceHappyPath(t *testing.T) {
	artifact, err := resolveRuntimeArtifact("linux", "amd64")
	if err != nil {
		t.Fatalf("unexpected artifact resolution error: %v", err)
	}

	const version = "1.99.11"
	officialChecksum := strings.Repeat("e", 64)
	archiveName := artifact.archiveFilename(version)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v"+version {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"assets":[{"name":"%s","digest":"sha256:%s"}]}`, archiveName, officialChecksum)
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		version:            version,
		baseURL:            defaultBootstrapBaseURL,
		releaseMetadataURL: server.URL,
		httpClient:         server.Client(),
	}

	checksum, err := resolveRuntimeArchiveChecksum(cfg, artifact)
	if err != nil {
		t.Fatalf("unexpected checksum resolution error: %v", err)
	}
	if checksum != officialChecksum {
		t.Fatalf("unexpected official checksum: got %q, want %q", checksum, officialChecksum)
	}
}

func TestResolveRuntimeArchiveChecksumNonOfficialMirrorSkipsMetadataLookup(t *testing.T) {
	artifact, err := resolveRuntimeArtifact("linux", "amd64")
	if err != nil {
		t.Fatalf("unexpected artifact resolution error: %v", err)
	}

	pinnedChecksum := strings.Repeat("f", 64)
	var metadataRequests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		metadataRequests.Add(1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"assets":[]}`))
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		version:            "1.99.12",
		baseURL:            "https://mirror.example.com/onnxruntime/releases/download",
		releaseMetadataURL: server.URL,
		expectedSHA256:     pinnedChecksum,
		httpClient:         server.Client(),
	}

	checksum, err := resolveRuntimeArchiveChecksum(cfg, artifact)
	if err != nil {
		t.Fatalf("unexpected mirror checksum resolution error: %v", err)
	}
	if checksum != pinnedChecksum {
		t.Fatalf("unexpected checksum for mirror source: got %q, want %q", checksum, pinnedChecksum)
	}
	if got := metadataRequests.Load(); got != 0 {
		t.Fatalf("expected no metadata requests for non-official mirror, got %d", got)
	}
}

func TestShouldResolveChecksumFromReleaseMetadata(t *testing.T) {
	tests := []struct {
		name        string
		baseURL     string
		metadataURL string
		want        bool
	}{
		{
			name:        "official source with metadata",
			baseURL:     defaultBootstrapBaseURL,
			metadataURL: defaultBootstrapReleaseMetadataURL,
			want:        true,
		},
		{
			name:        "official source with trim and trailing slash",
			baseURL:     " " + defaultBootstrapBaseURL + "/ ",
			metadataURL: " " + defaultBootstrapReleaseMetadataURL + "/ ",
			want:        true,
		},
		{
			name:        "official source without metadata",
			baseURL:     defaultBootstrapBaseURL,
			metadataURL: "",
			want:        false,
		},
		{
			name:        "non-official source",
			baseURL:     "https://mirror.example.com/onnxruntime/releases/download",
			metadataURL: defaultBootstrapReleaseMetadataURL,
			want:        false,
		},
		{
			name:        "official source with whitespace metadata",
			baseURL:     defaultBootstrapBaseURL,
			metadataURL: "   ",
			want:        false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := shouldResolveChecksumFromReleaseMetadata(tc.baseURL, tc.metadataURL)
			if got != tc.want {
				t.Fatalf("unexpected shouldResolveChecksumFromReleaseMetadata result: got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestNewBootstrapHTTPClientPreservesProxyFromEnvironment(t *testing.T) {
	t.Setenv("HTTPS_PROXY", "http://proxy.example:8080")
	t.Setenv("NO_PROXY", "")

	client := newBootstrapHTTPClient()
	transport, ok := client.Transport.(*http.Transport)
	if !ok || transport == nil {
		t.Fatalf("expected *http.Transport, got %T", client.Transport)
	}
	if transport.Proxy == nil {
		t.Fatalf("expected transport proxy function to be set")
	}

	req, err := http.NewRequest(http.MethodGet, "https://example.com/resource", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	proxyURL, err := transport.Proxy(req)
	if err != nil {
		t.Fatalf("unexpected proxy resolution error: %v", err)
	}
	if proxyURL == nil {
		t.Fatalf("expected proxy URL from environment")
	}
	if got, want := proxyURL.Host, "proxy.example:8080"; got != want {
		t.Fatalf("unexpected proxy host: got %q, want %q", got, want)
	}
}

func TestIsRetryableBootstrapHTTPStatus(t *testing.T) {
	tests := []struct {
		statusCode int
		want       bool
	}{
		{statusCode: http.StatusRequestTimeout, want: true},
		{statusCode: http.StatusTooManyRequests, want: true},
		{statusCode: http.StatusInternalServerError, want: true},
		{statusCode: http.StatusServiceUnavailable, want: true},
		{statusCode: http.StatusBadRequest, want: false},
		{statusCode: http.StatusUnauthorized, want: false},
		{statusCode: http.StatusForbidden, want: false},
		{statusCode: http.StatusNotFound, want: false},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("status-%d", tc.statusCode), func(t *testing.T) {
			if got := isRetryableBootstrapHTTPStatus(tc.statusCode); got != tc.want {
				t.Fatalf("unexpected retryable status decision for %d: got %v, want %v", tc.statusCode, got, tc.want)
			}
		})
	}
}

func TestIsRetryableGitHubMetadataStatusForbiddenRateLimited(t *testing.T) {
	headers := make(http.Header)
	headers.Set("X-RateLimit-Remaining", "0")

	if !isRetryableGitHubMetadataStatus(http.StatusForbidden, headers, "API rate limit exceeded") {
		t.Fatalf("expected forbidden rate-limited metadata response to be retryable")
	}
}

func TestIsRetryableGitHubMetadataStatusForbiddenNonRateLimit(t *testing.T) {
	t.Run("no headers", func(t *testing.T) {
		if isRetryableGitHubMetadataStatus(http.StatusForbidden, nil, "forbidden") {
			t.Fatalf("expected non-rate-limited forbidden metadata response to be non-retryable")
		}
	})

	t.Run("with non-exhausted rate-limit headers", func(t *testing.T) {
		headers := make(http.Header)
		headers.Set("X-RateLimit-Remaining", "42")
		headers.Set("X-RateLimit-Reset", fmt.Sprintf("%d", time.Now().Add(time.Minute).Unix()))

		if isRetryableGitHubMetadataStatus(http.StatusForbidden, headers, "forbidden") {
			t.Fatalf("expected forbidden metadata response with remaining rate limit to be non-retryable")
		}
	})
}

func TestParseSHA256Digest(t *testing.T) {
	tests := []struct {
		name    string
		digest  string
		want    string
		wantErr bool
	}{
		{name: "valid lower", digest: "sha256:" + strings.Repeat("a", 64), want: strings.Repeat("a", 64)},
		{name: "valid upper prefix and hex", digest: "SHA256:" + strings.Repeat("B", 64), want: strings.Repeat("b", 64)},
		{name: "empty", digest: "", wantErr: true},
		{name: "wrong prefix", digest: "md5:" + strings.Repeat("a", 64), wantErr: true},
		{name: "short", digest: "sha256:" + strings.Repeat("a", 63), wantErr: true},
		{name: "non-hex", digest: "sha256:" + strings.Repeat("z", 64), wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseSHA256Digest(tc.digest)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected parse error for %q", tc.digest)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected parse error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("unexpected digest parse result: got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestLooksLikeSHA256(t *testing.T) {
	tests := []struct {
		name   string
		input  string
		expect bool
	}{
		{name: "lowercase", input: strings.Repeat("a", 64), expect: true},
		{name: "uppercase", input: strings.Repeat("B", 64), expect: true},
		{name: "mixed", input: strings.Repeat("1a", 32), expect: true},
		{name: "too short", input: strings.Repeat("a", 63), expect: false},
		{name: "too long", input: strings.Repeat("a", 65), expect: false},
		{name: "invalid character", input: strings.Repeat("g", 64), expect: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := looksLikeSHA256(tc.input); got != tc.expect {
				t.Fatalf("unexpected looksLikeSHA256 result for %q: got %v, want %v", tc.input, got, tc.expect)
			}
		})
	}
}

func TestResolveGitHubToken(t *testing.T) {
	t.Run("GITHUB_TOKEN preferred", func(t *testing.T) {
		t.Setenv("GITHUB_TOKEN", "preferred-token")
		t.Setenv("GH_TOKEN", "fallback-token")

		if got := resolveGitHubToken(); got != "preferred-token" {
			t.Fatalf("expected GITHUB_TOKEN to be preferred, got %q", got)
		}
	})

	t.Run("fallback to GH_TOKEN", func(t *testing.T) {
		t.Setenv("GITHUB_TOKEN", " ")
		t.Setenv("GH_TOKEN", "gh-token")

		if got := resolveGitHubToken(); got != "gh-token" {
			t.Fatalf("expected GH_TOKEN fallback, got %q", got)
		}
	})

	t.Run("empty when neither set", func(t *testing.T) {
		t.Setenv("GITHUB_TOKEN", "")
		t.Setenv("GH_TOKEN", "")

		if got := resolveGitHubToken(); got != "" {
			t.Fatalf("expected empty token when env vars are unset, got %q", got)
		}
	})
}

func TestMetadataStatusErrorTokenHintAvoidsCredentialLikeLiteralStrings(t *testing.T) {
	t.Setenv("GITHUB_TOKEN", "test-token")
	t.Setenv("GH_TOKEN", "")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte("unauthorized"))
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		httpClient: server.Client(),
	}

	_, err := fetchRuntimeArchiveChecksumFromReleaseMetadataURL(cfg, server.URL+"/v1.2.3", "archive.tgz")
	if err == nil {
		t.Fatalf("expected metadata auth error")
	}
	message := err.Error()
	if !strings.Contains(message, "GitHub settings from environment") {
		t.Fatalf("expected environment credentials hint in error, got: %v", err)
	}
	if strings.Contains(message, "GITHUB_TOKEN") || strings.Contains(message, "GH_TOKEN") {
		t.Fatalf("error message should not contain credential-like literal env var names, got: %v", err)
	}
	if !strings.Contains(message, "HTTP 401") {
		t.Fatalf("expected HTTP status code in error, got: %v", err)
	}
}

func TestRejectHTTPSDowngradeRedirect(t *testing.T) {
	prevURL, err := url.Parse("https://example.com/a")
	if err != nil {
		t.Fatalf("failed to parse previous URL: %v", err)
	}
	nextURL, err := url.Parse("http://example.com/b")
	if err != nil {
		t.Fatalf("failed to parse next URL: %v", err)
	}

	err = rejectHTTPSDowngradeRedirect(
		&http.Request{URL: nextURL},
		[]*http.Request{{URL: prevURL}},
	)
	if err == nil {
		t.Fatalf("expected redirect downgrade rejection")
	}
	if !isBootstrapRedirectPolicyError(err) {
		t.Fatalf("expected redirect downgrade error to be tagged as redirect policy error, got: %v", err)
	}
	if !strings.Contains(err.Error(), "not allowed") {
		t.Fatalf("unexpected redirect validation error: %v", err)
	}
}

func TestRejectHTTPSDowngradeRedirectAllowsSafeCases(t *testing.T) {
	nextURL, err := url.Parse("https://example.com/b")
	if err != nil {
		t.Fatalf("failed to parse next URL: %v", err)
	}
	if err := rejectHTTPSDowngradeRedirect(&http.Request{URL: nextURL}, nil); err != nil {
		t.Fatalf("expected no error for first request with empty redirect chain, got: %v", err)
	}

	prevURL, err := url.Parse("https://example.com/a")
	if err != nil {
		t.Fatalf("failed to parse previous URL: %v", err)
	}
	if err := rejectHTTPSDowngradeRedirect(&http.Request{URL: nextURL}, []*http.Request{{URL: prevURL}}); err != nil {
		t.Fatalf("expected no error for HTTPS to HTTPS redirect, got: %v", err)
	}
}

func TestRejectHTTPSDowngradeRedirectRejectsNilURL(t *testing.T) {
	prevURL, err := url.Parse("https://example.com/a")
	if err != nil {
		t.Fatalf("failed to parse previous URL: %v", err)
	}
	if err := rejectHTTPSDowngradeRedirect(
		&http.Request{},
		[]*http.Request{{URL: prevURL}},
	); err == nil {
		t.Fatalf("expected nil URL rejection when request URL is nil")
	} else if !isBootstrapRedirectPolicyError(err) {
		t.Fatalf("expected nil-URL redirect rejection to be tagged as redirect policy error, got: %v", err)
	}

	nextURL, err := url.Parse("https://example.com/b")
	if err != nil {
		t.Fatalf("failed to parse next URL: %v", err)
	}
	if err := rejectHTTPSDowngradeRedirect(
		&http.Request{URL: nextURL},
		[]*http.Request{{}},
	); err == nil {
		t.Fatalf("expected nil URL rejection when previous redirect URL is nil")
	} else if !isBootstrapRedirectPolicyError(err) {
		t.Fatalf("expected nil-URL redirect rejection to be tagged as redirect policy error, got: %v", err)
	}
}

func TestRejectHTTPSDowngradeRedirectLimit(t *testing.T) {
	nextURL, err := url.Parse("https://example.com/final")
	if err != nil {
		t.Fatalf("failed to parse final URL: %v", err)
	}
	chain := make([]*http.Request, 10)
	for i := range chain {
		stepURL, parseErr := url.Parse(fmt.Sprintf("https://example.com/%d", i))
		if parseErr != nil {
			t.Fatalf("failed to parse step URL: %v", parseErr)
		}
		chain[i] = &http.Request{URL: stepURL}
	}

	err = rejectHTTPSDowngradeRedirect(&http.Request{URL: nextURL}, chain)
	if err == nil {
		t.Fatalf("expected redirect limit rejection")
	}
	if !isBootstrapRedirectPolicyError(err) {
		t.Fatalf("expected redirect limit error to be tagged as redirect policy error, got: %v", err)
	}
	if !strings.Contains(err.Error(), "stopped after 10 redirects") {
		t.Fatalf("unexpected redirect limit error: %v", err)
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
		cacheDir:      cacheDir,
		httpClient:    server.Client(),
		retryAttempts: 1,
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

func TestDownloadRuntimeArchiveCleansTempFileOnResponseCloseError(t *testing.T) {
	clearBootstrapEnv(t)

	cacheDir := t.TempDir()
	closeCause := errors.New("synthetic response close failure")
	payload := []byte("onnxruntime-archive")
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			Header:        make(http.Header),
			Body:          &closeErrorReadCloser{Reader: bytes.NewReader(payload), closeErr: closeCause},
			ContentLength: int64(len(payload)),
			Request:       req,
		}, nil
	})}
	cfg := bootstrapConfig{
		cacheDir:        cacheDir,
		httpClient:      client,
		maxDownloadSize: 1024,
		retryAttempts:   1,
	}

	_, _, err := downloadRuntimeArchive(cfg, "https://example.invalid/archive")
	if !errors.Is(err, closeCause) {
		t.Fatalf("download error = %v, want response close cause", err)
	}
	matches, globErr := filepath.Glob(filepath.Join(cacheDir, "onnxruntime-*.archive"))
	if globErr != nil {
		t.Fatalf("glob temporary archives: %v", globErr)
	}
	if len(matches) != 0 {
		t.Fatalf("temporary archives remained after response close failure: %v", matches)
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
		retryAttempts:   1,
	}

	_, _, err := downloadRuntimeArchive(cfg, server.URL+"/archive")
	if err == nil {
		t.Fatalf("expected HTTP status download error")
	}
	if !strings.Contains(err.Error(), "HTTP 503") {
		t.Fatalf("expected HTTP status in error, got: %v", err)
	}
}

func TestDownloadRuntimeArchiveRetriesTransientThenSucceeds(t *testing.T) {
	clearBootstrapEnv(t)

	cacheDir := t.TempDir()
	payload := []byte("onnxruntime-archive")
	wantSum := sha256.Sum256(payload)
	wantChecksum := hex.EncodeToString(wantSum[:])
	var hits atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		current := hits.Add(1)
		if current == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte("service unavailable"))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		cacheDir:        cacheDir,
		httpClient:      server.Client(),
		maxDownloadSize: 1024,
		retryAttempts:   3,
	}

	archivePath, checksum, err := downloadRuntimeArchive(cfg, server.URL+"/archive")
	if err != nil {
		t.Fatalf("expected retry then success, got error: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Remove(archivePath)
	})
	if checksum != wantChecksum {
		t.Fatalf("unexpected checksum after retry success: got %q, want %q", checksum, wantChecksum)
	}
	if got := hits.Load(); got != 2 {
		t.Fatalf("expected exactly two attempts (one retry), got %d", got)
	}
}

func TestDownloadRuntimeArchivePermanent404StopsImmediately(t *testing.T) {
	clearBootstrapEnv(t)

	cacheDir := t.TempDir()
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte("not found"))
	}))
	t.Cleanup(server.Close)

	cfg := bootstrapConfig{
		cacheDir:        cacheDir,
		httpClient:      server.Client(),
		maxDownloadSize: 1024,
		retryAttempts:   3,
	}

	_, _, err := downloadRuntimeArchive(cfg, server.URL+"/archive")
	if err == nil {
		t.Fatalf("expected permanent HTTP 404 error")
	}
	if !strings.Contains(err.Error(), "HTTP 404") {
		t.Fatalf("expected HTTP 404 in error, got: %v", err)
	}
	if got := hits.Load(); got != 1 {
		t.Fatalf("expected a single attempt for permanent 404, got %d", got)
	}
}

func TestDownloadRuntimeArchiveRedirectPolicyStopsImmediately(t *testing.T) {
	clearBootstrapEnv(t)

	cacheDir := t.TempDir()
	var httpsHits atomic.Int32
	var httpHits atomic.Int32

	httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		httpHits.Add(1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("archive"))
	}))
	t.Cleanup(httpServer.Close)

	httpsServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		httpsHits.Add(1)
		http.Redirect(w, r, httpServer.URL+"/archive", http.StatusFound)
	}))
	t.Cleanup(httpsServer.Close)

	client := httpsServer.Client()
	client.CheckRedirect = rejectHTTPSDowngradeRedirect

	cfg := bootstrapConfig{
		cacheDir:      cacheDir,
		httpClient:    client,
		retryAttempts: 3,
	}

	_, _, err := downloadRuntimeArchive(cfg, httpsServer.URL+"/archive")
	if err == nil {
		t.Fatalf("expected redirect policy rejection")
	}
	if !strings.Contains(err.Error(), "HTTPS to HTTP is not allowed") {
		t.Fatalf("unexpected redirect policy error: %v", err)
	}
	if got := httpsHits.Load(); got != 1 {
		t.Fatalf("expected single HTTPS attempt for permanent redirect-policy failure, got %d", got)
	}
	if got := httpHits.Load(); got != 0 {
		t.Fatalf("expected no HTTP downgrade request to be issued, got %d", got)
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
	if errors.Is(err, ErrSharedLibraryNotFound) {
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
	if !errors.Is(err, ErrSharedLibraryNotFound) {
		t.Fatalf("expected not-found error, got: %v", err)
	}
}

func TestWithBootstrapVersionRejectsEmpty(t *testing.T) {
	var cfg bootstrapConfig
	if err := WithBootstrapVersion("   ")(&cfg); err == nil {
		t.Fatalf("expected empty version validation error")
	} else {
		if !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("expected ErrInvalidArgument, got: %v", err)
		}
		if !strings.Contains(err.Error(), "version") {
			t.Fatalf("expected version identifier in error, got: %v", err)
		}
	}
}

func TestWithBootstrapLibraryPathAndCacheDirRejectEmpty(t *testing.T) {
	var cfg bootstrapConfig

	if err := WithBootstrapLibraryPath("   ")(&cfg); err == nil {
		t.Fatalf("expected empty library path validation error")
	} else if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for library path, got: %v", err)
	}
	if err := WithBootstrapCacheDir("   ")(&cfg); err == nil {
		t.Fatalf("expected empty cache directory validation error")
	} else if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for cache directory, got: %v", err)
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
				if !errors.Is(err, ErrInvalidArgument) {
					t.Fatalf("expected ErrInvalidArgument for checksum %q, got: %v", tc.checksum, err)
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
			if tc.wantErr && !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("expected ErrInvalidArgument for %q, got: %v", tc.baseURL, err)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected validation error for %q: %v", tc.baseURL, err)
			}
		})
	}
}

func TestBootstrapOptionsReusableConcurrently(t *testing.T) {
	const (
		workers           = 16
		iterationsPerWork = 64
		libraryPath       = "/tmp/onnxruntime/libonnxruntime.so"
		cacheDir          = "/tmp/onnxruntime-cache"
		version           = "1.24.1"
		baseURL           = "https://example.com/onnxruntime"
	)
	wantChecksum := strings.Repeat("a", 64)

	opts := []BootstrapOption{
		WithBootstrapLibraryPath("  " + libraryPath + "  "),
		WithBootstrapCacheDir("  " + cacheDir + "  "),
		WithBootstrapVersion("  " + version + "  "),
		WithBootstrapExpectedSHA256(strings.ToUpper(wantChecksum)),
		withBootstrapBaseURL("  " + baseURL + "  "),
	}

	start := make(chan struct{})
	errCh := make(chan error, workers)
	var wg sync.WaitGroup
	wg.Add(workers)
	for range workers {
		go func() {
			defer wg.Done()
			<-start

			for i := 0; i < iterationsPerWork; i++ {
				var cfg bootstrapConfig
				for _, opt := range opts {
					if err := opt(&cfg); err != nil {
						errCh <- fmt.Errorf("apply bootstrap option: %w", err)
						return
					}
				}
				if cfg.libraryPath != libraryPath || cfg.cacheDir != cacheDir || cfg.version != version || cfg.expectedSHA256 != wantChecksum || cfg.baseURL != baseURL {
					errCh <- fmt.Errorf("normalized bootstrap config = %+v", cfg)
					return
				}
			}
		}()
	}

	close(start)
	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Error(err)
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
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument, got: %v", err)
	}
}

func TestResolveBootstrapConfigAllowSharedCacheEnvironment(t *testing.T) {
	for _, test := range []struct {
		value string
		want  bool
	}{
		{value: "true", want: true},
		{value: "false", want: false},
		{value: "1", want: true},
		{value: "0", want: false},
	} {
		t.Run(test.value, func(t *testing.T) {
			clearBootstrapEnv(t)
			t.Setenv("ONNXRUNTIME_ALLOW_SHARED_CACHE", test.value)

			cfg, err := resolveBootstrapConfig()
			if err != nil {
				t.Fatalf("resolve bootstrap config: %v", err)
			}
			if cfg.allowSharedCache != test.want {
				t.Fatalf("allowSharedCache = %t, want %t", cfg.allowSharedCache, test.want)
			}
		})
	}
}

func TestResolveBootstrapConfigAllowSharedCacheOptionPrecedence(t *testing.T) {
	for _, test := range []struct {
		name string
		env  string
		opt  bool
		want bool
	}{
		{name: "option disables environment", env: "true", opt: false, want: false},
		{name: "option enables environment", env: "false", opt: true, want: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			clearBootstrapEnv(t)
			t.Setenv("ONNXRUNTIME_ALLOW_SHARED_CACHE", test.env)

			cfg, err := resolveBootstrapConfig(WithBootstrapAllowSharedCache(test.opt))
			if err != nil {
				t.Fatalf("resolve bootstrap config: %v", err)
			}
			if cfg.allowSharedCache != test.want {
				t.Fatalf("allowSharedCache = %t, want %t", cfg.allowSharedCache, test.want)
			}
		})
	}
}

func TestResolveBootstrapConfigRejectsInvalidAllowSharedCacheEnvironment(t *testing.T) {
	clearBootstrapEnv(t)
	t.Setenv("ONNXRUNTIME_ALLOW_SHARED_CACHE", "sometimes")

	_, err := resolveBootstrapConfig()
	if err == nil {
		t.Fatal("expected invalid shared-cache environment error")
	}
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("shared-cache environment error = %v, want ErrInvalidArgument", err)
	}
	if !strings.Contains(err.Error(), "ONNXRUNTIME_ALLOW_SHARED_CACHE") {
		t.Fatalf("shared-cache environment error = %v, want variable name", err)
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

	symlinkPath := filepath.Join(dir, "libonnxruntime-link.so")
	if err := os.Symlink(validPath, symlinkPath); err != nil {
		t.Skipf("cannot create symlink on this platform: %v", err)
	}
	if _, err := validateLibraryFile(symlinkPath); err == nil || !strings.Contains(err.Error(), "symbolic link") {
		t.Fatalf("symlink validation error = %v, want symbolic-link rejection", err)
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
		holderErrCh <- withProcessFileLock(lockPath, false, func() error {
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

	err := withProcessFileLock(lockPath, false, func() error { return nil })
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
	err := withProcessFileLock(lockPath, false, nil)
	if err == nil || !strings.Contains(err.Error(), "lock callback is nil") {
		t.Fatalf("expected nil callback error, got: %v", err)
	}
}

func TestBootstrapCreatedFilePermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX file mode assertions are not portable to Windows ACLs")
	}

	t.Run("bootstrap cache install and lock paths", func(t *testing.T) {
		clearBootstrapEnv(t)

		artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
		if err != nil {
			t.Skipf("unsupported runtime for bootstrap permission test: %v", err)
		}

		const version = "1.99.13"
		archiveRoot := artifact.archiveName(version)
		libraryEntry := path.Join(archiveRoot, "lib", artifact.primaryLibrary)
		archiveBytes := buildArchiveWithFileMode(
			t,
			artifact.archiveExtension,
			libraryEntry,
			0o777,
		)
		sum := sha256.Sum256(archiveBytes)
		server, _ := newArchiveServer(t, artifact, version, archiveBytes)
		cacheDir := filepath.Join(t.TempDir(), "cache")

		libraryPath, err := EnsureOnnxRuntimeSharedLibrary(
			WithBootstrapCacheDir(cacheDir),
			WithBootstrapVersion(version),
			WithBootstrapExpectedSHA256(hex.EncodeToString(sum[:])),
			withBootstrapBaseURL(server.URL),
			withBootstrapHTTPClient(server.Client()),
		)
		if err != nil {
			t.Fatalf("unexpected bootstrap error: %v", err)
		}

		installDir := filepath.Join(cacheDir, artifact.archiveName(version))
		lockDir := filepath.Join(cacheDir, ".locks")
		lockPath := filepath.Join(lockDir, fmt.Sprintf("%s-%s.lock", artifact.platform, version))
		for _, dir := range []string{cacheDir, installDir, lockDir} {
			assertBootstrapDirectoryMode(t, dir)
		}
		assertBootstrapLibraryMode(t, libraryPath)
		assertBootstrapLockMode(t, lockPath)
	})

	for _, extension := range []string{"tgz", "zip"} {
		t.Run(extension+" permissive library entry", func(t *testing.T) {
			const entryName = "onnxruntime-test/lib/libonnxruntime-test"
			archivePath := filepath.Join(t.TempDir(), "runtime."+extension)
			archiveBytes := buildArchiveWithFileMode(t, extension, entryName, 0o777)
			if err := os.WriteFile(archivePath, archiveBytes, 0o600); err != nil {
				t.Fatalf("failed to write %s archive: %v", extension, err)
			}

			destinationDir := filepath.Join(t.TempDir(), "extract")
			if _, err := extractArchiveFile(archivePath, destinationDir, extension, ""); err != nil {
				t.Fatalf("failed to extract %s archive: %v", extension, err)
			}

			assertBootstrapDirectoryMode(t, filepath.Join(destinationDir, "onnxruntime-test", "lib"))
			assertBootstrapLibraryMode(t, filepath.Join(destinationDir, filepath.FromSlash(entryName)))
		})
	}

	if got, want := safeArchiveFileMode(0o777), os.FileMode(0o755); got != want {
		t.Fatalf("safe archive mode = %#o, want %#o", got, want)
	}
	if got, want := safeArchiveFileMode(0o644), os.FileMode(0o644); got != want {
		t.Fatalf("safe archive mode changed ordinary file mode: got %#o, want %#o", got, want)
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
		{name: "canonicalizes segments", in: "v01.002.0003", want: "1.2.3"},
		{name: "empty", in: "", expectErr: true},
		{name: "too few segments", in: "1.2", expectErr: true},
		{name: "too many segments", in: "1.2.3.4", expectErr: true},
		{name: "empty segment", in: "1..3", expectErr: true},
		{name: "non-numeric", in: "1.a.3", expectErr: true},
		{name: "negative major", in: "-1.23.1", expectErr: true},
		{name: "negative minor", in: "1.-23.1", expectErr: true},
		{name: "negative patch", in: "1.23.-1", expectErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := normalizeRuntimeVersion(tc.in)
			if tc.expectErr {
				if err == nil {
					t.Fatalf("expected error for %q", tc.in)
				}
				if !errors.Is(err, ErrInvalidArgument) {
					t.Fatalf("normalizeRuntimeVersion(%q) error = %v, want ErrInvalidArgument", tc.in, err)
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

func TestDiagnosticCallSites(t *testing.T) {
	var legacyOutput bytes.Buffer
	previousLogWriter := log.Writer()
	previousLogFlags := log.Flags()
	previousLogPrefix := log.Prefix()
	log.SetOutput(&legacyOutput)
	log.SetFlags(0)
	log.SetPrefix("")
	t.Cleanup(func() {
		log.SetOutput(previousLogWriter)
		log.SetFlags(previousLogFlags)
		log.SetPrefix(previousLogPrefix)
	})

	handler := &bootstrapDiagnosticRecordingHandler{}
	t.Cleanup(func() { SetDiagnosticHandler(nil) })

	t.Run("temporary archive cleanup failure", func(t *testing.T) {
		cfg, artifact, installDir, checksum := newBootstrapDiagnosticDownloadFixture(t, false)
		cfg.expectedSHA256 = checksum
		removeErr := errors.New("synthetic archive removal failure")
		previousRemove := bootstrapRemove
		bootstrapRemove = func(string) error { return removeErr }
		t.Cleanup(func() { bootstrapRemove = previousRemove })

		assertBootstrapDiagnosticCase(t, &legacyOutput, handler, func(t *testing.T) {
			if err := os.RemoveAll(installDir); err != nil {
				t.Fatalf("reset diagnostic install directory: %v", err)
			}
			if err := downloadAndInstallRuntime(cfg, artifact, installDir); err != nil {
				t.Fatalf("downloadAndInstallRuntime: %v", err)
			}
		}, []bootstrapDiagnosticExpectation{{
			level:   slog.LevelWarn,
			message: "temporary bootstrap archive cleanup failed",
			attrs: map[string]string{
				"operation": "remove temporary archive",
				"path":      "",
				"error":     "",
			},
		}})
	})

	t.Run("download without checksum redacts URL", func(t *testing.T) {
		cfg, artifact, installDir, checksum := newBootstrapDiagnosticDownloadFixture(t, true)
		parsedURL, err := url.Parse(cfg.baseURL)
		if err != nil {
			t.Fatalf("parse fixture URL: %v", err)
		}
		wantURL := parsedURL.Redacted() + "/v" + cfg.version + "/" + artifact.archiveFilename(cfg.version)

		assertBootstrapDiagnosticCase(t, &legacyOutput, handler, func(t *testing.T) {
			if err := os.RemoveAll(installDir); err != nil {
				t.Fatalf("reset diagnostic install directory: %v", err)
			}
			if err := downloadAndInstallRuntime(cfg, artifact, installDir); err != nil {
				t.Fatalf("downloadAndInstallRuntime: %v", err)
			}
		}, []bootstrapDiagnosticExpectation{{
			level:   slog.LevelWarn,
			message: "bootstrap download continued without checksum verification",
			attrs: map[string]string{
				"url":               wantURL,
				"checksum_verified": "false",
				"observed_sha256":   checksum,
			},
			forbidden: []string{"bootstrap-secret"},
		}})
	})

	t.Run("staging cleanup failure", func(t *testing.T) {
		cfg, artifact, installDir, checksum := newBootstrapDiagnosticDownloadFixture(t, false)
		cfg.expectedSHA256 = checksum
		removeErr := errors.New("synthetic staging removal failure")
		previousRemoveAll := bootstrapRemoveAll
		removeCalls := 0
		bootstrapRemoveAll = func(path string) error {
			removeCalls++
			if removeCalls == 2 && strings.Contains(path, ".staging-") {
				return removeErr
			}
			return os.RemoveAll(path)
		}
		t.Cleanup(func() { bootstrapRemoveAll = previousRemoveAll })

		assertBootstrapDiagnosticCase(t, &legacyOutput, handler, func(t *testing.T) {
			removeCalls = 0
			if err := os.RemoveAll(installDir); err != nil {
				t.Fatalf("reset diagnostic install directory: %v", err)
			}
			if err := downloadAndInstallRuntime(cfg, artifact, installDir); err != nil {
				t.Fatalf("downloadAndInstallRuntime: %v", err)
			}
		}, []bootstrapDiagnosticExpectation{{
			level:   slog.LevelWarn,
			message: "bootstrap staging cleanup failed",
			attrs: map[string]string{
				"operation": "remove staging directory",
				"path":      "",
				"error":     "",
			},
		}})
	})

	t.Run("metadata checksum fallback", func(t *testing.T) {
		artifact, err := resolveRuntimeArtifact("linux", "amd64")
		if err != nil {
			t.Fatalf("resolveRuntimeArtifact: %v", err)
		}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte("rate limited"))
		}))
		t.Cleanup(server.Close)
		cfg := bootstrapConfig{
			version:            "1.99.15",
			baseURL:            defaultBootstrapBaseURL,
			releaseMetadataURL: server.URL,
			expectedSHA256:     strings.Repeat("b", 64),
			httpClient:         server.Client(),
			retryAttempts:      1,
		}

		assertBootstrapDiagnosticCase(t, &legacyOutput, handler, func(t *testing.T) {
			if _, err := resolveRuntimeArchiveChecksum(cfg, artifact); err != nil {
				t.Fatalf("resolveRuntimeArchiveChecksum: %v", err)
			}
		}, []bootstrapDiagnosticExpectation{{
			level:   slog.LevelWarn,
			message: "bootstrap checksum metadata lookup failed; using pinned checksum",
			attrs: map[string]string{
				"operation": "resolve release metadata checksum",
				"error":     "",
			},
		}})
	})

	t.Run("tar glob failure and link skip", func(t *testing.T) {
		const (
			regularEntry = "runtime/lib/libonnxruntime-real.so"
			linkEntry    = "runtime/lib/libonnxruntime.so"
			invalidGlob  = "["
		)
		archivePath := writeBootstrapDiagnosticTGZ(t, []bootstrapDiagnosticArchiveEntry{
			{name: regularEntry, mode: 0o644, content: "library"},
			{name: linkEntry, mode: 0o777, typeflag: tar.TypeSymlink, linkname: "libonnxruntime-real.so"},
		})

		assertBootstrapDiagnosticCase(t, &legacyOutput, handler, func(t *testing.T) {
			if _, err := extractTGZArchive(archivePath, t.TempDir(), invalidGlob); err != nil {
				t.Fatalf("extractTGZArchive: %v", err)
			}
		}, []bootstrapDiagnosticExpectation{
			{
				level:   slog.LevelWarn,
				message: "bootstrap tar library glob match failed",
				attrs: map[string]string{
					"archive_entry": linkEntry,
					"library_glob":  invalidGlob,
					"error":         "",
				},
			},
			{
				level:   slog.LevelWarn,
				message: "bootstrap tar link entry skipped",
				attrs: map[string]string{
					"archive_entry": linkEntry,
					"entry_type":    fmt.Sprint(tar.TypeSymlink),
				},
			},
		})
	})

	t.Run("tar unsupported entry skip", func(t *testing.T) {
		const unsupportedEntry = "runtime/lib/pipe"
		archivePath := writeBootstrapDiagnosticTGZ(t, []bootstrapDiagnosticArchiveEntry{
			{name: "runtime/lib/libonnxruntime.so", mode: 0o644, content: "library"},
			{name: unsupportedEntry, mode: 0o600, typeflag: tar.TypeFifo},
		})

		assertBootstrapDiagnosticCase(t, &legacyOutput, handler, func(t *testing.T) {
			if _, err := extractTGZArchive(archivePath, t.TempDir(), ""); err != nil {
				t.Fatalf("extractTGZArchive: %v", err)
			}
		}, []bootstrapDiagnosticExpectation{{
			level:   slog.LevelWarn,
			message: "bootstrap tar archive entry skipped",
			attrs: map[string]string{
				"archive_entry": unsupportedEntry,
				"entry_type":    fmt.Sprint(tar.TypeFifo),
			},
		}})
	})

	t.Run("zip glob failure and symlink skip", func(t *testing.T) {
		const (
			regularEntry = "runtime/lib/onnxruntime-real.dll"
			linkEntry    = "runtime/lib/onnxruntime.dll"
			invalidGlob  = "["
		)
		archivePath := writeBootstrapDiagnosticZIP(t, []bootstrapDiagnosticArchiveEntry{
			{name: regularEntry, mode: 0o644, content: "library"},
			{name: linkEntry, mode: os.ModeSymlink | 0o777, content: "onnxruntime-real.dll"},
		})

		assertBootstrapDiagnosticCase(t, &legacyOutput, handler, func(t *testing.T) {
			if _, err := extractZIPArchive(archivePath, t.TempDir(), invalidGlob); err != nil {
				t.Fatalf("extractZIPArchive: %v", err)
			}
		}, []bootstrapDiagnosticExpectation{
			{
				level:   slog.LevelWarn,
				message: "bootstrap ZIP library glob match failed",
				attrs: map[string]string{
					"archive_entry": linkEntry,
					"library_glob":  invalidGlob,
					"error":         "",
				},
			},
			{
				level:   slog.LevelWarn,
				message: "bootstrap ZIP symlink entry skipped",
				attrs: map[string]string{
					"archive_entry": linkEntry,
					"entry_type":    "symlink",
				},
			},
		})
	})

	t.Run("lock wait", func(t *testing.T) {
		previousTimeout := bootstrapLockAcquireTimeout
		previousRetry := bootstrapLockRetryInterval
		previousLogInterval := bootstrapLockLogInterval
		bootstrapLockAcquireTimeout = 45 * time.Millisecond
		bootstrapLockRetryInterval = 5 * time.Millisecond
		bootstrapLockLogInterval = 25 * time.Millisecond
		t.Cleanup(func() {
			bootstrapLockAcquireTimeout = previousTimeout
			bootstrapLockRetryInterval = previousRetry
			bootstrapLockLogInterval = previousLogInterval
		})

		assertBootstrapDiagnosticCase(t, &legacyOutput, handler, func(t *testing.T) {
			lockPath := filepath.Join(t.TempDir(), "bootstrap.lock")
			locked := make(chan struct{})
			release := make(chan struct{})
			holderErr := make(chan error, 1)
			go func() {
				holderErr <- withProcessFileLock(lockPath, false, func() error {
					close(locked)
					<-release
					return nil
				})
			}()
			select {
			case <-locked:
			case <-time.After(time.Second):
				t.Fatal("timed out waiting for lock holder")
			}

			err := withProcessFileLock(lockPath, false, func() error { return nil })
			if err == nil || !strings.Contains(err.Error(), "timed out acquiring lock") {
				t.Fatalf("lock wait: got %v, want timeout", err)
			}
			close(release)
			if err := <-holderErr; err != nil {
				t.Fatalf("lock holder: %v", err)
			}
		}, []bootstrapDiagnosticExpectation{{
			level:   slog.LevelWarn,
			message: "waiting for bootstrap lock",
			attrs: map[string]string{
				"path":          "",
				"wait_duration": "",
			},
		}})
	})

	t.Run("user cache lookup failure fallback", func(t *testing.T) {
		previousUserCacheDir := bootstrapUserCacheDir
		cacheErr := errors.New("synthetic user cache lookup failure")
		bootstrapUserCacheDir = func() (string, error) { return "", cacheErr }
		t.Cleanup(func() {
			bootstrapUserCacheDir = previousUserCacheDir
			bootstrapCacheFallbackWarnOnce = sync.Once{}
		})
		fallback := filepath.Join(os.TempDir(), "onnx-purego", "onnxruntime")

		assertBootstrapDiagnosticCase(t, &legacyOutput, handler, func(t *testing.T) {
			bootstrapCacheFallbackWarnOnce = sync.Once{}
			if got := defaultBootstrapCacheDir(); got != fallback {
				t.Fatalf("defaultBootstrapCacheDir = %q, want %q", got, fallback)
			}
		}, []bootstrapDiagnosticExpectation{{
			level:   slog.LevelWarn,
			message: "bootstrap user cache lookup failed; using temporary cache",
			attrs: map[string]string{
				"path":  fallback,
				"error": "",
			},
		}})
	})

	t.Run("empty user cache fallback", func(t *testing.T) {
		previousUserCacheDir := bootstrapUserCacheDir
		bootstrapUserCacheDir = func() (string, error) { return "", nil }
		t.Cleanup(func() {
			bootstrapUserCacheDir = previousUserCacheDir
			bootstrapCacheFallbackWarnOnce = sync.Once{}
		})
		fallback := filepath.Join(os.TempDir(), "onnx-purego", "onnxruntime")

		assertBootstrapDiagnosticCase(t, &legacyOutput, handler, func(t *testing.T) {
			bootstrapCacheFallbackWarnOnce = sync.Once{}
			if got := defaultBootstrapCacheDir(); got != fallback {
				t.Fatalf("defaultBootstrapCacheDir = %q, want %q", got, fallback)
			}
		}, []bootstrapDiagnosticExpectation{{
			level:   slog.LevelWarn,
			message: "bootstrap user cache path empty; using temporary cache",
			attrs: map[string]string{
				"path": fallback,
			},
		}})
	})

	t.Run("consumer handler panic propagates", func(t *testing.T) {
		const panicValue = "bootstrap diagnostic handler panic"
		previousUserCacheDir := bootstrapUserCacheDir
		bootstrapUserCacheDir = func() (string, error) { return "", nil }
		bootstrapCacheFallbackWarnOnce = sync.Once{}
		SetDiagnosticHandler(diagnosticPanicHandler{value: panicValue})
		t.Cleanup(func() {
			SetDiagnosticHandler(nil)
			bootstrapUserCacheDir = previousUserCacheDir
			bootstrapCacheFallbackWarnOnce = sync.Once{}
		})

		var recovered any
		func() {
			defer func() { recovered = recover() }()
			_ = defaultBootstrapCacheDir()
		}()
		if recovered != panicValue {
			t.Fatalf("recovered panic = %v, want %q", recovered, panicValue)
		}
	})
}

// diagnosticReentrantCacheDirHandler re-enters bootstrap from inside Handle to
// prove the emit happens after sync.Once releases its internal mutex.
type diagnosticReentrantCacheDirHandler struct {
	reentered chan<- string
}

func (h diagnosticReentrantCacheDirHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h diagnosticReentrantCacheDirHandler) Handle(context.Context, slog.Record) error {
	select {
	case h.reentered <- defaultBootstrapCacheDir():
	default:
	}
	return nil
}

func (h diagnosticReentrantCacheDirHandler) WithAttrs([]slog.Attr) slog.Handler { return h }

func (h diagnosticReentrantCacheDirHandler) WithGroup(string) slog.Handler { return h }

func TestDefaultBootstrapCacheDirEmitsFallbackWarningOutsideOnce(t *testing.T) {
	previousUserCacheDir := bootstrapUserCacheDir
	bootstrapUserCacheDir = func() (string, error) { return "", nil }
	bootstrapCacheFallbackWarnOnce = sync.Once{}
	t.Cleanup(func() {
		SetDiagnosticHandler(nil)
		bootstrapUserCacheDir = previousUserCacheDir
		bootstrapCacheFallbackWarnOnce = sync.Once{}
	})

	reentered := make(chan string, 1)
	SetDiagnosticHandler(diagnosticReentrantCacheDirHandler{reentered: reentered})

	fallback := filepath.Join(os.TempDir(), "onnx-purego", "onnxruntime")
	resolved := make(chan string, 1)
	go func() { resolved <- defaultBootstrapCacheDir() }()

	select {
	case got := <-resolved:
		if got != fallback {
			t.Fatalf("defaultBootstrapCacheDir = %q, want %q", got, fallback)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("defaultBootstrapCacheDir deadlocked against its own sync.Once")
	}

	select {
	case got := <-reentered:
		if got != fallback {
			t.Fatalf("reentrant defaultBootstrapCacheDir = %q, want %q", got, fallback)
		}
	default:
		t.Fatal("diagnostic handler never re-entered defaultBootstrapCacheDir")
	}
}

func TestEnsureOnnxRuntimeSharedLibraryMemoizesVerifiedInstall(t *testing.T) {
	clearBootstrapEnv(t)

	artifact, err := resolveRuntimeArtifact(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("unsupported runtime for bootstrap test: %v", err)
	}

	cacheDir := t.TempDir()
	version := "1.99.71"
	archiveBytes := buildORTArchive(t, artifact, version, true)
	sum := sha256.Sum256(archiveBytes)
	server, hits := newArchiveServer(t, artifact, version, archiveBytes)
	opts := []BootstrapOption{
		WithBootstrapCacheDir(cacheDir),
		WithBootstrapVersion(version),
		WithBootstrapExpectedSHA256(hex.EncodeToString(sum[:])),
		withBootstrapBaseURL(server.URL),
		withBootstrapHTTPClient(server.Client()),
	}

	resolved, err := EnsureOnnxRuntimeSharedLibrary(opts...)
	if err != nil {
		t.Fatalf("initial bootstrap: %v", err)
	}

	installDir := filepath.Join(cacheDir, artifact.archiveName(version))
	manifestPath := filepath.Join(installDir, bootstrapManifestFilename)
	info, err := os.Stat(manifestPath)
	if err != nil {
		t.Fatalf("stat manifest: %v", err)
	}
	original, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}

	// Corrupt the manifest while preserving size and mtime. A memoized install
	// keeps its fingerprint, so the second call must not re-read the manifest.
	if err := os.WriteFile(manifestPath, bytes.Repeat([]byte("x"), len(original)), 0o600); err != nil {
		t.Fatalf("corrupt manifest: %v", err)
	}
	if err := os.Chtimes(manifestPath, info.ModTime(), info.ModTime()); err != nil {
		t.Fatalf("restore manifest mtime: %v", err)
	}

	memoized, err := EnsureOnnxRuntimeSharedLibrary(opts...)
	if err != nil {
		t.Fatalf("memoized cache hit revalidated the corrupt manifest: %v", err)
	}
	if memoized != resolved {
		t.Fatalf("memoized path = %q, want %q", memoized, resolved)
	}
	if got := hits.Load(); got != 1 {
		t.Fatalf("archive download count = %d, want 1 while the memo is valid", got)
	}

	// Any metadata change drops the memo and forces full verification, which now
	// fails because the manifest is corrupt.
	shifted := info.ModTime().Add(2 * time.Second)
	if err := os.Chtimes(manifestPath, shifted, shifted); err != nil {
		t.Fatalf("shift manifest mtime: %v", err)
	}
	if _, err := EnsureOnnxRuntimeSharedLibrary(append(opts, WithBootstrapDisableDownload(true))...); err == nil {
		t.Fatal("changed fingerprint returned a memoized install without revalidating")
	}
}

func TestReturnedErrorsDoNotEmit(t *testing.T) {
	handler := &bootstrapDiagnosticRecordingHandler{}
	SetDiagnosticHandler(handler)
	t.Cleanup(func() { SetDiagnosticHandler(nil) })

	assertNoDiagnostic := func(t *testing.T, operation func() error) {
		t.Helper()
		handler.reset()
		if err := operation(); err == nil {
			t.Fatal("operation returned nil, want error")
		}
		if records := handler.snapshot(); len(records) != 0 {
			t.Fatalf("returned error emitted %d diagnostics: %+v", len(records), records)
		}
	}

	t.Run("validation", func(t *testing.T) {
		assertNoDiagnostic(t, func() error {
			return WithBootstrapVersion("")(&bootstrapConfig{})
		})
	})

	t.Run("network", func(t *testing.T) {
		networkErr := errors.New("synthetic network failure")
		cfg := bootstrapConfig{
			cacheDir: t.TempDir(),
			httpClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, networkErr
			})},
			retryAttempts: 1,
		}
		assertNoDiagnostic(t, func() error {
			_, _, err := downloadRuntimeArchive(cfg, "https://example.invalid/runtime.tgz")
			return err
		})
	})

	t.Run("checksum", func(t *testing.T) {
		cfg, artifact, installDir, _ := newBootstrapDiagnosticDownloadFixture(t, false)
		cfg.expectedSHA256 = strings.Repeat("0", 64)
		assertNoDiagnostic(t, func() error {
			return downloadAndInstallRuntime(cfg, artifact, installDir)
		})
	})

	t.Run("archive", func(t *testing.T) {
		archivePath := filepath.Join(t.TempDir(), "runtime.tgz")
		if err := os.WriteFile(archivePath, []byte("not a gzip archive"), 0o600); err != nil {
			t.Fatalf("write invalid archive: %v", err)
		}
		assertNoDiagnostic(t, func() error {
			_, err := extractArchiveFile(archivePath, t.TempDir(), "tgz", "")
			return err
		})
	})

	t.Run("lock timeout before notice interval", func(t *testing.T) {
		previousTimeout := bootstrapLockAcquireTimeout
		previousRetry := bootstrapLockRetryInterval
		previousLogInterval := bootstrapLockLogInterval
		bootstrapLockAcquireTimeout = 30 * time.Millisecond
		bootstrapLockRetryInterval = 5 * time.Millisecond
		bootstrapLockLogInterval = time.Second
		t.Cleanup(func() {
			bootstrapLockAcquireTimeout = previousTimeout
			bootstrapLockRetryInterval = previousRetry
			bootstrapLockLogInterval = previousLogInterval
		})

		lockPath := filepath.Join(t.TempDir(), "bootstrap.lock")
		locked := make(chan struct{})
		release := make(chan struct{})
		holderErr := make(chan error, 1)
		go func() {
			holderErr <- withProcessFileLock(lockPath, false, func() error {
				close(locked)
				<-release
				return nil
			})
		}()
		select {
		case <-locked:
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for lock holder")
		}

		assertNoDiagnostic(t, func() error {
			return withProcessFileLock(lockPath, false, func() error { return nil })
		})
		close(release)
		if err := <-holderErr; err != nil {
			t.Fatalf("lock holder: %v", err)
		}
	})
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

func TestInitializeEnvironmentWithBootstrapLoadsSelectedPathAtomically(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)
	clearBootstrapEnv(t)
	t.Setenv("ONNXRUNTIME_SKIP_VERSION_CHECK", "1")

	dir := t.TempDir()
	bootstrapLib := filepath.Join(dir, "lib-bootstrap.so")
	if err := os.WriteFile(bootstrapLib, []byte("bootstrap"), 0o600); err != nil {
		t.Fatalf("write bootstrap library: %v", err)
	}
	otherLib := filepath.Join(dir, "lib-other.so")
	if err := os.WriteFile(otherLib, []byte("other"), 0o600); err != nil {
		t.Fatalf("write competing library: %v", err)
	}
	resolvedBootstrapLib, err := filepath.EvalSymlinks(bootstrapLib)
	if err != nil {
		t.Fatalf("resolve bootstrap library path: %v", err)
	}
	otherLib, _ = filepath.Abs(otherLib)

	noOp := purego.NewCallback(func() uintptr { return 0 })
	api := &OrtApi{
		GetErrorCode:                   noOp,
		GetErrorMessage:                noOp,
		ReleaseStatus:                  noOp,
		CreateMemoryInfo:               noOp,
		ReleaseMemoryInfo:              noOp,
		CreateTensorWithDataAsOrtValue: noOp,
		ReleaseValue:                   noOp,
		CreateSessionOptions:           noOp,
		ReleaseSessionOptions:          noOp,
		CreateSession:                  noOp,
		Run:                            noOp,
		ReleaseSession:                 noOp,
		ReleaseEnv:                     purego.NewCallback(func(uintptr) uintptr { return 0 }),
	}
	api.CreateEnv = purego.NewCallback(func(_ int32, _ uintptr, out uintptr) uintptr {
		//nolint:govet // The purego callback ABI supplies the native output address as uintptr; the test writes the fake OrtEnv handle through it.
		*(*uintptr)(unsafe.Pointer(out)) = 707
		return 0
	})
	apiBase := &OrtApiBase{
		GetApi: purego.NewCallback(func(uint32) uintptr {
			return uintptr(unsafe.Pointer(api))
		}),
		GetVersionString: noOp,
	}
	getAPIBase := purego.NewCallback(func() uintptr {
		return uintptr(unsafe.Pointer(apiBase))
	})

	loadEntered := make(chan string, 1)
	allowLoad := make(chan struct{})
	installEnvironmentLibraryHooks(
		func(path string) (uintptr, error) {
			loadEntered <- path
			<-allowLoad
			return 606, nil
		},
		func(uintptr, string) (uintptr, error) {
			return getAPIBase, nil
		},
		func(uintptr) error { return nil },
	)

	initDone := make(chan error, 1)
	go func() {
		initDone <- InitializeEnvironmentWithBootstrap(WithBootstrapLibraryPath(bootstrapLib))
	}()

	select {
	case loadedPath := <-loadEntered:
		if loadedPath != resolvedBootstrapLib {
			t.Fatalf("loaded path = %q, want resolved bootstrap path %q", loadedPath, resolvedBootstrapLib)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("bootstrap initialization did not reach the library loader")
	}

	setPathDone := make(chan error, 1)
	go func() {
		setPathDone <- SetSharedLibraryPath(otherLib)
	}()
	select {
	case err := <-setPathDone:
		t.Fatalf("competing path mutation completed during bootstrap load: %v", err)
	case <-time.After(50 * time.Millisecond):
	}

	close(allowLoad)
	if err := <-initDone; err != nil {
		t.Fatalf("bootstrap initialization failed: %v", err)
	}
	if err := <-setPathDone; err == nil || !strings.Contains(err.Error(), "environment is initialized") {
		t.Fatalf("competing path mutation error = %v, want initialized-environment rejection", err)
	}
	if err := DestroyEnvironment(); err != nil {
		t.Fatalf("destroy environment: %v", err)
	}
}

func clearBootstrapEnv(t *testing.T) {
	t.Helper()
	t.Setenv("ONNXRUNTIME_LIB_PATH", "")
	t.Setenv("ONNXRUNTIME_CACHE_DIR", "")
	t.Setenv("ONNXRUNTIME_VERSION", "")
	t.Setenv("ONNXRUNTIME_DISABLE_DOWNLOAD", "")
	t.Setenv("ONNXRUNTIME_ALLOW_SHARED_CACHE", "")
}

func makeBootstrapTreeReadOnly(t *testing.T, root string) {
	t.Helper()

	if err := filepath.Walk(root, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() {
			return os.Chmod(path, 0o550)
		}
		return os.Chmod(path, 0o440)
	}); err != nil {
		t.Fatalf("make bootstrap tree read-only: %v", err)
	}
	t.Cleanup(func() {
		_ = filepath.Walk(root, func(path string, info os.FileInfo, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			if info.IsDir() {
				return os.Chmod(path, 0o750)
			}
			return os.Chmod(path, 0o600)
		})
	})
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

func buildArchiveWithFileMode(t *testing.T, extension, name string, mode os.FileMode) []byte {
	t.Helper()

	const content = "synthetic library"
	var buf bytes.Buffer
	switch extension {
	case "tgz":
		gz := gzip.NewWriter(&buf)
		tw := tar.NewWriter(gz)
		if err := tw.WriteHeader(&tar.Header{
			Name: name,
			Mode: int64(mode.Perm()),
			Size: int64(len(content)),
		}); err != nil {
			t.Fatalf("failed to write tar header: %v", err)
		}
		if _, err := tw.Write([]byte(content)); err != nil {
			t.Fatalf("failed to write tar content: %v", err)
		}
		if err := tw.Close(); err != nil {
			t.Fatalf("failed to close tar writer: %v", err)
		}
		if err := gz.Close(); err != nil {
			t.Fatalf("failed to close gzip writer: %v", err)
		}
	case "zip":
		zw := zip.NewWriter(&buf)
		header := &zip.FileHeader{Name: name, Method: zip.Deflate}
		header.SetMode(mode)
		entry, err := zw.CreateHeader(header)
		if err != nil {
			t.Fatalf("failed to create zip entry: %v", err)
		}
		if _, err := entry.Write([]byte(content)); err != nil {
			t.Fatalf("failed to write zip content: %v", err)
		}
		if err := zw.Close(); err != nil {
			t.Fatalf("failed to close zip writer: %v", err)
		}
	default:
		t.Fatalf("unsupported archive extension %q", extension)
	}
	return buf.Bytes()
}

func assertBootstrapDirectoryMode(t *testing.T, path string) {
	t.Helper()

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("failed to stat directory %q: %v", path, err)
	}
	if !info.IsDir() {
		t.Fatalf("expected directory at %q", path)
	}
	perm := info.Mode().Perm()
	if perm&0o700 != 0o700 {
		t.Fatalf("directory %q mode %#o does not retain owner access", path, perm)
	}
	if perm&0o027 != 0 {
		t.Fatalf("directory %q mode %#o grants group write or other-user access", path, perm)
	}
}

func assertBootstrapLibraryMode(t *testing.T, path string) {
	t.Helper()

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("failed to stat library %q: %v", path, err)
	}
	perm := info.Mode().Perm()
	if perm&0o500 != 0o500 {
		t.Fatalf("library %q mode %#o does not retain owner read/execute", path, perm)
	}
	if perm&0o022 != 0 {
		t.Fatalf("library %q mode %#o grants group/other write", path, perm)
	}
}

func assertBootstrapLockMode(t *testing.T, path string) {
	t.Helper()

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("failed to stat lock file %q: %v", path, err)
	}
	perm := info.Mode().Perm()
	if perm&0o600 != 0o600 {
		t.Fatalf("lock file %q mode %#o does not retain owner read/write", path, perm)
	}
	if perm&0o077 != 0 {
		t.Fatalf("lock file %q mode %#o grants group/other access", path, perm)
	}
}

type bootstrapDiagnosticExpectation struct {
	level     slog.Level
	message   string
	attrs     map[string]string
	forbidden []string
}

type bootstrapDiagnosticRecord struct {
	level   slog.Level
	message string
	attrs   map[string]any
}

type bootstrapDiagnosticRecordingHandler struct {
	mu      sync.Mutex
	records []bootstrapDiagnosticRecord
}

func (*bootstrapDiagnosticRecordingHandler) Enabled(context.Context, slog.Level) bool {
	return true
}

func (h *bootstrapDiagnosticRecordingHandler) Handle(_ context.Context, record slog.Record) error {
	attrs := make(map[string]any, record.NumAttrs())
	record.Attrs(func(attr slog.Attr) bool {
		attrs[attr.Key] = attr.Value.Resolve().Any()
		return true
	})

	h.mu.Lock()
	h.records = append(h.records, bootstrapDiagnosticRecord{
		level:   record.Level,
		message: record.Message,
		attrs:   attrs,
	})
	h.mu.Unlock()
	return nil
}

func (h *bootstrapDiagnosticRecordingHandler) WithAttrs([]slog.Attr) slog.Handler {
	return h
}

func (h *bootstrapDiagnosticRecordingHandler) WithGroup(string) slog.Handler {
	return h
}

func (h *bootstrapDiagnosticRecordingHandler) reset() {
	h.mu.Lock()
	h.records = nil
	h.mu.Unlock()
}

func (h *bootstrapDiagnosticRecordingHandler) snapshot() []bootstrapDiagnosticRecord {
	h.mu.Lock()
	defer h.mu.Unlock()

	records := make([]bootstrapDiagnosticRecord, len(h.records))
	copy(records, h.records)
	return records
}

func assertBootstrapDiagnosticCase(
	t *testing.T,
	legacyOutput *bytes.Buffer,
	handler *bootstrapDiagnosticRecordingHandler,
	exercise func(*testing.T),
	want []bootstrapDiagnosticExpectation,
) {
	t.Helper()

	SetDiagnosticHandler(nil)
	legacyOutput.Reset()
	exercise(t)
	if output := legacyOutput.String(); output != "" {
		t.Fatalf("nil diagnostic handler produced legacy output: %q", output)
	}

	handler.reset()
	SetDiagnosticHandler(handler)
	exercise(t)
	got := handler.snapshot()
	if len(got) != len(want) {
		t.Fatalf("diagnostic count = %d, want %d: %+v", len(got), len(want), got)
	}
	for index, expectation := range want {
		record := got[index]
		if record.level != expectation.level {
			t.Errorf("record %d level = %s, want %s", index, record.level, expectation.level)
		}
		if record.message != expectation.message {
			t.Errorf("record %d message = %q, want %q", index, record.message, expectation.message)
		}
		if len(record.attrs) != len(expectation.attrs) {
			t.Errorf("record %d attrs = %+v, want keys %+v", index, record.attrs, expectation.attrs)
		}
		for key, wantValue := range expectation.attrs {
			gotValue, ok := record.attrs[key]
			if !ok {
				t.Errorf("record %d missing attr %q", index, key)
				continue
			}
			if wantValue != "" && fmt.Sprint(gotValue) != wantValue {
				t.Errorf("record %d attr %q = %v, want %q", index, key, gotValue, wantValue)
			}
		}
		serialized := fmt.Sprint(record)
		for _, forbidden := range expectation.forbidden {
			if strings.Contains(serialized, forbidden) {
				t.Errorf("record %d contains forbidden value %q: %s", index, forbidden, serialized)
			}
		}
	}
	SetDiagnosticHandler(nil)
}

func newBootstrapDiagnosticDownloadFixture(
	t *testing.T,
	withCredentials bool,
) (bootstrapConfig, runtimeArtifact, string, string) {
	t.Helper()

	artifact, err := resolveRuntimeArtifact("linux", "amd64")
	if err != nil {
		t.Fatalf("resolveRuntimeArtifact: %v", err)
	}
	const version = "1.99.14"
	archive := buildORTArchive(t, artifact, version, true)
	sum := sha256.Sum256(archive)
	checksum := hex.EncodeToString(sum[:])
	server, _ := newArchiveServer(t, artifact, version, archive)
	baseURL := server.URL
	if withCredentials {
		parsedURL, err := url.Parse(baseURL)
		if err != nil {
			t.Fatalf("parse archive server URL: %v", err)
		}
		parsedURL.User = url.UserPassword("bootstrap-user", "bootstrap-secret")
		baseURL = parsedURL.String()
	}

	cacheDir := filepath.Join(t.TempDir(), "cache")
	return bootstrapConfig{
		cacheDir:      cacheDir,
		version:       version,
		baseURL:       baseURL,
		httpClient:    server.Client(),
		retryAttempts: 1,
	}, artifact, filepath.Join(cacheDir, artifact.archiveName(version)), checksum
}

type bootstrapDiagnosticArchiveEntry struct {
	name     string
	mode     os.FileMode
	typeflag byte
	linkname string
	content  string
}

func writeBootstrapDiagnosticTGZ(t *testing.T, entries []bootstrapDiagnosticArchiveEntry) string {
	t.Helper()

	var buffer bytes.Buffer
	gzipWriter := gzip.NewWriter(&buffer)
	tarWriter := tar.NewWriter(gzipWriter)
	for _, entry := range entries {
		typeflag := entry.typeflag
		if typeflag == 0 {
			typeflag = tar.TypeReg
		}
		header := &tar.Header{
			Name:     entry.name,
			Mode:     int64(entry.mode.Perm()),
			Typeflag: typeflag,
			Linkname: entry.linkname,
		}
		if typeflag == tar.TypeReg {
			header.Size = int64(len(entry.content))
		}
		if err := tarWriter.WriteHeader(header); err != nil {
			t.Fatalf("write tar header %q: %v", entry.name, err)
		}
		if typeflag == tar.TypeReg {
			if _, err := tarWriter.Write([]byte(entry.content)); err != nil {
				t.Fatalf("write tar content %q: %v", entry.name, err)
			}
		}
	}
	if err := tarWriter.Close(); err != nil {
		t.Fatalf("close tar writer: %v", err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatalf("close gzip writer: %v", err)
	}

	archivePath := filepath.Join(t.TempDir(), "diagnostic.tgz")
	if err := os.WriteFile(archivePath, buffer.Bytes(), 0o600); err != nil {
		t.Fatalf("write diagnostic tgz: %v", err)
	}
	return archivePath
}

func writeBootstrapDiagnosticZIP(t *testing.T, entries []bootstrapDiagnosticArchiveEntry) string {
	t.Helper()

	var buffer bytes.Buffer
	zipWriter := zip.NewWriter(&buffer)
	for _, entry := range entries {
		header := &zip.FileHeader{Name: entry.name, Method: zip.Deflate}
		header.SetMode(entry.mode)
		writer, err := zipWriter.CreateHeader(header)
		if err != nil {
			t.Fatalf("create ZIP entry %q: %v", entry.name, err)
		}
		if _, err := writer.Write([]byte(entry.content)); err != nil {
			t.Fatalf("write ZIP entry %q: %v", entry.name, err)
		}
	}
	if err := zipWriter.Close(); err != nil {
		t.Fatalf("close ZIP writer: %v", err)
	}

	archivePath := filepath.Join(t.TempDir(), "diagnostic.zip")
	if err := os.WriteFile(archivePath, buffer.Bytes(), 0o600); err != nil {
		t.Fatalf("write diagnostic zip: %v", err)
	}
	return archivePath
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type failingReadCloser struct {
	readErr  error
	closeErr error
}

func (r *failingReadCloser) Read([]byte) (int, error) {
	return 0, r.readErr
}

func (r *failingReadCloser) Close() error {
	return r.closeErr
}

type closeErrorReadCloser struct {
	io.Reader
	closeErr error
}

func (r *closeErrorReadCloser) Close() error {
	return r.closeErr
}
