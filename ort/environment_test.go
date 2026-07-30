package ort

import (
	"bytes"
	"context"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"log/slog"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"
)

// resetEnvironmentState resets global state for testing
func resetEnvironmentState() {
	mu.Lock()
	defer mu.Unlock()
	refCount = 0
	ortLib = 0
	ortAPI = nil
	ortEnv = 0
	libPath = ""
	logLevel = LoggingLevelWarning
	getVersionStringFunc = nil
	getErrorCodeFunc = nil
	getErrorMessageFunc = nil
	releaseStatusFunc = nil
	createMemoryInfoFunc = nil
	releaseMemoryInfoFunc = nil
	createTensorWithDataAsOrtValueFunc = nil
	releaseValueFunc = nil
	createSessionOptionsFunc = nil
	releaseSessionOptionsFunc = nil
	createSessionFunc = nil
	runSessionFunc = nil
	releaseSessionFunc = nil
	environmentLoadLibrary = loadLibrary
	environmentGetSymbol = getSymbol
	environmentCloseLibrary = closeLibrary
}

func installEnvironmentLibraryHooks(
	load func(string) (uintptr, error),
	symbol func(uintptr, string) (uintptr, error),
	close func(uintptr) error,
) {
	mu.Lock()
	environmentLoadLibrary = load
	environmentGetSymbol = symbol
	environmentCloseLibrary = close
	mu.Unlock()
}

func TestEnvironmentErrorFunctionRegistration(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	registerTestFunctions := func() {
		mu.Lock()
		getErrorCodeFunc = func(uintptr) ErrorCode { return ErrorCodeFail }
		getErrorMessageFunc = func(uintptr) uintptr { return 0 }
		releaseStatusFunc = func(uintptr) {}
		mu.Unlock()
	}
	assertRegistered := func(want bool) {
		t.Helper()
		mu.Lock()
		functions := []struct {
			name       string
			registered bool
		}{
			{name: "GetErrorCode", registered: getErrorCodeFunc != nil},
			{name: "GetErrorMessage", registered: getErrorMessageFunc != nil},
			{name: "ReleaseStatus", registered: releaseStatusFunc != nil},
		}
		mu.Unlock()
		for _, function := range functions {
			if function.registered != want {
				t.Fatalf("%s registered = %t, want %t", function.name, function.registered, want)
			}
		}
	}

	registerTestFunctions()
	assertRegistered(true)

	mu.Lock()
	clearORTGlobalsLocked()
	mu.Unlock()
	assertRegistered(false)

	registerTestFunctions()
	resetEnvironmentState()
	assertRegistered(false)
}

func TestEnvironmentErrorChains(t *testing.T) {
	t.Run("load failure preserves OS cause", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		handler := &diagnosticCountingHandler{}
		SetDiagnosticHandler(handler)
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		loadCause := &os.PathError{
			Op:   "dlopen",
			Path: "/missing/libonnxruntime.so",
			Err:  os.ErrNotExist,
		}
		installEnvironmentLibraryHooks(
			func(string) (uintptr, error) { return 0, loadCause },
			func(uintptr, string) (uintptr, error) {
				t.Fatal("symbol lookup called after load failure")
				return 0, nil
			},
			func(uintptr) error {
				t.Fatal("close called after load failure")
				return nil
			},
		)

		if err := SetSharedLibraryPath(loadCause.Path); err != nil {
			t.Fatalf("set library path: %v", err)
		}
		err := InitializeEnvironment()
		if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("initialization error = %v, want os.ErrNotExist", err)
		}
		var pathErr *os.PathError
		if !errors.As(err, &pathErr) {
			t.Fatalf("initialization error = %v, want *os.PathError", err)
		}
		if pathErr != loadCause {
			t.Fatalf("path error = %p, want original cause %p", pathErr, loadCause)
		}
		if got := handler.count.Load(); got != 0 {
			t.Fatalf("returned load error emitted %d diagnostics, want 0", got)
		}
	})

	t.Run("symbol failure preserves cause and closes library", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		handler := &diagnosticCountingHandler{}
		SetDiagnosticHandler(handler)
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		symbolCause := errors.New("symbol resolution failed")
		var closes int
		installEnvironmentLibraryHooks(
			func(string) (uintptr, error) { return 101, nil },
			func(handle uintptr, symbol string) (uintptr, error) {
				if handle != 101 || symbol != "OrtGetApiBase" {
					t.Errorf("symbol lookup = (%d, %q), want (101, OrtGetApiBase)", handle, symbol)
				}
				return 0, symbolCause
			},
			func(handle uintptr) error {
				if handle != 101 {
					t.Errorf("close handle = %d, want 101", handle)
				}
				closes++
				return nil
			},
		)

		if err := SetSharedLibraryPath("fake-runtime"); err != nil {
			t.Fatalf("set library path: %v", err)
		}
		err := InitializeEnvironment()
		if !errors.Is(err, symbolCause) {
			t.Fatalf("initialization error = %v, want symbol cause", err)
		}
		if closes != 1 {
			t.Fatalf("library close count = %d, want 1", closes)
		}
		if got := handler.count.Load(); got != 0 {
			t.Fatalf("returned symbol error emitted %d diagnostics, want 0", got)
		}
	})

	t.Run("primary and cleanup failures remain independently reachable", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		handler := &diagnosticCountingHandler{}
		SetDiagnosticHandler(handler)
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		primaryCause := errors.New("primary initialization failure")
		cleanupCause := errors.New("library cleanup failure")
		installEnvironmentLibraryHooks(
			func(string) (uintptr, error) { return 202, nil },
			func(uintptr, string) (uintptr, error) { return 0, primaryCause },
			func(uintptr) error { return cleanupCause },
		)

		if err := SetSharedLibraryPath("fake-runtime"); err != nil {
			t.Fatalf("set library path: %v", err)
		}
		err := InitializeEnvironment()
		if !errors.Is(err, primaryCause) {
			t.Fatalf("joined error = %v, want primary cause", err)
		}
		if !errors.Is(err, cleanupCause) {
			t.Fatalf("joined error = %v, want cleanup cause", err)
		}
		if got := handler.count.Load(); got != 0 {
			t.Fatalf("returned joined error emitted %d diagnostics, want 0", got)
		}
	})

	t.Run("initialization failure with old runtime emits nothing", func(t *testing.T) {
		handler := &diagnosticCountingHandler{}
		SetDiagnosticHandler(handler)
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		wantErr := errors.New("CreateEnv failed")
		err := completeEnvironmentInitialization("1.21.4", false, wantErr)
		if !errors.Is(err, wantErr) {
			t.Fatalf("initialization error = %v, want %v", err, wantErr)
		}
		if got := handler.count.Load(); got != 0 {
			t.Fatalf("returned initialization error emitted %d diagnostics, want 0", got)
		}
	})
}

func TestInitializeEnvironmentRejectsUnsupportedAPIVersion(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	apiBase := &OrtApiBase{
		GetApi: purego.NewCallback(func(uint32) uintptr {
			return 0
		}),
		GetVersionString: purego.NewCallback(func() uintptr {
			return 0
		}),
	}
	getAPIBase := purego.NewCallback(func() uintptr {
		return uintptr(unsafe.Pointer(apiBase))
	})

	var closes int
	installEnvironmentLibraryHooks(
		func(string) (uintptr, error) { return 404, nil },
		func(handle uintptr, symbol string) (uintptr, error) {
			if handle != 404 || symbol != "OrtGetApiBase" {
				t.Fatalf("symbol lookup = (%d, %q), want (404, OrtGetApiBase)", handle, symbol)
			}
			return getAPIBase, nil
		},
		func(handle uintptr) error {
			if handle != 404 {
				t.Fatalf("close handle = %d, want 404", handle)
			}
			closes++
			return nil
		},
	)

	if err := SetSharedLibraryPath("unsupported-runtime"); err != nil {
		t.Fatalf("set shared library path: %v", err)
	}

	err := InitializeEnvironment()
	if !errors.Is(err, ErrUnsupportedRuntime) {
		t.Fatalf("initialization error = %v, want ErrUnsupportedRuntime", err)
	}
	if closes != 1 {
		t.Fatalf("library close count = %d, want 1", closes)
	}
}

func TestEnvironmentStatusConversion(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	handler := &diagnosticCountingHandler{}
	SetDiagnosticHandler(handler)
	t.Cleanup(func() { SetDiagnosticHandler(nil) })

	const statusHandle = uintptr(303)
	var releases int
	mu.Lock()
	getErrorCodeFunc = func(status uintptr) ErrorCode {
		if status != statusHandle {
			t.Errorf("GetErrorCode status = %d, want %d", status, statusHandle)
		}
		return ErrorCodeRuntimeException
	}
	getErrorMessageFunc = func(status uintptr) uintptr {
		if status != statusHandle {
			t.Errorf("GetErrorMessage status = %d, want %d", status, statusHandle)
		}
		// Keep the production call-site proof race/checkptr safe. Non-empty
		// copy-before-release semantics are covered by TestStatusToError.
		return 0
	}
	releaseStatusFunc = func(status uintptr) {
		if status != statusHandle {
			t.Errorf("ReleaseStatus status = %d, want %d", status, statusHandle)
		}
		releases++
	}
	mu.Unlock()

	ortCallMu.RLock()
	handle, err := createEnvironment(
		func(level int32, logID uintptr, out *uintptr) uintptr {
			if level != int32(LoggingLevelWarning) {
				t.Errorf("log level = %d, want %d", level, LoggingLevelWarning)
			}
			if logID == 0 {
				t.Error("log ID pointer is nil")
			}
			if out == nil {
				t.Error("environment output pointer is nil")
			}
			return statusHandle
		},
		LoggingLevelWarning,
	)
	ortCallMu.RUnlock()

	if handle != 0 {
		t.Fatalf("environment handle = %d, want 0 on status failure", handle)
	}
	var nativeErr *ORTError
	if !errors.As(err, &nativeErr) {
		t.Fatalf("errors.As(%v, *ORTError) = false", err)
	}
	if nativeErr.Operation != "create ONNX Runtime environment" {
		t.Fatalf("operation = %q, want create ONNX Runtime environment", nativeErr.Operation)
	}
	if nativeErr.Code != ErrorCodeRuntimeException {
		t.Fatalf("code = %d, want %d", nativeErr.Code, ErrorCodeRuntimeException)
	}
	if nativeErr.Message != "" {
		t.Fatalf("message = %q, want empty race-safe probe message", nativeErr.Message)
	}
	if releases != 1 {
		t.Fatalf("status release count = %d, want 1", releases)
	}
	if got := handler.count.Load(); got != 0 {
		t.Fatalf("returned status error emitted %d diagnostics, want 0", got)
	}
}

func TestCreateEnvironmentRejectsZeroHandle(t *testing.T) {
	handle, err := createEnvironment(
		func(_ int32, _ uintptr, _ *uintptr) uintptr {
			return 0
		},
		LoggingLevelWarning,
	)
	if handle != 0 {
		t.Fatalf("environment handle = %d, want 0", handle)
	}
	if !errors.Is(err, ErrNativeContract) {
		t.Fatalf("createEnvironment error = %v, want ErrNativeContract", err)
	}
}

func TestDiagnosticRuntimeVersion(t *testing.T) {
	t.Run("old runtime emits one structured warning", func(t *testing.T) {
		var output bytes.Buffer
		SetDiagnosticHandler(slog.NewJSONHandler(&output, nil))
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		emitRuntimeVersionWarning("1.21.4")

		record := decodeDiagnosticRecord(t, &output)
		if got := record["level"]; got != "WARN" {
			t.Fatalf("level = %v, want WARN", got)
		}
		if got := record["runtime_version"]; got != "1.21.4" {
			t.Fatalf("runtime_version = %v, want 1.21.4", got)
		}
		if got := record["api_version"]; got != float64(ORT_API_VERSION) {
			t.Fatalf("api_version = %v, want %d", got, ORT_API_VERSION)
		}
	})

	t.Run("supported runtime emits nothing", func(t *testing.T) {
		handler := &diagnosticCountingHandler{}
		SetDiagnosticHandler(handler)
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		emitRuntimeVersionWarning("1.22.0")

		if got := handler.count.Load(); got != 0 {
			t.Fatalf("supported runtime emitted %d diagnostics, want 0", got)
		}
	})

	t.Run("future major runtime emits nothing", func(t *testing.T) {
		handler := &diagnosticCountingHandler{}
		SetDiagnosticHandler(handler)
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		emitRuntimeVersionWarning("2.0.0")

		if got := handler.count.Load(); got != 0 {
			t.Fatalf("future major runtime emitted %d diagnostics, want 0", got)
		}
	})

	t.Run("older major runtime emits warning", func(t *testing.T) {
		handler := &diagnosticCountingHandler{}
		SetDiagnosticHandler(handler)
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		emitRuntimeVersionWarning("0.99.0")

		if got := handler.count.Load(); got != 1 {
			t.Fatalf("older major runtime emitted %d diagnostics, want 1", got)
		}
	})

	t.Run("malformed runtime emits diagnostic", func(t *testing.T) {
		var output bytes.Buffer
		SetDiagnosticHandler(slog.NewJSONHandler(&output, nil))
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		emitRuntimeVersionWarning("not-a-version")

		record := decodeDiagnosticRecord(t, &output)
		if got := record["msg"]; got != "Could not parse ONNX Runtime version" {
			t.Fatalf("message = %v, want parse diagnostic", got)
		}
		if got := record["runtime_version"]; got != "not-a-version" {
			t.Fatalf("runtime_version = %v, want not-a-version", got)
		}
	})

	t.Run("nil handler restores warning stderr default", func(t *testing.T) {
		handler := &diagnosticCountingHandler{}
		SetDiagnosticHandler(handler)
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		output := captureDiagnosticStderr(t, func() {
			SetDiagnosticHandler(nil)
			emitRuntimeVersionWarning("1.21.4")
		})

		if got := handler.count.Load(); got != 0 {
			t.Fatalf("nil handler emitted %d diagnostics, want 0", got)
		}
		if !strings.Contains(output, "ONNX Runtime version is older") {
			t.Fatalf("nil-handler stderr output = %q, want runtime warning", output)
		}
	})

	t.Run("consumer handler panic propagates synchronously", func(t *testing.T) {
		const panicValue = "runtime warning handler panic"
		SetDiagnosticHandler(diagnosticPanicHandler{value: panicValue})
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		var recovered any
		func() {
			defer func() {
				recovered = recover()
			}()
			emitRuntimeVersionWarning("1.21.4")
		}()

		if recovered != panicValue {
			t.Fatalf("recovered panic = %v, want %q", recovered, panicValue)
		}
	})

	t.Run("rollback failure is reported without masking handler panic", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		panicValue := &struct{ label string }{label: "handler panic"}
		closeCause := errors.New("rollback close failed")
		var closeCount int
		installEnvironmentLibraryHooks(
			func(string) (uintptr, error) { return 0, nil },
			func(uintptr, string) (uintptr, error) { return 0, nil },
			func(handle uintptr) error {
				closeCount++
				if handle != 912 {
					t.Errorf("closed handle = %d, want 912", handle)
				}
				return closeCause
			},
		)
		mu.Lock()
		refCount = 1
		ortLib = 912
		runSessionFunc = func(uintptr, uintptr, *uintptr, *uintptr, uintptr, *uintptr, uintptr, *uintptr) uintptr {
			return 0
		}
		mu.Unlock()

		var recovered any
		output := captureDiagnosticStderr(t, func() {
			SetDiagnosticHandler(diagnosticPanicHandler{value: panicValue})
			func() {
				defer func() {
					recovered = recover()
				}()
				_ = completeEnvironmentInitialization("1.21.4", true, nil)
			}()
		})

		if recovered != panicValue {
			t.Fatalf("recovered panic = %#v, want original sentinel %#v", recovered, panicValue)
		}
		if closeCount != 1 {
			t.Fatalf("library close count = %d, want 1", closeCount)
		}
		if !strings.Contains(output, closeCause.Error()) {
			t.Fatalf("rollback stderr output = %q, want close cause", output)
		}
		mu.Lock()
		defer mu.Unlock()
		if refCount != 0 || ortLib != 0 || runSessionFunc != nil {
			t.Fatalf(
				"rollback state = refCount %d, ortLib %d, runSessionFunc nil %t",
				refCount,
				ortLib,
				runSessionFunc == nil,
			)
		}
	})
}

func TestLifecycleLockHierarchyDocumentation(t *testing.T) {
	parsed, err := parser.ParseFile(
		token.NewFileSet(),
		"environment.go",
		nil,
		parser.ParseComments,
	)
	if err != nil {
		t.Fatalf("parse environment.go: %v", err)
	}

	var documentation string
	ast.Inspect(parsed, func(node ast.Node) bool {
		spec, ok := node.(*ast.ValueSpec)
		if !ok || len(spec.Names) == 0 || spec.Names[0].Name != "mu" || spec.Doc == nil {
			return true
		}
		documentation = spec.Doc.Text()
		return false
	})
	if documentation == "" {
		t.Fatal("mu declaration has no lock hierarchy documentation")
	}

	for _, fact := range []string{
		"partial order",
		"AdvancedSession.runMu -> ortCallMu",
		"ortCallMu -> SessionOptions.handleMu, mu, Tensor.runMu, MemoryInfo.handleMu",
		"SessionOptions.handleMu -> mu only when both are held",
		"mu is released before Tensor.runMu or MemoryInfo.handleMu",
		"Tensor.runMu and MemoryInfo.handleMu are never nested with each other",
	} {
		if !strings.Contains(documentation, fact) {
			t.Errorf("mu documentation missing %q:\n%s", fact, documentation)
		}
	}
}

func TestInitializeEnvironmentDiagnosticHandlerCanQueryRuntime(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)
	t.Setenv("ONNXRUNTIME_SKIP_VERSION_CHECK", "")

	versionBytes, versionPtr := GoToCstring("1.21.4")
	noOp := purego.NewCallback(func() {})
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
		ReleaseEnv:                     purego.NewCallback(func(uintptr) {}),
	}
	api.CreateEnv = purego.NewCallback(func(_ int32, _ uintptr, out uintptr) uintptr {
		//nolint:govet // The purego callback ABI supplies the native output address as uintptr; the test writes the fake OrtEnv handle through it.
		*(*uintptr)(unsafe.Pointer(out)) = 1001
		return 0
	})
	apiBase := &OrtApiBase{
		GetApi: purego.NewCallback(func(uint32) uintptr {
			return uintptr(unsafe.Pointer(api))
		}),
		GetVersionString: purego.NewCallback(func() uintptr {
			return versionPtr
		}),
	}
	getAPIBase := purego.NewCallback(func() uintptr {
		return uintptr(unsafe.Pointer(apiBase))
	})

	installEnvironmentLibraryHooks(
		func(string) (uintptr, error) { return 1002, nil },
		func(uintptr, string) (uintptr, error) { return getAPIBase, nil },
		func(uintptr) error { return nil },
	)
	if err := SetSharedLibraryPath("reentrant-diagnostic-runtime"); err != nil {
		t.Fatalf("set shared library path: %v", err)
	}

	handler := &runtimeQueryDiagnosticHandler{
		handled:      make(chan struct{}),
		queryTimeout: time.Second,
	}
	SetDiagnosticHandler(handler)
	t.Cleanup(func() { SetDiagnosticHandler(nil) })

	initDone := make(chan error, 1)
	go func() {
		initDone <- InitializeEnvironment()
	}()

	select {
	case err := <-initDone:
		if err != nil {
			t.Fatalf("initialize environment: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("InitializeEnvironment blocked in the diagnostic handler")
	}

	select {
	case <-handler.handled:
	case <-time.After(2 * time.Second):
		t.Fatal("runtime-version diagnostic was not handled")
	}
	if handler.queryTimedOut {
		t.Fatal("diagnostic handler runtime queries blocked on lifecycle locks")
	}
	if !handler.initialized {
		t.Fatal("IsInitialized returned false from the diagnostic handler")
	}
	if handler.version != "1.21.4" {
		t.Fatalf("GetVersionString returned %q from the diagnostic handler, want 1.21.4", handler.version)
	}

	if err := DestroyEnvironment(); err != nil {
		t.Fatalf("destroy environment: %v", err)
	}
	runtime.KeepAlive(versionBytes)
	runtime.KeepAlive(api)
	runtime.KeepAlive(apiBase)
}

type runtimeQueryDiagnosticHandler struct {
	handled       chan struct{}
	queryTimeout  time.Duration
	queryTimedOut bool
	initialized   bool
	version       string
}

func (*runtimeQueryDiagnosticHandler) Enabled(context.Context, slog.Level) bool {
	return true
}

func (h *runtimeQueryDiagnosticHandler) Handle(context.Context, slog.Record) error {
	type queryResult struct {
		initialized bool
		version     string
	}
	result := make(chan queryResult, 1)
	go func() {
		result <- queryResult{
			initialized: IsInitialized(),
			version:     GetVersionString(),
		}
	}()

	select {
	case query := <-result:
		h.initialized = query.initialized
		h.version = query.version
	case <-time.After(h.queryTimeout):
		h.queryTimedOut = true
	}
	close(h.handled)
	return nil
}

func (h *runtimeQueryDiagnosticHandler) WithAttrs([]slog.Attr) slog.Handler {
	return h
}

func (h *runtimeQueryDiagnosticHandler) WithGroup(string) slog.Handler {
	return h
}

func TestIsInitialized(t *testing.T) {
	resetEnvironmentState()

	if IsInitialized() {
		t.Error("expected environment to not be initialized")
	}

	// Manually set refCount to simulate initialization
	mu.Lock()
	refCount = 1
	mu.Unlock()

	if !IsInitialized() {
		t.Error("expected environment to be initialized")
	}

	// Reset
	resetEnvironmentState()
}

func TestSetSharedLibraryPath(t *testing.T) {
	resetEnvironmentState()

	if err := SetSharedLibraryPath(""); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("empty path error = %v, want ErrInvalidArgument", err)
	}

	path := "/test/path/libonnxruntime.so"
	err := SetSharedLibraryPath(path)
	if err != nil {
		t.Errorf("unexpected error setting library path: %v", err)
	}

	mu.Lock()
	if libPath != path {
		t.Errorf("expected libPath to be %q, got %q", path, libPath)
	}
	mu.Unlock()

	// Test that changing path after init returns an error
	mu.Lock()
	refCount = 1
	mu.Unlock()

	newPath := "/different/path.so"
	err = SetSharedLibraryPath(newPath)
	if err == nil {
		t.Error("expected error when setting library path after initialization")
	}

	mu.Lock()
	if libPath != path {
		t.Errorf("expected libPath to remain %q after init, got %q", path, libPath)
	}
	mu.Unlock()

	resetEnvironmentState()
}

func TestSetLogLevel(t *testing.T) {
	resetEnvironmentState()

	if err := SetLogLevel(LoggingLevel(-1)); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("invalid log level error = %v, want ErrInvalidArgument", err)
	}

	tests := []LoggingLevel{
		LoggingLevelVerbose,
		LoggingLevelInfo,
		LoggingLevelWarning,
		LoggingLevelError,
		LoggingLevelFatal,
	}

	for _, level := range tests {
		err := SetLogLevel(level)
		if err != nil {
			t.Errorf("unexpected error setting log level: %v", err)
		}

		mu.Lock()
		if logLevel != level {
			t.Errorf("expected logLevel to be %d, got %d", level, logLevel)
		}
		mu.Unlock()
	}

	// Test that changing level after init returns an error
	err := SetLogLevel(LoggingLevelWarning)
	if err != nil {
		t.Errorf("unexpected error setting log level: %v", err)
	}
	mu.Lock()
	refCount = 1
	mu.Unlock()

	err = SetLogLevel(LoggingLevelError)
	if err == nil {
		t.Error("expected error when setting log level after initialization")
	}

	mu.Lock()
	if logLevel != LoggingLevelWarning {
		t.Errorf("expected logLevel to remain Warning after init, got %d", logLevel)
	}
	mu.Unlock()

	resetEnvironmentState()
}

func TestGetVersionStringWhenNotInitialized(t *testing.T) {
	resetEnvironmentState()

	version := GetVersionString()
	if version != "0.0.0-dev" {
		t.Errorf("expected version to be '0.0.0-dev' when not initialized, got %q", version)
	}

	resetEnvironmentState()
}

func TestInitializeEnvironmentWithoutLibraryPath(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	handler := &diagnosticCountingHandler{}
	SetDiagnosticHandler(handler)
	t.Cleanup(func() { SetDiagnosticHandler(nil) })

	err := InitializeEnvironment()
	if err == nil {
		t.Error("expected error when library path not set")
	}

	if !errors.Is(err, ErrNotInitialized) {
		t.Fatalf("initialization error = %v, want ErrNotInitialized", err)
	}
	if !strings.Contains(err.Error(), "SetSharedLibraryPath") {
		t.Fatalf("initialization error = %v, want SetSharedLibraryPath guidance", err)
	}
	if got := handler.count.Load(); got != 0 {
		t.Fatalf("returned initialization error emitted %d diagnostics, want 0", got)
	}
}

func TestReferenceCountingLogic(t *testing.T) {
	resetEnvironmentState()

	// Simulate initialized state
	mu.Lock()
	refCount = 1
	mu.Unlock()

	// First init increments
	err := InitializeEnvironment()
	if err != nil {
		t.Errorf("unexpected error on second init: %v", err)
	}

	mu.Lock()
	if refCount != 2 {
		t.Errorf("expected refCount to be 2, got %d", refCount)
	}
	mu.Unlock()

	// Third init increments again
	err = InitializeEnvironment()
	if err != nil {
		t.Errorf("unexpected error on third init: %v", err)
	}

	mu.Lock()
	if refCount != 3 {
		t.Errorf("expected refCount to be 3, got %d", refCount)
	}
	mu.Unlock()

	resetEnvironmentState()
}

func TestDestroyEnvironmentWhenNotInitialized(t *testing.T) {
	resetEnvironmentState()

	err := DestroyEnvironment()
	if err != nil {
		t.Errorf("unexpected error when destroying non-initialized environment: %v", err)
	}

	resetEnvironmentState()
}

func TestDestroyEnvironmentDecrements(t *testing.T) {
	resetEnvironmentState()

	// Simulate initialized state with refCount=3
	mu.Lock()
	refCount = 3
	mu.Unlock()

	// First destroy
	err := DestroyEnvironment()
	if err != nil {
		t.Errorf("unexpected error on destroy: %v", err)
	}

	mu.Lock()
	if refCount != 2 {
		t.Errorf("expected refCount to be 2, got %d", refCount)
	}
	mu.Unlock()

	// Second destroy
	err = DestroyEnvironment()
	if err != nil {
		t.Errorf("unexpected error on destroy: %v", err)
	}

	mu.Lock()
	if refCount != 1 {
		t.Errorf("expected refCount to be 1, got %d", refCount)
	}
	mu.Unlock()

	resetEnvironmentState()
}

func TestConcurrentInitialization(t *testing.T) {
	resetEnvironmentState()

	// Set a dummy library path
	if err := SetSharedLibraryPath("/nonexistent/path.so"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	var wg sync.WaitGroup
	concurrency := 10

	// Simulate initialized state first
	mu.Lock()
	refCount = 1
	mu.Unlock()

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = InitializeEnvironment()
		}()
	}

	wg.Wait()

	mu.Lock()
	expectedCount := 1 + concurrency
	if refCount != expectedCount {
		t.Errorf("expected refCount to be %d after concurrent inits, got %d", expectedCount, refCount)
	}
	mu.Unlock()

	resetEnvironmentState()
}

func TestConcurrentDestroy(t *testing.T) {
	resetEnvironmentState()

	concurrency := 10

	// Set initial refCount
	mu.Lock()
	refCount = concurrency
	mu.Unlock()

	var wg sync.WaitGroup

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = DestroyEnvironment()
		}()
	}

	wg.Wait()

	mu.Lock()
	if refCount != 0 {
		t.Errorf("expected refCount to be 0 after concurrent destroys, got %d", refCount)
	}
	mu.Unlock()

	resetEnvironmentState()
}

// TestInitializeWithActualLibrary tests with a real ONNX Runtime library if available
func TestInitializeWithActualLibrary(t *testing.T) {
	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		t.Skip("Skipping integration test: ONNXRUNTIME_LIB_PATH not set")
	}

	resetEnvironmentState()

	if err := SetSharedLibraryPath(libPath); err != nil {
		t.Fatalf("failed to set library path: %v", err)
	}

	err := InitializeEnvironment()
	if err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}

	if !IsInitialized() {
		t.Error("expected environment to be initialized")
	}

	// Test version string
	version := GetVersionString()
	if version == "0.0.0-dev" || version == "" {
		t.Errorf("expected valid version string, got %q", version)
	}
	t.Logf("ONNX Runtime version: %s", version)

	// Test double initialization (should increment ref count)
	err = InitializeEnvironment()
	if err != nil {
		t.Errorf("failed second initialization: %v", err)
	}

	// First destroy (should decrement)
	err = DestroyEnvironment()
	if err != nil {
		t.Errorf("failed first destroy: %v", err)
	}

	// Should still be initialized
	if !IsInitialized() {
		t.Error("expected environment to still be initialized after first destroy")
	}

	// Second destroy (should actually destroy)
	err = DestroyEnvironment()
	if err != nil {
		t.Errorf("failed second destroy: %v", err)
	}

	// Now should be uninitialized
	if IsInitialized() {
		t.Error("expected environment to be uninitialized after final destroy")
	}

	resetEnvironmentState()
}

func TestErrorMessageIntegrationWithFailedInit(t *testing.T) {
	// Test that error messages are properly extracted during failed initialization
	resetEnvironmentState()

	if err := SetSharedLibraryPath("/nonexistent/path/libonnxruntime.so"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	err := InitializeEnvironment()
	if err == nil {
		t.Fatal("expected error when loading non-existent library")
	}

	// Verify error message contains helpful information
	errMsg := err.Error()
	if !strings.Contains(errMsg, "failed to load ONNX Runtime library") {
		t.Errorf("expected error message to mention library loading failure, got: %v", errMsg)
	}

	// Verify the error provides context about what went wrong
	if errMsg == "" {
		t.Error("expected non-empty error message")
	}

	resetEnvironmentState()
}

func TestErrorMessageFormattingQuality(t *testing.T) {
	// Test that error messages follow good practices
	resetEnvironmentState()

	testCases := []struct {
		name         string
		setup        func() error
		shouldError  bool
		errorPattern string
	}{
		{
			name: "missing library path",
			setup: func() error {
				return InitializeEnvironment()
			},
			shouldError:  true,
			errorPattern: "library path not set",
		},
		{
			name: "cannot change path after init",
			setup: func() error {
				mu.Lock()
				refCount = 1
				mu.Unlock()
				return SetSharedLibraryPath("/new/path.so")
			},
			shouldError:  true,
			errorPattern: "cannot change library path after environment is initialized",
		},
		{
			name: "cannot change log level after init",
			setup: func() error {
				mu.Lock()
				refCount = 1
				mu.Unlock()
				return SetLogLevel(LoggingLevelError)
			},
			shouldError:  true,
			errorPattern: "cannot change log level after environment is initialized",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resetEnvironmentState()

			err := tc.setup()

			if tc.shouldError {
				if err == nil {
					t.Errorf("expected error but got nil")
					return
				}

				errMsg := err.Error()

				// Check that error message matches expected pattern
				if !strings.Contains(errMsg, tc.errorPattern) {
					t.Errorf("expected error message to contain %q, got: %v", tc.errorPattern, errMsg)
				}

				// Check error message quality
				if len(errMsg) < 10 {
					t.Errorf("error message too short (< 10 chars): %q", errMsg)
				}

				// Error messages should start with lowercase (Go convention for wrapped errors)
				// or be a complete sentence
				if errMsg[0] >= 'A' && errMsg[0] <= 'Z' {
					// Capital letter is OK if it's a proper noun or acronym
					if !strings.HasPrefix(errMsg, "ONNX") && !strings.HasPrefix(errMsg, "ORT") {
						// This is fine, just noting that it starts with capital
					}
				}
			} else {
				if err != nil {
					t.Errorf("expected no error but got: %v", err)
				}
			}

			resetEnvironmentState()
		})
	}
}

// Error path tests with real failure conditions

func TestInitializeWithNonExistentLibrary(t *testing.T) {
	resetEnvironmentState()

	if err := SetSharedLibraryPath("/nonexistent/path/libonnxruntime.so"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	err := InitializeEnvironment()
	if err == nil {
		t.Error("expected error when loading non-existent library")
	}
	if err != nil && !strings.Contains(err.Error(), "failed to load ONNX Runtime library") {
		t.Errorf("expected load error, got: %v", err)
	}

	resetEnvironmentState()
}

func TestInitializeWithInvalidLibrary(t *testing.T) {
	resetEnvironmentState()

	// Use the test binary itself as an invalid library
	// It exists as a file but doesn't have the ONNX Runtime symbols
	if err := SetSharedLibraryPath("/bin/sh"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	err := InitializeEnvironment()
	if err == nil {
		t.Error("expected error when loading invalid library")
		_ = DestroyEnvironment() // Clean up if it somehow succeeded
	}

	resetEnvironmentState()
}

func TestMultipleInitializeAfterDestroy(t *testing.T) {
	resetEnvironmentState()

	// Set library path
	if err := SetSharedLibraryPath("/nonexistent/path.so"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	// Simulate a successful initialization
	mu.Lock()
	refCount = 1
	mu.Unlock()

	// Destroy
	err := DestroyEnvironment()
	if err != nil {
		t.Errorf("unexpected error on destroy: %v", err)
	}

	// Should be able to set library path again after destroy
	if err := SetSharedLibraryPath("/different/path.so"); err != nil {
		t.Errorf("expected to be able to change library path after destroy, got error: %v", err)
	}

	mu.Lock()
	if libPath != "/different/path.so" {
		t.Errorf("expected libPath to be updated after destroy, got %q", libPath)
	}
	mu.Unlock()

	resetEnvironmentState()
}

// Benchmarks

func BenchmarkInitializeEnvironment(b *testing.B) {
	// Benchmark the reference counting path (already initialized)
	// This is the fast path that most applications will hit
	resetEnvironmentState()

	// Simulate already initialized state
	mu.Lock()
	refCount = 1
	mu.Unlock()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = InitializeEnvironment()
	}
	b.StopTimer()

	resetEnvironmentState()
}

func BenchmarkDestroyEnvironment(b *testing.B) {
	// Benchmark the reference counting path (decrement without actual cleanup)
	// This is the fast path when refCount > 1
	resetEnvironmentState()

	// Set high refCount so we never reach zero
	mu.Lock()
	refCount = b.N + 1
	mu.Unlock()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DestroyEnvironment()
	}
	b.StopTimer()

	resetEnvironmentState()
}

func BenchmarkInitializeDestroyPair(b *testing.B) {
	// Benchmark a complete init/destroy pair
	// This measures the full lifecycle with reference counting
	resetEnvironmentState()

	// Start with refCount=1 to avoid actual library operations
	mu.Lock()
	refCount = 1
	mu.Unlock()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = InitializeEnvironment() // Increments to 2
		_ = DestroyEnvironment()    // Decrements back to 1
	}
	b.StopTimer()

	resetEnvironmentState()
}

func BenchmarkSetSharedLibraryPath(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		resetEnvironmentState()
		b.StartTimer()

		_ = SetSharedLibraryPath("/path/to/library.so")
	}
}

func BenchmarkIsInitialized(b *testing.B) {
	resetEnvironmentState()

	// Test both initialized and uninitialized states
	b.Run("uninitialized", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = IsInitialized()
		}
	})

	b.Run("initialized", func(b *testing.B) {
		mu.Lock()
		refCount = 1
		mu.Unlock()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = IsInitialized()
		}
	})

	resetEnvironmentState()
}

func BenchmarkGetVersionString(b *testing.B) {
	resetEnvironmentState()

	// Test uninitialized path (fast path - no C call)
	b.Run("uninitialized", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = GetVersionString()
		}
	})

	// Note: We can't easily benchmark the initialized path without a real library
	// That would require integration testing with actual ONNX Runtime

	resetEnvironmentState()
}
