package ort

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

const (
	// defaultLogID is the default log identifier used when creating the ONNX Runtime environment
	defaultLogID = "onnx-purego"
)

var (
	// Lock hierarchy across ORT lifecycle and calls:
	// 1) AdvancedSession.runMu (session-local serialization)
	// 2) ortCallMu (RLock for regular ORT calls; Lock for environment init/destroy
	//    and selected object releases that must not overlap in-flight ORT use)
	// 3) mu (global runtime pointers/function snapshots)
	// 4) Tensor.runMu (value-local run lease lock; only acquired while ortCallMu is held)
	//
	// Keep this order to avoid deadlocks.
	mu                                 sync.Mutex
	ortCallMu                          sync.RWMutex
	refCount                           int
	ortLib                             uintptr
	ortAPI                             *OrtApi
	ortEnv                             uintptr
	libPath                            string
	logLevel                           LoggingLevel = LoggingLevelWarning // Default to Warning
	getVersionStringFunc               func() uintptr
	getErrorCodeFunc                   func(uintptr) ErrorCode
	getErrorMessageFunc                func(uintptr) uintptr
	releaseStatusFunc                  func(uintptr)
	createMemoryInfoFunc               func(name uintptr, allocatorType AllocatorType, deviceID int32, memType MemType, out *uintptr) uintptr
	releaseMemoryInfoFunc              func(uintptr)
	createTensorWithDataAsOrtValueFunc func(info uintptr, pData uintptr, pDataLen uintptr, shape *int64, shapeLen uintptr, dataType TensorElementDataType, out *uintptr) uintptr
	releaseValueFunc                   func(uintptr)
	createSessionOptionsFunc           func(out *uintptr) uintptr
	releaseSessionOptionsFunc          func(uintptr)
	createSessionFunc                  func(env uintptr, modelPath uintptr, sessionOptions uintptr, out *uintptr) uintptr
	runSessionFunc                     func(session uintptr, runOptions uintptr, inputNames *uintptr, inputValues *uintptr, inputLen uintptr, outputNames *uintptr, outputLen uintptr, outputValues *uintptr) uintptr
	releaseSessionFunc                 func(uintptr)
	environmentLoadLibrary             = loadLibrary
	environmentGetSymbol               = getSymbol
	environmentCloseLibrary            = closeLibrary
)

func clearORTGlobalsLocked() {
	ortAPI = nil
	ortEnv = 0
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
}

func emitRuntimeVersionWarning(version string) {
	version = strings.TrimSpace(version)
	if version == "" {
		return
	}

	parts := strings.Split(version, ".")
	if len(parts) < 2 {
		emitDiagnostic(
			context.Background(),
			slog.LevelWarn,
			"Could not parse ONNX Runtime version",
			slog.String("runtime_version", version),
			slog.Int("api_version", int(ORT_API_VERSION)),
		)
		return
	}

	major, majorErr := strconv.Atoi(parts[0])
	minor, minorErr := strconv.Atoi(parts[1])
	if majorErr != nil || minorErr != nil || major < 0 || minor < 0 {
		emitDiagnostic(
			context.Background(),
			slog.LevelWarn,
			"Could not parse ONNX Runtime version",
			slog.String("runtime_version", version),
			slog.Int("api_version", int(ORT_API_VERSION)),
		)
		return
	}

	if major > 1 || (major == 1 && minor >= 22) {
		return
	}

	emitDiagnostic(
		context.Background(),
		slog.LevelWarn,
		"ONNX Runtime version is older than the supported runtime",
		slog.String("runtime_version", version),
		slog.Int("api_version", int(ORT_API_VERSION)),
	)
}

func createEnvironment(
	createEnv func(logLevel int32, logID uintptr, out *uintptr) uintptr,
	level LoggingLevel,
) (uintptr, error) {
	logIDBytes, logIDPtr := GoToCstring(defaultLogID)

	var environment uintptr
	// #nosec G115 -- LoggingLevel values are validated to the native 0-4 range.
	status := createEnv(int32(level), logIDPtr, &environment)
	runtime.KeepAlive(logIDBytes)
	if err := statusToError(status, "create ONNX Runtime environment"); err != nil {
		return 0, err
	}

	return environment, nil
}

// InitializeEnvironment initializes the ONNX Runtime environment.
func InitializeEnvironment() error {
	return initializeEnvironmentAt("")
}

// initializeEnvironmentAt initializes the runtime with path as one atomic
// lifecycle transition. An empty path keeps the value configured by
// SetSharedLibraryPath.
func initializeEnvironmentAt(path string) (err error) {
	runtimeVersion, newlyInitialized, err := initializeEnvironmentAtLocked(path)
	if runtimeVersion == "" {
		return err
	}

	if newlyInitialized {
		defer func() {
			if recovered := recover(); recovered != nil {
				_ = DestroyEnvironment()
				panic(recovered)
			}
		}()
	}
	emitRuntimeVersionWarning(runtimeVersion)
	return err
}

func initializeEnvironmentAtLocked(path string) (runtimeVersion string, newlyInitialized bool, err error) {
	ortCallMu.Lock()
	defer ortCallMu.Unlock()

	mu.Lock()
	defer mu.Unlock()

	if refCount > 0 {
		if path != "" && libPath != path {
			return "", false, fmt.Errorf(
				"cannot change library path after environment is initialized: configured %q, requested %q",
				libPath,
				path,
			)
		}
		refCount++
		return "", false, nil
	}

	if path != "" {
		libPath = path
	}
	if libPath == "" {
		return "", false, fmt.Errorf(
			"library path not set; call SetSharedLibraryPath or InitializeEnvironmentWithBootstrap: %w",
			ErrNotInitialized,
		)
	}

	// Setup centralized cleanup for error paths
	var cleanupNeeded = true
	defer func() {
		if cleanupNeeded {
			if ortLib != 0 {
				if closeErr := environmentCloseLibrary(ortLib); closeErr != nil {
					closeErr = fmt.Errorf("failed to close ONNX Runtime library during initialization cleanup: %w", closeErr)
					if err == nil {
						err = closeErr
					} else {
						err = errors.Join(err, closeErr)
					}
				}
				ortLib = 0
			}
			clearORTGlobalsLocked()
		}
	}()

	ortLib, err = environmentLoadLibrary(libPath)
	if err != nil {
		return "", false, fmt.Errorf("failed to load ONNX Runtime library: %w", err)
	}

	sym, err := environmentGetSymbol(ortLib, "OrtGetApiBase")
	if err != nil {
		return "", false, fmt.Errorf("failed to get OrtGetApiBase symbol: %w", err)
	}

	var ortGetApiBase func() *OrtApiBase
	purego.RegisterFunc(&ortGetApiBase, sym)
	apiBase := ortGetApiBase()
	if apiBase == nil {
		return "", false, fmt.Errorf("OrtGetApiBase returned nil: %w", ErrUnsupportedRuntime)
	}
	if apiBase.GetApi == 0 {
		return "", false, fmt.Errorf("OrtApiBase.GetApi is nil: %w", ErrUnsupportedRuntime)
	}

	purego.RegisterFunc(&getVersionStringFunc, apiBase.GetVersionString)

	var getApi func(uint32) uintptr
	purego.RegisterFunc(&getApi, apiBase.GetApi)
	apiPtr := getApi(ORT_API_VERSION)
	if apiPtr == 0 {
		return "", false, fmt.Errorf(
			"runtime does not support ONNX Runtime API version %d: %w",
			ORT_API_VERSION,
			ErrUnsupportedRuntime,
		)
	}
	// #nosec G103 -- This unsafe conversion is required for purego FFI.
	// The OrtApi struct layout exactly matches the C API struct returned by GetApi.
	// This pattern is the standard way to use purego for calling C libraries without CGO.
	ortAPI = (*OrtApi)(unsafe.Pointer(apiPtr))

	// Register frequently-used API functions once to avoid repeated RegisterFunc calls
	purego.RegisterFunc(&getErrorCodeFunc, ortAPI.GetErrorCode)
	purego.RegisterFunc(&getErrorMessageFunc, ortAPI.GetErrorMessage)
	purego.RegisterFunc(&releaseStatusFunc, ortAPI.ReleaseStatus)
	purego.RegisterFunc(&createMemoryInfoFunc, ortAPI.CreateMemoryInfo)
	purego.RegisterFunc(&releaseMemoryInfoFunc, ortAPI.ReleaseMemoryInfo)
	purego.RegisterFunc(&createTensorWithDataAsOrtValueFunc, ortAPI.CreateTensorWithDataAsOrtValue)
	purego.RegisterFunc(&releaseValueFunc, ortAPI.ReleaseValue)
	purego.RegisterFunc(&createSessionOptionsFunc, ortAPI.CreateSessionOptions)
	purego.RegisterFunc(&releaseSessionOptionsFunc, ortAPI.ReleaseSessionOptions)
	purego.RegisterFunc(&createSessionFunc, ortAPI.CreateSession)
	purego.RegisterFunc(&runSessionFunc, ortAPI.Run)
	purego.RegisterFunc(&releaseSessionFunc, ortAPI.ReleaseSession)

	// Validate ONNX Runtime version (warn if mismatch, unless explicitly skipped)
	if os.Getenv("ONNXRUNTIME_SKIP_VERSION_CHECK") == "" {
		versionPtr := getVersionStringFunc()
		runtimeVersion = CstringToGo(versionPtr)
	}

	var createEnv func(logLevel int32, logID uintptr, out *uintptr) uintptr
	purego.RegisterFunc(&createEnv, ortAPI.CreateEnv)

	ortEnv, err = createEnvironment(createEnv, logLevel)
	if err != nil {
		return runtimeVersion, false, err
	}

	// Success - prevent cleanup
	cleanupNeeded = false
	refCount = 1
	return runtimeVersion, true, nil
}

// DestroyEnvironment cleans up the ONNX Runtime environment
func DestroyEnvironment() error {
	ortCallMu.Lock()
	defer ortCallMu.Unlock()

	mu.Lock()
	defer mu.Unlock()

	if refCount == 0 {
		return nil
	}

	refCount--
	if refCount > 0 {
		return nil
	}

	if ortAPI != nil && ortEnv != 0 {
		// Now that we have the complete OrtApi struct layout (all 305 functions),
		// we can properly call ReleaseEnv
		var releaseEnv func(uintptr)
		purego.RegisterFunc(&releaseEnv, ortAPI.ReleaseEnv)
		releaseEnv(ortEnv)
		ortEnv = 0
	}

	var closeErr error
	if ortLib != 0 {
		if err := environmentCloseLibrary(ortLib); err != nil {
			closeErr = fmt.Errorf("failed to close ONNX Runtime library: %w", err)
		}
		// Clear the handle even when close fails to avoid reusing stale symbols.
		ortLib = 0
	}

	// Always clear function pointers/state after environment destruction. If
	// closeLibrary fails, stale pointers must still be removed.
	clearORTGlobalsLocked()
	return closeErr
}

// IsInitialized returns true if the environment is initialized
func IsInitialized() bool {
	mu.Lock()
	defer mu.Unlock()
	return refCount > 0
}

// SetSharedLibraryPath sets the path to the ONNX Runtime shared library.
// This must be called before InitializeEnvironment().
// Returns an error if the environment is already initialized.
func SetSharedLibraryPath(path string) error {
	if path == "" {
		return fmt.Errorf("set shared library path: %w", ErrInvalidArgument)
	}

	mu.Lock()
	defer mu.Unlock()
	if refCount > 0 {
		return fmt.Errorf("cannot change library path after environment is initialized")
	}
	libPath = path
	return nil
}

// SetLogLevel sets the logging level for the ONNX Runtime environment.
// This must be called before InitializeEnvironment() to take effect.
// Valid levels are: LoggingLevelVerbose, LoggingLevelInfo, LoggingLevelWarning, LoggingLevelError, LoggingLevelFatal.
// Default is LoggingLevelWarning.
// Returns an error if the environment is already initialized.
func SetLogLevel(level LoggingLevel) error {
	if level < LoggingLevelVerbose || level > LoggingLevelFatal {
		return fmt.Errorf("set log level %d: %w", level, ErrInvalidArgument)
	}

	mu.Lock()
	defer mu.Unlock()
	if refCount > 0 {
		return fmt.Errorf("cannot change log level after environment is initialized")
	}
	logLevel = level
	return nil
}

// GetVersionString returns the ONNX Runtime version string.
// Returns "0.0.0-dev" if the environment is not initialized.
//
// Thread-safety: This function is safe to call concurrently from multiple goroutines.
// It acquires ortCallMu.RLock to prevent concurrent environment teardown, snapshots
// the function pointer under mu, then calls it after releasing mu.
func GetVersionString() string {
	ortCallMu.RLock()
	defer ortCallMu.RUnlock()

	mu.Lock()
	if refCount == 0 || getVersionStringFunc == nil {
		mu.Unlock()
		return "0.0.0-dev"
	}
	versionStringFunc := getVersionStringFunc
	mu.Unlock()

	versionPtr := versionStringFunc()
	return CstringToGo(versionPtr)
}
