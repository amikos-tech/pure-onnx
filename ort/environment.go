package ort

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

const (
	// defaultLogID is the default log identifier used when creating the ONNX Runtime environment
	defaultLogID = "onnx-purego"
)

var (
	mu                   sync.Mutex
	refCount             int
	ortLib               uintptr
	ortAPI               *OrtApi
	ortEnv               uintptr
	libPath              string
	logLevel             LoggingLevel = LoggingLevelWarning // Default to Warning
	getVersionStringFunc func() uintptr
	getErrorMessageFunc  func(uintptr) uintptr
	releaseStatusFunc    func(uintptr)
)

// getErrorMessage extracts the error message from an ORT status code.
// Returns empty string if status is 0 (success) or if the function is not initialized.
func getErrorMessage(status uintptr) string {
	if status == 0 || getErrorMessageFunc == nil {
		return ""
	}

	msgPtr := getErrorMessageFunc(status)
	return CstringToGo(msgPtr)
}

// releaseStatus releases an ORT status object to prevent memory leaks.
func releaseStatus(status uintptr) {
	if status == 0 || releaseStatusFunc == nil {
		return
	}

	releaseStatusFunc(status)
}

// InitializeEnvironment initializes the ONNX Runtime environment
func InitializeEnvironment() error {
	mu.Lock()
	defer mu.Unlock()

	if refCount > 0 {
		refCount++
		return nil
	}

	if libPath == "" {
		return fmt.Errorf("library path not set, call SetSharedLibraryPath first")
	}

	// Setup centralized cleanup for error paths
	var cleanupNeeded = true
	defer func() {
		if cleanupNeeded {
			if ortLib != 0 {
				_ = closeLibrary(ortLib)
				ortLib = 0
			}
			ortAPI = nil
			getVersionStringFunc = nil
			getErrorMessageFunc = nil
			releaseStatusFunc = nil
		}
	}()

	var err error
	ortLib, err = loadLibrary(libPath)
	if err != nil {
		return fmt.Errorf("failed to load ONNX Runtime library: %w", err)
	}

	sym, err := getSymbol(ortLib, "OrtGetApiBase")
	if err != nil {
		return fmt.Errorf("failed to get OrtGetApiBase symbol: %w", err)
	}

	var ortGetApiBase func() *OrtApiBase
	purego.RegisterFunc(&ortGetApiBase, sym)
	apiBase := ortGetApiBase()

	purego.RegisterFunc(&getVersionStringFunc, apiBase.GetVersionString)

	var getApi func(uint32) uintptr
	purego.RegisterFunc(&getApi, apiBase.GetApi)
	apiPtr := getApi(ORT_API_VERSION)
	// #nosec G103 -- This unsafe conversion is required for purego FFI.
	// The OrtApi struct layout exactly matches the C API struct returned by GetApi.
	// This pattern is the standard way to use purego for calling C libraries without CGO.
	ortAPI = (*OrtApi)(unsafe.Pointer(apiPtr))

	// Register frequently-used API functions once to avoid repeated RegisterFunc calls
	purego.RegisterFunc(&getErrorMessageFunc, ortAPI.GetErrorMessage)
	purego.RegisterFunc(&releaseStatusFunc, ortAPI.ReleaseStatus)

	var createEnv func(logLevel int32, logID uintptr, out *uintptr) uintptr
	purego.RegisterFunc(&createEnv, ortAPI.CreateEnv)

	logIDBytes, logIDPtr := GoToCstring(defaultLogID)
	// #nosec G115 -- LoggingLevel values are constrained to 0-4 by type definition, no overflow possible
	status := createEnv(int32(logLevel), logIDPtr, &ortEnv)
	runtime.KeepAlive(logIDBytes) // Prevent GC from collecting bytes during C call
	if status != 0 {
		errMsg := getErrorMessage(status)
		releaseStatus(status)
		return fmt.Errorf("failed to create ONNX Runtime environment: %s", errMsg)
	}

	// Success - prevent cleanup
	cleanupNeeded = false
	refCount = 1
	return nil
}

// DestroyEnvironment cleans up the ONNX Runtime environment
func DestroyEnvironment() error {
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
		// TODO(memory-leak): ReleaseEnv currently disabled due to OrtApi struct layout mismatch.
		// The OrtApi struct definition in types.go is incomplete - it only defines a subset
		// of the ~200+ functions in the full ONNX Runtime C API. This causes incorrect offsets
		// when accessing function pointers beyond the first few functions.
		//
		// To fix this properly, we need to either:
		// 1. Define ALL functions in the C API struct in correct order (tedious but complete), OR
		// 2. Use individual GetApi() calls for each function we need (cleaner but more calls)
		//
		// For now, the OS will clean up the environment on process exit. This is acceptable
		// for short-lived processes but should be fixed for long-running applications.
		// See issue: https://github.com/amikos-tech/pure-onnx/issues/20
		ortEnv = 0
	}

	if ortLib != 0 {
		if err := closeLibrary(ortLib); err != nil {
			return fmt.Errorf("failed to close ONNX Runtime library: %w", err)
		}
		ortLib = 0
	}

	ortAPI = nil
	getVersionStringFunc = nil
	getErrorMessageFunc = nil
	releaseStatusFunc = nil

	return nil
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
// It uses a mutex to protect access to the version string function pointer, ensuring
// that the pointer remains valid throughout the call even if another goroutine calls
// DestroyEnvironment() concurrently. The mutex guarantees that either:
// 1. GetVersionString() completes before DestroyEnvironment() starts, or
// 2. DestroyEnvironment() completes before GetVersionString() starts
func GetVersionString() string {
	mu.Lock()
	defer mu.Unlock()

	if refCount == 0 || getVersionStringFunc == nil {
		return "0.0.0-dev"
	}

	versionPtr := getVersionStringFunc()
	return CstringToGo(versionPtr)
}
