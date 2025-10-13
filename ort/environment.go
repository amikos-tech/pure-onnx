package ort

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

var (
	mu                   sync.Mutex
	refCount             int
	ortLib               uintptr
	ortAPI               *OrtApi
	ortEnv               uintptr
	libPath              string
	getVersionStringFunc func() uintptr
)

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

	var err error
	ortLib, err = loadLibrary(libPath)
	if err != nil {
		return fmt.Errorf("failed to load ONNX Runtime library: %w", err)
	}

	sym, err := getSymbol(ortLib, "OrtGetApiBase")
	if err != nil {
		_ = closeLibrary(ortLib)
		ortLib = 0
		return fmt.Errorf("failed to get OrtGetApiBase symbol: %w", err)
	}

	var ortGetApiBase func() *OrtApiBase
	purego.RegisterFunc(&ortGetApiBase, sym)
	apiBase := ortGetApiBase()

	purego.RegisterFunc(&getVersionStringFunc, apiBase.GetVersionString)

	var getApi func(uint32) uintptr
	purego.RegisterFunc(&getApi, apiBase.GetApi)
	apiPtr := getApi(ORT_API_VERSION)
	ortAPI = (*OrtApi)(unsafe.Pointer(apiPtr))

	var createEnv func(logLevel int32, logID uintptr, out *uintptr) uintptr
	purego.RegisterFunc(&createEnv, ortAPI.CreateEnv)

	logIDBytes, logIDPtr := GoToCstring("onnx-purego")
	status := createEnv(int32(LoggingLevelWarning), logIDPtr, &ortEnv)
	_ = logIDBytes // Keep bytes alive during C call
	if status != 0 {
		_ = closeLibrary(ortLib)
		ortLib = 0
		ortAPI = nil
		return fmt.Errorf("failed to create ONNX Runtime environment (status: %d)", status)
	}

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
		// TODO: Fix OrtApi struct layout to properly call ReleaseEnv
		// For now, skip calling ReleaseEnv as the struct offsets may not match the C API exactly
		// This will be addressed when we properly verify the OrtApi struct layout
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

	return nil
}

// IsInitialized returns true if the environment is initialized
func IsInitialized() bool {
	mu.Lock()
	defer mu.Unlock()
	return refCount > 0
}

// SetSharedLibraryPath sets the path to the ONNX Runtime shared library
func SetSharedLibraryPath(path string) {
	mu.Lock()
	defer mu.Unlock()
	libPath = path
}

// GetVersionString returns the ONNX Runtime version string
func GetVersionString() string {
	mu.Lock()
	defer mu.Unlock()

	if refCount == 0 || getVersionStringFunc == nil {
		return "0.0.0-dev"
	}

	versionPtr := getVersionStringFunc()
	return CstringToGo(versionPtr)
}
