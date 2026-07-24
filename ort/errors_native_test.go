//go:build !windows

package ort

import (
	"errors"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

func TestNativeORTStatusRoundTrip(t *testing.T) {
	libraryPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libraryPath == "" {
		t.Skip("set ONNXRUNTIME_LIB_PATH to an ONNX Runtime shared library to exercise the native status API")
	}

	resetEnvironmentState()

	library, err := purego.Dlopen(libraryPath, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		t.Fatalf("load ONNX Runtime library: %v", err)
	}
	t.Cleanup(func() {
		resetEnvironmentState()
		if closeErr := purego.Dlclose(library); closeErr != nil {
			t.Errorf("close ONNX Runtime library: %v", closeErr)
		}
	})

	baseSymbol, err := purego.Dlsym(library, "OrtGetApiBase")
	if err != nil {
		t.Fatalf("resolve OrtGetApiBase: %v", err)
	}

	var getAPIBase func() *OrtApiBase
	purego.RegisterFunc(&getAPIBase, baseSymbol)
	apiBase := getAPIBase()
	if apiBase == nil {
		t.Fatal("OrtGetApiBase returned nil")
	}

	var getAPI func(uint32) uintptr
	purego.RegisterFunc(&getAPI, apiBase.GetApi)
	apiPointer := getAPI(ORT_API_VERSION)
	if apiPointer == 0 {
		t.Fatalf("GetApi(%d) returned nil", ORT_API_VERSION)
	}
	// #nosec G103 -- purego returns the C API table as an integer address.
	api := (*OrtApi)(unsafe.Pointer(apiPointer))

	var createStatus func(ErrorCode, uintptr) uintptr
	var getErrorCode func(uintptr) ErrorCode
	var getErrorMessage func(uintptr) uintptr
	var releaseStatus func(uintptr)
	purego.RegisterFunc(&createStatus, api.CreateStatus)
	purego.RegisterFunc(&getErrorCode, api.GetErrorCode)
	purego.RegisterFunc(&getErrorMessage, api.GetErrorMessage)
	purego.RegisterFunc(&releaseStatus, api.ReleaseStatus)

	mu.Lock()
	getErrorCodeFunc = getErrorCode
	getErrorMessageFunc = getErrorMessage
	releaseStatusFunc = releaseStatus
	mu.Unlock()

	messageBacking, messagePointer := GoToCstring("native status survives release")
	status := createStatus(ErrorCodeInvalidGraph, messagePointer)
	runtime.KeepAlive(messageBacking)
	if status == 0 {
		t.Fatal("CreateStatus returned nil")
	}

	var converted error
	func() {
		ortCallMu.RLock()
		defer ortCallMu.RUnlock()
		converted = statusToError(status, "load model")
	}()

	var got *ORTError
	if !errors.As(converted, &got) {
		t.Fatalf("errors.As did not find *ORTError in %T", converted)
	}
	if got.Code != ErrorCodeInvalidGraph {
		t.Fatalf("native code mismatch: got %d", got.Code)
	}
	if got.Message != "native status survives release" {
		t.Fatalf("native message changed after ReleaseStatus: got %q", got.Message)
	}
	if got.Operation != "load model" {
		t.Fatalf("operation mismatch: got %q", got.Operation)
	}

	runtime.KeepAlive(api)
}
