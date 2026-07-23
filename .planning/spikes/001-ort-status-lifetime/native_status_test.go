//go:build !windows

package ortstatuslifetime

import (
	"errors"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"

	ort "github.com/amikos-tech/pure-onnx/ort"
)

func TestNativeORTStatusRoundTrip(t *testing.T) {
	if os.Getenv("ORT_SPIKE_NATIVE") == "" {
		t.Skip("set ORT_SPIKE_NATIVE=1 to bootstrap ONNX Runtime and exercise its real status API")
	}

	libraryPath, err := ort.EnsureOnnxRuntimeSharedLibrary()
	if err != nil {
		t.Fatalf("resolve ONNX Runtime library: %v", err)
	}

	library, err := purego.Dlopen(libraryPath, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		t.Fatalf("load ONNX Runtime library: %v", err)
	}
	defer func() {
		if closeErr := purego.Dlclose(library); closeErr != nil {
			t.Errorf("close ONNX Runtime library: %v", closeErr)
		}
	}()

	baseSymbol, err := purego.Dlsym(library, "OrtGetApiBase")
	if err != nil {
		t.Fatalf("resolve OrtGetApiBase: %v", err)
	}

	var getAPIBase func() *ort.OrtApiBase
	purego.RegisterFunc(&getAPIBase, baseSymbol)
	apiBase := getAPIBase()
	if apiBase == nil {
		t.Fatal("OrtGetApiBase returned nil")
	}

	var getAPI func(uint32) uintptr
	purego.RegisterFunc(&getAPI, apiBase.GetApi)
	apiPointer := getAPI(ort.ORT_API_VERSION)
	if apiPointer == 0 {
		t.Fatalf("GetApi(%d) returned nil", ort.ORT_API_VERSION)
	}
	api := (*ort.OrtApi)(unsafe.Pointer(apiPointer))

	var createStatus func(ort.ErrorCode, uintptr) uintptr
	var getErrorCode func(uintptr) ort.ErrorCode
	var getErrorMessage func(uintptr) uintptr
	var releaseStatus func(uintptr)
	purego.RegisterFunc(&createStatus, api.CreateStatus)
	purego.RegisterFunc(&getErrorCode, api.GetErrorCode)
	purego.RegisterFunc(&getErrorMessage, api.GetErrorMessage)
	purego.RegisterFunc(&releaseStatus, api.ReleaseStatus)

	messageBacking, messagePointer := ort.GoToCstring("native status survives release")
	status := createStatus(ort.ErrorCodeInvalidGraph, messagePointer)
	runtime.KeepAlive(messageBacking)
	if status == 0 {
		t.Fatal("CreateStatus returned nil")
	}

	err = statusToErrorPrototype(status, "load model", statusOps{
		getCode: getErrorCode,
		copyMessage: func(status uintptr) string {
			return ort.CstringToGo(getErrorMessage(status))
		},
		release: releaseStatus,
	})

	var got *statusError
	if !errors.As(err, &got) {
		t.Fatalf("errors.As did not find *statusError in %T", err)
	}
	if got.Code != ort.ErrorCodeInvalidGraph {
		t.Fatalf("native code mismatch: got %d", got.Code)
	}
	if got.Message != "native status survives release" {
		t.Fatalf("native message changed after ReleaseStatus: got %q", got.Message)
	}
	if got.Op != "load model" {
		t.Fatalf("operation mismatch: got %q", got.Op)
	}

	runtime.KeepAlive(api)
}
