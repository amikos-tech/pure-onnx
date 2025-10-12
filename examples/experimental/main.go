// Package main contains experimental code for testing ONNX Runtime C API bindings
// This code is preserved for reference during development
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"log"
	"unsafe"

	"github.com/ebitengine/purego"
)

// This file contains the experimental code from main2.go
// It demonstrates direct C API usage and will be refactored into the ort package

type OrtApiBase struct {
	GetApi           uintptr
	GetVersionString uintptr
}

type OrtApi struct {
	CreateEnv      uintptr
	ReleaseEnv     uintptr
	CreateSession  uintptr
	ReleaseSession uintptr
	// ... other fields omitted for brevity
}

type OrtEnv uintptr
type OrtStatus uintptr
type OrtSession uintptr

func main() {
	// Example of direct ONNX Runtime library usage
	// This will be refactored into the ort package

	// Load the ONNX Runtime shared library
	ort, err := purego.Dlopen("/Users/tazarov/Downloads/onnxruntime-osx-arm64-1.21.0/lib/libonnxruntime.1.21.0.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		panic(err)
	}
	defer purego.Dlclose(ort)

	// Get the OrtApiBase
	sym, err := purego.Dlsym(ort, "OrtGetApiBase")
	if err != nil {
		log.Fatal(err)
	}
	var OrtGetApiBase func() *OrtApiBase
	purego.RegisterFunc(&OrtGetApiBase, sym)

	apiBase := OrtGetApiBase()

	// Get version string
	var GetVersionString func() *C.char
	purego.RegisterFunc(&GetVersionString, apiBase.GetVersionString)
	version := C.GoString(GetVersionString())
	fmt.Println("ONNX Runtime version:", version)

	// Get the OrtApi
	var GetApi func(uint32) uintptr
	purego.RegisterFunc(&GetApi, apiBase.GetApi)
	ORT_API_VERSION := uint32(21)
	api := (*OrtApi)(unsafe.Pointer(GetApi(ORT_API_VERSION)))

	// Create environment
	var CreateEnv func(logLevel int32, logID *C.char, out **OrtEnv) OrtStatus
	purego.RegisterFunc(&CreateEnv, api.CreateEnv)

	var env *OrtEnv
	logID := C.CString("onnx_env")
	defer C.free(unsafe.Pointer(logID))

	status := CreateEnv(2, logID, &env)
	if status != 0 {
		fmt.Println("Error creating environment:", status)
		return
	}

	fmt.Println("Environment created successfully")

	// Release environment
	var ReleaseEnv func(env *OrtEnv)
	purego.RegisterFunc(&ReleaseEnv, api.ReleaseEnv)
	ReleaseEnv(env)
	fmt.Println("Environment released successfully")
}
