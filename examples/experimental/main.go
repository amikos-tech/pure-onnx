// Package main contains experimental code for testing ONNX Runtime C API bindings
// This code is preserved for reference during development
package main

import (
	"fmt"
	"log"
	"runtime"
	"unsafe"

	"github.com/amikos-tech/pure-onnx/ort"
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

	// Try to load the ONNX Runtime shared library from various paths
	var ortLibHandle uintptr
	var err error

	for _, path := range getTestLibraryPaths() {
		if isLibraryValid(path) {
			fmt.Printf("Attempting to load ONNX Runtime from: %s\n", path)
			ortLibHandle, err = loadLibrary(path)
			if err == nil {
				fmt.Printf("Successfully loaded ONNX Runtime from: %s\n", path)
				break
			}
			fmt.Printf("Failed to load from %s: %v\n", path, err)
		}
	}

	if ortLibHandle == 0 {
		log.Fatal("Could not load ONNX Runtime library from any of the attempted paths")
	}
	defer func() {
		if err := closeLibrary(ortLibHandle); err != nil {
			log.Printf("Failed to close library: %v", err)
		}
	}()

	// Get the OrtApiBase
	sym, err := getSymbol(ortLibHandle, "OrtGetApiBase")
	if err != nil {
		log.Fatal(err)
	}
	var OrtGetApiBase func() *OrtApiBase
	purego.RegisterFunc(&OrtGetApiBase, sym)

	apiBase := OrtGetApiBase()

	// Get version string
	var GetVersionString func() uintptr
	purego.RegisterFunc(&GetVersionString, apiBase.GetVersionString)
	versionPtr := GetVersionString()
	version := ort.CstringToGo(versionPtr)
	fmt.Println("ONNX Runtime version:", version)

	// Get the OrtApi
	var GetApi func(uint32) uintptr
	purego.RegisterFunc(&GetApi, apiBase.GetApi)
	ORT_API_VERSION := uint32(21)
	api := (*OrtApi)(unsafe.Pointer(GetApi(ORT_API_VERSION)))

	// Create environment
	var CreateEnv func(logLevel int32, logID uintptr, out **OrtEnv) OrtStatus
	purego.RegisterFunc(&CreateEnv, api.CreateEnv)

	var env *OrtEnv
	logIDBytes, logIDPtr := ort.GoToCstring("onnx_env")
	status := CreateEnv(2, logIDPtr, &env)
	runtime.KeepAlive(logIDBytes) // Prevent GC from collecting bytes during C call
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
