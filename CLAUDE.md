# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a pure-Go implementation of Microsoft ONNX Runtime (ORT) bindings that uses `purego` instead of cgo to dynamically load and call the ONNX Runtime C API. The project aims to provide a Go-native interface to ONNX Runtime without requiring CGO compilation.

## Key Architecture

- **Dynamic Library Loading**: Uses `github.com/ebitengine/purego` to load the ONNX Runtime shared library (`.dylib`/`.so`/`.dll`) at runtime
- **C API Mapping**: Maps ONNX Runtime C API structures (`OrtApiBase`, `OrtApi`) directly to Go structs with uintptr fields for function pointers
- **Function Registration**: Uses `purego.RegisterFunc()` to bind C function pointers to Go function types
- **Memory Management**: Uses C.CString for string conversion and manual memory management with C.free

## Build and Development Commands

```bash
# Build the project
go build .

# Format code
go fmt ./...

# Run static analysis (note: will show unsafe.Pointer warnings which are expected)
go vet ./...

# Run the main executable
go run main2.go
```

## Important Technical Details

- The main working implementation is in `main2.go` (main.go is commented out/experimental)
- Requires the ONNX Runtime library to be installed locally (currently hardcoded path: `/Users/tazarov/Downloads/onnxruntime-osx-arm64-1.21.0/lib/libonnxruntime.1.21.0.dylib`)
- The `OrtApi` struct contains all ONNX Runtime API function pointers organized by API version (versions 1-21)
- Uses API version 21 (`ORT_API_VERSION := uint32(21)`)
- C header files (`onnxruntime_c_api.h`, `ort_apis.h`) are included for reference but not used in compilation

## Current Implementation Status

- Successfully loads ONNX Runtime library
- Can create and release ORT environments
- Can create and release ORT sessions
- Loads ONNX models from file paths
- Basic error handling through OrtStatus returns