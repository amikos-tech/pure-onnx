# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a pure-Go implementation of Microsoft ONNX Runtime (ORT) bindings that uses `purego` instead of cgo to dynamically load and call the ONNX Runtime C API. The project aims to provide a Go-native interface to ONNX Runtime without requiring CGO compilation.

## 🚨 CRITICAL: NO CGO POLICY

**ABSOLUTELY NO CGO ALLOWED IN THE `ort/` PACKAGE!**

- ❌ **NEVER** use `import "C"`
- ❌ **NEVER** use CGO types (`*C.char`, `C.int`, etc.)
- ❌ **NEVER** use CGO functions (`C.CString`, `C.GoString`, `C.free`, etc.)
- ✅ **ALWAYS** use pure Go with `unsafe` package for C interop
- ✅ **ALWAYS** use `uintptr` for all C pointers
- ✅ **ALWAYS** use custom string conversion functions (see `ort/cstring.go`)

**Why?** The entire purpose of this project is to avoid CGO compilation. Using CGO defeats the core value proposition: no C compiler needed, cross-compilation support, faster builds, cleaner dependencies.

## Key Architecture

- **Dynamic Library Loading**: Uses `github.com/ebitengine/purego` to load the ONNX Runtime shared library (`.dylib`/`.so`/`.dll`) at runtime
- **C API Mapping**: Maps ONNX Runtime C API structures (`OrtApiBase`, `OrtApi`) directly to Go structs with uintptr fields for function pointers
- **Function Registration**: Uses `purego.RegisterFunc()` to bind C function pointers to Go function types
- **String Conversion**: Uses pure Go functions in `ort/cstring.go` to convert between Go strings and C null-terminated strings without CGO
- **Memory Management**: Manual byte slice management with careful GC pinning to prevent premature collection

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
- Requires the ONNX Runtime library to be installed locally (example path: `/Users/tazarov/Downloads/onnxruntime-osx-arm64-1.22.0/lib/libonnxruntime.1.22.0.dylib`)
- The `OrtApi` struct contains all ONNX Runtime API function pointers organized by API version (versions 1-22)
- Uses API version 22 (`ORT_API_VERSION := uint32(22)`)
- C header files (`onnxruntime_c_api.h`, `ort_apis.h`) are included for reference but not used in compilation

## Current Implementation Status

- Successfully loads ONNX Runtime library
- Can create and release ORT environments
- Can create and release ORT sessions
- Loads ONNX models from file paths
- Basic error handling through OrtStatus returns
- always use feature branches for all changes and create PRs for those