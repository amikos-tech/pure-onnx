# Testing Guide

This document provides guidance on running tests for the onnx-purego library.

## Quick Start

Run basic unit tests (no ONNX Runtime library required):
```bash
go test ./ort/...
```

## Test Categories

### 1. Unit Tests (No External Dependencies)

These tests run without requiring the ONNX Runtime library and cover:
- String conversion utilities (`CstringToGo`, `GoToCstring`)
- Reference counting logic
- Configuration functions (`SetSharedLibraryPath`, `SetLogLevel`)
- Concurrent access patterns
- Error handling paths

Run with:
```bash
go test -v ./ort/...
```

### 2. Integration Tests (Requires ONNX Runtime)

Integration tests verify actual FFI interactions with the ONNX Runtime library.

#### Setup

1. Download ONNX Runtime from [official releases](https://github.com/microsoft/onnxruntime/releases)

2. Extract the archive and note the library path:
   - **Linux**: `libonnxruntime.so` (in `lib/` directory)
   - **macOS**: `libonnxruntime.dylib` (in `lib/` directory)
   - **Windows**: `onnxruntime.dll` (in `lib/` directory)

3. Set the environment variable:
   ```bash
   # Linux/macOS
   export ONNXRUNTIME_LIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so

   # macOS specific example
   export ONNXRUNTIME_LIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.1.23.1.dylib

   # Windows (PowerShell)
   $env:ONNXRUNTIME_LIB_PATH="C:\path\to\onnxruntime\lib\onnxruntime.dll"
   ```

4. Run tests:
   ```bash
   go test -v ./ort/...
   ```

#### Integration Test Coverage

When `ONNXRUNTIME_LIB_PATH` is set, the following additional tests run:
- `TestInitializeWithActualLibrary`: Tests actual library loading, environment creation, version retrieval, and proper cleanup
- Tests all FFI interactions including:
  - Dynamic library loading
  - Symbol resolution
  - ORT environment creation and destruction
  - Version string retrieval
  - Error message extraction
  - Reference counting with real library

### 3. Benchmark Tests

Run performance benchmarks:
```bash
# Benchmark string conversion
go test -bench=. -benchmem ./ort/...

# Specific benchmarks
go test -bench=BenchmarkGoToCstring -benchmem ./ort/...
go test -bench=BenchmarkCstringToGo -benchmem ./ort/...
```

## Continuous Integration

### GitHub Actions

The CI pipeline runs tests in multiple configurations:
- **Unit Tests**: Run on all platforms (Linux, macOS, Windows) with Go 1.23.x and 1.24.x
- **Integration Tests**: Skipped in CI (no ONNX Runtime library available)
- **Race Detection**: Partially disabled due to checkptr incompatibility with purego FFI

### Local CI Simulation

To test all platforms locally using Docker:

```bash
# Linux
docker run --rm -v $(pwd):/work -w /work golang:1.24 go test ./ort/...

# With ONNX Runtime (mount library)
docker run --rm \
  -v $(pwd):/work \
  -v /path/to/onnxruntime:/ort \
  -e ONNXRUNTIME_LIB_PATH=/ort/lib/libonnxruntime.so \
  -w /work \
  golang:1.24 go test -v ./ort/...
```

## Troubleshooting

### "Skipping integration test: ONNXRUNTIME_LIB_PATH not set"

This is expected when running without the ONNX Runtime library. Unit tests still provide good coverage.

To run integration tests, download ONNX Runtime and set the environment variable as described above.

### Segmentation Faults

If you encounter segfaults during testing:
1. Verify you're using a compatible ONNX Runtime version (1.19+)
2. Ensure the library path points to the correct architecture (arm64, x86_64, etc.)
3. Check that library dependencies are satisfied (`ldd` on Linux, `otool -L` on macOS)

### Library Not Found Errors

**Linux**: Add library directory to `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

**macOS**: Add library directory to `DYLD_LIBRARY_PATH`:
```bash
export DYLD_LIBRARY_PATH=/path/to/onnxruntime/lib:$DYLD_LIBRARY_PATH
```

**Windows**: Add library directory to `PATH`:
```powershell
$env:PATH="C:\path\to\onnxruntime\lib;$env:PATH"
```

## Test Coverage

Generate coverage report:
```bash
go test -coverprofile=coverage.out ./ort/...
go tool cover -html=coverage.out -o coverage.html
```

View coverage summary:
```bash
go test -cover ./ort/...
```

## Race Detection

Race detection is partially disabled due to checkptr incompatibility with purego's FFI layer. However, concurrency tests still verify thread-safety:

```bash
# Run concurrency tests
go test -v -run Concurrent ./ort/...
```

## Memory Leak Detection

While ReleaseEnv is currently disabled (see [issue #20](https://github.com/amikos-tech/pure-onnx/issues/20)), you can check for other memory leaks:

### Using Valgrind (Linux)

```bash
go test -c ./ort
valgrind --leak-check=full ./ort.test -test.run=TestInitializeWithActualLibrary
```

### Using Address Sanitizer

```bash
CGO_ENABLED=1 go test -asan ./ort/...
```

Note: ASAN requires CGO, which this project avoids. This is primarily useful for checking test infrastructure.

## Writing New Tests

### Test Naming Convention

- `Test*`: Standard unit/integration tests
- `Test*WithActualLibrary`: Integration tests requiring ONNX Runtime (should check `ONNXRUNTIME_LIB_PATH`)
- `Benchmark*`: Performance benchmarks
- `Example*`: Runnable examples (also serve as documentation)

### Integration Test Template

```go
func TestMyFeatureWithActualLibrary(t *testing.T) {
    libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
    if libPath == "" {
        t.Skip("Skipping integration test: ONNXRUNTIME_LIB_PATH not set")
    }

    resetEnvironmentState() // If testing environment management

    if err := SetSharedLibraryPath(libPath); err != nil {
        t.Fatalf("failed to set library path: %v", err)
    }

    // Your test code here

    resetEnvironmentState() // Clean up
}
```

## References

- [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
- [Go Testing Package](https://pkg.go.dev/testing)
- [purego Documentation](https://pkg.go.dev/github.com/ebitengine/purego)
