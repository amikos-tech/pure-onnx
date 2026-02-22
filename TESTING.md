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
   export ONNXRUNTIME_LIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.1.22.0.dylib

   # Windows (PowerShell)
   $env:ONNXRUNTIME_LIB_PATH="C:\path\to\onnxruntime\lib\onnxruntime.dll"
   ```

4. Optional: configure all-MiniLM integration test model settings:
   ```bash
   # Use a pre-downloaded all-MiniLM model (skips network download)
   export ONNXRUNTIME_TEST_ALL_MINILM_MODEL_PATH=/path/to/all-MiniLM-L6-v2.onnx

   # Override sequence length for ./ort real-model tests only
   # (CI uses 8; minimum is 6 in ort/minilm helper tests)
   export ONNXRUNTIME_TEST_ALL_MINILM_SEQUENCE_LENGTH=8

   # Optional integrity check for custom model path/URL.
   # When unset, the default HuggingFace URL is verified against a built-in SHA-256.
   export ONNXRUNTIME_TEST_ALL_MINILM_MODEL_SHA256=<expected_sha256>
   ```

   Note: `embeddings/minilm` integration tests do not read
   `ONNXRUNTIME_TEST_ALL_MINILM_SEQUENCE_LENGTH`; they use the embedder default
   sequence length (`256`) unless `WithSequenceLength(...)` is set in code.

5. Optional: configure SPLADE integration test settings:
   ```bash
   # Optional local overrides (if unset, test defaults to:
   # prithivida/Splade_PP_en_v1@762be6a... with cached download)
   export ONNXRUNTIME_TEST_SPLADE_MODEL_PATH=/path/to/splade.onnx
   export ONNXRUNTIME_TEST_SPLADE_TOKENIZER_PATH=/path/to/tokenizer.json

   # Optional overrides:
   # export ONNXRUNTIME_TEST_SPLADE_MODEL_URL=https://...
   # export ONNXRUNTIME_TEST_SPLADE_MODEL_SHA256=<expected_sha256>
   # export ONNXRUNTIME_TEST_SPLADE_TOKENIZER_URL=https://...
   # export ONNXRUNTIME_TEST_SPLADE_TOKENIZER_SHA256=<expected_sha256>
   # export ONNXRUNTIME_TEST_SPLADE_OUTPUT_NAME=output
   # export ONNXRUNTIME_TEST_SPLADE_OUTPUT_LAYOUT=token_logits   # or document_logits
   # export ONNXRUNTIME_TEST_SPLADE_VOCAB_SIZE=30522
   # export ONNXRUNTIME_TEST_SPLADE_DISABLE_TOKEN_TYPE_IDS=1
   # export ONNXRUNTIME_TEST_SPLADE_INPUT_IDS_NAME=input_ids
   # export ONNXRUNTIME_TEST_SPLADE_ATTENTION_MASK_NAME=input_mask
   # export ONNXRUNTIME_TEST_SPLADE_TOKEN_TYPE_IDS_NAME=segment_ids
   ```

6. Run tests:
   ```bash
   go test -v ./ort/...
   go test -v ./embeddings/minilm
   go test -v ./embeddings/splade
   ```

#### Integration Test Coverage

When `ONNXRUNTIME_LIB_PATH` is set, the following additional tests run:
- `TestInitializeWithActualLibrary`: Tests actual library loading, environment creation, version retrieval, and proper cleanup
- `TestAdvancedSessionRunWithAllMiniLML6V2`: Downloads (or reuses cached) `all-MiniLM-L6-v2` ONNX model and runs end-to-end multi-input inference
- `TestEmbedDocumentsWithAllMiniLML6V2`: Runs dense embedding end-to-end through `embeddings/minilm`
- `TestEmbedDocumentsWithSPLADEModel`: Runs sparse embedding end-to-end through `embeddings/splade` when SPLADE env vars are set
- `TestSPLADEGoldenRegressionTopK16WithLabels`: Verifies pinned `prithivida/Splade_PP_en_v1` outputs against a golden sparse vector set (indices/values/labels)
- `TestSPLADERepeatabilityTopK16`: Re-runs SPLADE inference and asserts stable outputs across repeated calls
- Tests all FFI interactions including:
  - Dynamic library loading
  - Symbol resolution
  - ORT environment creation and destruction
  - Version string retrieval
  - Error message extraction
  - Reference counting with real library

The all-MiniLM model is cached under your user cache directory by default (`.../onnx-purego/models/all-MiniLM-L6-v2.onnx`).
Use `ONNXRUNTIME_TEST_MODEL_CACHE_DIR` to override cache location and
`ONNXRUNTIME_TEST_ALL_MINILM_MODEL_URL` to override the download URL.
For custom URLs, set `ONNXRUNTIME_TEST_ALL_MINILM_MODEL_SHA256` to enable checksum verification.

### 3. Benchmark Tests

Run performance benchmarks:
```bash
# Benchmark string conversion
go test -bench=. -benchmem ./ort/...

# Specific benchmarks
go test -bench=BenchmarkGoToCstring -benchmem ./ort/...
go test -bench=BenchmarkCstringToGo -benchmem ./ort/...
```

Run real-model all-MiniLM benchmarks (requires ONNX Runtime library path):
```bash
export ONNXRUNTIME_LIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so
export ONNXRUNTIME_TEST_ALL_MINILM_MODEL_PATH=/path/to/all-MiniLM-L6-v2.onnx

go test -run '^$' \
  -bench 'BenchmarkAdvancedSessionRunWarmWithAllMiniLML6V2|BenchmarkAdvancedSessionCreateRunDestroyWithAllMiniLML6V2' \
  -benchmem \
  ./ort/...
```

Run SPLADE benchmarks (requires ONNX Runtime library path):
```bash
export ONNXRUNTIME_LIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so

go test -run '^$' \
  -bench 'BenchmarkSPLADEEmbedDocumentsWarmTopK128|BenchmarkSPLADEEmbedDocumentsWarmTopK128WithLabels|BenchmarkSPLADECreateRunDestroyTopK128' \
  -benchmem \
  ./embeddings/splade
```

Run SPLADE golden-dataset parity check (optional):
```bash
export ONNXRUNTIME_LIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so
export HF_DATASET_REPO=tazarov/pure-onnx
# Optional for private/gated dataset access only:
# export HF_TOKEN=<hf_token_with_dataset_read_access>
# Optional direct override:
# export ONNXRUNTIME_TEST_SPLADE_GOLDEN_JSONL_URL=https://huggingface.co/datasets/tazarov/pure-onnx/resolve/main/splade_endpoint_golden/v1/splade_pp_en_v1_endpoint_topk24_labels_v1.jsonl
# Compatibility alias still supported:
# export ONNXRUNTIME_TEST_SPLADE_PRIVATE_GOLDEN_JSONL_URL=https://huggingface.co/datasets/tazarov/pure-onnx/resolve/main/splade_endpoint_golden/v1/splade_pp_en_v1_endpoint_topk24_labels_v1.jsonl
# Optional tolerance override (default 1e-4):
# export ONNXRUNTIME_TEST_SPLADE_GOLDEN_TOLERANCE=0.0001
# Compatibility alias still supported:
# export ONNXRUNTIME_TEST_SPLADE_PRIVATE_GOLDEN_TOLERANCE=0.0001

go test -v ./embeddings/splade -run TestSPLADEGoldenDatasetParity -count=1
```

Generate a SPLADE golden dataset locally (no hosted endpoint required):
```bash
# texts.txt: one input document per line
python3 ./tools/splade_generate_golden.py \
  --texts-file ./texts.txt \
  --output-jsonl ./splade_endpoint_golden/v1/splade_pp_en_v1_local_topk24_v1.jsonl \
  --metadata-path ./splade_endpoint_golden/v1/metadata.json \
  --model-name prithivida/Splade_PP_en_v1 \
  --top-k 24
```

## Continuous Integration

### GitHub Actions

The CI pipeline runs tests in multiple configurations:
- **Unit Tests**: Run on all platforms (Linux, macOS, Windows) with Go 1.24.x
- **Integration Tests (matrix job)**: Skipped in the cross-platform matrix (no ONNX Runtime library preinstalled)
- **Real-model Integration Job**: Linux job downloads ONNX Runtime, runs all-MiniLM integration + memory stability tests, runs SPLADE integration (`prithivida/Splade_PP_en_v1` defaults), and runs all-MiniLM benchmarks
- **Race Detection**: Partially disabled due to checkptr incompatibility with purego FFI
- **Vulnerability Check**: Runs `make vulncheck` with a patched Go baseline (`go1.24.13+auto`)

### Local Pre-commit Checks

Install repo-managed hooks once:

```bash
make install-hooks
```

Run the same checks on demand:

```bash
make precommit
```

`make precommit` runs:
- `make fmt-check`
- `make vet`
- `make precommit-lint-new` (golangci-lint only for issues introduced since merge-base with `PRECOMMIT_BASE_REF`, default `origin/main`)
- `make gosec` (blocking security scan, excludes `examples/experimental`)
- `go test ./...`
- `make check-mod-tidy`
- `make vulncheck`

Optional environment flags:
- `SKIP_LINT_NEW=1`
- `SKIP_GOSEC=1`
- `SKIP_VULNCHECK=1`

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

### Integration Memory Stability Test

```bash
export ONNXRUNTIME_LIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so
export ONNXRUNTIME_TEST_ALL_MINILM_MODEL_PATH=/path/to/all-MiniLM-L6-v2.onnx
export ONNXRUNTIME_TEST_LEAK_ITERATIONS=80
export ONNXRUNTIME_TEST_LEAK_MAX_GROWTH_MB=96

go test -v ./ort/... -run TestAdvancedSessionRunWithAllMiniLML6V2MemoryStability
```

The test executes repeated all-MiniLM inference create/run/destroy cycles and fails if post-GC heap growth exceeds the configured threshold.
It only measures Go heap growth and does not detect native allocator leaks inside ONNX Runtime.

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
