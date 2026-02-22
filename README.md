# ONNX-PureGo

Pure-Go bindings for Microsoft ONNX Runtime using [purego](https://github.com/ebitengine/purego) - no CGO required!

## Overview

This library provides Go bindings for [ONNX Runtime](https://onnxruntime.ai/) without requiring CGO, making it easier to build and deploy Go applications that use ONNX models. It uses `purego` to dynamically load and call the ONNX Runtime C API directly.

## Features

- ✅ Pure Go implementation - no CGO required
- ✅ Cross-platform support (Linux, macOS, Windows)
- ✅ Type-safe tensor operations with generics
- ✅ Simple API similar to existing Go ONNX libraries
- 🚧 Environment and session management (in progress)
- 🚧 Tensor operations (in progress)
- 🚧 Model inference (in progress)

## Installation

```bash
go get github.com/amikos-tech/pure-onnx
```

## Runtime Setup

You can initialize ONNX Runtime in two modes.

### 1. Explicit library path (existing behavior)

```go
import "github.com/amikos-tech/pure-onnx/ort"

func main() {
    // Set the path to your ONNX Runtime library
    if err := ort.SetSharedLibraryPath("/path/to/libonnxruntime.so"); err != nil { // Linux
        log.Fatal(err)
    }
    // ort.SetSharedLibraryPath("/path/to/libonnxruntime.dylib") // macOS
    // ort.SetSharedLibraryPath("/path/to/onnxruntime.dll") // Windows

    // Initialize the environment
    err := ort.InitializeEnvironment()
    if err != nil {
        log.Fatal(err)
    }
    defer ort.DestroyEnvironment()

    // Your code here...
}
```

### 2. Bootstrap mode (pure-Go auto-download + cache)

```go
import "github.com/amikos-tech/pure-onnx/ort"

func main() {
    if err := ort.InitializeEnvironmentWithBootstrap(); err != nil {
        log.Fatal(err)
    }
    defer ort.DestroyEnvironment()
}
```

Bootstrap downloads official Microsoft ONNX Runtime artifacts and caches them locally.

Optional bootstrap environment variables:
- `ONNXRUNTIME_VERSION` (default: `1.23.1`)
- `ONNXRUNTIME_CACHE_DIR` (default: user cache dir under `onnx-purego/onnxruntime`)
- `ONNXRUNTIME_DISABLE_DOWNLOAD=1` (fail if library is not already cached)
- `ONNXRUNTIME_LIB_PATH` (if set, explicit path mode is used)

## Usage Example

```go
package main

import (
    "fmt"
    "log"
    "github.com/amikos-tech/pure-onnx/ort"
)

func main() {
    if err := ort.SetSharedLibraryPath("/path/to/libonnxruntime.so"); err != nil {
        log.Fatal(err)
    }
    if err := ort.InitializeEnvironment(); err != nil {
        log.Fatal(err)
    }
    defer ort.DestroyEnvironment()

    fmt.Println("ONNX Runtime version:", ort.GetVersionString())
}
```

### End-to-end Inference Example

A runnable inference example lives at:

- `examples/inference/main.go`
- `examples/inference/README.md`

Run it with:

```bash
go run ./examples/inference
```

### Optional Dense Embeddings Layer (`embeddings/minilm`)

For local dense embedding workflows, use:
`github.com/amikos-tech/pure-onnx/embeddings/minilm`.

It adds:
- tokenizer loading (`tokenizer.json`)
- truncation/padding to `256` (configurable)
- ONNX multi-input assembly (`input_ids`, `attention_mask`, optional `token_type_ids`)
- configurable post-processing:
  - `WithMeanPooling()` (default)
  - `WithCLSPooling()`
  - `WithNoPooling()`
  - `WithL2Normalization()` / `WithoutL2Normalization()`
- configurable embedding width via `WithEmbeddingDimension(...)`
- LRU-bounded per-batch session cache (default `8`, override with `WithMaxCachedBatchSessions`)

```go
package main

import (
    "log"

    "github.com/amikos-tech/pure-onnx/embeddings/minilm"
    "github.com/amikos-tech/pure-onnx/ort"
)

func main() {
    if err := ort.SetSharedLibraryPath("/path/to/libonnxruntime.so"); err != nil {
        log.Fatal(err)
    }
    if err := ort.InitializeEnvironment(); err != nil {
        log.Fatal(err)
    }
    defer ort.DestroyEnvironment()

    embedder, err := minilm.NewEmbedder(
        "/path/to/all-MiniLM-L6-v2.onnx",
        "/path/to/tokenizer.json",
        minilm.WithMeanPooling(),
        minilm.WithL2Normalization(),
    )
    if err != nil {
        log.Fatal(err)
    }
    defer embedder.Close()

    vectors, err := embedder.EmbedDocuments([]string{"hello world", "local inference only"})
    if err != nil {
        log.Fatal(err)
    }

    _ = vectors // [][]float32
}
```

### Optional Sparse Embeddings Layer (`embeddings/splade`)

For sparse embedding workflows (e.g. SPLADE-like models), use:
`github.com/amikos-tech/pure-onnx/embeddings/splade`.

Current defaults are aligned with `prithivida/Splade_PP_en_v1`
(`input_ids`, `input_mask`, `segment_ids`, output `output`).
For other ONNX exports, override names with `splade.WithInputOutputNames(...)`.

```go
package main

import (
    "log"

    "github.com/amikos-tech/pure-onnx/embeddings/splade"
    "github.com/amikos-tech/pure-onnx/ort"
)

func main() {
    if err := ort.SetSharedLibraryPath("/path/to/libonnxruntime.so"); err != nil {
        log.Fatal(err)
    }
    if err := ort.InitializeEnvironment(); err != nil {
        log.Fatal(err)
    }
    defer ort.DestroyEnvironment()

    embedder, err := splade.NewEmbedder(
        "/path/to/splade-model.onnx",
        "/path/to/tokenizer.json",
        splade.WithTopK(128),
        splade.WithPruneThreshold(0.0),
        splade.WithPreProcessor(func(input string) string {
            // Optional: normalize/strip noisy structured content before tokenization.
            return input
        }),
        splade.WithReturnLabels(), // Optional: decode token labels for each index
    )
    if err != nil {
        log.Fatal(err)
    }
    defer embedder.Close()

    sparseVectors, err := embedder.EmbedDocuments([]string{"hello world"})
    if err != nil {
        log.Fatal(err)
    }

    _ = sparseVectors // []splade.SparseVector{Indices, Values, Labels}
}
```

## Project Status

This project is under active development. See our [GitHub Issues](https://github.com/amikos-tech/pure-onnx/issues) for the development roadmap.

### Current Focus

We're focusing on providing a drop-in replacement for common ONNX Runtime use cases, particularly for embeddings and inference tasks.

## Local CI Guardrails

Install repository-managed git hooks once per clone:

```bash
make install-hooks
```

The pre-commit hook runs:
- `make fmt-check`
- `make vet`
- `make precommit-lint-new` (golangci-lint new issues vs `PRECOMMIT_BASE_REF`, default `origin/main`)
- `make gosec`
- `go test ./...`
- `make check-mod-tidy`
- `make vulncheck` (with patched Go toolchain baseline `go1.24.13+auto`)

Optional skip knobs:
- `SKIP_LINT_NEW=1` (skip new-issues lint check)
- `SKIP_GOSEC=1` (skip gosec)
- `SKIP_VULNCHECK=1` (skip govulncheck)

You can run the same sequence manually:

```bash
make precommit
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## References

- [ONNX Runtime C API](https://onnxruntime.ai/docs/get-started/with-c.html)
- [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
- [purego](https://github.com/ebitengine/purego)
