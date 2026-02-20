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

## Prerequisites

You need to have ONNX Runtime installed on your system. Download the appropriate version for your platform:

- [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)

Set the library path before running your application:

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

### Optional all-MiniLM Embeddings Layer

For local embedding workflows, use the optional high-level package:
`github.com/amikos-tech/pure-onnx/embeddings/minilm`.

It adds:
- tokenizer loading (`tokenizer.json`)
- truncation/padding to `256`
- ONNX multi-input assembly (`input_ids`, `attention_mask`, `token_type_ids`)
- mean pooling + L2 normalization

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
    )
    if err != nil {
        log.Fatal(err)
    }
    defer embedder.Close()

    vectors, err := embedder.EmbedDocuments([]string{"hello world", "local inference only"})
    if err != nil {
        log.Fatal(err)
    }

    _ = vectors // [][]float32, shape N x 384
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
- `go test ./...`
- `make check-mod-tidy`
- `make vulncheck` (with patched Go toolchain baseline `go1.24.13+auto`)

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
