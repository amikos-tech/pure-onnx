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
    ort.SetSharedLibraryPath("/path/to/libonnxruntime.so") // Linux
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
    // Initialize ONNX Runtime
    err := ort.InitializeEnvironment()
    if err != nil {
        log.Fatal(err)
    }
    defer ort.DestroyEnvironment()

    // Create a session (coming soon)
    // session, err := ort.NewSession("model.onnx", nil)
    // ...
}
```

## Project Status

This project is under active development. See our [GitHub Issues](https://github.com/amikos-tech/pure-onnx/issues) for the development roadmap.

### Current Focus

We're focusing on providing a drop-in replacement for common ONNX Runtime use cases, particularly for embeddings and inference tasks.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## References

- [ONNX Runtime C API](https://onnxruntime.ai/docs/get-started/with-c.html)
- [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
- [purego](https://github.com/ebitengine/purego)