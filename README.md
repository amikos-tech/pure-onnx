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
For official upstream downloads, bootstrap resolves the release asset SHA256 from GitHub release metadata and verifies the archive checksum before extraction.

Optional bootstrap environment variables:
- `ONNXRUNTIME_VERSION` (default: `1.23.1`)
- `ONNXRUNTIME_CACHE_DIR` (default: user cache dir under `onnx-purego/onnxruntime`)
- `ONNXRUNTIME_DISABLE_DOWNLOAD=1` (fail if library is not already cached)
- `ONNXRUNTIME_LIB_PATH` (if set, explicit path mode is used)
- `GITHUB_TOKEN` / `GH_TOKEN` (optional; helps avoid GitHub API rate limits during checksum metadata lookup)

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

### Optional OpenCLIP Embeddings Layer (`embeddings/openclip`)

For local CLIP text + image embeddings, use:
`github.com/amikos-tech/pure-onnx/embeddings/openclip`.

Expected artifacts from the OpenCLIP export tooling:
- `text_model.onnx`
- `vision_model.onnx`
- `tokenizer.json`
- `preprocessor_config.json`

Defaults are aligned with the pinned OpenCLIP export contract:
- text inputs: `input_ids`, `attention_mask`; output: `text_embeds`
- vision input: `pixel_values`; output: `image_embeds`
- sequence length `77`, image size `224`, embedding width `512`
- L2 normalization enabled by default (toggle with `WithoutL2Normalization()`)
- per-modality LRU session cache (default `8` per modality, configurable)

Built-in bootstrap can download and cache the default model bundle:
- repo: `amikos/openclip-vit-b-32-laion2b-s34b-b79k-onnx`
- revision: `248a2ed76a7189fc080e654e36930171331ef085`
- cache directory env var: `ONNXRUNTIME_OPENCLIP_CACHE_DIR` (defaults to user cache, e.g. `~/.cache/onnx-purego/openclip`)
- optional auth token env var: `HF_TOKEN` (adds Hugging Face bearer token for gated/private downloads)

When `HF_TOKEN` is set, downloads require `https://` base URLs to avoid leaking credentials.

```go
package main

import (
    "log"

    "github.com/amikos-tech/pure-onnx/embeddings/openclip"
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

    assets, err := openclip.EnsureDefaultAssets()
    if err != nil {
        log.Fatal(err)
    }

    embedder, err := openclip.NewEmbedder(
        assets.TextModelPath,
        assets.VisionModelPath,
        assets.TokenizerPath,
        assets.PreprocessorConfigPath,
    )
    if err != nil {
        log.Fatal(err)
    }
    defer embedder.Close()

    textEmbeds, err := embedder.EmbedTexts([]string{"a photo of a cat", "a photo of a dog"})
    if err != nil {
        log.Fatal(err)
    }
    _ = textEmbeds // [][]float32
}
```

Similarity helpers are also available:
- `openclip.CosineSimilarity(a, b)`
- `openclip.CLIPSimilarityLogits(imageEmbeddings, textEmbeddings, openclip.DefaultCLIPLogitScale)`

Runnable OpenCLIP example:
- `examples/openclip/main.go`
- `examples/openclip/README.md`
- `examples/openclip/ATTRIBUTION.md`

Run it with:

```bash
cd examples/openclip
go run .
```

OpenCLIP test instructions and commands are documented in [`TESTING.md`](TESTING.md).

### OpenCLIP ONNX Export Tooling (`tools/openclip_export_onnx.py`)

To generate pinned OpenCLIP ONNX artifacts (split text + vision encoders):

Detailed runbook: [`docs/openclip-export.md`](docs/openclip-export.md)

```bash
pip install torch transformers huggingface_hub numpy onnxruntime==1.23.1

python3 ./tools/openclip_export_onnx.py \
  --output-dir ./build/openclip-vit-b-32-laion2b-s34b-b79k-onnx
```

Defaults are pinned to:
- model: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
- revision: `1a25a446712ba5ee05982a381eed697ef9b435cf`

The export writes:
- `text_model.onnx`
- `vision_model.onnx`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `preprocessor_config.json`
- `manifest.json` (SHA256 + export metadata)

Optional copied files when available upstream:
- `special_tokens_map.json`
- `open_clip_config.json`
- `vocab.json`
- `merges.txt`
- `README.md`

If `onnxruntime` is unavailable in your Python env, add `--skip-verify`.

Optional: upload generated artifacts directly to Hugging Face Hub:

```bash
export HF_TOKEN=<your_hf_token>

python3 ./tools/openclip_export_onnx.py \
  --output-dir ./build/openclip-vit-b-32-laion2b-s34b-b79k-onnx \
  --push-to-hub-repo <your-org-or-user>/<repo-name>
```

## Project Status

This project is under active development. See our [GitHub Issues](https://github.com/amikos-tech/pure-onnx/issues) for the development roadmap.

### Current Focus

We're focusing on providing a drop-in replacement for common ONNX Runtime use cases, particularly for embeddings and inference tasks.

## Releases

Tagged releases (`v*`) publish signed artifacts to `releases.amikos.tech`.

- Runbook: [docs/releases.md](docs/releases.md)
- Metadata endpoint: `https://releases.amikos.tech/pure-onnx/latest.json`

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
- `make vulncheck` (with patched Go toolchain baseline `go1.25.8+auto`)

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
