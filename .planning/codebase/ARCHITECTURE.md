# Architecture

**Analysis Date:** 2026-03-18

## Pattern Overview

**Overall:** Layered Go library for native ONNX Runtime interop, with a low-level FFI core in `ort/` and model-specific convenience layers in `embeddings/`.

**Key Characteristics:**
- No-CGO design: native symbols are loaded dynamically through `purego`
- Global runtime lifecycle managed once per process in `ort/environment.go`
- Explicit resource ownership for tensors, sessions, memory info, and embedders
- Higher-level embedding packages build reusable batching and post-processing on top of the raw ORT session API

## Layers

**FFI Runtime Layer (`ort/`):**
- Purpose: map the ONNX Runtime C API into Go and expose safe-ish wrappers for environment, tensor, memory, and session lifecycle
- Contains: `InitializeEnvironment`, `DestroyEnvironment`, `AdvancedSession`, `Tensor[T]`, bootstrap/download logic, OS-specific library loading
- Depends on: `purego`, `unsafe`, native ONNX Runtime shared libraries, and generated `OrtApi` bindings
- Used by: all example programs and all packages under `embeddings/`

**Model Adapter Layer (`embeddings/*`):**
- Purpose: hide raw tensor/session wiring behind model-specific APIs
- Contains: `minilm.Embedder`, `splade.Embedder`, `openclip.Embedder`, session cache management, tokenizer usage, pooling/post-processing
- Depends on: `ort/`, `pure-tokenizers`, and model-specific local artifacts
- Used by: example programs and downstream applications that want dense, sparse, or CLIP embeddings

**Utility Layer (`embeddings/internal/ortutil`):**
- Purpose: small shared helpers for resource cleanup
- Contains: `DestroyAll`
- Depends on: only local interfaces and the Go standard library
- Used by: embedding packages when tearing down grouped ORT resources

**Executable/Tooling Layer (`examples/`, `tools/`, `.github/workflows/`):**
- Purpose: provide runnable demos, generators, and CI/release automation
- Contains: basic/inference/openclip examples, OpenCLIP export tooling, `gen_ortapi.go`, GitHub Actions workflows
- Depends on: lower library layers plus external services such as GitHub and Hugging Face
- Used by: maintainers, CI, and users validating real-world flows

## Data Flow

**Core ORT Inference Flow:**

1. Caller resolves a runtime library path explicitly or via `EnsureOnnxRuntimeSharedLibrary()` in `ort/bootstrap.go`
2. `ort.InitializeEnvironment()` loads the native library, registers function pointers, and creates a process-global ORT environment
3. Caller creates tensors via `ort.NewTensor` / `ort.NewEmptyTensor`
4. Caller constructs an `ort.AdvancedSession` with model path, input names, and output names
5. `AdvancedSession.Run()` acquires session-local and global runtime locks, converts names/values into native handles, and invokes ORT
6. Callers read tensor-backed output slices and then destroy sessions/tensors/environment in reverse order

**Embedding Flow:**

1. Caller initializes `ort` once for the process
2. Embedder loads tokenizer/model metadata and validates artifact paths
3. Inputs are tokenized or preprocessed into reusable backing buffers
4. Package-specific session caches keyed by batch size reuse tensors and `AdvancedSession` instances
5. Post-processing converts raw outputs into dense vectors, sparse vectors, or CLIP similarity matrices

**Bootstrap/Asset Resolution Flow:**

1. Bootstrap resolves cache directory and target artifact names from platform/runtime config
2. Downloads are guarded by file/process locks
3. Archives/files are validated for size, checksum, and path traversal
4. Resolved local file paths are returned to the caller for normal `ort` or embedding setup

**State Management:**
- Process-global ORT state lives in package globals in `ort/environment.go`
- Session-level mutable state lives on `AdvancedSession` and embedder caches
- Persistent state is file-based only (user cache directories and generated artifacts)

## Key Abstractions

**`AdvancedSession`:**
- Purpose: wrap an ONNX Runtime session plus fixed input/output bindings
- Examples: `ort/session.go`, used directly by `examples/inference/main.go` and all embedding packages
- Pattern: stateful handle wrapper with explicit `Run()` / `Destroy()`

**`Tensor[T]`:**
- Purpose: represent ORT values backed by Go slices pinned for native access
- Examples: `ort/tensor.go`
- Pattern: generic resource wrapper with finalizer safety net and explicit destroy semantics

**`BootstrapOption` / embedding `Option`:**
- Purpose: configure bootstrap and embedder behavior without large constructors
- Examples: `ort/bootstrap.go`, `embeddings/minilm/embedder.go`, `embeddings/splade/embedder.go`, `embeddings/openclip/embedder.go`
- Pattern: functional options

## Entry Points

**Library Entry Points:**
- `ort/environment.go` - global runtime initialization and teardown
- `ort/session.go` - model session creation and inference
- `ort/tensor.go` - tensor allocation and lifecycle

**Example Programs:**
- `examples/basic/main.go` - minimal runtime initialization example
- `examples/inference/main.go` - end-to-end single-model inference flow driven by env vars
- `examples/openclip/main.go` - OpenCLIP text/image embedding demo with manifest-backed fixtures

**Tooling:**
- `tools/gen_ortapi.go` - regenerates `ort/ortapi_generated.go` from ONNX Runtime headers

## Error Handling

**Strategy:** return Go `error` values aggressively, validate inputs early, and use deferred cleanup to keep native handles from leaking.

**Patterns:**
- Wrap lower-level failures with `fmt.Errorf(...: %w)`
- Translate ORT `OrtStatus` handles into Go strings via helper functions in `ort/environment.go`
- Use `errors.Join` when cleanup steps can fail independently
- Treat nil receivers as safe no-ops for most `Destroy()` methods

## Cross-Cutting Concerns

**Concurrency:**
- `ort/environment.go` defines a lock hierarchy spanning global runtime state, session runs, and tensor lifetimes
- Several tests in `ort/session_test.go` and `ort/environment_test.go` exist specifically to protect these invariants

**Memory Management:**
- The code pins Go slice backing arrays while native ORT uses them (`ort/tensor.go`)
- Finalizers are present as leak backstops, but the design still expects explicit `Destroy()` calls

**Security and Integrity:**
- Bootstrap code validates checksums, path traversal, redirect safety, and download size limits in both `ort/bootstrap.go` and `embeddings/openclip/bootstrap.go`

---

*Architecture analysis: 2026-03-18*
*Update when major patterns change*
