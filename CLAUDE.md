# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 CRITICAL: NO CGO POLICY

**ABSOLUTELY NO CGO ALLOWED IN THE `ort/` PACKAGE!**

- ❌ **NEVER** use `import "C"`
- ❌ **NEVER** use CGO types (`*C.char`, `C.int`, etc.)
- ❌ **NEVER** use CGO functions (`C.CString`, `C.GoString`, `C.free`, etc.)
- ✅ **ALWAYS** use pure Go with `unsafe` package for C interop
- ✅ **ALWAYS** use `uintptr` for all C pointers
- ✅ **ALWAYS** use custom string conversion functions (see `ort/cstring.go`)

**Why?** The entire purpose of this project is to avoid CGO compilation. Using CGO defeats the core value proposition: no C compiler needed, cross-compilation support, faster builds, cleaner dependencies.

## Build and Development Commands

```bash
# Build everything
go build ./...

# Run tests (add -race for concurrency-sensitive packages)
go test ./...

# Format and static analysis
go fmt ./...
go vet ./...

# Full local pre-commit gate (mirrors CI: fmt, vet, lint, tests)
make precommit
```

- Requires an ONNX Runtime shared library: set `ONNXRUNTIME_LIB_PATH`, or let bootstrap download/cache one (`ort/bootstrap.go`).
- Always use feature branches and open PRs for all changes; conventional commits; never push to `main` directly.

<!-- GSD:project-start source:PROJECT.md -->
## Project

**onnx-purego (pure-onnx)**

A pure-Go binding for Microsoft ONNX Runtime that loads and calls the ONNX Runtime C API through `purego` instead of CGO — no C compiler, clean cross-compilation, faster builds. It ships a low-level FFI core (`ort/`) plus model-specific embedding adapters (`minilm` dense, `splade` sparse, `openclip` text/image) for Go applications that need ONNX inference or embeddings without a CGO toolchain.

**Core Value:** Run ONNX Runtime inference from Go with zero CGO — if that stops working, nothing else matters.

### Constraints

- **Tech stack**: Pure Go + `unsafe`, no CGO in `ort/` — core value proposition
- **Interop**: All C pointers as `uintptr`; custom string conversion (`ort/cstring.go`), never `C.CString`/`C.GoString`
- **Dependency**: `github.com/ebitengine/purego` is load-bearing for the entire binding strategy
- **Compatibility**: Must keep working across Linux/macOS/Windows amd64+arm64 — the supported artifact matrix
- **Workflow**: Feature branches + PRs for all changes; conventional commits; never push to `main` directly
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Go 1.25.0 - All library code, examples, and tests live under `ort/`, `embeddings/`, `examples/`, and `tools/gen_ortapi.go`.
- Python 3.10+ - Export and golden-data tooling in `tools/openclip_export_onnx.py`, `tools/openclip_generate_golden.py`, and `tools/splade_generate_golden.py`.
- Shell/Make - Local automation and CI entry points in `Makefile`, `.githooks/pre-commit`, and `.github/workflows/*.yml`.
- C headers (reference only) - ONNX Runtime API definitions in `internal/c_api/onnxruntime_c_api.h` and `internal/c_api/ort_apis.h`; these are not compiled directly.
## Runtime
- Go 1.25.x for normal development and CI (`go.mod`, `.github/workflows/ci.yml`).
- Patched Go 1.25.12+auto only for vulnerability scanning via `govulncheck` (`Makefile`, CI env `GO_VULNCHECK_TOOLCHAIN`).
- Native ONNX Runtime shared libraries are loaded dynamically at runtime through `purego`, so consumers still need a platform-specific `.so`, `.dylib`, or `.dll`.
- Go modules
- Lockfile: `go.sum` present
## Frameworks
- `github.com/ebitengine/purego` v0.10.0 - CGO-free dynamic library loading and symbol binding in `ort/environment.go`, `ort/library_unix.go`, and `ort/library_windows.go`.
- Microsoft ONNX Runtime C API v22 - exposed through generated bindings in `ort/ortapi_generated.go` and constants in `ort/constants.go`.
- `github.com/amikos-tech/pure-tokenizers` v0.1.4 - tokenizer support for `embeddings/minilm`, `embeddings/splade`, and `embeddings/openclip`.
- Go `testing` package - unit, integration, race, and benchmark coverage across `ort/` and `embeddings/`.
- `github.com/stretchr/testify` v1.11.1 - supplemental assertions in parts of the test suite.
- `gofmt`, `goimports`, `go vet`, `go test` - standard local verification, wired through `Makefile`.
- `golangci-lint` v2.8.0 and `gosec` v2.23.0 - optional but expected pre-commit and CI tooling.
## Key Dependencies
- `github.com/ebitengine/purego` v0.10.0 - the entire no-CGO binding strategy depends on it.
- `github.com/amikos-tech/pure-tokenizers` v0.1.4 - used by all higher-level embedding packages for tokenizer loading and preprocessing.
- `golang.org/x/sys` v0.41.0 - OS-specific helpers for file locking and native runtime interactions.
- `github.com/Masterminds/semver/v3` v3.4.0 - version parsing in runtime/bootstrap flows.
- Python packages `torch`, `transformers`, `huggingface_hub`, `numpy`, and `onnxruntime==1.23.1` - used only by OpenCLIP export tooling in `tools/requirements-openclip.txt`.
## Configuration
- Runtime library resolution: `ONNXRUNTIME_LIB_PATH`, `ONNXRUNTIME_VERSION`, `ONNXRUNTIME_CACHE_DIR`, `ONNXRUNTIME_DISABLE_DOWNLOAD`, `ONNXRUNTIME_SKIP_VERSION_CHECK`.
- GitHub-backed bootstrap and checksum lookup: `GITHUB_TOKEN` / `GH_TOKEN`.
- Hugging Face-backed asset downloads: `HF_TOKEN`, `ONNXRUNTIME_OPENCLIP_CACHE_DIR`.
- Test-specific overrides are documented in `TESTING.md` and consumed throughout `ort/*_test.go` and `embeddings/*_test.go`.
- `Makefile` - main task runner for build, test, lint, release, and pre-commit flows.
- `.golangci.yml` - linter and formatter configuration.
- `.github/workflows/ci.yml` and `.github/workflows/release.yml` - CI, integration, and release pipelines.
## Platform Requirements
- macOS, Linux, or Windows with Go 1.25+.
- Either an explicit ONNX Runtime shared library or network access for bootstrap download/caching.
- Python is optional unless working on export or golden-dataset tooling in `tools/`.
- Intended as an embeddable Go library plus example binaries built from `examples/basic` and `examples/inference`.
- Target platforms currently mirror ONNX Runtime artifact handling in `ort/bootstrap.go`: Linux amd64/arm64, macOS amd64/arm64, and Windows amd64/arm64.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- lower_snake_case for implementation files: `environment.go`, `bootstrap_lock_unix.go`, `golden_dataset_parity_test.go`
- `*_test.go` for all test files, often with specialized suffixes like `*_integration_test.go` or `*_benchmark_test.go`
- generated code is named explicitly with `_generated`: `ort/ortapi_generated.go`
- Exported APIs use Go-standard PascalCase: `InitializeEnvironment`, `EnsureDefaultAssets`, `WithSequenceLength`
- Unexported helpers use lowerCamelCase: `resolveRuntimeArtifact`, `validateFilePath`, `setupORTTestEnvironment`
- Functional option constructors consistently start with `With` / `Without`
- lowerCamelCase for locals and fields
- package-level constants are mostly PascalCase when exported and lowerCamelCase when internal
- all-caps names are reserved for compatibility constants or env-oriented identifiers such as `ORT_API_VERSION`
- PascalCase for structs, interfaces, and enums: `AdvancedSession`, `MemoryInfo`, `PoolingStrategy`
- Package names stay short and lowercase: `ort`, `minilm`, `splade`, `openclip`
## Code Style
- `gofmt` and `goimports` are the formatting source of truth (`.golangci.yml`, `Makefile`)
- Standard Go formatting conventions apply: tabs, grouped imports, no manual alignment
- Error strings are generally lowercase and sentence-fragment style
- `go vet`, `golangci-lint`, and `gosec` are the main static checks
- `precommit` in `Makefile` mirrors CI blockers, with opt-out env vars for local workflows
- `.golangci.yml` keeps the rule set small and targeted instead of enabling everything
## Import Organization
- One blank line between import groups
- No path aliases beyond short clarity-driven aliases like `tokenizers`
- Side-effect imports are only used when required, for example image codecs in `examples/openclip/main.go`
- None; imports use full module paths
## Error Handling
- Validate aggressively at function entry and return early on bad inputs
- Wrap underlying failures with `fmt.Errorf(...: %w)`
- Join cleanup failures with `errors.Join` where multiple destroy steps can fail
- Native-handle wrappers usually treat nil receivers as safe no-ops on `Destroy()`
- Most code returns plain `error` rather than custom error structs
- A few package-private sentinel or wrapper types exist where retry/permanence matters, such as `permanentBootstrapError` in `ort/bootstrap.go`
- Tests assert on message content frequently, so wording changes can be user-visible
## Logging
- Standard library `log` package only
- Library code logs sparingly, mainly for warnings or bootstrap path visibility
- Examples use `log.Fatal` / `log.Printf` for CLI-style behavior
- There is no structured logging abstraction
## Comments
- Explain FFI safety assumptions, lock ordering, and lifetime rules
- Document why `unsafe` usage is acceptable in specific places
- Keep obvious code uncommented
- Exported types, constants, and functions are usually documented
- Internal helpers receive comments when concurrency, lifecycle, or security behavior is non-obvious
- `#nosec` annotations are used narrowly to justify intentional `unsafe` pointer conversions or checksum-like constants
## Function Design
- Public APIs tend to be moderate-sized with early validation and deferred cleanup
- The longest functions are concentrated in bootstrap/download code and embedder setup paths
- Constructors prefer explicit required parameters plus functional options
- Helpers avoid large configuration structs at call sites unless state needs to persist
- APIs nearly always return `(value, error)` or `error`
- Cleanup methods return `error` rather than panicking
## Module Design
- Packages expose focused public APIs and keep helpers unexported
- `internal/` is used only where cross-package reuse should remain private
- `ort/ortapi_generated.go` is treated as generated output; changes should come from `tools/gen_ortapi.go` and header inputs, not hand edits
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- No-CGO design: native symbols are loaded dynamically through `purego`
- Global runtime lifecycle managed once per process in `ort/environment.go`
- Explicit resource ownership for tensors, sessions, memory info, and embedders
- Higher-level embedding packages build reusable batching and post-processing on top of the raw ORT session API
## Layers
- Purpose: map the ONNX Runtime C API into Go and expose safe-ish wrappers for environment, tensor, memory, and session lifecycle
- Contains: `InitializeEnvironment`, `DestroyEnvironment`, `AdvancedSession`, `Tensor[T]`, bootstrap/download logic, OS-specific library loading
- Depends on: `purego`, `unsafe`, native ONNX Runtime shared libraries, and generated `OrtApi` bindings
- Used by: all example programs and all packages under `embeddings/`
- Purpose: hide raw tensor/session wiring behind model-specific APIs
- Contains: `minilm.Embedder`, `splade.Embedder`, `openclip.Embedder`, session cache management, tokenizer usage, pooling/post-processing
- Depends on: `ort/`, `pure-tokenizers`, and model-specific local artifacts
- Used by: example programs and downstream applications that want dense, sparse, or CLIP embeddings
- Purpose: small shared helpers for resource cleanup
- Contains: `DestroyAll`
- Depends on: only local interfaces and the Go standard library
- Used by: embedding packages when tearing down grouped ORT resources
- Purpose: provide runnable demos, generators, and CI/release automation
- Contains: basic/inference/openclip examples, OpenCLIP export tooling, `gen_ortapi.go`, GitHub Actions workflows
- Depends on: lower library layers plus external services such as GitHub and Hugging Face
- Used by: maintainers, CI, and users validating real-world flows
## Data Flow
- Process-global ORT state lives in package globals in `ort/environment.go`
- Session-level mutable state lives on `AdvancedSession` and embedder caches
- Persistent state is file-based only (user cache directories and generated artifacts)
## Key Abstractions
- Purpose: wrap an ONNX Runtime session plus fixed input/output bindings
- Examples: `ort/session.go`, used directly by `examples/inference/main.go` and all embedding packages
- Pattern: stateful handle wrapper with explicit `Run()` / `Destroy()`
- Purpose: represent ORT values backed by Go slices pinned for native access
- Examples: `ort/tensor.go`
- Pattern: generic resource wrapper with finalizer safety net and explicit destroy semantics
- Purpose: configure bootstrap and embedder behavior without large constructors
- Examples: `ort/bootstrap.go`, `embeddings/minilm/embedder.go`, `embeddings/splade/embedder.go`, `embeddings/openclip/embedder.go`
- Pattern: functional options
## Entry Points
- `ort/environment.go` - global runtime initialization and teardown
- `ort/session.go` - model session creation and inference
- `ort/tensor.go` - tensor allocation and lifecycle
- `examples/basic/main.go` - minimal runtime initialization example
- `examples/inference/main.go` - end-to-end single-model inference flow driven by env vars
- `examples/openclip/main.go` - OpenCLIP text/image embedding demo with manifest-backed fixtures
- `tools/gen_ortapi.go` - regenerates `ort/ortapi_generated.go` from ONNX Runtime headers
## Error Handling
- Wrap lower-level failures with `fmt.Errorf(...: %w)`
- Translate ORT `OrtStatus` handles into Go strings via helper functions in `ort/environment.go`
- Use `errors.Join` when cleanup steps can fail independently
- Treat nil receivers as safe no-ops for most `Destroy()` methods
## Cross-Cutting Concerns
- `ort/environment.go` defines a lock hierarchy spanning global runtime state, session runs, and tensor lifetimes
- Several tests in `ort/session_test.go` and `ort/environment_test.go` exist specifically to protect these invariants
- The code pins Go slice backing arrays while native ORT uses them (`ort/tensor.go`)
- Finalizers are present as leak backstops, but the design still expects explicit `Destroy()` calls
- Bootstrap code validates checksums, path traversal, redirect safety, and download size limits in both `ort/bootstrap.go` and `embeddings/openclip/bootstrap.go`
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
