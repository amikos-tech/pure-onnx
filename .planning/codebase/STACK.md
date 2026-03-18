# Technology Stack

**Analysis Date:** 2026-03-18

## Languages

**Primary:**
- Go 1.24.0 - All library code, examples, and tests live under `ort/`, `embeddings/`, `examples/`, and `tools/gen_ortapi.go`.

**Secondary:**
- Python 3.10+ - Export and golden-data tooling in `tools/openclip_export_onnx.py`, `tools/openclip_generate_golden.py`, and `tools/splade_generate_golden.py`.
- Shell/Make - Local automation and CI entry points in `Makefile`, `.githooks/pre-commit`, and `.github/workflows/*.yml`.
- C headers (reference only) - ONNX Runtime API definitions in `internal/c_api/onnxruntime_c_api.h` and `internal/c_api/ort_apis.h`; these are not compiled directly.

## Runtime

**Environment:**
- Go 1.24.x for normal development and CI (`go.mod`, `.github/workflows/ci.yml`).
- Patched Go 1.25.8+auto only for vulnerability scanning via `govulncheck` (`Makefile`, CI env `GO_VULNCHECK_TOOLCHAIN`).
- Native ONNX Runtime shared libraries are loaded dynamically at runtime through `purego`, so consumers still need a platform-specific `.so`, `.dylib`, or `.dll`.

**Package Manager:**
- Go modules
- Lockfile: `go.sum` present

## Frameworks

**Core:**
- `github.com/ebitengine/purego` v0.10.0 - CGO-free dynamic library loading and symbol binding in `ort/environment.go`, `ort/library_unix.go`, and `ort/library_windows.go`.
- Microsoft ONNX Runtime C API v22 - exposed through generated bindings in `ort/ortapi_generated.go` and constants in `ort/constants.go`.
- `github.com/amikos-tech/pure-tokenizers` v0.1.4 - tokenizer support for `embeddings/minilm`, `embeddings/splade`, and `embeddings/openclip`.

**Testing:**
- Go `testing` package - unit, integration, race, and benchmark coverage across `ort/` and `embeddings/`.
- `github.com/stretchr/testify` v1.11.1 - supplemental assertions in parts of the test suite.

**Build/Dev:**
- `gofmt`, `goimports`, `go vet`, `go test` - standard local verification, wired through `Makefile`.
- `golangci-lint` v2.8.0 and `gosec` v2.23.0 - optional but expected pre-commit and CI tooling.

## Key Dependencies

**Critical:**
- `github.com/ebitengine/purego` v0.10.0 - the entire no-CGO binding strategy depends on it.
- `github.com/amikos-tech/pure-tokenizers` v0.1.4 - used by all higher-level embedding packages for tokenizer loading and preprocessing.
- `golang.org/x/sys` v0.41.0 - OS-specific helpers for file locking and native runtime interactions.

**Infrastructure:**
- `github.com/Masterminds/semver/v3` v3.4.0 - version parsing in runtime/bootstrap flows.
- Python packages `torch`, `transformers`, `huggingface_hub`, `numpy`, and `onnxruntime==1.23.1` - used only by OpenCLIP export tooling in `tools/requirements-openclip.txt`.

## Configuration

**Environment:**
- Runtime library resolution: `ONNXRUNTIME_LIB_PATH`, `ONNXRUNTIME_VERSION`, `ONNXRUNTIME_CACHE_DIR`, `ONNXRUNTIME_DISABLE_DOWNLOAD`, `ONNXRUNTIME_SKIP_VERSION_CHECK`.
- GitHub-backed bootstrap and checksum lookup: `GITHUB_TOKEN` / `GH_TOKEN`.
- Hugging Face-backed asset downloads: `HF_TOKEN`, `ONNXRUNTIME_OPENCLIP_CACHE_DIR`.
- Test-specific overrides are documented in `TESTING.md` and consumed throughout `ort/*_test.go` and `embeddings/*_test.go`.

**Build:**
- `Makefile` - main task runner for build, test, lint, release, and pre-commit flows.
- `.golangci.yml` - linter and formatter configuration.
- `.github/workflows/ci.yml` and `.github/workflows/release.yml` - CI, integration, and release pipelines.

## Platform Requirements

**Development:**
- macOS, Linux, or Windows with Go 1.24+.
- Either an explicit ONNX Runtime shared library or network access for bootstrap download/caching.
- Python is optional unless working on export or golden-dataset tooling in `tools/`.

**Production:**
- Intended as an embeddable Go library plus example binaries built from `examples/basic` and `examples/inference`.
- Target platforms currently mirror ONNX Runtime artifact handling in `ort/bootstrap.go`: Linux amd64/arm64, macOS amd64/arm64, and Windows amd64/arm64.

---

*Stack analysis: 2026-03-18*
*Update after major dependency or toolchain changes*
