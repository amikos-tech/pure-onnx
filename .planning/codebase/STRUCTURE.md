# Codebase Structure

**Analysis Date:** 2026-03-18

## Directory Layout

```text
onnx-purego/
├── .github/workflows/        # CI, review, and release automation
├── .githooks/                # Repo-managed git hooks
├── .planning/codebase/       # Generated codebase map documents
├── docs/                     # Runbooks and release/export documentation
├── embeddings/               # High-level embedding packages on top of ort
│   ├── internal/ortutil/     # Shared cleanup helper
│   ├── minilm/               # Dense sentence embedding adapter
│   ├── openclip/             # Text + image embedding adapter and asset bootstrap
│   └── splade/               # Sparse embedding adapter
├── examples/                 # Runnable sample programs
│   ├── basic/                # Minimal initialization example
│   ├── experimental/         # Unsafe/manual experiment code paths
│   ├── inference/            # End-to-end inference example
│   └── openclip/             # OpenCLIP example plus committed fixture assets
├── internal/c_api/           # ONNX Runtime header snapshots for reference/generation
├── ort/                      # Low-level runtime bindings and wrappers
├── tools/                    # Generators and Python-based model tooling
├── Makefile                  # Primary task runner
├── README.md                 # User-facing package overview
└── TESTING.md                # Test setup and matrix documentation
```

## Directory Purposes

**`ort/`:**
- Purpose: core ONNX Runtime binding and lifecycle package
- Contains: environment/session/tensor/memory code, bootstrap/download logic, generated API bindings, platform-specific library loaders
- Key files: `ort/environment.go`, `ort/session.go`, `ort/tensor.go`, `ort/bootstrap.go`, `ort/ortapi_generated.go`
- Subdirectories: none; package is flat with `_unix.go` and `_windows.go` file suffixes for platform-specific behavior

**`embeddings/`:**
- Purpose: higher-level model adapters that turn raw ORT sessions into convenient embedding APIs
- Contains: three public subpackages plus `internal/ortutil`
- Key files: `embeddings/minilm/embedder.go`, `embeddings/splade/embedder.go`, `embeddings/openclip/embedder.go`, `embeddings/openclip/bootstrap.go`
- Subdirectories: `minilm/`, `splade/`, `openclip/`, `internal/ortutil/`

**`examples/`:**
- Purpose: runnable demonstrations for common usage patterns
- Contains: `main.go` programs, example READMEs, and OpenCLIP sample assets
- Key files: `examples/basic/main.go`, `examples/inference/main.go`, `examples/openclip/main.go`
- Subdirectories: `openclip/assets/` contains committed PNG fixtures and `manifest.jsonl`

**`tools/`:**
- Purpose: repository maintenance and offline artifact tooling
- Contains: header/code generators plus Python scripts for OpenCLIP export and golden dataset generation
- Key files: `tools/gen_ortapi.go`, `tools/openclip_export_onnx.py`, `tools/openclip_generate_golden.py`, `tools/splade_generate_golden.py`
- Subdirectories: none

**`docs/`:**
- Purpose: maintainer-facing runbooks
- Contains: release and export documentation
- Key files: `docs/openclip-export.md`, `docs/releases.md`
- Subdirectories: none

## Key File Locations

**Entry Points:**
- `examples/basic/main.go` - minimal library smoke test
- `examples/inference/main.go` - env-driven inference example
- `examples/openclip/main.go` - OpenCLIP demo application
- `tools/gen_ortapi.go` - generator entry point for `ort/ortapi_generated.go`

**Configuration:**
- `go.mod` / `go.sum` - module and dependency state
- `Makefile` - local build/test/release commands
- `.golangci.yml` - lint/formatter configuration
- `.github/workflows/ci.yml` - CI matrix and integration setup
- `.gitignore` - generated artifact and local cache rules

**Core Logic:**
- `ort/` - raw bindings and runtime bootstrap
- `embeddings/minilm/` - dense text embedding flow
- `embeddings/splade/` - sparse text embedding flow
- `embeddings/openclip/` - multimodal embedding flow and default asset bootstrap

**Testing:**
- `ort/*_test.go` - the deepest unit/race/integration coverage
- `embeddings/*/*_test.go` - package-specific behavior, integration, parity, and benchmark coverage
- `examples/openclip/main_test.go` - example-level validation

**Documentation:**
- `README.md` - main package and examples overview
- `TESTING.md` - full test environment documentation
- `docs/*.md` - specialized maintainer runbooks

## Naming Conventions

**Files:**
- lower_snake_case for Go source files: `shape_parse.go`, `session_benchmark_test.go`
- `*_test.go` for tests
- generated files are called out explicitly: `ort/ortapi_generated.go`

**Directories:**
- lowercase package names: `ort`, `minilm`, `openclip`, `splade`
- nested `internal/` only where visibility should be restricted (`embeddings/internal/ortutil`)

**Special Patterns:**
- OS-specific implementations use suffixes such as `_unix.go` and `_windows.go`
- examples consistently use `main.go`
- docs and repo control files are uppercase where conventional: `README.md`, `TESTING.md`, `CLAUDE.md`

## Where to Add New Code

**New ORT capability:**
- Primary code: `ort/`
- Tests: matching `ort/*_test.go`
- Reference header updates: `internal/c_api/` plus `tools/gen_ortapi.go` if API shape changes

**New embedding model adapter:**
- Primary code: `embeddings/<model>/`
- Shared cleanup helpers: `embeddings/internal/ortutil/` only if reused across packages
- Tests: colocated `*_test.go`, `*_integration_test.go`, and optional benchmarks

**New example:**
- Implementation: `examples/<name>/main.go`
- Docs: `examples/<name>/README.md` when setup is non-trivial

**New tooling/runbook:**
- Automation/scripts: `tools/`
- Human-facing docs: `docs/`

## Special Directories

**`internal/c_api/`:**
- Purpose: checked-in ONNX Runtime headers and related reference material
- Source: upstream ONNX Runtime C API snapshots
- Committed: yes

**`examples/openclip/assets/`:**
- Purpose: committed demo fixtures for the OpenCLIP example
- Source: curated PNG assets plus `manifest.jsonl`
- Committed: yes

**Generated/ignored local dirs (`build/`, `third_party/`, `.cache/`):**
- Purpose: downloaded artifacts, temporary builds, local caches
- Source: bootstrap flows, manual tooling, or local development
- Committed: no, ignored by `.gitignore`

---

*Structure analysis: 2026-03-18*
*Update when directory structure changes*
