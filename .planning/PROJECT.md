# onnx-purego (pure-onnx)

## What This Is

A pure-Go binding for Microsoft ONNX Runtime that loads and calls the ONNX Runtime C API through `purego` instead of CGO — no C compiler, clean cross-compilation, faster builds. It ships a low-level FFI core (`ort/`) plus model-specific embedding adapters (`minilm` dense, `splade` sparse, `openclip` text/image) for Go applications that need ONNX inference or embeddings without a CGO toolchain.

## Core Value

Run ONNX Runtime inference from Go with zero CGO — if that stops working, nothing else matters.

## Requirements

### Validated

<!-- Inferred from existing code (brownfield). Shipped and relied upon. -->

- ✓ No-CGO dynamic loading of ONNX Runtime via `purego` on Linux/macOS/Windows (amd64+arm64) — existing
- ✓ Process-global ORT environment lifecycle (`InitializeEnvironment`/`DestroyEnvironment`) with lock hierarchy — existing
- ✓ Generic `Tensor[T]` allocation with GC pinning and explicit `Destroy()` — existing
- ✓ `AdvancedSession` model load + `Run()` inference with fixed input/output bindings — existing
- ✓ Runtime bootstrap: download/cache/checksum/lock of ONNX Runtime shared libs (`EnsureOnnxRuntimeSharedLibrary`) — existing
- ✓ Embedding adapters: `minilm` (dense), `splade` (sparse), `openclip` (CLIP) with tokenizer + pooling — existing
- ✓ Runnable examples (`basic`, `inference`, `openclip`) and ORT API generation tooling (`gen_ortapi.go`) — existing
- ✓ Comprehensive, inspectable error handling across the public `ort` API — validated in Phase 2: Core API — Errors & Values
- ✓ Sealed `Value` interface with polymorphic session tensor handling — validated in Phase 2: Core API — Errors & Values
- ✓ Generalized embedder API spanning dense MiniLM/OpenCLIP and sparse SPLADE, with a zero-import typed contract and revision-bound native parity evidence — validated in Phase 3: Generalized Embedder API

### Active

<!-- v0.1.0 milestone — harden and ship. Hypotheses until shipped. -->

**DX & test hardening**
- [ ] Inference example fails fast with actionable message on unsupported platforms (#42)
- [ ] Deterministic (non-timing) concurrency assertions in tests (#43)
- [ ] Stress tests for concurrent init/destroy (#24)

**Documentation**
- [ ] Comprehensive documentation and examples (#9)
- [ ] Document API function maturity and testing status (#30)
- [ ] Verify and document error message string lifetime semantics (#25)
- [ ] Document ReleaseEnv memory leak limitation before v1.0 (#21)

**Quality gate**
- [ ] Enable full linting — remove `continue-on-error` from golangci-lint (#23)

### Out of Scope

- CGO / `import "C"` anywhere in `ort/` — defeats the project's entire premise (no C compiler, cross-compilation)
- Changing `ort` runtime/library-loading behavior for #42 — that issue is example UX only
- OpenCLIP end-to-end tracker (#68) and numerical-equivalence tightening (#76) — separate milestone, not gating v0.1.0
- tree-sitter-based C API auto-generation (#29) — future tooling milestone
- Advanced features / Phase 2 (#10) — post-v0.1.0

## Context

- **Brownfield.** Mature codebase already mapped in `.planning/codebase/` (2026-03-18). Layered: FFI core (`ort/`) → embedding adapters (`embeddings/*`) → examples/tooling.
- The public `CLAUDE.md` predates the embeddings work and under-describes the current surface; treat the codebase map + this doc as ground truth.
- Issue #42 is partially resolved by prior work: the bootstrap rewrite (`resolveRuntimeArtifact`) already emits a fail-fast `GOOS=/GOARCH=`-labeled error. The remaining gap is example-only — surface a "set `ONNXRUNTIME_LIB_PATH`" hint when bootstrap fails.
- ONNX Runtime C API version 22; default bootstrap runtime version tracks CI (currently `1.24.1`, asserted by the version-match step in `ci.yml`).
- CI runs on Go 1.24.x across Linux/macOS/Windows (amd64+arm64); `govulncheck` uses a patched Go 1.25.x toolchain.
- Phase 2 is complete: public `ort` failures are inspectable, polymorphic `Value` handling is available, and race/native CI lanes cover the new contracts.
- Phase 3 is complete: `embeddings.Embedder[T]` unifies dense and sparse API conformance; native CI evidence records 12 named PASS events with zero skips for its final revision.

## Constraints

- **Tech stack**: Pure Go + `unsafe`, no CGO in `ort/` — core value proposition
- **Interop**: All C pointers as `uintptr`; custom string conversion (`ort/cstring.go`), never `C.CString`/`C.GoString`
- **Dependency**: `github.com/ebitengine/purego` is load-bearing for the entire binding strategy
- **Compatibility**: Must keep working across Linux/macOS/Windows amd64+arm64 — the supported artifact matrix
- **Workflow**: Feature branches + PRs for all changes; conventional commits; never push to `main` directly; squash merge on integration

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| v0.1.0 = harden full milestone (all 11 issues), not a feature-tight subset | User chose to include features (#49/#7/#6) alongside DX/docs/lint | — Pending |
| Definition of Done = tagged + documented release | All issues closed, full lint gate green, docs complete, CI green on all platforms, v0.1.0 tag + release notes | — Pending |
| #42 fix stays in the example, not `ort/` | Issue is scoped "example UX only; no ort runtime changes" | — Pending |
| Separate local error categories from native runtime detail | `errors.Is` remains stable for local lifecycle/validation failures while `errors.As` exposes `ORTError` operation, code, and message | ✓ Validated in Phase 2 |
| Seal `Value` and keep tensor extraction exact | Only package-owned values can enter the native lease protocol; `AsTensor[T]` performs no coercion, copying, reflection, or allocation | ✓ Validated in Phase 2 |
| Keep diagnostics opt-in and avoid logging returned errors | Silent defaults prevent surprise output; structured diagnostics are reserved for non-returnable notices and finalizer failures | ✓ Validated in Phase 2 |
| Borrow per-call values through the existing session run core | `RunWithValues` preserves caller ownership and existing lock, lease, lifetime, and `Run` behavior | ✓ Validated in Phase 2 |
| Keep race and native ABI verification in separate live-counted CI lanes | Exact selector counts prevent renamed tests from silently reducing coverage without disabling checkptr | ✓ Validated in Phase 2 |
| Keep the generalized embedding API compile-time typed and its root contract import-free | Dense and sparse result types stay concrete without runtime tags, registries, wrappers, or copies | ✓ Validated in Phase 3 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-08-02 after Phase 3 completion*
