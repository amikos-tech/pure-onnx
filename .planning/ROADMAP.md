# Roadmap: onnx-purego (pure-onnx) — v0.1.0

## Overview

This is a brownfield hardening milestone for a mature pure-Go ONNX Runtime binding. The `ort/` FFI core and `embeddings/` adapters already exist and work. The journey to v0.1.0 closes 12 open issues in thematic technical clusters: first shore up developer-facing example UX and make concurrency tests deterministic under the race detector; then land the code-heavy public API work (wrapped errors, a polymorphic `Value` interface, and a generalized dense/sparse embedder API); then document the now-settled surface (GoDoc, maturity status, lifetime semantics, known limitations); then flip the full lint gate on so it audits the final tree; and finally cut a tagged, documented v0.1.0 with CI green across the full platform matrix.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: DX & Test Hardening** - Fail-fast example UX plus deterministic, race-clean concurrency tests
- [ ] **Phase 2: Core API — Errors & Values** - Wrapped errors across the surface and a polymorphic `Value` interface
- [ ] **Phase 3: Generalized Embedder API** - Unified dense/sparse embedder API including SPLADE
- [ ] **Phase 4: Documentation** - GoDoc, maturity status, lifetime semantics, and known limitations
- [ ] **Phase 5: Full Lint Gate** - Remove `continue-on-error` and pass the full golangci-lint gate
- [ ] **Phase 6: Release v0.1.0** - Tag, release notes, green CI matrix, all issues closed

## Phase Details

### Phase 1: DX & Test Hardening
**Goal**: The inference example fails fast with actionable guidance on unsupported platforms, and concurrency tests prove correctness deterministically under the race detector.
**Depends on**: Nothing (first phase)
**Requirements**: DX-01, TST-01, TST-02
**Success Criteria** (what must be TRUE):
  1. Running the inference example when bootstrap fails exits non-zero with a message that names `GOOS`/`GOARCH` and instructs the user to set `ONNXRUNTIME_LIB_PATH` (change stays in the example, not `ort/`).
  2. Concurrency tests assert correctness via synchronization primitives (channels, `WaitGroup`, atomics) with no remaining sleep/timing-based assertions.
  3. Stress tests exercise many concurrent `InitializeEnvironment`/`DestroyEnvironment` cycles and pass under `go test -race`.
**Plans**: 3 plans

Plans:
- [ ] 01-01-PLAN.md — DX-01: exported ErrUnsupportedPlatform sentinel + inference example fail-fast hint
- [ ] 01-02-PLAN.md — TST-01: convert the 3 named concurrency tests to deterministic event-order assertions
- [ ] 01-03-PLAN.md — TST-02: stress tests for concurrent init/destroy, dedicated CI job, -short wiring, TESTING.md

### Phase 2: Core API — Errors & Values
**Goal**: The `ort` core returns comprehensive wrapped errors and exposes a `Value` interface for polymorphic tensor handling.
**Depends on**: Phase 1
**Requirements**: API-02, API-03
**Success Criteria** (what must be TRUE):
  1. Environment, tensor, session, and bootstrap failures return errors wrapped with actionable context and are inspectable via `errors.Is`/`errors.As`.
  2. A `Value` interface lets heterogeneous tensor types be passed as session inputs and returned as outputs.
  3. `AdvancedSession.Run` accepts and returns values through the `Value` interface without breaking existing typed `Tensor[T]` usage.
  4. Existing `ort` unit and integration tests pass against the new error and `Value` surfaces.
**Plans**: TBD

### Phase 3: Generalized Embedder API
**Goal**: A unified embedder API serves both dense and sparse embeddings, including SPLADE, on top of the settled core API.
**Depends on**: Phase 2
**Requirements**: API-01
**Success Criteria** (what must be TRUE):
  1. A common embedder interface produces both dense (minilm) and sparse (splade) embeddings through one consistent API shape.
  2. SPLADE sparse embeddings are produced and validated against the existing golden parity dataset.
  3. Existing `minilm` and `openclip` embedders conform to (or adapt cleanly to) the generalized API with no functional regression in their tests.
**Plans**: TBD

### Phase 4: Documentation
**Goal**: Public packages are comprehensively documented, including per-function maturity, error-string lifetime semantics, and the known environment-leak limitation.
**Depends on**: Phase 3
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04
**Success Criteria** (what must be TRUE):
  1. The `ort` core and each embedder package carry GoDoc plus runnable `Example` functions that `go test` executes.
  2. Documentation states the maturity and testing status of each public API function.
  3. Documentation verifies and states who owns and frees ORT status strings (error-message lifetime semantics).
  4. The `ReleaseEnv`/environment memory-leak limitation is documented in package docs before v1.0.
**Plans**: TBD

### Phase 5: Full Lint Gate
**Goal**: CI enforces the full `golangci-lint` gate with no escape hatch, and the tree passes it cleanly.
**Depends on**: Phase 4
**Requirements**: CLN-01
**Success Criteria** (what must be TRUE):
  1. `continue-on-error` is removed from the `golangci-lint` step in CI.
  2. `golangci-lint run` passes clean on the full tree locally and in CI.
  3. A lint failure blocks the pipeline — the gate is enforcing, not advisory.
**Plans**: TBD

### Phase 6: Release v0.1.0
**Goal**: v0.1.0 is cut as a tagged, documented release with CI green across all supported platforms and every milestone issue closed.
**Depends on**: Phase 5 (and transitively all prior phases)
**Requirements**: REL-01
**Success Criteria** (what must be TRUE):
  1. CI is green across Linux/macOS/Windows × amd64/arm64 on the release commit.
  2. A `v0.1.0` git tag exists with release notes summarizing the milestone.
  3. All v1 milestone issues (#42, #43, #24, #9, #30, #25, #21, #23, #49, #7, #6) are closed.
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. DX & Test Hardening | 0/TBD | Not started | - |
| 2. Core API — Errors & Values | 0/TBD | Not started | - |
| 3. Generalized Embedder API | 0/TBD | Not started | - |
| 4. Documentation | 0/TBD | Not started | - |
| 5. Full Lint Gate | 0/TBD | Not started | - |
| 6. Release v0.1.0 | 0/TBD | Not started | - |
