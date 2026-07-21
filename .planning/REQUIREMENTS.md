# Requirements: onnx-purego (pure-onnx) — v0.1.0

**Defined:** 2026-07-21
**Core Value:** Run ONNX Runtime inference from Go with zero CGO.

> "User" here means a Go developer consuming this library. Requirements map to the
> open v0.1.0 milestone issues (GitHub #). This is a brownfield hardening milestone.

## v1 Requirements

### Developer Experience

- [ ] **DX-01**: When ONNX Runtime cannot be bootstrapped on an unsupported platform, the inference example exits fast with an actionable message that names `GOOS`/`GOARCH` and instructs the user to set `ONNXRUNTIME_LIB_PATH` (#42)

### Testing & Concurrency

- [ ] **TST-01**: Concurrency tests assert correctness via deterministic synchronization instead of timing/sleep-based checks (#43)
- [ ] **TST-02**: Stress tests exercise concurrent `InitializeEnvironment`/`DestroyEnvironment` cycles and pass under the race detector (#24)

### Documentation

- [ ] **DOC-01**: Public packages carry comprehensive GoDoc and runnable examples covering the `ort` core and each embedder (#9)
- [ ] **DOC-02**: Documentation states the maturity and testing status of each public API function (#30)
- [ ] **DOC-03**: Error-message string lifetime semantics are verified and documented — who owns and frees ORT status strings (#25)
- [ ] **DOC-04**: The `ReleaseEnv`/environment memory-leak limitation is documented before v1.0 (#21)

### Quality Gate

- [ ] **CLN-01**: `golangci-lint` runs without `continue-on-error` in CI and the tree passes the full lint gate (#23)

### Public API

- [ ] **API-01**: A generalized embedder API supports both dense and sparse embeddings, including SPLADE (#49)
- [ ] **API-02**: The public API returns comprehensive, wrapped errors with actionable context across environment, tensor, session, and bootstrap flows (#7)
- [ ] **API-03**: A `Value` interface enables polymorphic tensor handling for session inputs and outputs (#6)

### Release

- [ ] **REL-01**: v0.1.0 is cut as a git tag with release notes, with CI green on all supported platforms (Linux/macOS/Windows × amd64/arm64) and all v1 requirements closed

## v2 Requirements

Deferred beyond v0.1.0. Tracked, not in this roadmap.

### OpenCLIP

- **OCLIP-01**: End-to-end OpenCLIP support tracker (#68)
- **OCLIP-02**: Tighten numerical equivalence against pinned ONNX artifacts (#76)

### Tooling

- **TOOL-01**: tree-sitter-based robust C API parsing and auto-generation (#29)
- **PHASE2-01**: Advanced features / Phase 2 (#10)

## Out of Scope

| Feature | Reason |
|---------|--------|
| CGO / `import "C"` in `ort/` | Defeats the project's core value (no C compiler, cross-compilation) |
| `ort` runtime/library-loading changes for #42 | Issue is scoped "example UX only; no ort runtime changes" |
| OpenCLIP end-to-end (#68) and numeric-equivalence (#76) | Separate milestone; not gating v0.1.0 |
| tree-sitter C API generation (#29) | Future tooling milestone |

## Traceability

Populated during roadmap creation. Each requirement maps to exactly one phase.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DX-01 | — | Pending |
| TST-01 | — | Pending |
| TST-02 | — | Pending |
| DOC-01 | — | Pending |
| DOC-02 | — | Pending |
| DOC-03 | — | Pending |
| DOC-04 | — | Pending |
| CLN-01 | — | Pending |
| API-01 | — | Pending |
| API-02 | — | Pending |
| API-03 | — | Pending |
| REL-01 | — | Pending |

**Coverage:**
- v1 requirements: 12 total
- Mapped to phases: 0 (roadmap pending)
- Unmapped: 12 ⚠️

---
*Requirements defined: 2026-07-21*
*Last updated: 2026-07-21 after initialization*
