---
phase: 03-generalized-embedder-api
plan: 01
subsystem: api
tags: [go, generics, embeddings, minilm, splade, openclip, native-verification]

requires:
  - phase: 02-core-api-errors-values
    provides: settled public error, value, ownership, and cleanup contracts
provides:
  - dependency-free `embeddings.Embedder[T any]` contract for dense and sparse embedding rows
  - OpenCLIP document/query text forwarders that retain its concrete text and image API
  - revision-bound native selector evidence with exact PASS and SKIP accounting
affects: [04-documentation, API-01, embeddings]

tech-stack:
  added: []
  patterns:
    - compile-time generic interface conformance and exact public-signature pins
    - durable native JSON evidence bound to the tested Git revision

key-files:
  created:
    - embeddings/embedder.go
    - embeddings/openclip/generalized_embedder.go
    - embeddings/embedder_test.go
    - .planning/phases/03-generalized-embedder-api/evidence/
  modified: []

key-decisions:
  - Keep the root embeddings package as a zero-import generic contract with no factory, registry, or model-specific behavior.
  - Treat revision-bound CI JSON streams as the native gate: 12 named PASS events and zero named SKIP events are required.

patterns-established:
  - Shared embedding abstractions remain compile-time typed; dense and sparse result types are never coerced or wrapped.
  - Environment-dependent native verification is preserved as exact-revision evidence with a non-secret target identifier.

requirements-completed: [API-01]

metrics:
  duration: 1127 min
  completed: 2026-08-02
---

# Phase 03 Plan 01: Generalized Embedder API Summary

**A zero-import generic embedding contract now unifies dense MiniLM/OpenCLIP rows and sparse SPLADE rows, backed by revision-bound native parity evidence.**

## Performance

- **Duration:** 1127 min
- **Started:** 2026-08-01T14:44:51+03:00
- **Completed:** 2026-08-02T06:32:17Z
- **Tasks:** 3/3
- **Files modified:** 9 implementation/evidence files; this summary is plan metadata.

## Accomplishments

- Added the dependency-free `embeddings.Embedder[T any]` contract without changing MiniLM or SPLADE APIs.
- Added direct OpenCLIP document/query text forwarders and external-package conformance, dispatch, forwarding, and signature proofs.
- Preserved six CI-produced evidence files for committed revision `7f7e475f218060438264d18b3844aa79a7b84810`: 12 named PASS events and zero named SKIP events.

## Task Commits

1. **Task 1: Add the leaf generic contract and direct OpenCLIP text forwarders** — `4300965` (feat)
2. **Task 2: Add external conformance and exact-signature regression proof** — `79438c5` (test)
3. **Task 3: Verify the exact-revision native selector gate** — `a087e74` (docs)

## Native Evidence

- Target: `ci:integration-real-model:30735708401`
- Manifest: `.planning/phases/03-generalized-embedder-api/evidence/03-01-native-7f7e475f218060438264d18b3844aa79a7b84810-manifest.env`
- Selectors:
  - `01-minilm` — `TestEmbedDocumentsWithAllMiniLML6V2` (1 PASS)
  - `02-splade-regression` — `TestEmbedDocumentsWithSPLADEModel`, `TestSPLADEGoldenRegressionTopK16WithLabels`, and `TestSPLADERepeatabilityTopK16` (3 PASS)
  - `03-splade-golden` — `TestSPLADEGoldenDatasetParity` (1 PASS)
  - `04-openclip-integration` — six named integration tests (6 PASS)
  - `05-openclip-golden` — `TestOpenCLIPGoldenDatasetParity` (1 PASS)

The manifest records Linux amd64, Go 1.25.12, and an available `ONNXRUNTIME_LIB_PATH`. Each copied evidence file is byte-for-byte identical to the downloaded artifact.

## Verification

- Committed-HEAD preflight passed: the three implementation paths are clean, tracked, and retrievable from `7f7e475f218060438264d18b3844aa79a7b84810`.
- The artifact filenames and manifest `GIT_REVISION` match that committed revision, and the target identifier is non-secret.
- JSON event accounting passed exactly: MiniLM 1, SPLADE regression 3, SPLADE golden 1, OpenCLIP integration 6, and OpenCLIP golden 1; zero named SKIP or FAIL events.
- No named selector was rerun locally.

## Files Created/Modified

- `embeddings/embedder.go` — public typed three-method contract.
- `embeddings/openclip/generalized_embedder.go` — direct OpenCLIP text forwarders.
- `embeddings/embedder_test.go` — external compile-time compatibility proof.
- `.planning/phases/03-generalized-embedder-api/evidence/` — manifest plus five revision-specific JSONL streams.

## Decisions Made

- Kept the root `embeddings` package import-free and free of runtime abstraction machinery.
- Used the configured CI target's exact revision-bound JSON streams as the native verification record.

## Deviations from Plan

None - plan executed exactly as written.

**Total deviations:** 0 auto-fixed.
**Impact on plan:** None.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

API-01's plan output is ready for Phase 4 documentation work. Phase-level completion gates remain with the orchestrator.

## Self-Check: PASSED

- All three implementation files and all six revision-bound evidence files exist.
- Task commits `4300965`, `79438c5`, and `a087e74` exist.
