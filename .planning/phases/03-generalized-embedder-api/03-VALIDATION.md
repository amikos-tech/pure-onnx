---
phase: 3
slug: generalized-embedder-api
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-08-01
last_updated: 2026-08-01
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Go standard-library `testing` with the module's Go 1.25 baseline |
| **Config file** | none — package tests and command-line selectors are already used |
| **Quick run command** | `go test -count=1 ./embeddings` |
| **Full suite command** | `go test -count=1 -short ./...` |
| **Estimated runtime** | under 30 seconds for the root contract test; under 2 minutes for the full short suite; configured native parity runtime varies |

---

## Sampling Rate

- **After each implementation task:** Run `go test -count=1 ./embeddings` once the root package exists; this is the under-30-second feedback sample.
- **At Task 03-01-03:** Stop at the blocking configured-environment checkpoint because the current shell lacks the native runtime/model/golden assets recorded in `03-RESEARCH.md`.
- **After the checkpoint / before `$gsd-verify-work`:** Run `go test -count=1 -short ./embeddings/...`, `go vet ./embeddings/...`, the complete short module suite, and every named native MiniLM, SPLADE, and OpenCLIP command below as the separate final phase gate. Native evidence must show `PASS`, not `SKIP`.
- **Max feedback latency:** 30 seconds for task-level contract tests; comprehensive and native suites run at the phase gate.

---

## Canonical Threat Register

| Threat ID | STRIDE Category | Severity | Scope | Disposition | Automated Evidence |
|-----------|-----------------|----------|-------|-------------|--------------------|
| T-03-01 | Tampering / Denial of Service | HIGH | OpenCLIP forwarding names bypass existing text validation, state checks, or inference behavior | mitigate | `go test -count=1 ./embeddings -run '^TestOpenCLIPForwardersPreserveExistingValidation$'` plus unchanged OpenCLIP integration/parity suites |
| T-03-02 | Tampering | HIGH | A runtime-tagged or weak result shape accepts the wrong dense/sparse type | mitigate | Compile-time assertions for `Embedder[[]float32]` and `Embedder[splade.SparseVector]` in `embeddings/embedder_test.go` |
| T-03-03 | Denial of Service | HIGH | The generalized contract drops `Close` and loses explicit resource cleanup | mitigate | Compile-time conformance assertions require `Close() error`; exact existing method signatures remain pinned |
| T-03-04 | Denial of Service / Supply-chain expansion | MEDIUM | The root contract becomes a model factory or imports child implementations | mitigate | `test -z "$(go list -f '{{join .Imports " "}}' ./embeddings)"` and source review for a three-method interface only |
| T-03-05 | Denial of Service | MEDIUM | Existing constructors, result types, or model-specific methods regress | mitigate | Exact function-type pins plus MiniLM, SPLADE, OpenCLIP, vet, and full short-module regression commands |

No HIGH threat is accepted or unresolved.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 03-01 | 1 | API-01 | T-03-01 / T-03-02 / T-03-03 / T-03-04 | The zero-import `Embedder[T any]` contract exposes only `EmbedDocuments`, `EmbedQuery`, and `Close`; OpenCLIP forwards directly to its existing text methods | static/build | `test -z "$(go list -f '{{join .Imports " "}}' ./embeddings)" && go test -run '^$' ./embeddings/...` | ❌ W0 production files | ⬜ pending |
| 03-01-02 | 03-01 | 1 | API-01 | T-03-01 / T-03-02 / T-03-03 / T-03-05 | All three concrete embedders satisfy the correct instantiated interface; existing public signatures stay exact; forwarding preserves direct validation | compile/unit | `go test -count=1 ./embeddings` | ❌ W0 `embeddings/embedder_test.go` | ⬜ pending |
| 03-01-03 | 03-01 | 1 | API-01 | T-03-01 / T-03-05 | A configured native target is made available after one fast root-contract sample; comprehensive regression/parity evidence runs only as the final phase gate | checkpoint + fast unit sample | `go test -count=1 ./embeddings -run '^(TestTypedInterfaceDispatchReachesExistingValidation\|TestOpenCLIPForwardersPreserveExistingValidation)$'`; blocking environment checkpoint follows | ✅ existing suites | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

### Native Acceptance Commands

Each command runs with `-json` in the separate final phase-verification gate after Task 03-01-03 resumes. The gate captures each stream and fails unless the listed top-level tests emit the exact PASS count and none emit `Action:"skip"`; package exit status alone is not evidence.

| Capability | Automated Command | Required named PASS events | SKIP allowed |
|------------|-------------------|----------------------------|--------------|
| MiniLM dense document/query behavior | `go test -count=1 -json ./embeddings/minilm -run '^TestEmbedDocumentsWithAllMiniLML6V2$'` | 1 | no |
| SPLADE sparse regression and repeatability | `go test -count=1 -json ./embeddings/splade -run '^(TestEmbedDocumentsWithSPLADEModel|TestSPLADEGoldenRegressionTopK16WithLabels|TestSPLADERepeatabilityTopK16)$'` | 3 | no |
| SPLADE hosted golden parity | `go test -count=1 -json ./embeddings/splade -run '^TestSPLADEGoldenDatasetParity$'` | 1 | no |
| OpenCLIP text/image integration behavior | `go test -count=1 -json ./embeddings/openclip -run '^(TestEmbedTextsAndImagesWithOpenCLIPModel|TestOpenCLIPFailsWithWrongInputOutputNames|TestOpenCLIPFailsWithWrongEmbeddingDimension|TestOpenCLIPFailsWithImageSizeMismatch|TestOpenCLIPErrorsAfterClose|TestOpenCLIPCloseIsIdempotent)$'` | 6 | no |
| OpenCLIP hosted golden parity | `go test -count=1 -json ./embeddings/openclip -run '^TestOpenCLIPGoldenDatasetParity$'` | 1 | no |

---

## Wave 0 Requirements

- [ ] `embeddings/embedder.go` — dependency-free public generic contract for API-01.
- [ ] `embeddings/openclip/generalized_embedder.go` — direct document/query forwarding methods for API-01.
- [ ] `embeddings/embedder_test.go` — external-package exact-signature, generic-conformance, typed-dispatch, and forwarding proof for API-01.
- [x] Existing Go test infrastructure, MiniLM integration coverage, SPLADE regression/parity suites, and OpenCLIP integration/parity suites need no new dependency, fixture, or configuration.

---

## Manual-Only Verifications

*None — Task 03-01-03 requires the user only to make an existing configured native target accessible; the executor runs every verification command. Native tests may skip in an unconfigured local shell, but the final phase gate requires non-skipped output from the configured environment.*

---

## Validation Sign-Off

- [x] Every implementation task has under-30-second automated feedback; Task 03-01-03 has a fast sample plus an explicit blocking environment checkpoint before the comprehensive automated phase gate.
- [x] Sampling continuity: no three consecutive tasks lack automated verification.
- [x] Wave 0 identifies every currently missing production/test file.
- [x] No watch-mode flags.
- [x] Task-level feedback latency target is under 30 seconds.
- [x] Provisional task IDs reconciled with `03-01-PLAN.md` tasks 03-01-01 through 03-01-03.
- [x] `nyquist_compliant: true` set in frontmatter after plan-checker approval.

**Approval:** approved 2026-08-01 (gsd-plan-checker verified Phase 3 plan structure, sampling, native checkpoint, and threat coverage)
