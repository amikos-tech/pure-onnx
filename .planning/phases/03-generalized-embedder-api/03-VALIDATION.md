---
phase: 3
slug: generalized-embedder-api
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-01
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

- **After every task commit:** Run `go test -count=1 ./embeddings` once the root package exists.
- **After the plan wave:** Run `go test -count=1 -short ./embeddings/... && go vet ./embeddings/...`.
- **Before `$gsd-verify-work`:** Prove the root package has zero production imports, run `go test -count=1 -short ./...`, then run every named native MiniLM, SPLADE, and OpenCLIP command below in the configured integration environment. Native evidence must show `PASS`, not `SKIP`.
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
| 03-01-03 | 03-01 | 1 | API-01 | T-03-01 / T-03-05 | Dense, sparse, text, and image behavior remains unchanged; SPLADE parity runs rather than silently skipping | static/regression/native parity | `go vet ./embeddings/... && go test -count=1 -short ./...` plus the named native commands below | ✅ existing suites | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

### Native Acceptance Commands

| Capability | Automated Command |
|------------|-------------------|
| MiniLM dense document/query behavior | `go test -count=1 -v ./embeddings/minilm -run '^TestEmbedDocumentsWithAllMiniLML6V2$'` |
| SPLADE sparse regression and repeatability | `go test -count=1 -v ./embeddings/splade -run '^(TestEmbedDocumentsWithSPLADEModel|TestSPLADEGoldenRegressionTopK16WithLabels|TestSPLADERepeatabilityTopK16)$'` |
| SPLADE hosted golden parity | `go test -count=1 -v ./embeddings/splade -run '^TestSPLADEGoldenDatasetParity$'` |
| OpenCLIP text/image integration behavior | `go test -count=1 -v ./embeddings/openclip -run '^(TestEmbedTextsAndImagesWithOpenCLIPModel|TestOpenCLIPFailsWithWrongInputOutputNames|TestOpenCLIPFailsWithWrongEmbeddingDimension|TestOpenCLIPFailsWithImageSizeMismatch|TestOpenCLIPErrorsAfterClose|TestOpenCLIPCloseIsIdempotent)$'` |
| OpenCLIP hosted golden parity | `go test -count=1 -v ./embeddings/openclip -run '^TestOpenCLIPGoldenDatasetParity$'` |

---

## Wave 0 Requirements

- [ ] `embeddings/embedder.go` — dependency-free public generic contract for API-01.
- [ ] `embeddings/openclip/generalized_embedder.go` — direct document/query forwarding methods for API-01.
- [ ] `embeddings/embedder_test.go` — external-package exact-signature, generic-conformance, typed-dispatch, and forwarding proof for API-01.
- [x] Existing Go test infrastructure, MiniLM integration coverage, SPLADE regression/parity suites, and OpenCLIP integration/parity suites need no new dependency, fixture, or configuration.

---

## Manual-Only Verifications

*None — all phase behaviors have automated commands. Native tests may skip in an unconfigured local shell, but the phase gate requires non-skipped output from the configured integration environment.*

---

## Validation Sign-Off

- [ ] All final PLAN.md tasks have automated verification or create their missing Wave 0 files.
- [x] Sampling continuity: no three consecutive tasks lack automated verification.
- [x] Wave 0 identifies every currently missing production/test file.
- [x] No watch-mode flags.
- [x] Task-level feedback latency target is under 30 seconds.
- [ ] Provisional task IDs reconciled with the final PLAN.md.
- [ ] `nyquist_compliant: true` set in frontmatter after plan-checker approval.

**Approval:** pending
