---
phase: 1
slug: dx-test-hardening
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-21
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Go stdlib `testing` (+ `testify/require` for assertions not touched by this phase) |
| **Config file** | none — `go test` built-in; linter config at `.golangci.yml` |
| **Quick run command** | `go test -race -run 'TestAdvancedSessionRunAndDestroyConcurrent\|TestTensorDestroyWaitsForInFlightRun\|TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun' ./ort/...` |
| **Full suite command** | `make precommit` (fmt, vet, lint-new, gosec, `go test ./...`, mod-tidy, vulncheck) |
| **Estimated runtime** | ~30 seconds (quick), ~3-5 minutes (full precommit) |

---

## Sampling Rate

- **After every task commit:** Run `go test -race -run '<test-being-changed>' ./ort/...`
- **After every plan wave:** Run `make precommit` plus `go test -race -run TestStress -count=5 ./ort/...`
- **Before `/gsd:verify-work`:** Full `make precommit` green, plus a manual/CI run of the new dedicated stress job
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 0 | DX-01 | — / N/A | Unsupported-platform bootstrap failure produces GOOS/GOARCH + `ONNXRUNTIME_LIB_PATH` hint; other bootstrap failures unchanged | unit | `go test ./examples/inference/... -run TestDiagnosticFor` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 0/1 | DX-01 | — / N/A | `ort.ErrUnsupportedPlatform` correctly wraps `resolveRuntimeArtifact`'s unsupported-combo error | unit | `go test ./ort/... -run TestResolveRuntimeArtifact` | ✅ | ⬜ pending |
| 01-02-01 | 01 | 1 | TST-01 | — / N/A | 3 named tests assert via channel/handshake + event order, no sleep-based primary assertions remain | unit/race | `go test -race -run 'TestAdvancedSessionRunAndDestroyConcurrent\|TestTensorDestroyWaitsForInFlightRun\|TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun' ./ort/...` | ✅ | ⬜ pending |
| 01-03-01 | 01 | 0/1 | TST-02 | — / N/A | Stress tests exercise many concurrent init/destroy cycles and pass under `-race` | stress/race | `go test -race -run TestStress -count=10 ./ort/...` (quick local sanity; CI uses `-count=50`) | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `ort/environment_stress_test.go` — covers TST-02 (new file, does not exist)
- [ ] `examples/inference/main_test.go` — covers DX-01 (new file, does not exist; requires extracting `diagnosticFor` or equivalent testable helper out of `main()`)
- [ ] `Makefile`/`ci.yml` `-short` wiring — needed for D-06's "skip by default" to actually hold

---

## Manual-Only Verifications

*None — all phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
