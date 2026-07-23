---
phase: 2
slug: core-api-errors-values
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-23
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Go standard-library `testing` with the module's Go 1.25 baseline |
| **Config file** | none — commands and CI selection live in `Makefile` and `.github/workflows/ci.yml` |
| **Quick run command** | `go test -short ./ort -run 'Test(ORTError|StatusToError|Value|AdvancedSessionRunWithValues|Diagnostic)'` |
| **Full suite command** | `go test -short ./...` |
| **Estimated runtime** | ~30 seconds quick, ~2 minutes full; native integration runtime varies by environment |

---

## Sampling Rate

- **After every task commit:** Run `go test -short ./ort -run 'Test(ORTError|StatusToError|Value|AdvancedSessionRunWithValues|Diagnostic)'`
- **After every plan wave:** Run `go test -short ./...`
- **After race-sensitive changes:** Run `go test -race ./ort -run 'Test(StatusToError|Diagnostic|AdvancedSessionRunWithValues|ValuesToHandles|TensorDestroy)'`
- **Before `$gsd-verify-work`:** Full short suite, targeted race suite, lint, and native non-race tests must be green
- **Max feedback latency:** 30 seconds for the quick suite

---

## Per-Task Verification Map

The planner must replace the provisional task IDs below with the final PLAN.md task IDs while preserving every behavior and command.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| provisional-error-unit | TBD | 0 | API-02 | T-02-03 | Nonzero native status is copied before release and released exactly once; zero status returns nil | unit/race | `go test -race ./ort -run 'Test(StatusToError|ORTError)'` | ❌ W0 | ⬜ pending |
| provisional-error-native | TBD | 0/1 | API-02 | T-02-03 | Real ORT status preserves code and message through the native ABI without race/checkptr mixing | native integration | `ONNXRUNTIME_LIB_PATH="$ONNXRUNTIME_LIB_PATH" go test ./ort -run TestNativeORTStatusRoundTrip` | ❌ W0 | ⬜ pending |
| provisional-error-chain | TBD | 1 | API-02 | T-02-01 | Validation and lifecycle categories remain inspectable and lower-level causes remain reachable | unit | `go test -short ./ort -run 'Test(ErrorSentinel|Bootstrap.*Error|.*Destroyed|.*NotInitialized)'` | ⚠️ extend existing tests | ⬜ pending |
| provisional-diagnostics | TBD | 0/1 | API-02 | T-02-06 / T-02-07 / T-02-08 | Diagnostics default to silent, reconfigure safely, omit sensitive data, and never duplicate returned errors | unit/race | `go test -race ./ort -run TestDiagnostic` | ❌ W0 | ⬜ pending |
| provisional-value | TBD | 0/1 | API-03 | T-02-01 | Only package-created values cross the FFI boundary; tensor extraction is exact and never coerces | compile/unit | `go test -short ./ort -run TestValue` | ❌ W0 | ⬜ pending |
| provisional-run-values | TBD | 1 | API-03 | Per-call values validate counts and preserve ownership, handle leases, serialization, and lock order | unit/race | `go test -race ./ort -run 'Test(AdvancedSessionRunWithValues|ValuesToHandles|TensorDestroy)'` | ⚠️ extend existing tests | ⬜ pending |
| provisional-run-native | TBD | 1/2 | API-03 | Caller-preallocated per-call tensors produce the expected output against a real model | native integration | `ONNXRUNTIME_LIB_PATH="$ONNXRUNTIME_LIB_PATH" go test ./ort -run TestAdvancedSessionRunWithValuesRealModel` | ⚠️ extend existing fixtures | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `ort/errors_test.go` — fake status store; zero/nonzero conversion; accessor order; message copy; exact release; concurrent conversion; `errors.As`; sentinel wrapping for API-02
- [ ] `ort/errors_native_test.go` — real `CreateStatus` ABI round trip gated by `ONNXRUNTIME_LIB_PATH` for API-02
- [ ] `ort/diagnostics_test.go` — silent default; standard attributes/levels; nil reset; concurrent emit/reconfigure; returned-error zero-emission proof for API-02
- [ ] `ort/value_test.go` — kind check and exact generic extraction matrix for API-03
- [ ] `ort/session_test.go` additions — count validation; supplied handle arrays; bound-path compatibility; borrow/Destroy synchronization; per-call concurrency for API-03
- [ ] Flow-test additions in `environment_test.go`, `memory_test.go`, `tensor_test.go`, `session_test.go`, and `bootstrap_test.go` — `errors.Is`/`errors.As`, preserved causes, and approved diagnostic call sites for API-02
- [ ] `.github/workflows/ci.yml` — run fake-status and diagnostic concurrency tests in the targeted race job; run native-status and `RunWithValues` real-model tests in the existing integration job
- [ ] No framework installation or dependency change is required

---

## Manual-Only Verifications

*None — all phase behaviors have automated verification. Native cases may skip locally when `ONNXRUNTIME_LIB_PATH` is unavailable, but the existing integration CI environment must run them.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verification or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verification
- [x] Wave 0 identifies all currently missing test files and cases
- [x] No watch-mode flags
- [x] Feedback latency target is under 30 seconds
- [ ] Provisional task IDs replaced with final PLAN.md task IDs
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending plan creation and plan-checker verification
