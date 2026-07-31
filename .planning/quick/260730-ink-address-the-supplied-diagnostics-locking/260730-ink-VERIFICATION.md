---
phase: quick-260730-ink-address-the-supplied-diagnostics-locking
verified: 2026-07-30T11:44:05Z
status: human_needed
score: 6/7 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Run the committed NATIVE_SELECTOR in the Linux native CI lane with ONNXRUNTIME_LIB_PATH set."
    expected: "Exactly five tests are discovered and all five run without skips, including float32, float64, int32, and int64 tensor creation/destruction with non-zero native handles."
    why_human: "No ONNX Runtime shared library is installed in the local environment, workspace, or common local library paths; all five native tests therefore skipped locally."
---

# Quick Task 260730-ink Verification Report

**Task Goal:** Address all nine supplied findings covering fail-safe diagnostics, complete lock-order documentation, non-destructive cache validation, environment/session/tensor race coverage, immutable/shared cache usability, explicit Windows trust limitations, Unix negative trust tests, rollback cleanup reporting, and accurate lease comments.

**Verified:** 2026-07-30T11:44:05Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Unconfigured warnings reach stderr, info stays quiet, handler replacement is race-safe, returned errors are not duplicated, and finalizer handler panics fall back safely. | ✓ VERIFIED | `ort/diagnostics.go:21-92` installs a warning-level `os.Stderr` handler via an atomic store and recovers finalizer-handler panics into the emergency path. Diagnostics and returned-error race gates passed. |
| 2 | A post-initialization handler panic rolls back the environment, reports rollback failure, and preserves the original panic value. | ✓ VERIFIED | `ort/environment.go:157-190` captures rollback error/panic, emergency-reports it, then executes `panic(recovered)`. `TestDiagnosticRuntimeVersion/rollback_failure_is_reported_without_masking_handler_panic` passed under `-race`. |
| 3 | Operational cache validation errors preserve the install and original cause; only confirmed invalid/trust results can remove it. | ✓ VERIFIED | `ort/bootstrap.go:343-399` returns operational errors before locking/removal. The only cached-install deletion is `bootstrapRemoveAll(installDir)` inside the confirmed-invalid switch at line 378. Permission and EIO cases passed for both download modes with zero removal and intact sentinels. |
| 4 | Valid read-only/root-baked cache hits need no write or lock, shared trust is explicit, Unix world-write remains rejected, and Windows makes no Unix trust claim. | ✓ VERIFIED | Fast validation precedes `MkdirAll` and lock creation (`ort/bootstrap.go:342-367`). Unix strict/shared policy is implemented in `ort/bootstrap_trust_unix.go:11-36`; the Windows limitation is explicit in `ort/bootstrap_trust_other.go:10-25`. Read-only and Unix negative tests passed; Windows amd64 cross-compilation passed. |
| 5 | Environment teardown waits for in-flight session/tensor use, SessionOptions destruction waits for construction, and concurrent session destruction releases once. | ✓ VERIFIED | `AdvancedSession.runMu -> ortCallMu -> tensor leases` is implemented in `ort/session.go:254-354`; construction holds `SessionOptions.handleMu.RLock` through native creation (`ort/session.go:137-233`). The focused lifecycle race suite passed ten consecutive runs and the committed race selector passed. |
| 6 | Session lease comments and Tensor KeepAlive/Pinner comments match the actual ownership and lifetime behavior. | ✓ VERIFIED | Canonical constructor/run wording is adjacent to the relevant code in `ort/session.go:153-154,320`; Pinner and synchronous KeepAlive purposes are distinguished in `ort/tensor.go:16,128-146`. Source gates and the GC-pressure test passed. |
| 7 | CI selectors run all renamed/new race coverage and real-runtime coverage for every supported tensor type without weakening checkptr. | ? UNCERTAIN | The committed race selector discovered exactly 33 tests and passed under `-race`; no checkptr disable was found. The native selector discovered exactly five tests and is wired after the Linux runtime download, but all five skipped locally because no runtime library is installed. |

**Score:** 6/7 truths verified

## Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `ort/diagnostics.go` | Warning stderr default and panic-safe emergency fallback | ✓ VERIFIED | Substantive implementation; called by finalizers and initialization rollback. |
| `ort/bootstrap.go` | Cache-hit ordering, typed validation dispositions, shared-cache option | ✓ VERIFIED | Missing, confirmed-invalid, and operational outcomes drive distinct branches; the validator seam is used for every Ensure cache validation. |
| `ort/bootstrap_trust_unix.go` | Unix ownership/group/world-write policy | ✓ VERIFIED | Current/root owner strict mode, explicit shared mode, and unconditional world-write rejection are implemented and tested. |
| `ort/bootstrap_trust_other.go` | Explicit Windows/residual non-Unix limitation | ✓ VERIFIED | Build-tagged implementation compiles for Windows and explicitly declines fictitious UID/mode guarantees while leaving neutral integrity checks upstream. |
| `ort/session_test.go` | Deterministic lifecycle, options-borrow, and exact-once tests | ✓ VERIFIED | Tests use callback events and TryLock probes and are included in the committed race selector. |
| `ort/tensor_test.go` | GC-pressure pinning and native supported-type coverage | ✓ VERIFIED | GC-pressure test passed; native type test is substantive and selector-wired, with execution deferred to a runtime-equipped environment. |
| `.github/workflows/ci.yml` | Live-counted race/native selector wiring | ✓ VERIFIED | Exact 33/5 counts, required test symbols, unchanged action pins, and no checkptr disable. |

## Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `ort/diagnostics.go` | `os.Stderr` | Default handler and emergency fallback | ✓ WIRED | Both paths write directly to `os.Stderr`; no use of replaceable `slog.Default`. |
| `ort/environment.go` | `ort/diagnostics.go` | Rollback failure emergency report before original re-panic | ✓ WIRED | Error and rollback-panic branches call `emitEmergencyDiagnostic`, followed by `panic(recovered)`. |
| `ort/environment.go` | `ort/environment_test.go` | AST lock-contract gate | ✓ WIRED | `TestLifecycleLockHierarchyDocumentation` parses the `mu` declaration comment and passed. |
| `ort/bootstrap.go` | `ort/bootstrap_test.go` | Typed disposition and validator seam | ✓ WIRED | Operational EACCES/EIO tests exercise Ensure through `bootstrapValidateCachedRuntimeInstall`. |
| `ort/bootstrap.go` | `ort/bootstrap_test.go` | No-write validated cache hit | ✓ WIRED | `TestBootstrapReadOnlyCacheHit` proves success with no `.locks` path and zero removal calls. |
| `ort/session.go` | `ort/session_test.go` | Borrowed SessionOptions lease | ✓ WIRED | Deterministic native-callback block proves `Destroy` cannot release early. |
| `ort/session.go` | `ort/session_test.go` | Validation-versus-run-lease source gate | ✓ WIRED | Exact source phrases and dead-helper absence are checked. |
| `.github/workflows/ci.yml` | `ort/session_test.go` | Race selector names/count | ✓ WIRED | Required constructor/lifecycle symbols were among exactly 33 discovered tests. |
| `.github/workflows/ci.yml` | `ort/tensor_test.go` | Native selector | ✓ WIRED | `TestTensorSupportedElementTypesWithORT` is among exactly five discovered native tests. |

## Lock and Lifetime Review

- The implemented partial order matches the required contract: `AdvancedSession.runMu -> ortCallMu`; `ortCallMu` precedes global and resource locks; `SessionOptions.handleMu` precedes `mu` when nested; `mu` is released before tensor or memory-info handle locks; tensor and memory-info locks are not nested.
- `AdvancedSession.Run` retains the session mutex, ORT lifecycle read lock, and unique tensor leases for the complete native callback.
- `AdvancedSession.Destroy` owns all session fields under `runMu`, snapshots only the release function under global `mu`, clears fields once, and invokes the native release once.
- `runtime.Pinner` is retained on the Tensor for the native value lifetime and is unpinned on all post-pin construction failures and on Destroy. `runtime.KeepAlive` remains a synchronous call barrier.

## Cache and Trust Review

- The cache is validated before `MkdirAll`, `.locks`, or lock-file opening.
- Raw `Lstat`, `ReadFile`, walk, open, hash, and close failures remain untyped operational errors, preserving `errors.Is`.
- Missing manifest in an existing install, malformed metadata, symlinks, wrong types, ownership/mode rejection, and manifest/file/hash mismatch are explicitly marked confirmed-invalid.
- Cached-install deletion has one production call site and is reachable only from the confirmed-invalid switch. Staging/archive cleanup paths target task-owned temporary data, not a validated cache install.
- Shared mode relaxes owner/group-write policy only; world-write, symlink, type, manifest, metadata, and hash checks remain active.
- Explicit caller paths resolve ordinary symlinks before validating their target; cache-managed paths reject symlinks.

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Full short suite | `go test -count=1 -short ./...` | All packages passed | ✓ PASS |
| Diagnostic and rollback race gate | Plan Task 1 `go test -race` selector | Passed | ✓ PASS |
| Cache/trust gate | Plan Task 2 focused selector | Passed | ✓ PASS |
| Windows implementation compiles | `GOOS=windows GOARCH=amd64 go test -c -o /dev/null ./ort` | Exit 0 | ✓ PASS |
| Lifecycle races repeat deterministically | Plan Task 3 race selector with `-count=10` | Passed all ten runs | ✓ PASS |
| Documentation, pinning, public API | Focused non-race selector | Three tests passed; native test skipped | ⚠ ENVIRONMENT-LIMITED |
| Committed race selector | Discovery count plus `go test -race` | Exactly 33; passed | ✓ PASS |
| Committed native selector | Discovery count plus verbose run | Exactly 5; all skipped due missing runtime | ? NEEDS RUNTIME |
| Copylock and unsafe-pointer vet | `go vet -copylocks ./ort/...`; `go vet -unsafeptr=false ./ort/...` | Both exit 0 | ✓ PASS |
| New-issues lint | `make precommit-lint-new PRECOMMIT_BASE_REF=main` | 0 issues | ✓ PASS |
| Formatting | `gofmt -l` on all changed Go files | No paths | ✓ PASS |

## Probe Execution

Step 7c: SKIPPED — neither the plan nor the repository declares a probe script for this quick task.

## Requirements Coverage

The RF identifiers are local to the authoritative quick-task plan; they are not mapped in `.planning/REQUIREMENTS.md`.

| Requirement | Status | Evidence |
|---|---|---|
| RF-01 | ✓ SATISFIED | Warning stderr default, nil reset, panic fallback, race safety, and no duplicate returned-error diagnostics passed. |
| RF-02 | ✓ SATISFIED | AST lock documentation gate plus deterministic SessionOptions ordering test passed. |
| RF-03 | ✓ SATISFIED | Operational errors preserve cause/cache; only confirmed-invalid cache state reaches deletion. |
| RF-04 | ✓ SATISFIED | Environment teardown waits for session Run and tensor lease under deterministic race coverage. |
| RF-05 | ✓ SATISFIED | Read-only no-write hit and explicit shared-cache option/env precedence passed. |
| RF-06 | ✓ SATISFIED | Windows branch explicitly declines Unix ACL claims and cross-compiles; neutral validation remains upstream. |
| RF-07 | ✓ SATISFIED | Strict and shared Unix modes both reject planted `0777`; strict rejects group-write. |
| RF-08 | ✓ SATISFIED | Rollback close failure is emergency-reported without replacing the original panic sentinel. |
| RF-09 | ✓ SATISFIED | Validation is documented as a local check; unique value leases are documented and implemented as native-Run protection. |

## Review Suggestions and Explicit Deferral

| Item | Status | Evidence |
|---|---|---|
| Finalizer handler-panic stderr fallback | ✓ COVERED | Panic is recovered and original cleanup failure plus handler-panic context is emitted. |
| Explicit-path versus cache symlink documentation | ✓ COVERED | Public comments, README, and explicit/cache tests agree. |
| KeepAlive versus Pinner clarification | ✓ COVERED | Lifetime-specific comments are adjacent to the calls/field. |
| GC-pressure pinning invariant | ✓ COVERED | Pointer/content stability test passed through repeated GC/allocation pressure. |
| Concurrent AdvancedSession double Destroy | ✓ COVERED | 32-way race test passed with one native release. |
| Real-runtime supported tensor types | ? NEEDS RUNTIME | Test covers all four supported types and is CI-wired; local native run skipped. |
| Copylock-sensitive public pointer literals | ✓ COVERED | External-package pointer literals compile and copylocks vet passes. |
| Platform-native status infrastructure | ↪ PLAN-APPROVED DEFERRAL | Existing native-status test remains live in the Linux native selector. The workflow supplies a runtime only in that Ubuntu integration job; adding macOS/Windows runtime provisioning was explicitly outside this quick task. |
| Remove dead `valuesToHandles` | ✓ COVERED | Production helper is absent; renamed tests exercise `acquireValueLeases` plus `handlesFromLeasedValues`. |
| AdvancedSession field-lock comment | ✓ COVERED | `runMu` is documented and used as sole session-field owner; global `mu` only snapshots runtime functions. |

## Anti-Patterns and Policy Checks

| Check | Result | Impact |
|---|---|---|
| Added TODO/FIXME/XXX/HACK/placeholder markers | None | No new stub or debt marker was introduced. |
| Existing TODO in a changed file | `ort/tensor.go:108`, introduced before this task | Warning only: caller-configurable non-CPU allocation is pre-existing future scope, not a stub in this change. |
| Empty/stub implementations | None in the seven-commit diff | No goal artifact is hollow or orphaned. |
| Dependency drift | `go.mod` and `go.sum` diff empty | None. |
| Workflow action-pin drift | No added/removed `uses:` line | None. |
| Planning-state drift | ROADMAP, STATE, and PROJECT diff empty | None. |
| CGO/checkptr weakening | No `import "C"` in `ort`; no checkptr-disable setting | None. |
| Prohibited internal references | Zero in the seven commit messages, plan, and summary | None. |
| Commit scope | Seven linear commits, no merge commit | No merge was performed. |

## Disconfirmation Notes

- The operational permission/EIO behavior is injected at the plan-required validator seam rather than by forcing a real filesystem hash failure. Manual tracing confirms file inspection/read/walk/open/hash errors remain untyped and therefore select the operational, non-destructive branch.
- The GC-pressure test validates reachability and pointer stability, but a non-moving Go collector alone cannot prove why the pointer stayed stable. The required `runtime.Pinner.Pin`/`Unpin` lifecycle was therefore also verified directly in production source.
- The only unexecuted behavior is native-runtime integration. Selector discovery, count, test substance, and CI environment wiring are verified; actual local native calls remain unobserved.

## Human Verification Required

### 1. Runtime-equipped native selector

**Test:** In the Linux native CI lane, after `ONNXRUNTIME_LIB_PATH` is exported, run the committed `NATIVE_SELECTOR`.

**Expected:** Exactly five tests are discovered and none skip. `TestTensorSupportedElementTypesWithORT` creates and destroys float32, float64, int32, and int64 tensors with non-zero native handles, and the other four native tests pass.

**Why human:** The local machine has no ONNX Runtime library in its environment, workspace, or common library paths, so all five tests skipped.

## Gaps Summary

No code-level gap or blocker was found. Automated tests, race coverage, cross-compilation, vet, formatting, lint, dependency/action-pin checks, and policy checks all pass. Status is `human_needed` solely because real native execution requires a runtime-equipped environment.

---

_Verified: 2026-07-30T11:44:05Z_
_Verifier: the agent (gsd-verifier)_
