---
phase: 02-core-api-errors-values
plan: 04
subsystem: api
tags: [go, onnx-runtime, ffi, errors, diagnostics, concurrency]

# Dependency graph
requires:
  - phase: 02-core-api-errors-values
    provides: "Public error contracts, sealed Value inspection, and structured finalizer diagnostics from Plans 01-03"
provides:
  - "RunWithValues for caller-owned per-call input and output tensors"
  - "One serialized session run core shared by bound and supplied values"
  - "Typed native session failures and finalizer-only structured warnings"
affects: [02-05, 02-06, 02-07, 02-08, examples, embeddings]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Borrow caller-supplied Values for one run without storing or transferring ownership"
    - "Convert every session status once through statusToError and wrap it with %w"

key-files:
  created: []
  modified:
    - ort/session.go
    - ort/session_test.go

key-decisions:
  - "Keep NewAdvancedSession and Run constructor bindings intact while selecting RunWithValues arguments only inside the shared locked core"
  - "Use stable native operation names and emit diagnostics only when a session finalizer cannot return its Destroy error"

patterns-established:
  - "Both public session run APIs use runMu -> ortCallMu.RLock -> mu -> Tensor.runMu with deduplicated leases and post-call KeepAlive barriers"
  - "Returned session failures remain silent and machine-inspectable through errors.Is or errors.As"

requirements-completed: [API-02, API-03]

# Metrics
duration: 18min
completed: 2026-07-24
---

# Phase 2 Plan 04: Shared Session Run and Error Contracts Summary

**Caller-owned per-call inference through a shared serialized run core, with typed native errors and finalizer-only structured diagnostics.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-07-24T12:30:05Z
- **Completed:** 2026-07-24T12:48:17Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added `RunWithValues(inputs, outputs []Value) error` without changing constructor-bound `Run()` behavior or storing borrowed values on the session.
- Unified both public methods behind one lock-, lease-, FFI-, and lifetime-safe private run core.
- Categorized local failures with public sentinels and preserved native CreateSessionOptions, CreateSession, and Run details as wrapped `*ORTError` values.
- Replaced session finalizer text logging with one structured Warn diagnostic while keeping every returned failure silent.
- Added fake-callback, race, diagnostic, status-lifecycle, caller-ownership, and real-model integration coverage.

## Task Commits

Both TDD tasks were committed in atomic RED/GREEN gates:

1. **Task 02-04-01 RED: Add failing RunWithValues session tests** - `226cd72` (test)
2. **Task 02-04-01 GREEN: Add shared per-call session run path** - `dacbc73` (feat)
3. **Task 02-04-02 RED: Add failing session error and diagnostic tests** - `da82dea` (test)
4. **Task 02-04-02 GREEN: Convert session errors and finalizer diagnostics** - `9b92357` (feat)

## Files Created/Modified

- `ort/session.go` - Shared bound/per-call run core, public sentinel wrapping, centralized status conversion, and structured session finalizer handling.
- `ort/session_test.go` - RunWithValues behavior, concurrency, ownership, typed-error, diagnostic-policy, and real-model coverage.

## Decisions Made

- Fixed cloned input/output names remain session-owned; only value slices vary per call.
- Supplied values are borrowed until native return and lease release, then remain entirely caller-owned.
- The stable native operation fields are `create session options`, `create session`, and `run inference`; contextual compatibility text wraps each error with `%w`.
- A private finalizer adapter is used so its failure-only diagnostic policy is deterministic to test while remaining the function installed with `runtime.SetFinalizer`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed a checkptr-unsafe test assertion**
- **Found during:** Task 02-04-01 GREEN verification
- **Issue:** The first fake callback decoded a Go-backed C-string address under `-race`, triggering Go's checkptr guard even though the production run path was correct.
- **Fix:** Asserted non-null name pointers and unchanged fixed session names without converting the raw address back to a string.
- **Files modified:** `ort/session_test.go`
- **Verification:** The complete targeted `go test -race` selector passed.
- **Committed in:** `dacbc73`

---

**Total deviations:** 1 auto-fixed (1 Rule 1)
**Impact on plan:** The fix kept the required name and compatibility coverage while making the test safe for the mandated race/checkptr lane. No production scope changed.

## TDD Gate Compliance

- Task 1 RED failed because `RunWithValues` did not exist; its GREEN commit then passed the full session/value race selector.
- Task 2 RED failed on the missing finalizer adapter; its GREEN commit then passed the error, diagnostic, and per-call selectors.
- Git history contains a `test(02-04)` commit before each corresponding `feat(02-04)` commit; no refactor commit was needed.

## Verification Evidence

- Task 1 targeted `go test -race ./ort` selector — passed.
- Task 2 focused short unit selector — passed.
- Native real-model selector — exited 0; native cases skipped locally because `ONNXRUNTIME_LIB_PATH` was unset.
- `go test -short ./ort` — passed.
- `go test -run '^$' ./examples/... ./embeddings/...` — passed, confirming unchanged consumers compile.
- Source audits confirmed both public methods call the same private core, session status paths contain no direct message/release calls, and the old finalizer warning call is absent.
- The inference example and all three embedder source files remained unmodified.

## Known Stubs

None.

## Threat Flags

None - all new FFI, lifecycle, status, and diagnostic surfaces were explicitly covered by the plan threat model.

## Issues Encountered

- The native ABI lane could not execute locally without `ONNXRUNTIME_LIB_PATH`; its repository-standard skip path completed successfully.

## User Setup Required

None. Set `ONNXRUNTIME_LIB_PATH` only when running the optional native integration lane locally.

## Next Phase Readiness

- Plans 02-05 through 02-07 can apply the same public sentinel, status conversion, and finalizer diagnostic patterns to other resource types.
- Plan 02-08 can add the final anchored integration gates with both session run APIs already covered.
- No implementation blockers remain for the next plan.

## Self-Check: PASSED

- Both modified source files and this summary exist.
- Commits `226cd72`, `dacbc73`, `da82dea`, and `9b92357` are present in git history.
- Shared-core, status-source, and unchanged-consumer audits passed after the final GREEN commit.

---
*Phase: 02-core-api-errors-values*
*Completed: 2026-07-24*
