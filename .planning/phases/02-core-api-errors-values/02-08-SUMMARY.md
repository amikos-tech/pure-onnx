---
phase: 02-core-api-errors-values
plan: 08
subsystem: api
tags: [go, onnx-runtime, errors, diagnostics, race, github-actions]

# Dependency graph
requires:
  - phase: 02-core-api-errors-values
    provides: "Central status conversion, structured diagnostics, and migrated environment/session/tensor/memory/bootstrap call sites from Plans 01-07"
provides:
  - "One seven-site status ownership path with obsolete wrappers and finalizer logging facade removed"
  - "Exact 29-test native-free race lane with selector-liveness enforcement"
  - "Exact four-test native ABI and real-model lane with selector-liveness enforcement"
affects: [phase-03, ci, diagnostics, onnx-runtime-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Count exact anchored go test selectors before executing CI lanes"
    - "Keep fake callback ownership tests under race and real native ABI tests outside race"
    - "Use narrow documented lint suppressions for intentional boundary tests"

key-files:
  created: []
  modified:
    - .github/workflows/ci.yml
    - ort/environment.go
    - ort/environment_test.go
    - ort/session_test.go
    - ort/diagnostics_test.go
    - ort/errors_native_test.go
    - ort/finalizer_log.go

key-decisions:
  - "Fail CI unless the exact race and native selectors resolve to 29 and 4 top-level tests respectively"
  - "Keep call-site status probes checkptr-safe with null message pointers while central and native tests retain non-empty message-copy proof"
  - "Preserve intentional native pointer and nil-context tests with line-scoped lint annotations instead of weakening repository lint"

patterns-established:
  - "CI selector liveness is an executable coverage contract, not a comment or naming convention"
  - "Race and native ABI validation remain separate without disabling checkptr"

requirements-completed: [API-02, API-03]

# Metrics
duration: 18min
completed: 2026-07-24
---

# Phase 2 Plan 08: Convergence and CI Validation Summary

**Centralized seven-site status ownership with dead helper removal, a 29-test checkptr-safe race lane, and a four-test native ABI lane guarded by selector-liveness counts.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-07-24T14:04:45Z
- **Completed:** 2026-07-24T14:23:11Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Deleted the unused finalizer logging facade and removed the obsolete status-message and status-release wrappers while retaining the registered function pointers used by `statusToError`.
- Confirmed all seven production native status sites use the centralized converter under caller-held lifecycle protection and all three finalizers use the panic-containing structured diagnostic path.
- Expanded the native-free race lane to the exact 29-test Phase 2 ownership, error, diagnostic, and concurrency contract with a fail-fast liveness count.
- Added a separate exact four-test native ABI and real-model selector to the existing runtime-backed integration job without changing action pins, dependency versions, lint behavior, or checkptr settings.
- Closed the two earlier Phase 2 changed-code lint findings with narrowly documented test-only annotations; the convergence lint gate now reports zero issues.

## Task Commits

Each task and convergence fix was committed atomically:

1. **Task 02-08-01: Remove obsolete status helpers and converge call-site audits** - `a98e714` (refactor)
2. **Task 02-08-02: Wire the race/native CI split and preserve compatibility gates** - `645fdf1` (chore)
3. **Convergence fix: Document intentional boundary-test lint exceptions** - `c1547b0` (test)

## Files Created/Modified

- `.github/workflows/ci.yml` - Exact anchored race/native selectors, printed selected test names, and fail-fast counts of 29 and 4.
- `ort/environment.go` - Removed the obsolete `getErrorMessage` and `releaseStatus` wrappers while retaining status callback globals.
- `ort/environment_test.go` - Removed wrapper-specific tests that no longer described the public ownership path.
- `ort/session_test.go` - Kept the session call-site ownership probe checkptr-safe by using the established null-message callback pattern.
- `ort/diagnostics_test.go` - Documented the intentional nil-context fallback test for changed-code lint.
- `ort/errors_native_test.go` - Documented the intentional real native C API table conversion for changed-code lint.
- `ort/finalizer_log.go` - Deleted the unused formatting/logging compatibility facade.
- `.planning/phases/02-core-api-errors-values/deferred-items.md` - Marked both prior lint findings resolved.

## Decisions Made

- Exact selector counts are checked before test execution. A renamed or missing top-level test now fails CI instead of silently reducing coverage.
- Pure-Go callback, ownership, diagnostic, and concurrency tests stay in the race lane. Tests that cross the real ONNX Runtime ABI stay in the existing non-race integration lane.
- The central converter and native round-trip tests retain non-empty message-copy proof. Resource call-site tests use null messages where their purpose is exact release and wrapping, avoiding unsafe Go-pointer round trips under checkptr.
- Intentional test-boundary constructs receive line-scoped explanations. Global lint configuration, workflow enforcement, and vet defaults remain unchanged.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Made the session status call-site probe checkptr-safe**

- **Found during:** Task 02-08-01 full 29-test race convergence run
- **Issue:** The prerequisite session probe returned a Go-backed C-string address through `uintptr`, which failed checkptr when the complete selector ran under race.
- **Fix:** Switched that call-site probe to a null message pointer and asserted the empty message; the central converter and real native round-trip tests continue to prove non-empty message copying.
- **Files modified:** `ort/session_test.go`
- **Commit:** `a98e714`

**2. [Rule 3 - Blocking] Resolved the changed-code lint gate**

- **Found during:** Plan-level `make precommit-lint-new`
- **Issue:** Earlier Phase 2 tests intentionally convert a native C API address and pass nil context to verify fallback behavior, but lacked the narrow annotations required by the enforcing changed-code lint gate.
- **Fix:** Added documented line-scoped `govet` and `staticcheck` suppressions without changing test or production behavior.
- **Files modified:** `ort/errors_native_test.go`, `ort/diagnostics_test.go`
- **Commit:** `c1547b0`

## Verification Evidence

- Both exact task-level automated verification commands passed; the CI-focused check completed in 8 seconds.
- `go test -short ./...`, `go test -short ./ort`, and the fast Phase 2 selector passed.
- The exact 29-test selector passed under `go test -race`; the exact four-test native selector resolved and its local command exited successfully.
- `go test -run '^$' ./...` compiled every package and example without running tests.
- Separate `go doc` lookups resolved `ParseShape`, `ShapeElementCount`, `Value`, `IsTensor`, `AsTensor`, `AdvancedSession.RunWithValues`, and `SetDiagnosticHandler`.
- The focused bootstrap file-permission regression passed.
- `go vet -unsafeptr=false ./ort/...` and `make precommit-lint-new` passed; changed-code lint reported zero issues.
- Source audits found seven centralized status conversions, three finalizer diagnostic call sites, no obsolete production wrappers, and no direct production `log.Printf` or `logFinalizerWarning`.
- `go.mod`, `go.sum`, workflow `uses:` lines, lint `continue-on-error`, and checkptr settings remained unchanged.

## Known Stubs

None.

## Deferred Issues

None.

## Issues Encountered

- The full race selector exposed an unsafe test-only pointer round trip that narrower prerequisite runs had not combined with checkptr.
- The final changed-code lint gate surfaced the two previously deferred intentional test constructs; both are now locally documented and the gate is green.

## User Setup Required

None - CI configures ONNX Runtime before the native integration lane. Local native execution remains optional when `ONNXRUNTIME_LIB_PATH` is not set.

## Next Phase Readiness

- Phase 2's public errors, values, status ownership, structured diagnostics, and bootstrap integrity contracts are fully converged.
- CI now protects both native-free race behavior and real-runtime ABI/model behavior with live exact selectors.
- No Phase 2 blocker or deferred lint item remains; Phase 3 can build on the finalized core API.

## Self-Check: PASSED

- All six retained implementation files, the CI workflow, this summary, and the intentional `ort/finalizer_log.go` deletion were verified on disk.
- Commits `a98e714`, `645fdf1`, and `c1547b0` are present in git history.
- Fresh short, race, compile, API-documentation, bootstrap-permission, vet, changed-code lint, source-audit, module-integrity, and workflow-integrity checks passed.

---
*Phase: 02-core-api-errors-values*
*Completed: 2026-07-24*
