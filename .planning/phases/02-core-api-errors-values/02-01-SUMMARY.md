---
phase: 02-core-api-errors-values
plan: 01
subsystem: api
tags: [go, purego, onnx-runtime, errors, ffi]

# Dependency graph
requires:
  - phase: 01-dx-test-hardening
    provides: "Established ORT lifecycle locking, race-test selectors, and native status lifetime spike"
provides:
  - "Inspectable ORTError values with Go-owned operation, native code, and message fields"
  - "Lean local error sentinels for invalid, uninitialized, destroyed, and library-not-found states"
  - "Single exact-release native status conversion path with GetErrorCode lifecycle registration"
  - "Race-safe fake ownership proof plus optional Unix real-ABI round-trip coverage"
affects: [02-03, 02-04, 02-05, 02-06, 02-07, 02-08]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Use errors.Is for local sentinels and errors.As for native *ORTError details"
    - "Copy native status data before one deferred ReleaseStatus call"
    - "Caller-held ortCallMu protects registered function pointers throughout conversion"

key-files:
  created:
    - ort/errors.go
    - ort/errors_test.go
    - ort/errors_native_test.go
  modified:
    - ort/environment.go
    - ort/environment_test.go

key-decisions:
  - "Keep native ErrorCode values on ORTError instead of mapping them to local sentinels"
  - "Require callers to hold ortCallMu rather than acquiring another lock inside statusToError"
  - "Use the exact ONNXRUNTIME_LIB_PATH for optional Unix ABI evidence and keep the native loader test out of Windows builds"

patterns-established:
  - "Native status ownership: install deferred release before access, copy code and message, then return a Go-owned ORTError"
  - "Error inspection: errors.Is classifies local lifecycle errors; errors.As exposes native details"

requirements-completed: [API-02]

# Metrics
duration: 14min
completed: 2026-07-24
---

# Phase 2 Plan 01: Native Error Contracts Summary

**Typed, Go-owned ONNX Runtime errors with exact status release ownership, lifecycle-safe function registration, and separate race/native ABI verification paths.**

## Performance

- **Duration:** ~14 min
- **Completed:** 2026-07-24
- **Tasks:** 2
- **Files modified:** 5 (3 created, 2 modified)

## Accomplishments

- Added `ORTError` with inspectable operation, native `ErrorCode`, and copied message fields, without retaining native memory.
- Added four narrow local sentinels while preserving the existing unsupported-platform sentinel as a separate contract.
- Centralized nonzero status ownership so accessors run after the release defer is installed and each status is released exactly once.
- Registered `GetErrorCode` alongside `GetErrorMessage` and `ReleaseStatus`, including both production clear and test reset paths.
- Added 256-worker race coverage, panic-path release coverage, registration/reset coverage, and an environment-gated Unix native ABI round trip.

## Task Commits

Each TDD gate was committed atomically:

1. **Task 1 RED: Add failing error ownership and inspection tests** - `b9f9b10` (test)
2. **Task 1 GREEN: Define ORTError, sentinels, and the injectable ownership helper** - `41822de` (feat)
3. **Task 2 RED: Add failing registration and native ABI tests** - `89c44a0` (test)
4. **Task 2 GREEN: Register GetErrorCode and complete the production adapter** - `bd45ec8` (feat)

## Files Created/Modified

- `ort/errors.go` - public error contracts and the single native-status conversion owner
- `ort/errors_test.go` - synchronized fake-status store, exact-release race tests, and Is/As checks
- `ort/errors_native_test.go` - Unix-only real CreateStatus/GetErrorCode/GetErrorMessage/ReleaseStatus round trip
- `ort/environment.go` - GetErrorCode registration and lifecycle clearing
- `ort/environment_test.go` - complete reset behavior and pointer lifecycle assertions

## Decisions Made

- Native runtime codes remain inspectable through `*ORTError`; they do not implicitly match local sentinels.
- `statusToError` does not acquire a lock. Its documented contract requires the native caller to keep `ortCallMu` held through conversion.
- The native round trip uses only the caller-provided `ONNXRUNTIME_LIB_PATH`. Windows evidence is the cross-platform registration/reset contract plus package compilation, not a claimed DLL round trip.

## Deviations from Plan

None - plan executed exactly as written.

## TDD Gate Compliance

- Task 1 RED failed on the missing `statusOps`, `statusToErrorWithOps`, and `ORTError` contracts; Task 1 GREEN passed the anchored race suite.
- Task 2 RED failed on the missing `getErrorCodeFunc` and production `statusToError`; Task 2 GREEN passed registration, Unix-test gating, and Windows compile checks.
- Git history contains a `test(...)` commit followed by a `feat(...)` commit for each task.

## Verification Evidence

- `go test -race ./ort -run '^(TestStatusToError|TestORTError|TestErrorSentinel)$'` — passed.
- `go test -short ./ort` — passed.
- `GOOS=windows GOARCH=amd64 go test -c -o /dev/null ./ort` — passed.
- The Darwin Unix-loader test ran without `-race` and skipped with its actionable message because `ONNXRUNTIME_LIB_PATH` was unset; no real native round trip is claimed for this run.
- `go.mod` and `go.sum` remain unchanged, and no checkptr override was introduced.

## Issues Encountered

- No ONNX Runtime library path was configured in the execution environment. The optional real-ABI test skipped as designed; fake/race ownership and Windows compile evidence remain complete.

## User Setup Required

None for normal development. Set `ONNXRUNTIME_LIB_PATH` to a valid ONNX Runtime shared library to run the optional real native status round trip.

## Next Phase Readiness

- Plans 02-03 through 02-07 can use the shared error contracts while migrating resource call sites.
- Plan 02-08 must audit that every production status conversion remains inside its required `ortCallMu` lifecycle scope.

## Self-Check: PASSED

- All five created or modified implementation files exist.
- Commits `b9f9b10`, `41822de`, `89c44a0`, and `bd45ec8` are present in git history.

---
*Phase: 02-core-api-errors-values*
*Completed: 2026-07-24*
