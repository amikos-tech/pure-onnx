---
phase: 02-core-api-errors-values
plan: 05
subsystem: api
tags: [go, onnx-runtime, ffi, errors, tensors, diagnostics]

# Dependency graph
requires:
  - phase: 02-core-api-errors-values
    provides: "Typed native errors, sealed Value ownership, and structured finalizer diagnostics from Plans 01-03"
provides:
  - "Inspectable ParseShape and ShapeElementCount validation with preserved strconv causes"
  - "Typed tensor validation, lifecycle, and native status failures"
  - "Finalizer-only tensor diagnostics with preserved pinning and run-lease ownership"
affects: [02-06, 02-07, 02-08, tensors, sessions]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Use errors.Is for local tensor/shape categories and errors.As for native ORTError detail"
    - "Keep tensor data pinned and alive through native creation, then release status and memory ownership exactly once"

key-files:
  created: []
  modified:
    - ort/shape_parse.go
    - ort/shape_test.go
    - ort/tensor.go
    - ort/tensor_test.go

key-decisions:
  - "Preserve strconv.NumError alongside ErrInvalidArgument by wrapping both causes in one ParseShape error"
  - "Require the complete tensor/status callback set before crossing the native creation boundary"
  - "Use a null-message fake status in the race lane so ownership checks do not route a Go heap pointer through uintptr"

patterns-established:
  - "Shape parsing/counting keeps actionable dimension context while exposing ErrInvalidArgument"
  - "Tensor finalizers call a deterministic adapter that emits one structured warning only when Destroy cannot return its failure"

requirements-completed: [API-02, API-03]

# Metrics
duration: 12min
completed: 2026-07-24
---

# Phase 2 Plan 05: Shape and Tensor Error Contracts Summary

**Inspectable shape and tensor failures with typed native status ownership, preserved pinned-memory lifetimes, and finalizer-only structured diagnostics.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-07-24T12:54:05Z
- **Completed:** 2026-07-24T13:06:02Z
- **Tasks:** 1
- **Files modified:** 4

## Accomplishments

- Made every invalid `ParseShape` and `ShapeElementCount` result match `ErrInvalidArgument`, while retaining `*strconv.NumError` for failed integer parsing.
- Made tensor validation, missing-runtime, destroyed-value, and release-unavailable failures inspectable through the public sentinels without losing actionable context.
- Routed both tensor creation statuses through `statusToError`, preserving native code/operation detail and exact one-release ownership.
- Preserved idempotent and concurrent `Destroy`, run leases, memory-info cleanup, pin/unpin behavior, and post-native shape/data lifetime barriers.
- Replaced tensor finalizer text logging with one structured Warn diagnostic while keeping returned tensor failures silent.

## Task Commits

The TDD task was committed in atomic RED/GREEN gates:

1. **Task 02-05-01 RED: Add failing shape and tensor error tests** - `2c96cc6` (test)
2. **Task 02-05-01 GREEN: Migrate shape and tensor error contracts** - `665919e` (feat)

## Files Created/Modified

- `ort/shape_parse.go` - Public invalid-argument wrapping with preserved strconv parsing causes.
- `ort/shape_test.go` - Exported parse/count Is/As coverage, overflow cases, and width-conditional dimension validation.
- `ort/tensor.go` - Sentinel wrapping, centralized native status conversion, lifetime barriers, and structured finalizer cleanup.
- `ort/tensor_test.go` - Validation/lifecycle categories, exact status release, cleanup, diagnostics, and race-safe fake callbacks.

## Decisions Made

- `ParseShape` uses one multi-wrap `fmt.Errorf` call for integer failures so callers can independently match the public category and inspect the underlying `*strconv.NumError`.
- Tensor creation checks all functions needed for native calls and status conversion before crossing the FFI boundary.
- The race-lane tensor status probe returns a null message pointer. It still proves operation/code/exact release through the production converter without violating checkptr; non-empty copy-before-release is already covered by the prerequisite `TestStatusToError` seam and native ABI lane.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Replaced a checkptr-unsafe fake native message**
- **Found during:** Task 02-05-01 GREEN race verification
- **Issue:** The first tensor status probe returned a Go-backed C-string address through `uintptr`, which the required race/checkptr lane correctly rejected.
- **Fix:** Used a null-message fake status for production call-site ownership assertions, leaving non-empty copy semantics to the existing pure-Go and native ABI proofs.
- **Files modified:** `ort/tensor_test.go`
- **Verification:** The exact task-level `go test -race` selector passed.
- **Committed in:** `665919e`

**2. [Rule 2 - Missing Critical] Added an explicit tensor-data KeepAlive barrier**
- **Found during:** Task 02-05-01 source-lifetime audit
- **Issue:** The existing creation path kept shape memory alive after the native call but lacked the plan-required explicit barrier for the pinned data slice.
- **Fix:** Added `runtime.KeepAlive(data)` immediately after `CreateTensorWithDataAsOrtValue`, before any failure-path unpin.
- **Files modified:** `ort/tensor.go`
- **Verification:** Source audit confirmed both shape and data barriers remain after the native call; the targeted race and short suites passed.
- **Committed in:** `665919e`

---

**Total deviations:** 2 auto-fixed (1 Rule 1, 1 Rule 2)
**Impact on plan:** Both fixes strengthen the mandated race and native-memory safety evidence without changing the public API or ownership model.

## TDD Gate Compliance

- RED failed on the intentionally missing `finalizeTensor` adapter before production code changed.
- GREEN passed the exact targeted race selector, both exported `go doc` checks, the focused short suite, and unchanged-consumer compilation.
- Git history contains `test(02-05)` before `feat(02-05)`; no refactor commit was needed.

## Verification Evidence

- Exact task-level `go test -race ./ort` selector — passed.
- `go doc ./ort.ParseShape` — passed.
- `go doc ./ort.ShapeElementCount` — passed.
- `go test -short ./ort -run 'Test(ParseShape|ShapeElementCount|NewTensor|NewEmptyTensor|Tensor|Value)'` — passed.
- `go test -short ./ort` — passed.
- `go test -run '^$' ./...` — passed for all unchanged consumers.
- Source audits found no direct tensor status message/release block, old finalizer logger, or test `t.Parallel`; all new global-state tests restore state with cleanup.
- `ort/types.go`, `go.mod`, and `go.sum` remained unchanged.

## Known Stubs

- `ort/tensor.go:104` - The pre-existing TODO for caller-configurable allocator/memory type remains intentionally deferred until non-CPU providers are exposed. It does not affect this plan's error or ownership guarantees.

## Issues Encountered

- The first fake status message used a Go-backed pointer that was invalid for the race/checkptr lane. Replacing it with the race-safe null-message probe kept the call-site ownership proof deterministic.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02-06 can reuse the same sentinel, status ownership, lifecycle-lock, and finalizer diagnostic patterns for environment and MemoryInfo.
- Plan 02-08 can audit tensor call sites with no remaining direct status release or legacy finalizer logging.
- No blockers remain.

## Self-Check: PASSED

- All four task files and this summary exist.
- Commits `2c96cc6` and `665919e` are present in git history in RED-before-GREEN order.
- Fresh race, short, documentation, source-audit, unchanged-module, and unchanged-Value checks all passed from committed HEAD.

---
*Phase: 02-core-api-errors-values*
*Completed: 2026-07-24*
