---
phase: 02-core-api-errors-values
plan: 02
subsystem: api
tags: [go, generics, onnx-runtime, values, tensors]

# Dependency graph
requires:
  - phase: 02-core-api-errors-values
    provides: "Existing Tensor[T] ownership, private native-handle leases, and package-local session value doubles"
provides:
  - "Package-sealed Value contract for package-owned ONNX values"
  - "Kind-only IsTensor inspection and exact non-nil AsTensor[T] extraction"
  - "Focused generic match, mismatch, nil, typed-nil, allocation, and heterogeneous-value tests"
affects: [02-04, 02-05, 02-08]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Seal native-resource interfaces with a package-private marker"
    - "Use a direct generic type assertion for exact zero-copy tensor extraction"

key-files:
  created:
    - ort/value_test.go
  modified:
    - ort/types.go
    - ort/tensor.go
    - ort/session_test.go

key-decisions:
  - "Seal Value with ortValue so only package-created values can participate in private native-handle and run-lease protocols"
  - "Keep IsTensor as a value-kind check while AsTensor[T] requires an exact, non-nil *Tensor[T] and returns nil, false for typed nil"

patterns-established:
  - "Value implementations remain package-owned; raw native handles and run leases stay private"
  - "Generic tensor inspection performs no coercion, copy, reflection, or allocation"

requirements-completed: [API-03]

# Metrics
duration: 4min
completed: 2026-07-24
---

# Phase 2 Plan 02: Sealed Value and Tensor Inspection Summary

**A package-owned `Value` boundary with kind inspection and exact, zero-allocation generic tensor extraction.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-07-24T12:11:09Z
- **Completed:** 2026-07-24T12:15:14Z
- **Tasks:** 1
- **Files modified:** 4

## Accomplishments

- Sealed `Value` with an unexported marker while retaining only `Destroy`, `Type`, and the marker in its method set.
- Added `IsTensor` for ONNX kind checks and `AsTensor[T]` for exact, non-nil pointer extraction with no conversion or copy.
- Updated the existing tensor implementation and all four package-local session test doubles without changing their handle or lease behavior.
- Proved heterogeneous tensor types remain assignable to `[]Value`, and all existing examples and embedders compile unchanged.

## Task Commits

The TDD task was committed in two atomic gates:

1. **Task 02-02-01 RED: Add failing value inspection tests** - `a022db3` (test)
2. **Task 02-02-01 GREEN: Seal Value and add tensor inspection** - `0092fbd` (feat)

## Files Created/Modified

- `ort/value_test.go` - Match, mismatch, nil, typed-nil, kind-only, allocation, copy, and heterogeneous-value coverage.
- `ort/types.go` - Sealed `Value` contract plus `IsTensor` and exact `AsTensor[T]` helpers.
- `ort/tensor.go` - Package-private marker on `*Tensor[T]`.
- `ort/session_test.go` - Marker implementations on the four existing package-local value doubles.

## Decisions Made

- Sealing is an intentional compatibility tradeoff: external `Value` implementations are unsupported because they cannot safely join the private native-handle lease protocol.
- A typed-nil tensor still has tensor kind for `IsTensor`, but `AsTensor[T]` normalizes it to `nil, false`.
- Exact extraction uses one direct type assertion; mismatched numeric element types are never converted.

## Deviations from Plan

None - plan executed exactly as written.

## TDD Gate Compliance

- RED failed on the missing `IsTensor` and `AsTensor` symbols.
- GREEN passed the focused value tests, full short `ort` suite, and unchanged-consumer compile gate.
- Git history contains `test(02-02)` before `feat(02-02)`; no refactor commit was needed.

## Verification Evidence

- `go test -short ./ort -run TestValue` — passed.
- `go test -short ./ort` — passed.
- `go test -run '^$' ./...` — passed for every package, including unchanged examples and embedding adapters.
- `go doc ./ort.Value`, `go doc ./ort.IsTensor`, and `go doc ./ort.AsTensor` — exposed only the intended public surface.
- `git diff --exit-code -- go.mod go.sum` — passed; no dependency or module metadata changed.
- Source audit confirmed the exact `value.(*Tensor[T])` assertion, explicit typed-nil rejection, all four test-double markers, and no reflection or numeric conversion path.

## Known Stubs

- `ort/types.go:23` and `ort/types.go:33` - Pre-existing `Status` compatibility accessors still return generic fallback data; this plan neither uses nor changes them.
- `ort/tensor.go:89` - Pre-existing allocator configurability TODO remains deferred until non-CPU providers are exposed.

These stubs predate this plan and do not block the sealed `Value` or tensor-inspection goal.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02-04 can now accept only package-created values and reuse the existing private lease protocol for `RunWithValues`.
- Plan 02-08 can include the new exported helpers in its final API and unchanged-consumer compatibility audit.
- No blockers remain for Plan 02-03.

## Self-Check: PASSED

- All four created or modified task files exist.
- Commits `a022db3` and `0092fbd` are present in git history.
- Focused, full short-package, compile-only consumer, API documentation, and module-file checks all passed after the GREEN commit.

---
*Phase: 02-core-api-errors-values*
*Completed: 2026-07-24*
