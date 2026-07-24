---
phase: 02-core-api-errors-values
plan: 03
subsystem: api
tags: [go, slog, diagnostics, concurrency, finalizers]

# Dependency graph
requires:
  - phase: 02-core-api-errors-values
    provides: "Existing finalizer-only panic recovery behavior and the ORT lifecycle lock hierarchy"
provides:
  - "Silent-by-default process-wide diagnostics configured with a standard slog.Handler"
  - "Atomic immutable logger replacement outside the ORT lifecycle lock graph"
  - "Private structured emission with finalizer-only consumer panic containment"
affects: [02-04, 02-05, 02-06, 02-07, 02-08]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Store a fully constructed slog.Logger in an atomically replaced immutable state"
    - "Treat consumer handlers as trusted synchronous callbacks and recover only at finalizer boundaries"

key-files:
  created:
    - ort/diagnostics.go
    - ort/diagnostics_test.go
  modified: []

key-decisions:
  - "Use slog.DiscardHandler for both package initialization and nil reset so diagnostics remain silent until explicitly configured"
  - "Let non-finalizer handler panics propagate as normal synchronous callback behavior while containing them in best-effort finalizer diagnostics"

patterns-established:
  - "All internal diagnostics flow through private Logger.LogAttrs emission with standard slog levels and attributes"
  - "Diagnostic reconfiguration uses atomic pointer replacement and never joins the ORT lifecycle lock graph"

requirements-completed: [API-02]

# Metrics
duration: 4min
completed: 2026-07-24
---

# Phase 2 Plan 03: Consumer-Wired Diagnostic Hook Summary

**Silent standard-library diagnostics with atomic handler replacement, structured `LogAttrs` emission, and finalizer-only panic containment.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-07-24T12:19:17Z
- **Completed:** 2026-07-24T12:23:27Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- Added `SetDiagnosticHandler(slog.Handler)` as the only exported diagnostic API, with `nil` restoring a `slog.DiscardHandler`-backed logger.
- Added one private `Logger.LogAttrs` emitter backed by atomically replaced immutable logger state and no ORT lifecycle locks.
- Added a finalizer-specific Warn wrapper that records `resource` and `error` attributes while containing arbitrary consumer-handler panics.
- Proved silent defaults, standard JSON output, pre-bound handler attributes, concurrent reconfiguration, panic policy, and zero emission for ordinary returned errors under the race detector.

## Task Commits

The TDD task was committed in two atomic gates:

1. **Task 02-03-01 RED: Add failing diagnostic hook tests** - `730d0d1` (test)
2. **Task 02-03-01 GREEN: Add consumer-wired diagnostics** - `b8bc801` (feat)

## Files Created/Modified

- `ort/diagnostics.go` - Atomic silent-default logger state, public handler configuration, private structured emission, and finalizer-safe warning wrapper.
- `ort/diagnostics_test.go` - One anchored behavior matrix covering silence, standard handler wiring, attributes, races, panic boundaries, and returned-error policy.

## Decisions Made

- A consumer-installed handler is trusted synchronous callback code. Its panic propagates from ordinary diagnostics, matching normal `slog.Handler` behavior.
- Only the library-owned best-effort finalizer boundary recovers a consumer-handler panic because finalizers cannot return failures.
- Handler replacement stores a fully constructed `*slog.Logger` atomically, keeping diagnostics independent from environment, session, and value locks.

## Deviations from Plan

None - plan executed exactly as written.

## TDD Gate Compliance

- RED failed on the missing `SetDiagnosticHandler`, `emitDiagnostic`, `emitFinalizerDiagnostic`, and diagnostic state symbols.
- GREEN passed the complete `TestDiagnostic` behavior matrix under `go test -race`.
- Git history contains `test(02-03)` before `feat(02-03)`; no refactor commit was needed.

## Verification Evidence

- `go test -race ./ort -run '^TestDiagnostic$'` — passed, including concurrent emit/reconfigure and paired panic-policy tests.
- `go test -short ./ort` — passed.
- `go doc ./ort.SetDiagnosticHandler` — confirmed the exact `func(slog.Handler)` public signature and trusted synchronous callback policy.
- Source audits confirmed `SetDiagnosticHandler` is the only exported identifier in `ort/diagnostics.go`, emission uses `Logger.LogAttrs`, no direct `Handler.Handle` call exists, and no ORT lifecycle lock is acquired.
- `git diff --exit-code -- go.mod go.sum` — passed; no dependency or module metadata changed.

## Known Stubs

None.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plans 02-04 through 02-07 can now route only non-returnable resource and bootstrap notices through the private diagnostic helpers.
- Plan 02-08 can audit the final diagnostic call sites and add the anchored race selector without introducing a logging abstraction or dependency.
- No blockers remain for Plan 02-04.

## Self-Check: PASSED

- Both created files exist and contain the planned exported and private contracts.
- Commits `730d0d1` and `b8bc801` are present in git history.
- Focused race, full short-package, API documentation, source-audit, and unchanged-module checks all passed after the GREEN commit.

---
*Phase: 02-core-api-errors-values*
*Completed: 2026-07-24*
