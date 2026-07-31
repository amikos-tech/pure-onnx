---
phase: 02-core-api-errors-values
plan: 06
subsystem: api
tags: [go, onnx-runtime, ffi, errors, diagnostics, concurrency]

# Dependency graph
requires:
  - phase: 02-core-api-errors-values
    provides: "Typed status errors and lifecycle sentinels from Plan 01, plus structured diagnostic plumbing from Plan 03"
provides:
  - "Inspectable environment load, symbol, cleanup, CreateEnv, and configuration failures"
  - "Lifecycle-safe MemoryInfo creation with typed native status conversion"
  - "Structured runtime-version and finalizer-only memory diagnostics"
  - "Deterministic teardown-exclusion coverage across MemoryInfo native calls and status access"
affects: [02-07, 02-08, environment, memory-info, diagnostics]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Preserve independent initialization and cleanup causes with errors.Join"
    - "Hold ortCallMu across native execution, KeepAlive, status access, and release"
    - "Use null-message call-site probes under race while central tests prove non-empty status copying"

key-files:
  created: []
  modified:
    - ort/environment.go
    - ort/environment_test.go
    - ort/memory.go
    - ort/memory_test.go

key-decisions:
  - "Use private environment loader seams to prove exact load, symbol, and cleanup cause preservation without a real runtime"
  - "Keep race-lane call-site status probes null-message and rely on the central converter test for non-empty copy-before-release proof"
  - "Require the complete MemoryInfo create, release, and status callback set before entering native code"

patterns-established:
  - "Environment failures retain OS and cleanup causes while local configuration errors match public sentinels"
  - "MemoryInfo creation snapshots callbacks under mu while ortCallMu protects the full native status lifetime"

requirements-completed: [API-02]

# Metrics
duration: 18min
completed: 2026-07-24
---

# Phase 2 Plan 06: Environment and MemoryInfo Error Contracts Summary

**Inspectable environment and MemoryInfo failures with preserved cause chains, exact native status ownership, lifecycle-safe locking, and opt-in structured diagnostics.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-07-24T13:10:51Z
- **Completed:** 2026-07-24T13:28:53Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Made missing environment configuration and invalid explicit settings match public sentinels while retaining actionable setup guidance.
- Preserved original load, symbol, primary, and cleanup causes through `%w` and `errors.Join`.
- Routed CreateEnv and CreateMemoryInfo statuses through `statusToError`, retaining native operation/code detail and exact one-release ownership.
- Replaced the runtime-version text warning with one structured Warn notice and routed only finalizer-only MemoryInfo cleanup failures through diagnostics.
- Proved `CreateMemoryInfo` excludes environment teardown during both the native callback and status conversion without sleeps or polling.

## Task Commits

Each TDD gate was committed atomically:

1. **Task 02-06-01 RED: Add failing environment error tests** - `d7a767c` (test)
2. **Task 02-06-01 GREEN: Migrate environment error contracts** - `454b88d` (feat)
3. **Task 02-06-02 RED: Add failing MemoryInfo error tests** - `70a14aa` (test)
4. **Task 02-06-02 GREEN: Migrate MemoryInfo error contracts** - `31975c8` (feat)

## Files Created/Modified

- `ort/environment.go` - Sentinel wrapping, preserved loader/cleanup causes, CreateEnv conversion, and structured version warning.
- `ort/environment_test.go` - Cause-chain, status ownership, diagnostic policy, invalid-input, refcount, and concurrency coverage.
- `ort/memory.go` - Lifecycle-safe callback snapshotting, typed status conversion, sentinel wrapping, and finalizer diagnostic routing.
- `ort/memory_test.go` - Validation, ownership, teardown lock, idempotent destroy, and diagnostic policy coverage.

## Decisions Made

- Environment loader operations use private resettable seams so tests can assert identity-preserving `errors.Is`/`errors.As` behavior and independent cleanup causes without invoking purego on fake symbols.
- Race call-site tests return a null native message pointer. The central `TestStatusToError` matrix remains the authority for non-empty copy-before-release semantics.
- MemoryInfo creation requires the runtime API, create/release functions, and all status callbacks before crossing the native boundary, preventing an unreturnable or unconvertible live handle.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Replaced checkptr-unsafe fake native strings**
- **Found during:** Task 02-06-01 GREEN race verification
- **Issue:** Reading Go-backed log-ID and status-message pointers after a `uintptr` round trip triggered the race lane's checkptr guard.
- **Fix:** Asserted a nonzero log-ID pointer and used a null native status message for environment and MemoryInfo call-site probes. The existing central converter test continues to prove non-empty message copying before release.
- **Files modified:** `ort/environment_test.go`, `ort/memory_test.go`
- **Verification:** Both exact task-level race selectors passed without disabling checkptr.
- **Committed in:** `454b88d` and `31975c8`

**2. [Rule 2 - Missing Critical] Required complete MemoryInfo ownership callbacks**
- **Found during:** Task 02-06-02 GREEN callback-lifetime audit
- **Issue:** Checking only `CreateMemoryInfo` could create a live native handle without a release function or attempt status conversion with missing accessors.
- **Fix:** Required the runtime API, create/release functions, and code/message/release status callbacks before the native call.
- **Files modified:** `ort/memory.go`, `ort/memory_test.go`
- **Verification:** Before-init classification, exact status release, teardown exclusion, and full short-package tests passed.
- **Committed in:** `31975c8`

---

**Total deviations:** 2 auto-fixed (1 Rule 1, 1 Rule 2)
**Impact on plan:** Both fixes enforce the planned race/checkptr and native-ownership boundaries without changing exported APIs or resource ownership.

## TDD Gate Compliance

- Task 1 RED failed on the missing environment loader, CreateEnv, and runtime-warning seams; GREEN passed the exact environment race selector.
- Task 2 RED failed on the missing deterministic `finalizeMemoryInfo` adapter; GREEN passed the exact MemoryInfo race selector.
- Git history contains a RED `test(02-06)` commit before each corresponding GREEN `feat(02-06)` commit; no refactor commit was needed.

## Verification Evidence

- Both exact task-level `go test -race ./ort` selectors passed.
- `go test -short ./ort -run 'Test(InitializeEnvironment|DestroyEnvironment|CreateMemoryInfo|MemoryInfo)'` passed.
- `go test -race ./ort -run '^TestCreateMemoryInfoBlocksEnvironmentTeardown$'` passed.
- `go test -short ./...` passed across `ort`, embeddings, examples, and tooling.
- `go test -run '^$' ./...` passed as the unchanged-consumer compile gate.
- Source audits confirmed MemoryInfo lock order, post-native `KeepAlive`, centralized status conversion, approved diagnostic call sites, no timing-based lock assertions, and no new `t.Parallel`.
- `go.mod` and `go.sum` remained unchanged.

## Known Stubs

None.

## Issues Encountered

- The first environment status probe used Go-backed pointers that the required race/checkptr lane rejected. The race-safe null-message pattern resolved it while preserving the separate non-empty ownership proof.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02-07 can migrate bootstrap causes and approved notices against the settled error/diagnostic contracts.
- Plan 02-08 can remove the now-obsolete compatibility helpers and run the converged status/diagnostic audit.
- No blockers remain.

## Self-Check: PASSED

- All four modified task files and this summary exist.
- Commits `d7a767c`, `454b88d`, `70a14aa`, and `31975c8` are present in git history in RED-before-GREEN order.
- Fresh race, short, compile, source-audit, and unchanged-module checks all passed from committed production HEAD.

---
*Phase: 02-core-api-errors-values*
*Completed: 2026-07-24*
