---
phase: quick-260730-ink-address-the-supplied-diagnostics-locking
plan: 01
status: complete
subsystem: core-runtime-security
tags: [diagnostics, bootstrap-cache, concurrency, runtime-pinner, ci]

requires:
  - phase: quick-260730-gye-address-all-five-review-findings
    provides: Explicit-path trust separation, SessionOptions lifecycle state, and clean security gates
  - phase: 02-core-api-errors-values
    provides: Native error handling, resource lifecycles, bootstrap, sessions, and tensors
provides:
  - Warning-level stderr diagnostics with panic-safe emergency reporting
  - Non-destructive bootstrap validation dispositions and immutable/shared cache support
  - Executable environment, session-options, session, and tensor lease ordering
  - GC-pressure pinning and supported native tensor-type coverage
  - Live-counted 33-test race and 5-test native CI selectors
affects: [phase-03, diagnostics, bootstrap, runtime-lifecycle, ci]

tech-stack:
  added: []
  patterns:
    - Semantic cache failures are typed separately from operational filesystem failures
    - Valid cache hits return before any cache write or process-lock creation
    - Per-resource locks own resource fields while the global mutex only snapshots runtime functions
    - KeepAlive protects synchronous calls while Pinner protects retained native data

key-files:
  created: []
  modified:
    - README.md
    - .github/workflows/ci.yml
    - ort/diagnostics.go
    - ort/environment.go
    - ort/bootstrap.go
    - ort/bootstrap_trust_unix.go
    - ort/bootstrap_trust_other.go
    - ort/session.go
    - ort/tensor.go

key-decisions:
  - "Write unhandled warning diagnostics to stderr while keeping informational diagnostics opt-in."
  - "Remove cache installs only after a typed confirmed-invalid result; preserve operational failures and their causes."
  - "Require explicit shared-cache trust and continue rejecting world-writable or symlinked cache state."
  - "Use resource-local locks for native handle ownership and leases for the full native Run call."

patterns-established:
  - "Diagnostics fallback: bypass consumer handlers only for emergency reporting after a handler panic."
  - "Bootstrap validation: missing, confirmed-invalid, and operational outcomes drive distinct non-destructive actions."
  - "Native lifetime: reachability, pinning, and synchronous call barriers are documented and tested separately."

requirements-completed: [RF-01, RF-02, RF-03, RF-04, RF-05, RF-06, RF-07, RF-08, RF-09]

duration: 32min
completed: 2026-07-30
---

# Quick Task 260730-ink: Diagnostics, Cache Trust, and Lifecycle Ordering Summary

**Fail-safe stderr diagnostics, non-destructive trusted cache handling, and race-proven session/tensor lifecycle leases with explicit pinning coverage.**

## Performance

- **Duration:** 32 min
- **Started:** 2026-07-30T11:00:33Z
- **Completed:** 2026-07-30T11:32:41Z
- **Tasks:** 3
- **Files modified:** 15

## Accomplishments

- Restored visible warning diagnostics by default, added panic-safe finalizer fallback, and preserved the original panic when initialization rollback also fails.
- Split cache validation into missing, confirmed-invalid, and operational outcomes so transient filesystem failures cannot delete a cache; added no-write read-only hits and explicit controlled shared-cache trust.
- Added deterministic race coverage proving environment teardown waits for in-flight session/tensor use and SessionOptions destruction waits for native session construction.
- Removed the unused value-handle wrapper, made `AdvancedSession.runMu` the sole owner of session fields, and documented validation, leases, `KeepAlive`, and `Pinner` according to their actual lifetimes.
- Added GC-pressure pinning coverage, all four supported tensor types to the native selector, pointer-literal API checks, and exact CI selector counts.

## Task Commits

1. **Task 1 RED: diagnostics safety regressions** - `6578c68` (`test`)
2. **Task 1 GREEN: fail-safe diagnostics and rollback reporting** - `c7e5801` (`fix`)
3. **Task 2 RED: cache trust regressions** - `6239948` (`test`)
4. **Task 2 GREEN: non-destructive bootstrap cache policy** - `3b07268` (`fix`)
5. **Task 3 RED: lifecycle lease contracts and CI wiring** - `a3b58fb` (`test`)
6. **Task 3 GREEN: native handle lease ownership** - `a779f21` (`fix`)
7. **Task 3 follow-up: lint-safe pinning proof** - `e92b8a7` (`test`)

## TDD Gate Evidence

- Task 1 RED failed because warning diagnostics were discarded by default, finalizer handler panics had no emergency fallback, rollback failures were lost, and the authoritative lock documentation was incomplete.
- Task 2 RED failed to compile because the validator seam, typed dispositions, shared-cache option, and shared-policy parameters did not yet exist.
- Task 3 RED failed because the canonical lease documentation was absent and `valuesToHandles` still existed; the focused suite passed after the minimal production update.

## Verification

- `gofmt -l` across all changed Go files: no paths reported.
- `go test -count=1 -short ./...`: passed for all packages.
- Focused diagnostics race suite: passed with Go 1.25.12.
- Focused cache/trust suite: passed; Windows amd64 cross-compilation exited zero.
- Focused lifecycle race suite: passed 10 consecutive runs.
- Committed race selector: exactly 33 tests discovered; the full selector passed under `-race`.
- Committed native selector: exactly 5 tests discovered. Local execution passed with all five skipped because `ONNXRUNTIME_LIB_PATH` was unavailable; the existing Linux native CI lane supplies the runtime.
- `go vet -copylocks ./ort/...`: exited zero.
- `go vet -unsafeptr=false ./ort/...`: exited zero.
- `make precommit-lint-new PRECOMMIT_BASE_REF=main`: 0 issues.
- `go.mod`, `go.sum`, and workflow action `uses:` lines are unchanged by this task.

## Decisions Made

- The default diagnostic handler is a fresh warning-level stderr handler, not the replaceable global slog default; setting a nil handler restores this behavior.
- Emergency diagnostics never retry a failed consumer handler and never replace the original panic.
- Bootstrap repairs are destructive only for marked semantic corruption or trust failures; raw filesystem errors remain operational and preserve `errors.Is`.
- Shared Unix caches may accept group-writable or differently owned paths only through explicit opt-in, while world-writable state remains invalid.
- Windows and residual non-Unix builds retain platform-neutral integrity checks without claiming Unix ownership or mode enforcement.
- Session state is owned solely by `runMu`; the global mutex only snapshots native release functions.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Made the reentrant diagnostic-handler test checkptr-safe**

- **Found during:** Task 1 focused race verification
- **Issue:** The prior fake ABI fixture round-tripped a Go heap pointer through `uintptr`, which race-enabled checkptr rejects before the intended lock behavior can be tested.
- **Fix:** Seeded initialized runtime state directly while retaining handler calls to `IsInitialized` and `GetVersionString`.
- **Files modified:** `ort/environment_test.go`
- **Verification:** The exact Task 1 race selector passed without disabling checkptr.
- **Committed in:** `c7e5801`

**2. [Rule 3 - Blocking] Updated repeated diagnostic installation fixtures for collision-safe publishing**

- **Found during:** Task 2 broader bootstrap verification
- **Issue:** A diagnostic helper intentionally installed twice into one destination, while the strengthened publisher correctly refused to replace an existing path without confirmed-invalid validation.
- **Fix:** Reset only the test-owned destination before each diagnostic exercise and added a regression proving production preserves a destination that appears mid-download.
- **Files modified:** `ort/bootstrap.go`, `ort/bootstrap_test.go`
- **Verification:** Focused and broader bootstrap suites plus the diagnostic call-site suite passed.
- **Committed in:** `3b07268`

**3. [Rule 3 - Blocking] Reworked the GC-pressure assertion to satisfy repository lint**

- **Found during:** Final new-issues lint
- **Issue:** Reconstructing an `unsafe.Pointer` from the captured `uintptr` triggered govet, and an explicit nil assignment used to drop the caller slice triggered ineffassign.
- **Fix:** Ended the caller slice's lexical scope, then compared the tensor's live pointer and contents against the address captured by the fake native constructor on every GC cycle.
- **Files modified:** `ort/tensor_test.go`
- **Verification:** The focused pinning test and `make precommit-lint-new PRECOMMIT_BASE_REF=main` passed with 0 issues.
- **Committed in:** `e92b8a7`

---

**Total deviations:** 3 auto-fixed blocking issues.

**Impact on plan:** Each adjustment preserved or strengthened the planned proof without changing public API, dependencies, checkptr settings, or CI gates.

## Issues Encountered

- The local environment did not provide `ONNXRUNTIME_LIB_PATH`. Native selector discovery and skip behavior were verified locally; actual native execution remains covered by the existing Linux CI job that installs ONNX Runtime.

## Known Stubs

None. No placeholder data, TODO implementation, or unwired production path was introduced.

## User Setup Required

None. Optional local native verification requires setting `ONNXRUNTIME_LIB_PATH` to a compatible ONNX Runtime shared library.

## Next Phase Readiness

- Diagnostics, bootstrap trust, and lifecycle ordering now have focused behavioral and race regressions.
- CI is wired to run the exact 33 race tests and 5 native tests, including all supported tensor element types.
- No dependency, state, roadmap, merge, or action-pin change was made.

## Self-Check: PASSED

- All required source files and this summary exist.
- All seven task commits are present in git history.
- No new stub marker or unplanned threat surface was found.
- The planning directory remains uncommitted as requested.
