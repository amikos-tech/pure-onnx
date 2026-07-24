---
phase: 02-core-api-errors-values
plan: 07
subsystem: api
tags: [go, onnx-runtime, bootstrap, errors, slog, filesystem-security]

# Dependency graph
requires:
  - phase: 02-core-api-errors-values
    provides: "Public error sentinels from Plan 01 and atomic structured diagnostic plumbing from Plan 03"
provides:
  - "Public, inspectable bootstrap error categories with preserved filesystem, network, archive, loader, and cleanup causes"
  - "Least-privilege TGZ and ZIP regular-file extraction that strips group and other write bits"
  - "Twelve audited structured bootstrap notices with redacted URLs and zero duplicate emission for returned errors"
affects: [02-08, bootstrap, diagnostics, archive-extraction]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Classify local validation separately from unsupported platforms and missing shared libraries"
    - "Preserve independent primary and cleanup causes with errors.Join"
    - "Emit only non-returnable bootstrap notices through the private slog emitter"

key-files:
  created:
    - .planning/phases/02-core-api-errors-values/deferred-items.md
  modified:
    - ort/bootstrap.go
    - ort/bootstrap_test.go

key-decisions:
  - "Keep unsupported platforms distinct from supported-platform library absence through ErrUnsupportedPlatform and ErrSharedLibraryNotFound"
  - "Clamp only group and other write bits from archive-derived file modes so owner execute permissions survive"
  - "Allowlist structured bootstrap attributes, redact URLs before emission, and preserve trusted synchronous handler panic propagation"

patterns-established:
  - "Bootstrap validation wraps ErrInvalidArgument while OS, network, archive, and loader failures retain their lower causes"
  - "Bootstrap diagnostics are silent by default, structured when configured, and never duplicate a returned failure"

requirements-completed: [API-02]

# Metrics
duration: 23min
completed: 2026-07-24
---

# Phase 2 Plan 07: Bootstrap Error, Permission, and Diagnostic Contracts Summary

**Inspectable bootstrap failures, least-privilege archive extraction, and twelve audited structured notices with redacted sensitive context.**

## Performance

- **Duration:** 23 min
- **Started:** 2026-07-24T13:32:54Z
- **Completed:** 2026-07-24T13:56:10Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- Replaced the private missing-library category with public `ErrSharedLibraryNotFound`, kept it distinct from `ErrUnsupportedPlatform`, and classified local bootstrap validation with `ErrInvalidArgument`.
- Preserved representative filesystem, network, checksum metadata, archive, dynamic-loader, and independent cleanup causes through `%w` and `errors.Join`.
- Hardened TGZ and ZIP extraction by stripping group and other write bits while retaining required owner read and execute permissions.
- Migrated all twelve direct bootstrap log sites to structured Info/Warn diagnostics with allowlisted attributes and URL redaction.
- Proved every approved notice is silent with a nil handler, emits once when configured, propagates consumer-handler panics, and does not duplicate returned validation/network/checksum/archive/lock failures.

## Task Commits

Each TDD gate was committed atomically:

1. **Task 02-07-01 RED: Add failing bootstrap error-chain tests** - `5373bd6` (test)
2. **Task 02-07-01 GREEN: Expose bootstrap error contracts** - `c0adff1` (feat)
3. **Task 02-07-02 RED: Add failing bootstrap permission regression** - `ca3988e` (test)
4. **Task 02-07-02 GREEN: Harden bootstrap archive file modes** - `4d44a78` (fix)
5. **Task 02-07-03 RED: Add failing bootstrap diagnostic audit** - `2977451` (test)
6. **Task 02-07-03 GREEN: Structure bootstrap diagnostics** - `6adb371` (feat)

## Files Created/Modified

- `ort/bootstrap.go` - Public sentinel wrapping, lower-cause preservation, safe archive file modes, and twelve structured bootstrap diagnostic call sites.
- `ort/bootstrap_test.go` - Exact error-chain, Unix permission, Windows compile, structured call-site, URL-redaction, panic, and zero-duplicate-emission coverage.
- `.planning/phases/02-core-api-errors-values/deferred-items.md` - Two unrelated lint findings recorded outside this plan's file ownership.

## Decisions Made

- Unsupported host platforms continue to match `ErrUnsupportedPlatform`; supported-platform library absence now matches public `ErrSharedLibraryNotFound`. These categories remain independently actionable.
- Archive regular-file modes remove only group and other write bits. This is the narrowest hardening that blocks archive-supplied writable permissions without breaking executable shared-library layouts.
- Bootstrap notices use standard `slog` levels and attributes through `emitDiagnostic`. Credential-bearing URLs are redacted before emission, errors returned to callers are not logged again, and non-finalizer consumer-handler panics intentionally propagate across the trusted callback boundary.
- Resettable private filesystem/cache seams are limited to cleanup and user-cache lookup operations so deterministic tests can exercise otherwise platform-dependent notice paths.

## Deviations from Plan

None - plan executed exactly as written.

## TDD Gate Compliance

- Task 1 RED failed on missing public bootstrap categories and lower-cause wrapping; GREEN passed the exact error/security selector.
- Task 2 RED failed because archive-derived modes were not clamped; GREEN passed Unix permission assertions and the Windows compile gate.
- Task 3 RED failed on missing deterministic cleanup/cache seams and structured call sites; GREEN passed the exact race-backed diagnostic selector.
- Git history contains a RED `test(02-07)` commit before every corresponding GREEN implementation commit; no refactor commit was needed.

## Verification Evidence

- All three exact task-level selectors passed, including the race-backed diagnostic audit.
- `go test -short ./ort` and `go test -short ./...` passed.
- `GOOS=windows GOARCH=amd64 go test -c -o /dev/null ./ort` passed.
- The focused bootstrap integration selector passed after all three tasks converged.
- Source audits found exactly twelve `emitDiagnostic` invocations and no `log.Printf` or new `t.Parallel` in the owned files.
- `go.mod` and `go.sum` remained unchanged from the plan-start commit.

## Known Stubs

None.

## Deferred Issues

- `make precommit-lint-new` reports an `unsafeptr` warning in `ort/errors_native_test.go` from Plan 02-01 and staticcheck `SA1012` in `ort/diagnostics_test.go` from Plan 02-03. Both are outside Plan 02-07 ownership and are recorded in `deferred-items.md`.

## Issues Encountered

- The optional phase-wide new-issues lint target includes findings introduced by earlier plans. No owned-file lint finding was reported, so the unrelated items were deferred instead of expanding this plan's scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02-08 can run the convergence audit against public bootstrap categories and the complete structured diagnostic call-site set.
- Bootstrap integrity controls, archive permission hardening, URL redaction, and returned-error non-duplication are all covered by committed tests.
- No Plan 02-07 blocker remains.

## Self-Check: PASSED

- Both modified task files, the deferred-items record, and this summary exist.
- Commits `5373bd6`, `c0adff1`, `ca3988e`, `4d44a78`, `2977451`, and `6adb371` are present in git history in RED-before-GREEN order.
- Fresh task-level, race, short, Windows compile, source-audit, and unchanged-module checks passed from committed production HEAD.

---
*Phase: 02-core-api-errors-values*
*Completed: 2026-07-24*
