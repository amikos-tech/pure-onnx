---
phase: quick-260730-gye-address-all-five-review-findings
plan: 01
status: complete
subsystem: core-api-security
tags: [onnxruntime, bootstrap, gosec, golangci-lint, compatibility]

requires:
  - phase: 02-core-api-errors-values
    provides: Native error categories, resource lifecycle handling, and bootstrap hardening
provides:
  - Backward-compatible exported handle structs
  - Distinct invalid and destroyed SessionOptions errors
  - Trusted explicit symlink resolution with strict cache symlink rejection
  - Clean gosec v2.25.0 and PR new-issues lint gates
affects: [phase-03, public-api, bootstrap, ci]

tech-stack:
  added: []
  patterns:
    - Explicit caller paths resolve symlinks before strict target validation
    - Cache-managed paths continue to reject symlinks with Lstat
    - Resource lifecycle state is guarded by the same mutex as its native handle

key-files:
  created:
    - ort/public_api_compat_test.go
  modified:
    - Makefile
    - ort/types.go
    - ort/session.go
    - ort/session_test.go
    - ort/bootstrap.go
    - ort/bootstrap_test.go
    - ort/environment_test.go
    - embeddings/minilm/embedder.go
    - embeddings/openclip/bootstrap.go
    - embeddings/openclip/embedder.go
    - embeddings/splade/embedder.go
    - examples/openclip/main.go
    - tools/gen_ortapi.go

key-decisions:
  - "Mark SessionOptions destroyed only after it has owned a non-zero native handle."
  - "Follow symlinks only for caller-selected library paths; keep cache validation strict."
  - "Suppress only the exact reviewed scanner rules at the reported statements."

patterns-established:
  - "Explicit-path validation: normalize, resolve the full symlink chain, then reuse strict regular and non-empty file checks."
  - "Scanner exceptions: use rule-specific, line-scoped annotations with a concrete safety reason."

requirements-completed: [RF-1, RF-2, RF-3, RF-4, RF-5]

duration: 14min
completed: 2026-07-30
---

# Quick Task 260730-gye: Address All Five Review Findings Summary

**Restored public handle compatibility, separated SessionOptions lifecycle errors, accepted trusted soname links without weakening cache checks, and cleared the pinned security and lint gates.**

## Performance

- **Duration:** 14 min
- **Started:** 2026-07-30T09:25:55Z
- **Completed:** 2026-07-30T09:39:43Z
- **Tasks:** 3
- **Files modified:** 17

## Accomplishments

- Restored `Status`, `Environment`, and `Session` as exported structs with private fields, preserving downstream zero-value composite literals.
- Added mutex-protected `SessionOptions` lifecycle state so never-initialized values match `ErrInvalidArgument` and once-live destroyed values match `ErrDestroyed`.
- Added a separate explicit-library validator that resolves trusted symlinks to a validated target while cache-managed paths still reject symlinks.
- Aligned local gosec tooling with v2.25.0 and documented the exact 15 reviewed false positives without weakening global enforcement.
- Added two line-scoped govet annotations for the intentional purego callback output writes.

## Task Commits

1. **Task 1 RED: public handle and lifecycle regressions** - `88bf464` (`test`)
2. **Task 1 GREEN: restore handle structs and lifecycle errors** - `2d00f0d` (`fix`)
3. **Task 2 RED: explicit symlink regressions** - `ce58588` (`test`)
4. **Task 2 GREEN: resolve trusted explicit symlinks** - `7e6c68e` (`fix`)
5. **Task 3: clear scanner and lint baselines** - `6700ae9` (`chore`)
6. **Task 2 follow-up: canonical loader expectations** - `0618207` (`test`)

## TDD Gate Evidence

- Task 1 RED failed because the scalar handle types rejected struct literals and exposed no private `handle` field; its focused suite passed after the struct and lifecycle implementation.
- Task 2 RED failed because both explicit path sources rejected a normal soname link and a dangling link lost `ErrSharedLibraryNotFound`; its focused suite passed after adding `validateExplicitLibraryFile`.

## Verification

- Focused Task 1 suite: passed.
- Status accessor suite (`go test -count=1 ./ort -run '^TestStatus'`): passed.
- Focused Task 2 suite, including cached-symlink rejection: passed.
- `gofmt -l` across every changed Go file: no paths reported.
- `go test -count=1 -short ./...`: passed for all packages.
- `GOOS=windows GOARCH=amd64 go test -c -o /dev/null ./ort`: exited zero.
- `go run github.com/securego/gosec/v2/cmd/gosec@v2.25.0 -exclude-dir=examples/experimental ./...`: 0 issues.
- `make precommit-lint-new PRECOMMIT_BASE_REF=main`: 0 issues.
- `go vet -unsafeptr=false ./ort/...`: exited zero.
- Workflow and global lint policy diff: clean.

## Decisions Made

- A zero-value `SessionOptions` remains invalid, while only an object that previously held a native handle becomes destroyed.
- Explicit paths are trusted caller input and may resolve symlinks; downloaded cache state remains untrusted and symlink-rejecting.
- The dynamic loader receives the same fully resolved path that was validated.
- Security and lint exceptions remain local to the reviewed statements and name only the matching rules.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated canonical path expectations on macOS**

- **Found during:** Task 2 and combined repository verification
- **Issue:** `filepath.EvalSymlinks` correctly resolved the `/var` temp-directory prefix to `/private/var`, but existing loader assertions compared against the unresolved spelling. One background loader test then blocked cleanup after the assertion failed.
- **Fix:** Compare explicit loader paths with `filepath.EvalSymlinks` results in both new and existing tests.
- **Files modified:** `ort/bootstrap_test.go`
- **Verification:** The isolated loader tests, focused Task 2 suite, and full short suite all passed.
- **Committed in:** `7e6c68e`, `0618207`

---

**Total deviations:** 1 auto-fixed bug.

**Impact on plan:** The adjustment enforces the planned fully resolved path contract and adds no new scope.

## Issues Encountered

- The first combined test run exposed unresolved-versus-canonical temp path expectations. The isolated failures were corrected and the full suite then passed in 5.667 seconds for the `ort` package.

## Known Stubs

None. The changed production paths are fully wired; no placeholder data or deferred behavior was added.

## User Setup Required

None.

## Next Phase Readiness

- All five review findings have regression or scanner coverage.
- Public API compatibility, bootstrap trust separation, and lifecycle error classification are ready for downstream Phase 3 work.
- No merge, push, workflow weakening, dependency addition, or state update was performed.

## Self-Check: PASSED

- All required source and summary files exist.
- All six quick-task commits are present in git history.
- The explicit validator and SessionOptions lifecycle state are present.
- No new stub markers were introduced.
