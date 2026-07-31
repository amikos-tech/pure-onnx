---
phase: 260731-j0t-address-concurrency-regression-review-fi
plan: 01
status: complete
subsystem: ort-bootstrap-ci
tags: [go, concurrency, race-detector, github-actions]
requires:
  - 260731-fxc-address-issue-111
provides:
  - Automated race-detector enforcement for reusable ORT bootstrap options.
  - Documented concurrent-reuse guarantee for package-provided BootstrapOption values.
affects: [ort-bootstrap-options, ci-race-lane]
tech-stack:
  added: []
  patterns:
    - Keep purego-compatible race coverage in the explicit ORT concurrency selector.
key-files:
  modified:
    - .github/workflows/ci.yml
    - ort/bootstrap.go
    - ort/bootstrap_test.go
key-decisions:
  - Explain the regression's race-detector dependency in one source comment instead of adding build-tag support files.
  - Remove AGENTS.md from this PR and defer its complete canonical-guidance pointer to separate work.
requirements-completed: [ISSUE-111-REVIEW]
---

# Quick Task 260731-j0t Summary

**Reusable bootstrap options now have enforced race-detector coverage in CI, a documented concurrency contract, and lane-independent trimming assertions.**

## Implementation

- Appended `TestBootstrapOptionsReusableConcurrently` to the targeted ORT race selector and raised its live-count guard from 33 to 34.
- Documented that package-provided `BootstrapOption` values can be reused concurrently by separate bootstrap calls.
- Added an ordinary unit test for version, library-path, and cache-directory whitespace trimming.
- Explained the regression's `-race` dependency at the test, renamed `iterationsPerWorker`, and used the project's integer-range loop style.
- Removed the branch-only `AGENTS.md` addition so the source diff stays within the concurrency fix; the deletion remains recoverable from git history.
- Corrected the prior quick-task summary's plan-scope record.

## Verification

- Selector before: 33 tests; `TestBootstrapOptionsReusableConcurrently` absent.
- Selector after: 34 tests; `TestBootstrapOptionsReusableConcurrently` present.
- `go test -race ./ort -run "$RACE_SELECTOR" -count=1`
- `go test ./ort -run '^(TestWithBootstrapVersionRejectsEmpty|TestWithBootstrapLibraryPathAndCacheDirRejectEmpty|TestBootstrapOptionsTrimWhitespace)$' -count=1`
- `make precommit` (formatting, vet, new-code lint, gosec, short tests, module tidiness, and govulncheck).

## Source Commits

- `121e81f` — `test(ort): enforce bootstrap option race coverage`
- `b43a960` — `chore: remove unrelated repository instructions`

## Deferred

- Add a canonical-guidance `AGENTS.md` pointer in a separate PR.
- Open a separate issue for OpenCLIP cache-directory normalization.

## Deviations from Plan

None - plan executed as written.
