---
phase: 260731-fxc-address-issue-111
plan: 01
status: complete
subsystem: ort-bootstrap
tags: [go, concurrency, race-detector]
requires: []
provides:
  - Concurrent-safe reuse of ORT bootstrap option closures.
affects: [ort-bootstrap-options]
tech-stack:
  added: []
  patterns:
    - Normalize captured option strings into invocation-local variables.
key-files:
  created:
    - .planning/quick/260731-fxc-address-issue-111/260731-fxc-SUMMARY.md
  modified:
    - ort/bootstrap.go
    - ort/bootstrap_test.go
key-decisions:
  - Preserve validation timing by normalizing only when each option is applied.
requirements-completed: [ISSUE-111]
---

# Quick Task 260731-fxc Summary

**Reusable ORT bootstrap options now normalize strings per invocation, eliminating shared closure writes while preserving validation and configured values.**

## Implementation

- Added a network-free concurrent regression that shares one set of five normalized options across 16 workers.
- Changed the five ORT string-normalizing option closures to validate and assign invocation-local values.
- Audited OpenCLIP bootstrap constructors; they already avoid captured-string writes, so no OpenCLIP source changed.

## Tests

- Confirmed the regression reports `DATA RACE` before the production fix.
- `go test ./ort -race -run '^(TestBootstrapOptionsReusableConcurrently|TestEnsureOnnxRuntimeSharedLibraryConcurrentLockSingleDownload)$' -count=1`
- `go test ./ort -run '^(TestWithBootstrapVersionRejectsEmpty|TestWithBootstrapLibraryPathAndCacheDirRejectEmpty|TestWithBootstrapExpectedSHA256Validation|TestWithBootstrapBaseURLValidation)$' -count=1`
- `go test ./embeddings/openclip -run '^(TestEnsureDefaultAssetsValidation|TestEnsureDefaultAssetsCustomRepoRequiresChecksums)$' -count=1`

## Source Commits

- `7d76f03` — `test(ort): cover concurrent bootstrap option reuse`
- `0c637a9` — `fix(ort): make bootstrap option normalization concurrent-safe`

## Deviations from Plan

- `AGENTS.md` was added in a later branch commit even though the plan's source gate allowed only `ort/bootstrap.go` and `ort/bootstrap_test.go`. Quick task `260731-j0t` removes it from this PR and records the complete repository-guidance change as separate follow-up work.
