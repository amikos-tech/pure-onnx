---
phase: 01-dx-test-hardening
plan: 01
subsystem: testing
tags: [go, errors, sentinel-error, bootstrap, cli-ux, tdd]

# Dependency graph
requires: []
provides:
  - "ort.ErrUnsupportedPlatform exported sentinel + ort.IsUnsupportedPlatformError detector for unsupported-GOOS/GOARCH bootstrap failures"
  - "examples/inference fail-fast diagnostic that hints ONNXRUNTIME_LIB_PATH only on unsupported-platform failures"
affects: [01-02, 01-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Exported sentinel error + errors.Is detector (mirrors existing errBootstrapRedirectPolicy pattern) for cross-package error classification"

key-files:
  created:
    - examples/inference/main_test.go
  modified:
    - ort/bootstrap.go
    - ort/bootstrap_test.go
    - examples/inference/main.go

key-decisions:
  - "Wrapped unsupported-combo error with %w preserving byte-identical message text so existing message-content checks do not break"
  - "diagnosticFor's non-platform branch returns exactly today's message, satisfying D-01 'other bootstrap failures unchanged'"

patterns-established:
  - "Exported sentinel error detection across package boundary via errors.Is wrapper (IsUnsupportedPlatformError)"

requirements-completed: [DX-01]

# Metrics
duration: 12min
completed: 2026-07-22
---

# Phase 01 Plan 01: Fail-fast unsupported-platform diagnostic Summary

**Exported `ort.ErrUnsupportedPlatform` sentinel + `IsUnsupportedPlatformError` detector, with the inference example surfacing a GOOS/GOARCH + ONNXRUNTIME_LIB_PATH hint only for unsupported-platform bootstrap failures.**

## Performance

- **Duration:** ~12 min
- **Tasks:** 2 (both TDD: RED → GREEN)
- **Files modified:** 3 (+1 created)

## Accomplishments
- Added exported, doc-commented `ErrUnsupportedPlatform` sentinel and `IsUnsupportedPlatformError(err)` detector in `ort/bootstrap.go`, mirroring the existing unexported redirect-policy sentinel idiom (no string-matching, per D-02).
- `resolveRuntimeArtifact` now wraps its unsupported-combo return with `%w` and the sentinel; resulting error text is byte-identical to before.
- `examples/inference/main.go` gains `diagnosticFor(err, goos, goarch)`, wired into `main()`'s bootstrap-failure branch via `runtime.GOOS`/`runtime.GOARCH`. Only unsupported-platform failures get the `ONNXRUNTIME_LIB_PATH` hint; every other failure keeps its exact current message. All other `log.Fatalf` sites untouched.
- Added table-test assertion scoped to only the `wantUnsupportedPlatform: true` case, plus two `diagnosticFor` unit tests covering both branches.

## Task Commits

TDD tasks — test (RED) then implementation (GREEN):

1. **Task 1 (RED): failing ErrUnsupportedPlatform assertion** - `a10991a` (test)
2. **Task 1 (GREEN): sentinel + wrapped resolveRuntimeArtifact error** - `56e6efc` (feat)
3. **Task 2 (RED): failing diagnosticFor tests** - `0e93b4f` (test)
4. **Task 2 (GREEN): diagnosticFor helper wired into inference example** - `3e0d52b` (feat)

## Files Created/Modified
- `ort/bootstrap.go` - Added exported `ErrUnsupportedPlatform` + `IsUnsupportedPlatformError`; wrapped unsupported-combo return with `%w`.
- `ort/bootstrap_test.go` - Added `wantUnsupportedPlatform bool` field (set true only on the `"unsupported"` case) and a scoped `errors.Is` assertion.
- `examples/inference/main.go` - Added `runtime` import + `diagnosticFor`; replaced the bootstrap-failure `log.Fatalf` with `log.Fatal(diagnosticFor(...))`.
- `examples/inference/main_test.go` (created) - `TestDiagnosticForUnsupportedPlatform` and `TestDiagnosticForOtherBootstrapFailureUnchanged`.

## Decisions Made
- None beyond the plan — followed D-01/D-02 as specified (sentinel + `errors.Is`, message text preserved).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `go vet ./examples/...` reports a pre-existing "possible misuse of unsafe.Pointer" warning at `examples/experimental/main.go:85`. That file is untouched by this plan (last changed in commit e607906) and out of scope per the executor SCOPE BOUNDARY. Logged to `deferred-items.md`. All in-scope packages (`ort/`, `examples/inference/`) build and vet clean.

## Verification
- `go test ./ort/... -run TestResolveRuntimeArtifact -v` — PASS (incl. gated `errors.Is` assertion)
- `go test ./examples/inference/... -run TestDiagnosticFor -v` — PASS (both tests)
- `go build ./...` — clean
- `go vet -unsafeptr=false ./ort/...` — clean

## Known Stubs
None.

## Threat Flags
None — diagnostic text echoes only compile-time GOOS/GOARCH constants and the existing wrapped error string; no new external-input surface.

## Next Phase Readiness
- DX-01 closed. `ort.ErrUnsupportedPlatform` / `IsUnsupportedPlatformError` are available as a public error-classification primitive for later phases.
- No blockers.

---
*Phase: 01-dx-test-hardening*
*Completed: 2026-07-22*
