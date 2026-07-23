---
phase: 01-dx-test-hardening
plan: 02
subsystem: ort-concurrency-tests
tags: [testing, concurrency, determinism, TST-01]
requires:
  - "ort/session.go runMu (sync.Mutex) lock ordering"
  - "ort/tensor.go runMu (sync.RWMutex) lockForRun/Destroy paths"
provides:
  - "Deterministic TryLock()-based lock-contention proof for the 3 concurrency tests named in D-03"
  - "Event-order recording via slices.Equal replacing wall-clock polling assertions"
affects:
  - "ort/session_test.go"
tech-stack:
  added:
    - "stdlib slices (Go 1.21+) for deterministic event-order comparison"
  patterns:
    - "TryLock() contention probe on the main test goroutine (RESEARCH.md Pattern 2 hardening)"
    - "single-shot 500ms watchdog as deadlock safety net, never the passing condition"
key-files:
  created: []
  modified:
    - "ort/session_test.go"
decisions:
  - "TryLock() probe on the exact mutex Destroy() must acquire proves blocking structurally, independent of goroutine scheduling — stronger than a mock-side rendezvous"
  - "Watchdog select runs only on the main test goroutine, honoring testing.T.FailNow's same-goroutine contract"
metrics:
  duration: "~10m"
  completed: "2026-07-22"
  tasks: 2
  files: 1
---

# Phase 01 Plan 02: Deterministic Concurrency Assertions Summary

Converted the 3 concurrency tests named in issue #43 / D-03 from wall-clock polling (`require.Eventually`/`require.Never`) to deterministic synchronization: a `TryLock()` lock-contention probe proves the in-flight `Run()` holds the contended mutex before the destroy goroutine is spawned, event-order recording asserts exact `["run-returned", "destroy-released"]` ordering via `slices.Equal`, and a single-shot 500ms watchdog remains only as a deadlock safety net.

## What Was Built

### Task 1 — `TestAdvancedSessionRunAndDestroyConcurrent` + `TestTensorDestroyWaitsForInFlightRun`
- Added a deterministic lock-contention probe immediately after `<-runStarted` and before spawning the destroy goroutine:
  - Session test: `session.runMu.TryLock()` (a `sync.Mutex`) must return `false` — proving `Run()` holds it.
  - Tensor test: `inputTensor.runMu.TryLock()` (write-lock probe against the `sync.RWMutex`) must return `false` — proving `lockForRun()`'s RLock is held.
- Added a mutex-guarded `events []string` slice with a `record()` closure; `runSessionFunc` records `"run-returned"` after `<-allowRunReturn`, and `releaseSessionFunc`/`releaseValueFunc` record `"destroy-released"` as their first statement.
- Replaced each `require.Never(...)` block with an inline watchdog `select` (destroy-received branch = `t.Fatalf`, `time.After(500ms)` = expected), running on the main test goroutine.
- Added a `slices.Equal(got, want)` order assertion after the deterministic channel joins.
- Added the `"slices"` import.

### Task 2 — `TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun`
- Replaced the `require.Eventually(...)` polling block with a direct channel receive plus a 500ms timeout-as-failure watchdog — the deliberate mirror-image of Task 1 (here timeout = failure, receive = expected), since this test proves the ABSENCE of blocking rather than its presence. This test was not part of the 01-REVIEWS.md consensus finding and intentionally uses a direct-receive design, not a TryLock probe.

## Verification

- `go test -race -run 'TestAdvancedSessionRunAndDestroyConcurrent|TestTensorDestroyWaitsForInFlightRun|TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun' ./ort/... -v` — PASS, zero race warnings.
- Full CI `test-race-ort-concurrency` regex (8 tests) — PASS.
- `TestAdvancedSessionRunConcurrent` (D-05, out of scope) — unmodified, still PASS.
- `go build ./...` and `go vet -unsafeptr=false ./ort/...` — clean.
- `gofmt -l` — clean.

### Acceptance grep counts
- `require.Never`: 0 (both call sites removed)
- `runMu.TryLock()`: 2 · `TryLock unexpectedly succeeded`: 2
- `record("run-returned")`: 2 · `record("destroy-released")`: 2
- `slices.Equal`: 2 · `"slices"` import: 1
- Remaining `require.Eventually` calls (lines 138/165/194/657/1004) are all outside the 3 named tests — out of scope per D-05.

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- FOUND: ort/session_test.go
- FOUND commit 052b03c (Task 1)
- FOUND commit cc6f1f4 (Task 2)
