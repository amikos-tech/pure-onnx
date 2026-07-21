# Phase 1: DX & Test Hardening - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-21
**Phase:** 1-DX & Test Hardening
**Areas discussed:** DX-01 hint scope, TST-01 test scope & watchdog, TST-02 stress test surface & race-lane placement

---

## DX-01 hint scope

| Option | Description | Selected |
|--------|-------------|----------|
| Unsupported-platform only | Detect the specific "unsupported platform for ONNX Runtime bootstrap" error and append the hint only then; other bootstrap failures unchanged | ✓ |
| Any bootstrap failure | Append the hint whenever `EnsureOnnxRuntimeSharedLibrary()` fails, regardless of cause | |

**User's choice:** Unsupported-platform only (recommended).
**Notes:** Matches issue #42's literal scope; a checksum/network failure isn't fixed by `ONNXRUNTIME_LIB_PATH` so the hint would mislead in those cases.

---

## TST-01 watchdog value

| Option | Description | Selected |
|--------|-------------|----------|
| Keep 500ms | Reuse existing 500ms as safety-net ceiling, not primary assertion | ✓ |
| Shorter watchdog (~100ms) | Tighter safety net, faster failure on genuine deadlock | |
| Longer watchdog (~2s) | More slack for slow/loaded CI runners | |

**User's choice:** Keep 500ms (recommended).

## TST-01 scope boundary

| Option | Description | Selected |
|--------|-------------|----------|
| Only the 3 named tests | Matches issue #43's explicit list; other sleeps aren't concurrency-correctness assertions | ✓ |
| Sweep all sleep-based timing in the suite | Broader cleanup including retry-backoff sleeps, goes beyond #43's scope | |

**User's choice:** Only the 3 named tests (recommended).
**Notes:** `TestAdvancedSessionRunConcurrent`'s 1ms sleep is a mock work-duration simulator (already proven deterministic via atomic `maxInFlight`); retry-backoff sleeps in bootstrap/embedder integration tests test retry logic, not concurrency.

---

## TST-02 stress test surface

| Option | Description | Selected |
|--------|-------------|----------|
| Test file only, this phase | Just `environment_stress_test.go`, verified locally under `-race`; docs deferred to Phase 4 | |
| Test file + CI job + docs, all in Phase 1 | Full issue #24 proposal now: stress test file, dedicated CI job, TESTING.md section | ✓ |

**User's choice:** Full scope (test file + CI job + docs), all in Phase 1.
**Notes:** Pulls a slice of doc work forward from Phase 4 deliberately — completeness against issue #24 chosen over strict phase-boundary purity.

## TST-02 race-lane placement

| Option | Description | Selected |
|--------|-------------|----------|
| Own dedicated job only | New stress tests get their own CI job, separate from `test-race-ort-concurrency` | ✓ |
| Also add to existing curated regex | Extend the existing job's regex too, in addition to the dedicated job | |

**User's choice:** Own dedicated job only (recommended).
**Notes:** Avoids redundant double-running of the same tests across two CI jobs; keeps the existing job focused on today's FFI-adjacent tests.

---

## Claude's Discretion

- Exact sentinel-error naming/plumbing for DX-01 detection (follow existing `errBootstrapRedirectPolicy` pattern).
- Exact set/parameters of the 2+ additional stress tests beyond `TestStressConcurrentInitDestroy`, sourced from issue #24's "Mixed Operations Under Load" and "Potential Issues to Catch" sections.

## Deferred Ideas

None — discussion stayed within phase scope.
