# Phase 1: DX & Test Hardening - Context

**Gathered:** 2026-07-21
**Status:** Ready for planning

<domain>
## Phase Boundary

The inference example fails fast with actionable guidance when ONNX Runtime bootstrap fails on an unsupported platform, and concurrency tests prove correctness deterministically (channel/handshake synchronization) rather than via wall-clock timing, including new stress coverage for concurrent `InitializeEnvironment`/`DestroyEnvironment` cycles. Maps to DX-01, TST-01, TST-02.

</domain>

<decisions>
## Implementation Decisions

### DX-01 — Inference example fail-fast hint (#42)
- **D-01:** The "set `ONNXRUNTIME_LIB_PATH`" hint is added only for the unsupported-platform bootstrap failure — i.e. when the underlying error is the "unsupported platform for ONNX Runtime bootstrap: GOOS=... GOARCH=..." error from `resolveRuntimeArtifact` (`ort/bootstrap.go:522`). Other bootstrap failures (checksum mismatch, network error, etc.) keep their existing messages unchanged — setting `ONNXRUNTIME_LIB_PATH` wouldn't fix those and the hint would mislead.
- **D-02:** Detection should follow the existing codebase pattern for this kind of thing — a sentinel error checked with `errors.Is` (see `errBootstrapRedirectPolicy` / `isBootstrapRedirectPolicyError` in `ort/bootstrap.go`), not string-matching the error text. This is Claude's implementation call, not a re-litigated decision — noted here so research/planning don't reinvent it.
- **Change stays in `examples/inference/main.go`** (and `ort/bootstrap.go` only if a sentinel error needs to be exported/added there) — no other `ort/` runtime behavior changes, per PROJECT.md's explicit out-of-scope note for #42.

### TST-01 — Deterministic concurrency assertions (#43)
- **D-03:** Convert exactly the 3 tests issue #43 names — `TestAdvancedSessionRunAndDestroyConcurrent` and `TestTensorDestroyWaitsForInFlightRun` (both currently `require.Eventually(..., 500*time.Millisecond, 50*time.Millisecond, ...)` at `ort/session_test.go:730` and `:882`), plus `TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun`. Nothing else in the suite is in scope for this requirement.
- **D-04:** Replace the primary assertion with explicit channel/handshake rendezvous points proving when goroutines are blocked/unblocked. Keep a watchdog timeout of **500ms** (reusing the existing value) purely as a deadlock safety net — it should never be the thing that makes the test pass, only a ceiling that fails the test if something hangs.
- **D-05 (explicitly out of scope):** `TestAdvancedSessionRunConcurrent`'s `1*time.Millisecond` sleep (`ort/session_test.go:556`) is a mock work-duration simulator, not a timing-based correctness assertion — its concurrency proof already comes from atomic `maxInFlight` tracking. Retry-backoff sleeps in `ort/bootstrap_test.go:1795`, `ort/minilm_helpers_test.go:210`, `embeddings/minilm/embedder_integration_test.go:298`, and `embeddings/splade/embedder_integration_test.go:231` are retry-logic tests, not concurrency assertions. None of these should be touched under TST-01.

### TST-02 — Stress tests for concurrent init/destroy (#24)
- **D-06:** Add a new `ort/environment_stress_test.go` implementing issue #24's proposed stress tests (starting point: `TestStressConcurrentInitDestroy` at ~100 goroutines × 1000 iterations, plus at least 2 more per the issue's "Mixed Operations Under Load" section), gated behind `testing.Short()` so normal `go test ./...` / CI runs skip them by default.
- **D-07:** These tests build on the existing pattern in `TestConcurrentInitialization`/`TestConcurrentDestroy` (`ort/environment_test.go:243,279`) — refcount-only correctness against a non-existent library path, no real FFI/unsafe pointer traffic — so they are checkptr-safe and can run under `-race`.
- **D-08 (full issue #24 scope, done in this phase, not deferred to Phase 4/5):**
  - Add a **new, dedicated** CI job for the stress tests (separate from the existing `test-race-ort-concurrency` job in `.github/workflows/ci.yml`), following issue #24's proposal: `go test -v -race -run=TestStress -count=50 -parallel=4 ./ort/...` with a `timeout-minutes: 10` guard.
  - Do **not** also add the new `TestStress*` tests to the existing `test-race-ort-concurrency` job's curated regex — that job stays focused on today's FFI-adjacent concurrency tests; the new job owns its own invocation/tuning. No redundant double-running of the same tests across two jobs.
  - Update `TESTING.md` (the real docs file consumed by contributors, not the `.planning/codebase/TESTING.md` snapshot) with a section on how to run stress tests locally and what they guard against. This pulls a slice of doc work forward from Phase 4 deliberately — the user chose completeness over strict phase-boundary purity here.

### Claude's Discretion
- Exact sentinel-error naming/plumbing for D-02.
- Exact set and parameters of the "at least 2 more" stress tests beyond `TestStressConcurrentInitDestroy` for D-06 — follow issue #24's "Mixed Operations Under Load" and "Potential Issues to Catch" sections (deadlocks, refCount corruption, panics under load) as the source of truth.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### GitHub Issues (source of truth for scope and acceptance criteria)
- GitHub issue #42 — "[DX] Improve inference example behavior on unsupported platforms" — exact scope: example-UX only, no `ort/` runtime changes
- GitHub issue #43 — "[TST] Replace timing-based concurrency assertions with deterministic synchronization" — names the exact 3 tests in scope and the deadlock-safety-net requirement
- GitHub issue #24 — "[TST] Add stress tests for concurrent init/destroy" — full proposal including test names/parameters, CI job YAML, and TESTING.md doc requirement

### Project docs
- `.planning/PROJECT.md` — confirms #42's remaining gap is example-only (bootstrap already emits GOOS/GOARCH-labeled error)
- `.planning/ROADMAP.md` §Phase 1 — literal success criteria (goal-backward check for planning/verification)
- `.planning/REQUIREMENTS.md` — DX-01, TST-01, TST-02 requirement text

### Code (already inspected this session — locations for planner/researcher to start from)
- `examples/inference/main.go:157-177` (`initializeOrtEnvironment`) — where the DX-01 hint is added
- `ort/bootstrap.go:467-522` (`resolveRuntimeArtifact`) — existing GOOS/GOARCH-labeled unsupported-platform error
- `ort/bootstrap.go:463-465` (`errBootstrapRedirectPolicy` / `isBootstrapRedirectPolicyError`) — sentinel-error pattern to follow for D-02
- `ort/session_test.go:530` (`TestAdvancedSessionRunConcurrent`) — reference pattern for atomic-based concurrency proof (not in scope, but the pattern to reuse)
- `ort/session_test.go:730,882` — the two `require.Eventually(500ms, 50ms)` call sites named in #43
- `ort/environment_test.go:243,279` (`TestConcurrentInitialization`, `TestConcurrentDestroy`) — existing foundation TST-02 stress tests build on
- `.github/workflows/ci.yml:108-126` (`test-race-ort-concurrency` job) — existing curated `-race` allowlist; new stress job stays separate from this
- `Makefile:110-113` (`test-race` target) — local race-test entry point

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `resetEnvironmentState()` / `setupORTTestEnvironment` helpers already centralize ORT init/cleanup for concurrency tests — reuse for the new stress test file.
- The atomic-counter + `sync.WaitGroup` pattern from `TestAdvancedSessionRunConcurrent` and `TestConcurrentInitialization` is the established idiom for proving concurrency invariants without real FFI calls.

### Established Patterns
- Sentinel errors + `errors.Is` (`errBootstrapRedirectPolicy`) is the existing pattern for typed error detection in `ort/bootstrap.go` — follow it for DX-01 rather than string-matching.
- Concurrency tests that don't touch real native calls (fake/nonexistent library paths, mocked `ortAPI`) are checkptr-safe and already run under `-race` in CI's `test-race-ort-concurrency` job — this is why the new stress tests can also target `-race` despite the repo-wide checkptr/purego incompatibility noted in `ci.yml:91`.

### Integration Points
- `.github/workflows/ci.yml` — new stress-test CI job goes alongside (not merged into) the existing `test-race-ort-concurrency` job.
- `TESTING.md` (repo root, contributor-facing) — new "Running stress tests" section.

</code_context>

<specifics>
## Specific Ideas

No UI/UX-style specifics — this is a mechanical hardening phase. The specifics that matter are the exact test names and error-string boundaries captured in `<decisions>` above, all traceable to the three GitHub issues.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. The TESTING.md update for stress tests was explicitly pulled forward into this phase (not deferred) per D-08.

</deferred>

---

*Phase: 1-DX & Test Hardening*
*Context gathered: 2026-07-21*
