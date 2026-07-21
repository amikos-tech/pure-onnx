# Phase 1: DX & Test Hardening - Research

**Researched:** 2026-07-21
**Domain:** Go stdlib concurrency-test synchronization, sentinel-error CLI diagnostics, race-detector-safe stress testing
**Confidence:** HIGH

## Summary

This phase is a mechanical, stdlib-only hardening pass with **no new external dependencies**. All three requirements (DX-01, TST-01, TST-02) are already tightly scoped by `01-CONTEXT.md` down to exact files, line numbers, and test names — this research fills the remaining *implementation* gaps: (1) how to export a sentinel error across the `ort` → `examples/inference` package boundary following the codebase's existing `errBootstrapRedirectPolicy` pattern, (2) how to replace `require.Eventually`/`require.Never` wall-clock polling with genuinely deterministic event-ordering assertions while keeping a 500ms watchdog purely as a hang-detector, and (3) how to size and gate `testing.Short()`-based stress tests so the gate is not inert and the new CI job's `-count=50` multiplier stays inside its 10-minute budget.

I fetched the source GitHub issues (#42, #43, #24) directly via `gh issue view` — these are the canonical specs and I treat them as VERIFIED, not assumed. I also empirically measured the proposed stress workload (`100 goroutines × 1000 iterations` of refcount-only `InitializeEnvironment`/`DestroyEnvironment`) under `-race` on this machine: **0.13s per run**, confirming issue #24's premise that this workload is checkptr-safe and cheap — but this measurement also surfaces a real risk: issue #24's *second* proposed test (`TestStressMixedOperations`, "run for 10 seconds") would, under the new job's `-count=50`, add roughly 500 seconds by itself if implemented as a wall-clock-duration loop — dangerously close to the CI job's 10-minute ceiling once combined with the other stress tests and job overhead. This is a concrete, quantified pitfall the planner must account for (see Common Pitfalls #1).

**Primary recommendation:** Export a new `ort.ErrUnsupportedPlatform` sentinel (mirroring `errBootstrapRedirectPolicy`/`isBootstrapRedirectPolicyError`, but exported since it must be checked from `examples/inference`), convert the 3 named tests to event-order-recording assertions with a single-shot 500ms `select`/`time.After` watchdog run *only on the main test goroutine* (never a spawned watchdog goroutine — `t.Fatal` from another goroutine is invalid per `testing.T.FailNow` docs), and implement the 3 stress tests from issue #24 verbatim except redesign `TestStressMixedOperations` as a **fixed-iteration** loop (not a fixed 10-second wall-clock loop) to keep the new CI job's runtime bounded and deterministic.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Fail-fast unsupported-platform diagnostic (DX-01) | Example/CLI (`examples/inference/main.go`) | Core FFI (`ort/bootstrap.go` — sentinel export only) | Issue #42 explicitly scopes this "example UX only, no `ort` runtime changes"; the sentinel error must originate in `ort` (where the error is created) but the hint text/formatting stays in the example |
| Deterministic concurrency assertions (TST-01) | Test Infrastructure (`ort/session_test.go`) | Core FFI (`ort/session.go`, `ort/tensor.go` — unchanged, code-under-test) | Tests validate the existing lock hierarchy documented in `ort/environment.go`; no production lock/lifecycle code changes are needed or in scope |
| Concurrent init/destroy stress coverage (TST-02) | Test Infrastructure (new `ort/environment_stress_test.go`) | CI/CD (`.github/workflows/ci.yml` new job, `TESTING.md`) | Stress tests exercise `ort/environment.go`'s refcount/mutex logic only (no FFI/unsafe); CI layer owns scheduling parameters (`-count`, `-parallel`, `timeout-minutes`) |

## Standard Stack

### Core

No new libraries. This phase uses only Go stdlib packages already imported elsewhere in the repo:

| Package | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `testing` | stdlib (go1.25) | Test framework, `testing.Short()` gate | Already the project's only test framework |
| `sync`, `sync/atomic` | stdlib | `WaitGroup`, `Once`, atomic counters for event ordering | Already used throughout `ort/session_test.go` and `ort/environment_test.go` |
| `errors` | stdlib | `errors.New`, `errors.Is`, `%w` wrapping | Already the established pattern (`errBootstrapRedirectPolicy`) |
| `time` | stdlib | Single-shot watchdog (`time.After`), not polling | Replaces `require.Eventually`/`require.Never` polling loops |
| `runtime` | stdlib | `runtime.GOOS`/`runtime.GOARCH` for the DX-01 hint | Already imported in `ort/bootstrap.go`; needed in `examples/inference/main.go` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `github.com/stretchr/testify` (require) | v1.11.1 (already in go.sum) | Remains used elsewhere in `session_test.go` (6 other `require.Eventually` call sites outside D-03's scope — see D-05) | Do not remove the import; only the 3 named tests' assertions change |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Channel/event-order rendezvous (this research's recommendation) | `testing/synctest` (stable in Go 1.25, `go.mod` already declares `go 1.25.0`) [VERIFIED: go doc testing/synctest, go.dev/blog/synctest] | `synctest.Test` gives a fake clock + deterministic scheduling inside an isolated "bubble," which is architecturally a cleaner fit for "prove ordering without wall-clock." **Not recommended for this phase**: CONTEXT.md's D-04 already locks in "explicit channel/handshake rendezvous points" as the mechanism, and `synctest`'s bubble model requires all goroutines-under-test to be "durably blocked" to advance its fake clock — the code under test here blocks on real `sync.Mutex`/channel operations controlled by test-mock FFI functions (`runSessionFunc`, `releaseSessionFunc`), which is compatible in principle but would be a bigger refactor than the locked-in decision calls for. Worth flagging to the user as a future alternative for Phase 4/5 test-hardening follow-up, not this phase. |
| Fixed-duration stress loop (issue #24's literal "run for 10 seconds" proposal) | Fixed-iteration-count loop | Fixed duration is non-deterministic in coverage (different iteration counts on fast vs. slow CI runners) and, combined with `-count=50` in the new CI job, risks exceeding the 10-minute timeout (see Common Pitfalls #1). Fixed iteration count is both more deterministic and safely boundable. |

**Installation:** None — no new dependencies to install for this phase.

**Version verification:** Not applicable — no new packages. `go.mod` already pins `go 1.25.0`; confirmed locally with `go version` → `go1.26.5` (toolchain forward-compatible; CI matrix uses `1.25.x`/`1.24` per `.github/workflows/ci.yml`).

## Package Legitimacy Audit

**Not applicable.** This phase introduces zero new external packages — it only modifies test files (`ort/session_test.go`, new `ort/environment_stress_test.go`), one example file (`examples/inference/main.go`), one library file (`ort/bootstrap.go`, sentinel export only), one CI workflow file, and `TESTING.md`. The Package Legitimacy Gate protocol is skipped; there is nothing to audit.

## Architecture Patterns

### System Architecture Diagram

**DX-01 error flow (unsupported-platform diagnostic):**

```
examples/inference/main()
  └─> initializeOrtEnvironment()
        └─> ort.EnsureOnnxRuntimeSharedLibrary()          [ort/bootstrap.go]
              └─> resolveRuntimeArtifact(goos, goarch)     [ort/bootstrap.go:467]
                    └─> [unsupported combo] returns
                        fmt.Errorf("...: %w", ort.ErrUnsupportedPlatform, goos, goarch)
                                     │
                                     ▼ (propagates up unwrapped by %w chain)
        initializeOrtEnvironment() returns wrapped err
                                     │
                                     ▼
  main(): if err := initializeOrtEnvironment(); err != nil {
             if ort.IsUnsupportedPlatformError(err) {   ◄── errors.Is check (D-02)
                 print GOOS/GOARCH + "set ONNXRUNTIME_LIB_PATH" hint
             } else {
                 print err unchanged (checksum/network/etc. — D-01)
             }
             os.Exit(1) / log.Fatal(...)                ◄── already exits non-zero today
          }
```

**TST-01 deterministic rendezvous flow (per-test, e.g. `TestAdvancedSessionRunAndDestroyConcurrent`):**

```
main test goroutine
  │
  ├─ spawn Run() goroutine ──────► enters mock runSessionFunc
  │                                   │ closes runStarted (rendezvous #1)
  │                                   │ blocks on <-allowRunReturn
  │                                   │ (on unblock) records "run-returned" event
  │
  ├─ <-runStarted  (blocks until rendezvous #1 fires — no sleep)
  │
  ├─ spawn Destroy() goroutine ───► calls releaseSessionFunc
  │                                   │ (only reachable after Run()'s runMu releases)
  │                                   │ records "destroy-released" event
  │
  ├─ select {                       ◄── single-shot watchdog, NOT the assertion itself
  │     case <-destroyErrCh: t.Fatal("destroy returned before run completed")
  │     case <-time.After(500ms):   // expected path — proves nothing by itself
  │  }
  │
  ├─ close(allowRunReturn)  → unblocks Run()
  ├─ <-runErrCh; <-destroyErrCh     (deterministic joins, no polling)
  └─ assert recorded event order: "run-returned" appears before "destroy-released"
     ◄── THIS is the deterministic correctness proof, not the watchdog
```

### Recommended Project Structure

No new packages/directories — files touched or added:

```
ort/
├── bootstrap.go                    # add ErrUnsupportedPlatform + IsUnsupportedPlatformError (exported sentinel)
├── bootstrap_test.go                # add test asserting errors.Is(err, ort.ErrUnsupportedPlatform) for unsupported combos
├── session_test.go                  # modify 3 named tests: event-order recording + single-shot watchdog
├── environment_stress_test.go       # NEW — 3 TestStress* functions gated behind testing.Short()
examples/inference/
├── main.go                          # add errors.Is(err, ort.ErrUnsupportedPlatform) branch + hint text
├── main_test.go                     # NEW — unit test for the extracted diagnostic-formatting helper
.github/workflows/ci.yml              # add new dedicated stress-test job (separate from test-race-ort-concurrency)
TESTING.md                            # add "Running stress tests" section
Makefile / ci.yml unit-test steps     # add -short to default `go test ./...` invocations (see Pitfall #2)
```

### Pattern 1: Exported sentinel error across a package boundary
**What:** A package-level `errors.New` sentinel, wrapped with `%w` at the error's origin, exported (capitalized) so a *different* package can detect it via `errors.Is`, following the codebase's own established idiom.
**When to use:** Whenever `examples/inference` (or any external consumer) needs to distinguish one specific `ort` failure mode from others without string-matching.
**Example:**
```go
// Source: existing pattern at ort/bootstrap.go:50,463-465 (errBootstrapRedirectPolicy),
// adapted to be exported for cross-package use.
var ErrUnsupportedPlatform = errors.New("unsupported platform for ONNX Runtime bootstrap")

func IsUnsupportedPlatformError(err error) bool {
	return errors.Is(err, ErrUnsupportedPlatform)
}

// at the error's origin (ort/bootstrap.go:522), wrap with %w instead of a bare fmt.Errorf:
return runtimeArtifact{}, fmt.Errorf("%w: GOOS=%s GOARCH=%s", ErrUnsupportedPlatform, goos, goarch)
```
```go
// examples/inference/main.go — detection + hint
if err := initializeOrtEnvironment(); err != nil {
	if ort.IsUnsupportedPlatformError(err) {
		log.Fatalf("failed to initialize ONNX Runtime: %v\nGOOS=%s GOARCH=%s is not supported by automatic bootstrap; "+
			"set ONNXRUNTIME_LIB_PATH to a prebuilt ONNX Runtime shared library for this platform.", err, runtime.GOOS, runtime.GOARCH)
	}
	log.Fatalf("failed to initialize ONNX Runtime: %v", err)
}
```
Note: the existing unexported `isBootstrapRedirectPolicyError` (line 463) stays private — it is used only for internal retry-loop decisions in `ort/bootstrap.go` (redirect handling), a different concern from DX-01's cross-package detection. Do not conflate or reuse it; add a new, separate, exported sentinel.

### Pattern 2: Event-order recording instead of negative wall-clock assertions
**What:** Record ordered events from within test-controlled mock hooks (`runSessionFunc`, `releaseSessionFunc`, `releaseValueFunc` — all already test-injectable via package-level function variables in `ort/environment.go`) into a mutex-guarded slice, then assert the recorded order *after* both goroutines have joined — never while they're still running.
**When to use:** Any test currently using `require.Never(..., someWindow, somePoll, "X returned before Y completed")` or `require.Eventually` to prove ordering/blocking behavior.
**Why it's actually deterministic (unlike a single `select`+`time.After`):** The assertion is a plain slice-equality/order check performed only after `wg.Wait()`/channel joins complete — it has zero dependency on how long anything took, only on what happened first. A `select { case <-ch: ...; case <-time.After(d): ... }` is still fundamentally probabilistic (a slow CI runner could theoretically let the "wrong" branch win) — it should be scoped to *hang detection only*, exactly as CONTEXT.md's D-04 specifies.
**Example:**
```go
// Source: adapted from existing ort/session_test.go:679-751 (TestAdvancedSessionRunAndDestroyConcurrent)
var eventsMu sync.Mutex
var events []string
record := func(e string) {
	eventsMu.Lock()
	events = append(events, e)
	eventsMu.Unlock()
}

runSessionFunc = func(...) uintptr {
	closeRunStarted.Do(func() { close(runStarted) })
	<-allowRunReturn
	record("run-returned")
	return 0
}
releaseSessionFunc = func(handle uintptr) {
	record("destroy-released")
	atomic.AddInt32(&releasedCount, 1)
}

// ... spawn Run() and Destroy() goroutines, rendezvous on runStarted ...

// Watchdog: hang-detection ONLY. Must run on the main test goroutine —
// testing.T.FailNow() (which t.Fatal calls) is documented as invalid from
// other goroutines. [VERIFIED: go doc testing.T.FailNow]
select {
case err := <-destroyErrCh:
	t.Fatalf("destroy returned before run completed (err=%v) -- deadlock-safety-net fired unexpectedly early", err)
case <-time.After(500 * time.Millisecond):
	// expected: destroy is still blocked; continue.
}

close(allowRunReturn)
if err := <-runErrCh; err != nil { t.Fatalf("run failed: %v", err) }
if err := <-destroyErrCh; err != nil { t.Fatalf("destroy failed: %v", err) }

// Deterministic assertion — the actual correctness proof:
eventsMu.Lock()
got := append([]string(nil), events...)
eventsMu.Unlock()
want := []string{"run-returned", "destroy-released"}
if !slices.Equal(got, want) {
	t.Fatalf("expected event order %v, got %v", want, got)
}
```

### Pattern 3: `testing.Short()`-gated stress tests with fixed iteration counts
**What:** Long-running/high-iteration tests skip themselves when `-short` is passed, per Go's standard convention. Use fixed iteration counts, not wall-clock durations, so total CI runtime is boundable and coverage is deterministic across fast/slow runners.
**When to use:** `ort/environment_stress_test.go`'s 3 `TestStress*` functions.
**Example:**
```go
// Source: adapted from GitHub issue #24's proposal (verified via `gh issue view 24`),
// with TestStressMixedOperations redesigned to fixed iterations instead of "run for 10 seconds"
// (see Common Pitfalls #1 for why).
func TestStressConcurrentInitDestroy(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")
	}
	resetEnvironmentState()
	defer resetEnvironmentState()

	const goroutines = 100
	const iterations = 1000

	if err := SetSharedLibraryPath("/nonexistent/path.so"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				_ = InitializeEnvironment()
				_ = DestroyEnvironment()
			}
		}()
	}
	wg.Wait()

	mu.Lock()
	got := refCount
	mu.Unlock()
	if got != 0 {
		t.Fatalf("expected refCount == 0 after balanced init/destroy cycles, got %d", got)
	}
}
```

### Anti-Patterns to Avoid
- **Wall-clock-duration stress loops** ("run for N seconds"): non-deterministic iteration count, hard to bound under `-count=50` in CI. Use fixed iteration counts instead.
- **Watchdog goroutines calling `t.Fatal`**: invalid per stdlib docs — `FailNow`/`Fatal` must run on the test's own goroutine. Put the `select`+`time.After` watchdog inline in the main test body, not in a spawned goroutine.
- **`testing.Short()` without updating default invocations**: the gate only skips tests when some caller passes `-short`; if no default `go test ./...` invocation in `Makefile`/CI passes `-short`, the "skip by default" intent is not actually achieved (see Common Pitfalls #2).
- **Reusing `isBootstrapRedirectPolicyError`'s naming/scope for DX-01**: that helper is intentionally unexported and scoped to redirect-policy retry logic inside `ort/bootstrap.go`. DX-01 needs a *new*, separately-named, *exported* sentinel — don't overload the existing one.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cross-package error classification | String-matching on `err.Error()` | Exported sentinel + `errors.Is` | Already the codebase's established pattern (`errBootstrapRedirectPolicy`); string-matching breaks silently if the message text changes |
| Proving goroutine ordering deterministically | A "smarter" polling loop (shorter interval, more retries) | Event-order recording + single join | Polling loops are still fundamentally probabilistic — they can never make an assertion 100% deterministic, only reduce (not eliminate) flake probability |
| Bounding a hang-prone test | Manual `context.WithTimeout` + goroutine leak accounting | `select { case <-doneCh: ; case <-time.After(d): t.Fatal(...) }` on the main test goroutine | Simpler, stdlib-only, matches D-04's "reuse the existing 500ms value" instruction exactly |

**Key insight:** Nothing in this phase requires a new library. The entire "hardening" is a matter of applying Go stdlib idioms (`errors.Is`, channels, `sync.WaitGroup`, `testing.Short()`) more precisely than the current code does — consistent with the project's "radically simple" directive.

## Common Pitfalls

### Pitfall 1: `-count=50` CI job multiplier can blow the 10-minute budget if any stress test uses a wall-clock duration
**What goes wrong:** Issue #24's `TestStressMixedOperations` proposal literally says "Run for 10 seconds, verify no panics or deadlocks." The locked CI job (D-08) runs `go test -race -run=TestStress -count=50 ...` with `timeout-minutes: 10`. A 10-second-duration test repeated 50 times is ~500 seconds (8.3 minutes) from that one test alone, before adding `TestStressConcurrentInitDestroy` and `TestStressRapidInitDestroy`, plus checkout/setup/compile overhead. This risks intermittent CI timeouts that look like "flaky" failures but are actually a sizing bug.
**Why it happens:** The issue's proposed test descriptions predate the concrete `-count=50` CI parameters locked into D-08; the two were never reconciled against each other.
**How to avoid:** Implement `TestStressMixedOperations` (and any other stress test) with a **fixed iteration count** (e.g., a fixed number of mixed operations across goroutines, joined via `WaitGroup`) rather than a wall-clock duration. I empirically measured `TestStressConcurrentInitDestroy`'s exact proposed shape (100 goroutines × 1000 iterations, refcount-only, no `-race` real FFI) at **0.13s per run** on this machine [VERIFIED: measured locally] — so even a considerably larger fixed-iteration mixed-ops test (e.g., 50 goroutines × 500 mixed calls) should stay well under a few hundred milliseconds per run, keeping `-count=50` comfortably inside the 10-minute ceiling.
**Warning signs:** New CI job passes locally (`go test -race -run TestStress ./ort/...` once) but times out or is unreliable specifically in the dedicated stress job with `-count=50` applied.

### Pitfall 2: `testing.Short()` gate is inert unless `-short` is added to default test invocations
**What goes wrong:** `testing.Short()` returns `true` only when the `-short` flag is explicitly passed to `go test`. Grepping this repo confirms **no existing invocation currently passes `-short`** — not `Makefile`'s `test`/`test-race`/`precommit` targets, not either CI unit-test step (`ci.yml` lines ~88-97). If the new stress tests are gated purely with `if testing.Short() { t.Skip(...) }` and nothing else changes, they will still run on every default `go test ./...`, `make test`, and `make precommit` invocation — contradicting D-06's "normal `go test ./...` / CI runs skip them by default."
**Why it happens:** `testing.Short()`'s polarity is easy to misread — it's an opt-in "run less" flag, not an opt-in "run more" flag.
**How to avoid:** Add `-short` to the repo's default test invocations that should *not* run stress tests: `Makefile`'s `test:` target (line 107), `precommit`'s `go test ./...` step (line 361), and both CI unit-test steps (`ci.yml` lines ~90/97). Leave the new dedicated stress job (and `make test-race`, which already targets a curated concurrency subset) without `-short` so `testing.Short()` is `false` there and the stress tests run.
**Warning signs:** `make precommit` or the main CI unit-test job takes noticeably longer after this phase merges, or stress test output appears in a job that wasn't supposed to run it.

### Pitfall 3: `t.Fatal`/`t.FailNow` from a non-test goroutine is invalid
**What goes wrong:** If the watchdog is implemented as a separate goroutine that calls `t.Fatal(...)` when a timeout fires, this violates the documented contract: *"FailNow must be called from the goroutine running the test or benchmark function, not from other goroutines created during the test. Calling FailNow does not stop those other goroutines."* [VERIFIED: go doc testing.T.FailNow] The test can appear to pass, panic unpredictably, or produce a confusing `-race`-flagged data race on `t`'s internal state.
**Why it happens:** It's tempting to spawn a "supervisor" goroutine for hang detection, mirroring patterns seen in some server code.
**How to avoid:** Structure the watchdog as an inline `select` on the *main test goroutine* (see Pattern 2's example) — never spawn a separate watchdog goroutine that calls test-failing functions.
**Warning signs:** Flaky test failures under `-race` specifically, or test output showing failures attributed to the wrong test name.

### Pitfall 4: Conflating the redirect-policy sentinel's scope with the new DX-01 sentinel
**What goes wrong:** `errBootstrapRedirectPolicy`/`isBootstrapRedirectPolicyError` (unexported, `ort/bootstrap.go:50,463`) is scoped narrowly to HTTP redirect-policy rejection during download retries. It is tempting to reuse or extend it for the unsupported-platform case since D-02 calls out "the existing codebase pattern," but they are unrelated failure modes with different retry semantics (redirect-policy errors are marked permanent via `markPermanentBootstrapError`; unsupported-platform errors never reach the retry/download path at all — `resolveRuntimeArtifact` fails before any network I/O).
**Why it happens:** Surface-level pattern-matching ("there's already a sentinel-error helper, just add a case to it").
**How to avoid:** Add a **new**, separate, **exported** sentinel (`ErrUnsupportedPlatform`) and its own `IsUnsupportedPlatformError` helper. Don't touch `errBootstrapRedirectPolicy`.
**Warning signs:** A PR diff that changes `isBootstrapRedirectPolicyError`'s signature or the `errBootstrapRedirectPolicy` error text.

## Code Examples

### DX-01: testable diagnostic-formatting helper (avoids needing to exec the built binary)
```go
// Source: new pattern for examples/inference/main.go + examples/inference/main_test.go
// Extracting formatting into a pure function makes it unit-testable without
// spawning the compiled binary or faking runtime.GOOS/GOARCH (which are compile-time
// constants and can't be overridden in-process).
func diagnosticFor(err error, goos, goarch string) string {
	if ort.IsUnsupportedPlatformError(err) {
		return fmt.Sprintf(
			"failed to initialize ONNX Runtime: %v\n"+
				"GOOS=%s GOARCH=%s is not supported by automatic bootstrap; "+
				"set ONNXRUNTIME_LIB_PATH to a prebuilt ONNX Runtime shared library for this platform.",
			err, goos, goarch,
		)
	}
	return fmt.Sprintf("failed to initialize ONNX Runtime: %v", err)
}

// main.go:
if err := initializeOrtEnvironment(); err != nil {
	log.Fatal(diagnosticFor(err, runtime.GOOS, runtime.GOARCH))
}
```
```go
// examples/inference/main_test.go (new file)
func TestDiagnosticForUnsupportedPlatform(t *testing.T) {
	// Construct the same error resolveRuntimeArtifact would produce for an
	// unsupported combo, without depending on the actual host platform.
	err := fmt.Errorf("%w: GOOS=%s GOARCH=%s", ort.ErrUnsupportedPlatform, "plan9", "386")
	got := diagnosticFor(err, "plan9", "386")
	if !strings.Contains(got, "GOOS=plan9") || !strings.Contains(got, "GOARCH=386") {
		t.Fatalf("expected GOOS/GOARCH in diagnostic, got: %q", got)
	}
	if !strings.Contains(got, "ONNXRUNTIME_LIB_PATH") {
		t.Fatalf("expected ONNXRUNTIME_LIB_PATH hint, got: %q", got)
	}
}

func TestDiagnosticForOtherBootstrapFailureUnchanged(t *testing.T) {
	err := errors.New("checksum mismatch")
	got := diagnosticFor(err, "linux", "amd64")
	if strings.Contains(got, "ONNXRUNTIME_LIB_PATH") {
		t.Fatalf("expected no misleading hint for non-platform failures, got: %q", got)
	}
}
```

### Bootstrap sentinel test (mirrors existing `resolveRuntimeArtifact` table test)
```go
// Source: adapted from existing ort/bootstrap_test.go:100-105 ("unsupported" case)
{
    name:   "unsupported",
    goos:   "linux",
    goarch: "386",
    wantErr: true,
},
// ...in the loop, for the wantErr case, additionally assert:
if !errors.Is(err, ErrUnsupportedPlatform) {
    t.Fatalf("expected ErrUnsupportedPlatform, got: %v", err)
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| `require.Eventually`/`require.Never` with 500ms window + 50ms poll | Channel rendezvous + event-order recording, single-shot 500ms watchdog for hang detection only | This phase (TST-01) | Removes the last-known flake source under loaded CI runners while preserving a bounded failure mode for genuine deadlocks |
| No stress coverage for concurrent init/destroy | `testing.Short()`-gated `TestStress*` suite, run under `-race` in a dedicated CI job | This phase (TST-02) | `-race` was previously believed unusable for *any* `ort` concurrency test due to checkptr/purego incompatibility; this phase demonstrates (per D-07, confirmed by my local probe run) that refcount-only tests avoiding real FFI/unsafe pointers are fully `-race`-compatible |

**Deprecated/outdated:** None — no library deprecations are relevant; this is purely an internal-pattern change.

**Note on `testing/synctest`:** Go 1.25 (the version already pinned in `go.mod`) ships `testing/synctest` as a *stable* package (graduated from the Go 1.24 `GOEXPERIMENT=synctest` flag) [VERIFIED: `go doc testing/synctest` on this machine; go.dev/blog/synctest]. It is a legitimate alternative to hand-rolled channel rendezvous for future concurrency-test work, but CONTEXT.md's D-04 already locks in the channel/handshake approach for this phase — flagging `synctest` here as a documented option for a *future* phase, not a recommendation to change course now.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Recommended exported sentinel name `ErrUnsupportedPlatform` / helper name `IsUnsupportedPlatformError` (exact identifiers are Claude's discretion per CONTEXT.md) | Architecture Patterns, Code Examples | Low — purely a naming choice, CONTEXT.md explicitly delegates this to implementation discretion |
| A2 | Fixed-iteration redesign of `TestStressMixedOperations` (50 goroutines × 500 ops, or similar) will keep the CI job's `-count=50` run comfortably under 10 minutes | Common Pitfalls #1, Code Examples | Medium — this is an extrapolation from a measured 0.13s baseline for a *different*, simpler test shape; the planner/implementer should re-measure the actual mixed-ops test locally with `-race -count=50` before locking the CI job's timeout value |
| A3 | `Makefile`/CI default invocations need `-short` added for the `testing.Short()` gate to have any effect (Pitfall #2) | Common Pitfalls #2, Anti-Patterns | Low — directly confirmed by grepping the repo for existing `-short`/`testing.Short()` usage (found: none) and reading the stdlib `testing.Short()` doc; if this assumption is wrong the actual effect (stress tests always running) would still just be a (harmless, per measurement) perf footnote, not a correctness bug |

## Open Questions

1. **(RESOLVED) Should `Makefile`/CI default invocations be updated to pass `-short` as part of this phase's scope?**
   - What we know: D-06 requires the stress tests to be "gated behind `testing.Short()`, so normal `go test ./...` / CI runs skip them by default." CONTEXT.md's D-08 enumerates the CI/TESTING.md changes explicitly but does not mention updating `Makefile`'s `test`/`precommit` targets or the existing CI unit-test steps to pass `-short`.
   - What's unclear: Whether this is an oversight in CONTEXT.md's scoping or an intentional decision that "skip by default" only needs to hold true for the new dedicated stress job's sibling jobs, tolerating the (currently negligible, ~0.1-0.5s) extra runtime in the main unit-test job.
   - Recommendation: Treat this as in-scope for TST-02 — add `-short` to `Makefile`'s `test:` and `precommit`'s `go test ./...` step, and to both CI unit-test steps in `ci.yml`. This is the only way D-06's literal wording ("skip them by default") is actually achieved; leaving it out makes the `testing.Short()` gate cosmetic. Flag for the planner to confirm with the user if there's a reason to leave default invocations unchanged.
   - **Resolution:** Resolved explicitly in `01-03-PLAN.md`'s `<rationale>` block — `-short` is wired into `Makefile`'s `test`/`precommit` targets and both CI unit-test steps in `ci.yml`; `test-race`/`test-race-ort-concurrency` are left untouched since their curated regexes never match `TestStress*`.
2. **(RESOLVED) Exact iteration/goroutine counts for the redesigned `TestStressMixedOperations` beyond `TestStressConcurrentInitDestroy`.**
   - **Resolution:** Locked in `01-03-PLAN.md` Task 1 as concrete fixed-iteration parameters (100×1000, 200×500, 50×500) — avoids the wall-clock CI budget risk flagged above.

2. **Exact parameters for the fixed-iteration `TestStressMixedOperations`**
   - What we know: Issue #24 proposes concurrent `InitializeEnvironment`/`DestroyEnvironment`/`IsInitialized`/`GetVersionString`/`SetSharedLibraryPath`/`SetLogLevel` calls, verifying "no panics or deadlocks" — CONTEXT.md's D-06 explicitly leaves "exact set and parameters" to Claude's discretion.
   - What's unclear: The precise goroutine/iteration counts that balance meaningful stress coverage against the CI timeout budget.
   - Recommendation: Start with something in the same order of magnitude as `TestStressConcurrentInitDestroy` (tens of goroutines × hundreds of mixed calls), measure locally under `-race -count=50` before finalizing, and adjust down if needed to stay well under the 10-minute job ceiling with margin for CI variance.

## Environment Availability

Not applicable — this phase has no new external tool/service/runtime dependencies. `go`, `gofmt`, `go vet`, and `golangci-lint`/`gosec` (used by `make precommit`) are already required and available per existing project tooling; nothing new is introduced.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Go stdlib `testing` (+ `testify/require` for assertions not touched by this phase) |
| Config file | none — `go test` built-in; linter config at `.golangci.yml` |
| Quick run command | `go test -race -run 'TestAdvancedSessionRunAndDestroyConcurrent|TestTensorDestroyWaitsForInFlightRun|TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun' ./ort/...` |
| Full suite command | `make precommit` (fmt, vet, lint-new, gosec, `go test ./...`, mod-tidy, vulncheck) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DX-01 | Unsupported-platform bootstrap failure produces GOOS/GOARCH + `ONNXRUNTIME_LIB_PATH` hint; other bootstrap failures unchanged | unit | `go test ./examples/inference/... -run TestDiagnosticFor` | ❌ Wave 0 (new `examples/inference/main_test.go`) |
| DX-01 | `ort.ErrUnsupportedPlatform` correctly wraps `resolveRuntimeArtifact`'s unsupported-combo error | unit | `go test ./ort/... -run TestResolveRuntimeArtifact` | ✅ (modify existing `ort/bootstrap_test.go`) |
| TST-01 | 3 named tests assert via channel/handshake + event order, no sleep-based primary assertions remain | unit/race | `go test -race -run 'TestAdvancedSessionRunAndDestroyConcurrent|TestTensorDestroyWaitsForInFlightRun|TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun' ./ort/...` | ✅ (modify existing `ort/session_test.go`) |
| TST-02 | Stress tests exercise many concurrent init/destroy cycles and pass under `-race` | stress/race | `go test -race -run TestStress -count=10 ./ort/...` (quick local sanity; CI uses `-count=50`) | ❌ Wave 0 (new `ort/environment_stress_test.go`) |

### Sampling Rate
- **Per task commit:** `go test -race -run '<test-being-changed>' ./ort/...`
- **Per wave merge:** `make precommit` plus `go test -race -run TestStress -count=5 ./ort/...` (fast local sanity for the new stress suite before pushing)
- **Phase gate:** Full `make precommit` green, plus a manual/CI run of the new dedicated stress job before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `ort/environment_stress_test.go` — covers TST-02 (new file, does not exist)
- [ ] `examples/inference/main_test.go` — covers DX-01 (new file, does not exist; requires extracting `diagnosticFor` or equivalent testable helper out of `main()`)
- [ ] `Makefile`/`ci.yml` `-short` wiring — needed for D-06's "skip by default" to actually hold (see Open Question #1)

## Security Domain

This phase touches only test code, one CLI example's error-message formatting, and CI configuration — it introduces no new user-facing input surface, authentication, session, or persistence logic.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | No | N/A — no auth in this project |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | No (unchanged) | `ONNXRUNTIME_LIB_PATH`/`GOOS`/`GOARCH` are existing, already-handled inputs; this phase only changes *diagnostic output* formatting, not input parsing |
| V6 Cryptography | No | N/A — no crypto touched by this phase |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| Log/error-message injection via untrusted data | Tampering/Information Disclosure | Not applicable here — `runtime.GOOS`/`runtime.GOARCH` are compile-time constants, not attacker-controlled; no new untrusted data flows into formatted error strings in this phase |

## Sources

### Primary (HIGH confidence)
- `gh issue view 24/42/43 --repo amikos-tech/pure-onnx` — fetched directly, canonical specs for TST-02/DX-01/TST-01
- `go doc testing.Short`, `go doc testing.T.Deadline`, `go doc testing.T.FailNow`, `go doc errors.Is`, `go doc testing/synctest` — run locally against go1.26.5 toolchain
- Local repo inspection: `ort/bootstrap.go`, `ort/environment.go`, `ort/session_test.go`, `ort/environment_test.go`, `ort/bootstrap_test.go`, `examples/inference/main.go`, `.github/workflows/ci.yml`, `Makefile`, `TESTING.md`, `.golangci.yml`, `go.mod`
- Local empirical measurement: 100-goroutine × 1000-iteration refcount-only stress probe run under `go test -race` — 0.13s test runtime (16.9s total including compile), zero race warnings

### Secondary (MEDIUM confidence)
- [The Go blog: Testing concurrent code with testing/synctest](https://go.dev/blog/synctest) — cross-referenced with local `go doc` output confirming stable status in Go 1.25

### Tertiary (LOW confidence)
- None — all findings above were verified against either the local toolchain, the local repo, or the canonical GitHub issues.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all patterns confirmed against stdlib docs run locally
- Architecture: HIGH — every pattern traced to exact existing line numbers in this repo, plus canonical GitHub issue text fetched directly
- Pitfalls: HIGH — Pitfall 1 is a quantified arithmetic risk from a real local measurement; Pitfalls 2-4 are confirmed by direct repo inspection (grep) and stdlib doc text

**Research date:** 2026-07-21
**Valid until:** Stable — this is stdlib-only, internal-pattern research with no external API surface to go stale; re-verify only if `go.mod`'s Go version changes materially (e.g., a future migration to `testing/synctest`)
