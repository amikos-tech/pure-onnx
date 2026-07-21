---
phase: 01-dx-test-hardening
reviewers: [codex, opencode]
reviewed_at: 2026-07-21T08:54:43Z
plans_reviewed: [01-01-PLAN.md, 01-02-PLAN.md, 01-03-PLAN.md]
---

# Cross-AI Plan Review — Phase 1: DX & Test Hardening

## Codex Review

### 01-01 Plan Review

**Summary:** Strong, tightly scoped DX plan. It preserves existing error text, uses typed error detection across package boundaries, and adds focused tests for both the new and unchanged diagnostic paths.

**Strengths**
- Correctly limits the user-facing hint to unsupported bootstrap platforms.
- Uses `errors.Is` with an exported sentinel instead of brittle string matching.
- Keeps `ort/` behavior unchanged except for error classification.
- Tests both the sentinel wrapping and the example's final diagnostic.

**Concerns**
- **LOW — Fragile table-test assumption:** Task 1 adds the sentinel assertion to every future `wantErr` case in `TestResolveRuntimeArtifact`. If another unrelated failing case is added later, the test will incorrectly require `ErrUnsupportedPlatform`.

**Suggestions**
- Add a `wantUnsupportedPlatform bool` field to the table and assert the sentinel only for that case.
- Include one assertion that the wrapped error text remains exactly `unsupported platform for ONNX Runtime bootstrap: GOOS=... GOARCH=...`, protecting compatibility explicitly.

**Risk Assessment:** **LOW.** The implementation is minimal, correctly ordered, and directly satisfies DX-01.

---

### 01-02 Plan Review

**Summary:** The plan improves the existing polling assertions, but it does not fully achieve deterministic proof of blocking/unblocking. Its event-order checks can still pass if the destroy goroutine simply has not run before the test releases `allowRunReturn`; the healthy path also waits 500ms unnecessarily.

**Strengths**
- Scope is disciplined: exactly the three named tests are changed.
- Retains existing lock-ownership checks and race-detector verification.
- Event recording inside mocked release functions is a good way to assert completion order.
- Avoids invalid `t.Fatal` calls from a background watchdog goroutine.

**Concerns**
- **HIGH — The two "destroy waits" tests do not prove that `Destroy()` was actually contending before `Run()` is released.** In `TestAdvancedSessionRunAndDestroyConcurrent` and `TestTensorDestroyWaitsForInFlightRun`, the destroy goroutine may be scheduled only after `close(allowRunReturn)`. The recorded order will still be `run-returned`, then `destroy-released`, so a regression can pass without testing the required wait behavior.
- **MEDIUM — The proposed 500ms watchdog is on the normal passing path.** Each healthy test deliberately waits 500ms before unblocking `Run()`, making the test suite slower and retaining a timing-based assertion as part of the pass condition.
- **MEDIUM — The unrelated-session test still depends on a wall-clock liveness assertion.** A direct receive with `time.After(500ms)` is simpler than polling, but it is still a timing threshold that determines pass/fail, so it does not fully meet the stated "deterministic synchronization" goal.

**Suggestions**
- Introduce a deterministic test seam/rendezvous that signals after `Destroy()` has reached the relevant lock-acquisition point, then release `Run()` only after that signal. This may require a narrowly scoped test hook or a small injectable lock/wait hook in the package.
- Use the 500ms timeout only to fail a hung rendezvous; do not intentionally wait for it during a healthy run.
- For the unrelated-session case, signal entry to `releaseSessionFunc` for handle `222` and use that as the positive event. Retain the timeout solely as a deadlock guard.

**Risk Assessment:** **MEDIUM.** It will likely reduce flakes, but the core waiting guarantee is not deterministically established by the planned synchronization.

---

### 01-03 Plan Review

**Summary:** The CI and documentation portions are well specified, but the planned stress tests are currently ineffective for their intended refcount coverage and the `-short` strategy does not meet its stated default-run behavior. These issues should be resolved before implementation.

**Strengths**
- Correctly creates a dedicated stress CI job instead of expanding the existing FFI-adjacent concurrency allowlist.
- Fixed iteration counts are appropriate with `-count=50` and a ten-minute CI limit.
- The test names, workload sizes, local command, CI invocation, and contributor documentation are concrete.
- Updating the normal Makefile and CI unit-test commands to use `-short` is a sensible attempt to keep routine runs fast.

**Concerns**
- **HIGH — The stress tests do not exercise refcount-only init/destroy cycles as written.** Starting from `refCount == 0` with a nonexistent library path means every `InitializeEnvironment()` attempts `loadLibrary`, fails, and leaves `refCount` at zero. Each following `DestroyEnvironment()` is a no-op. The tests therefore repeatedly test failed dynamic-library loading, not concurrent reference-count transitions.
- **HIGH — `make test-race` is not a curated subset.** The actual target runs `go test -v -race ./ort/...` with no `-run` or `-short`, so it will run all new `TestStress*` tests. The plan's rationale and acceptance criteria incorrectly claim it cannot match them.
- **HIGH — `go test ./...` will not skip `testing.Short()`-gated tests by default.** `testing.Short()` is false unless the caller explicitly provides `-short`. Updating CI, `make test`, and `make precommit` does not change the behavior of a contributor running the literal `go test ./...` command stated in the plan's success criteria.
- **MEDIUM — The mixed-operations scenario is underspecified and its final `refCount == 0` check can be vacuous.** A rotating sequence of independently selected init/destroy operations does not establish a known expected balance. With the proposed failed initializations, it will always end at zero regardless of lifecycle correctness.
- **MEDIUM — Repeated failed `loadLibrary` calls can create CI-runtime variance.** The proposed 100k/100k/25k operation shapes become much more expensive under `-race -count=50` if each initialization invokes the platform loader rather than taking the established refcount fast path.

**Suggestions**
- Follow the existing concurrency-test foundation: set `refCount = 1` under `mu` before launching workers, so `InitializeEnvironment()` takes the increment-only path and never loads a library. After all balanced worker operations complete, assert `refCount == 1`, then call one final `DestroyEnvironment()` and assert zero.
- Define the mixed test's operation accounting explicitly. For example, keep each worker's init/destroy operations paired, interleave read-only operations, and verify the known baseline refcount after joining.
- Add `-short` to `make test-race`, or change that target to an explicit curated `-run` expression, if stress tests must be excluded from normal race runs.
- Reword the documentation and success criteria to say "repository default commands" rather than raw `go test ./...`, or introduce an explicit stress opt-in mechanism if the raw command must skip stress tests. `testing.Short()` alone cannot provide both behaviors.

**Risk Assessment:** **HIGH.** Without a baseline initialized refcount, the central stress coverage is largely vacuous; additionally, the documented/default test-run behavior differs from what Go actually does.

---

## OpenCode Review

### 01-01-PLAN.md

**Summary:** Strong, well-scoped plan for DX-01. It keeps the behavior change in the inference example while adding only the minimal `ort/` support needed for typed error detection. The sentinel approach matches the codebase pattern and avoids brittle string matching. Main risk is API surface creep from exporting both a sentinel and helper for an example-only issue, but that is acceptable because cross-package detection requires some exported contract.

**Strengths**
- Correctly avoids changing bootstrap/runtime behavior beyond wrapping the unsupported-platform error.
- Uses `errors.Is` and `%w`, which is the right Go pattern.
- Explicitly preserves non-platform bootstrap error messages.
- Adds focused unit tests for both the `ort` sentinel and example diagnostic formatting.
- Keeps the fix scoped to `examples/inference/main.go` except for the exported sentinel.

**Concerns**
- **MEDIUM:** Exporting `ErrUnsupportedPlatform` and `IsUnsupportedPlatformError` creates public API surface for an example UX issue. Once exported, it becomes harder to rename or remove.
- **LOW:** The plan says the resulting unsupported-platform error text stays identical, but changing from `"unsupported platform for ONNX Runtime bootstrap: GOOS=..."` to `"%w: GOOS=..."` only stays identical if the sentinel string is exactly `"unsupported platform for ONNX Runtime bootstrap"`. The plan does specify this, so this is mostly an implementation watchpoint.
- **LOW:** `log.Fatal(diagnosticFor(...))` changes from formatted fatal to direct fatal. Behavior is fine, but tests should confirm no accidental extra formatting changes for the fallback branch.

**Suggestions**
- Add a doc comment for exported `ErrUnsupportedPlatform` and `IsUnsupportedPlatformError` to satisfy Go lint expectations.
- Consider whether only `ErrUnsupportedPlatform` is enough, since callers can use `errors.Is(err, ort.ErrUnsupportedPlatform)` directly. The helper is convenient but expands API surface.
- Include `go test ./examples/inference/...` in verification, not only `-run TestDiagnosticFor`, to catch package-level compile/test regressions.

**Risk Assessment:** **LOW.** The plan is minimal, testable, and aligned with the phase goal. The only meaningful risk is public API expansion, but the exported sentinel is justified by the package boundary.

---

### 01-02-PLAN.md

**Summary:** The plan targets exactly the three concurrency tests in scope and correctly replaces polling-based assertions with explicit synchronization and event ordering. It is directionally strong, but there is one subtle concern: using a 500ms timeout as the expected path in the two "destroy waits" tests still relies on wall-clock time to prove "still blocked." The recorded event order improves determinism after unblocking, but the "destroy did not return early" assertion still depends on waiting 500ms.

**Strengths**
- Correctly limits scope to the three named tests.
- Avoids modifying `TestAdvancedSessionRunConcurrent` and retry-backoff tests that are explicitly out of scope.
- Uses event recording inside mock FFI hooks, which is a good deterministic proof of final ordering.
- Keeps watchdog failures on the main test goroutine, avoiding invalid `t.Fatal` usage from goroutines.
- Verifies under `-race`, which is important for these tests.

**Concerns**
- **MEDIUM:** In `TestAdvancedSessionRunAndDestroyConcurrent` and `TestTensorDestroyWaitsForInFlightRun`, the passing path still waits for `time.After(500ms)` to establish that destroy has not returned. That is better than polling, but it is still a timing-based negative assertion. The event-order assertion catches eventual ordering, but it does not by itself prove destroy was blocked before `allowRunReturn` unless the test observes that destroy has reached the blocking point.
- **LOW:** Adding `slices` is fine for Go 1.21+, and the repo supports newer Go, but make sure CI's actual Go version supports it.
- **LOW:** Acceptance criterion "grep `require.Never` returns 0 matches anywhere" may be broader than the task scope. If unrelated future tests use `require.Never`, this criterion becomes unnecessarily brittle.
- **LOW:** Event names `"destroy-released"` are used for both session release and value release. That is acceptable locally, but could make failures slightly less descriptive.

**Suggestions**
- For the two "destroy waits" tests, consider adding a deterministic "destroy attempted" rendezvous before the watchdog if possible, so the test proves the destroy goroutine is actually contending/blocking rather than not yet scheduled.
- Narrow the acceptance check for `require.Never` to the two converted test bodies, unless the file truly has no other uses today and the broader check is intentional.
- Use more specific event names where useful, e.g. `"session-released"` and `"tensor-released"`.
- Keep the timeout branch comments very explicit: in the "waits" tests timeout is expected only as a scheduling ceiling, while in the "does not block unrelated run" test timeout is failure.

**Risk Assessment:** **MEDIUM.** The plan materially improves the tests and probably satisfies the issue, but the "destroy remains blocked" proof still contains a timing-based negative assertion. If the strict requirement is "no timing-based correctness assertion," this should be tightened before execution.

---

### 01-03-PLAN.md

**Summary:** This is the most ambitious plan and mostly well thought out: it adds stress coverage, a dedicated CI job, documentation, and makes `testing.Short()` meaningful by wiring `-short` into default test paths. The main risk is scope expansion and behavior change: adding `-short` to default `go test ./...`, `make test`, `make precommit`, and CI changes the default test contract for the entire repository, not just the new stress tests. That may be correct per the plan rationale, but it is the highest-impact decision in Phase 1.

**Strengths**
- Correctly separates the new stress job from the existing curated race job.
- Avoids wall-clock-duration stress loops, which is important with `-count=50`.
- Documents local stress-test usage in `TESTING.md`.
- Uses nonexistent library path/refcount-only paths to avoid real FFI and checkptr/race incompatibility.
- Explicitly validates YAML with `yq`, matching repo guidance.
- Makes `testing.Short()` actually effective instead of cosmetic.

**Concerns**
- **HIGH:** Adding `-short` to default test invocations may skip any current or future tests that use `testing.Short()` for reasons unrelated to these stress tests. Today research says there are no existing uses, but this changes the repository's default testing semantics permanently.
- **MEDIUM:** `TestStressMixedOperationsUnderLoad` mixes `InitializeEnvironment()` and `DestroyEnvironment()` independently. Depending on the implementation, unbalanced destroy calls may drive behavior that is not representative or may mask refcount corruption. The final `refCount == 0` assertion is useful, but intermediate negative counts or invalid transitions may go unnoticed unless the production code prevents them.
- **MEDIUM:** `SetSharedLibraryPath` under concurrent `InitializeEnvironment`/`DestroyEnvironment` is intentionally not in the mixed test despite issue #24 mentioning it in research. That may be fine, but the plan should explain why it was excluded, especially since `SetSharedLibraryPath` is named in the interface/context.
- **MEDIUM:** `go test -v -race -run=TestStress -count=50 -parallel=4 ./ort/...` in CI may still be expensive because each test includes up to 100k init/destroy pairs, repeated 50 times under race. Local measurement helps, but actual CI runners can be much slower.
- **LOW:** Importing `sync/atomic` "only if used" is a minor ambiguity. The described tests do not require it.
- **LOW:** Acceptance criteria around exact `-short` match counts in `.github/workflows/ci.yml` and `Makefile` may be brittle if comments/docs or future jobs also include `-short`.

**Suggestions**
- Reconsider using `testing.Short()` as the only gate. A more isolated pattern would be an explicit env gate such as `ONNX_PUREGO_STRESS=1`, but that would deviate from the current plan. If keeping `testing.Short()`, document clearly that default CI now runs short mode.
- Add a grep/verification step before editing to confirm there are still no existing `testing.Short()` usages outside the new stress file.
- In `TestStressMixedOperationsUnderLoad`, track successful `InitializeEnvironment()` calls and pair destroys more deliberately, or assert that `refCount` never goes negative if the package state permits checking that safely.
- Either include `SetSharedLibraryPath` in the mixed scenario with careful expected-error handling, or explicitly state why it is excluded from the implemented mix.
- Locally run the exact CI command once, not only `-count=10`, before finalizing the CI job timeout if feasible.
- Consider reducing CI `-count=50` if exact-command runtime is close to the 10-minute timeout.

**Risk Assessment:** **MEDIUM.** The stress-test design is reasonable, but the plan changes default test behavior repo-wide by introducing `-short`. That is a deliberate choice with good rationale, but it carries more long-term risk than the code/test additions themselves.

---

### Overall (OpenCode)

**Summary:** The three plans are coherent and mostly achieve Phase 1's goals. 01-01 is low-risk and well scoped. 01-02 improves determinism but should tighten the remaining negative timing proof if the requirement is interpreted strictly. 01-03 covers the requested stress-test scope thoroughly, but its `-short` wiring changes the default test contract and should be treated as an intentional project-level decision, not just test plumbing.

**Cross-Plan Strengths**
- Requirements map cleanly to plans: DX-01, TST-01, and TST-02 are all covered.
- Scope boundaries are repeatedly called out, especially avoiding runtime changes for #42 and avoiding unrelated timing sleeps for #43.
- Verification commands are concrete and include `-race` where it matters.
- Plans use stdlib-only approaches and avoid new dependencies.
- CI and documentation are included for the stress-test work, not left as follow-up.

**Cross-Plan Concerns**
- **HIGH:** Plan 01-03's `-short` change affects all current and future tests, not only the new stress tests.
- **MEDIUM:** Plan 01-02 may still rely on time passing as the success path for "destroy remains blocked" assertions.
- **MEDIUM:** All three plans are marked wave 1 with no dependencies, but 01-01 Task 2 depends on Task 1's exported sentinel. That dependency is internal to the plan, so it is fine, but executors must not parallelize those two tasks blindly.
- **LOW:** Several acceptance criteria rely on exact grep counts, which can be brittle if nearby comments or future tests include the same strings.
- **LOW:** Summary artifact creation is requested for each plan, but verification criteria focus mostly on code/tests. Make sure executors do not skip the `.planning/...SUMMARY.md` outputs.

**Overall Suggestions**
- Before executing 01-03, explicitly confirm the repo is comfortable with default `go test ./...` becoming `go test -short ./...` in Makefile/CI.
- Tighten 01-02 with an additional deterministic rendezvous showing the destroy goroutine has reached the contested operation before waiting.
- For 01-03, run the exact stress CI command locally once if practical: `go test -v -race -run=TestStress -count=50 -parallel=4 ./ort/...`.
- Keep the implementation minimal and avoid using this phase to refactor production lifecycle code.

**Overall Risk Assessment:** **MEDIUM.** The phase is well planned and likely to succeed, but two decisions deserve care: the remaining timing dependency in 01-02 and the repo-wide default `-short` behavior in 01-03. 01-01 is low-risk; 01-02 and 01-03 are manageable with the suggested tightening.

---

## Consensus Summary

Both reviewers independently reached the same shape of verdict: **01-01 is low-risk and ready to execute as-is**, while **01-02 and 01-03 have a shared structural flaw worth fixing before execution**.

### Agreed Strengths
- DX-01 (01-01) correctly scopes the fix to the example, uses the existing sentinel + `errors.Is` idiom instead of string matching, and leaves `ort/` runtime behavior unchanged.
- All three plans map cleanly to their requirements (DX-01, TST-01, TST-02) and repeatedly call out what's explicitly out of scope (no `ort/` changes for #42, no touching `TestAdvancedSessionRunConcurrent`/retry-backoff tests for #43).
- 01-03's CI design correctly keeps the new stress job separate from the existing `test-race-ort-concurrency` regex, uses fixed iteration counts (not wall-clock duration) to bound CI runtime, and pulls TESTING.md docs into this phase rather than deferring them.
- Verification commands throughout are concrete and race-detector-aware.

### Agreed Concerns (highest priority — both reviewers raised these independently)

1. **[01-02, HIGH/MEDIUM] The "destroy waits for in-flight Run" tests don't deterministically prove destroy was blocked, only that it finished after.** Both reviewers flagged that the recorded event order (`run-returned` → `destroy-released`) is also what you'd observe if the destroy goroutine simply hadn't been scheduled yet before `allowRunReturn` was closed — a genuine regression (destroy not actually waiting on the lock) could still produce the same passing order. The 500ms wait remains part of the *passing* path, not just a deadlock ceiling, in the two "waits" tests. Codex rates this HIGH; OpenCode rates it MEDIUM. **Suggested fix (both agree in spirit):** add a rendezvous signal for "destroy/tensor-destroy has reached its blocking point" (e.g., a channel closed just before the mock blocks) so the test can prove contention independent of wall-clock timing.

2. **[01-03, HIGH] The `-short` wiring does not make `go test ./...` skip stress tests by default, contradicting the plan's own success criteria.** Both reviewers independently derived that `testing.Short()` is only false unless `-short` is explicitly passed — editing `Makefile`/CI to add `-short` doesn't change what a contributor gets by typing the literal `go test ./...`. Codex additionally caught that `make test-race` (an existing, unmodified target) has no `-run` filter and no `-short`, so it will also pick up and run all `TestStress*` tests, which the plan doesn't account for. **Suggested fix:** either reword success criteria to "repository default commands" (not raw `go test ./...`), add `-short` to `make test-race` too, or switch to an explicit opt-in env gate instead of relying solely on `testing.Short()`.

3. **[01-03, HIGH — Codex only, but structurally significant] The stress tests may not actually exercise concurrent refcount transitions.** Codex traced through the mechanics: starting from `refCount == 0` against a nonexistent library path means every `InitializeEnvironment()` call attempts (and fails) `loadLibrary`, leaving `refCount` at 0, so `DestroyEnvironment()` becomes a no-op every time — the stress tests would repeatedly exercise failed library loading, not the refcount increment/decrement path under contention. This is a correctness gap in the test design itself, not just a documentation nit. OpenCode's related-but-softer concern (MEDIUM) about `TestStressMixedOperationsUnderLoad`'s unbalanced init/destroy calls potentially masking refcount corruption points at the same root cause. **Suggested fix:** seed `refCount = 1` under `mu` before spawning workers (mirroring `TestConcurrentInitialization`'s existing pattern), so `InitializeEnvironment()` takes the fast increment-only path instead of attempting a real (failing) library load.

### Divergent Views

- **Overall phase risk:** Codex's plan-by-plan verdicts imply HIGH risk for 01-03 specifically (correctness gap in the stress test's core mechanism); OpenCode's synthesized overall risk lands at MEDIUM (treating `-short`'s repo-wide behavior change as the dominant risk driver rather than test vacuity). Worth resolving before execution — Codex's refcount-seeding finding (concern #3 above) is the more concrete, blocking issue and should be treated as the higher-priority fix.
- **API surface creep (01-01):** Only OpenCode raised exporting `ErrUnsupportedPlatform`/`IsUnsupportedPlatformError` as a MEDIUM concern (permanent public API for an example-only issue). Codex did not flag this — likely because CONTEXT.md's D-02 already explicitly calls for following the existing exported-sentinel pattern, making this an accepted tradeoff rather than an oversight. Worth a quick doc-comment addition (both reviewers suggest this) but not a blocker.
- **`make test-race` gap:** Only Codex identified that `make test-race` needs its own `-short`/`-run` treatment to avoid running `TestStress*`; OpenCode's `-short`-repo-wide concern is more general and didn't drill into this specific target.
