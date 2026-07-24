---
phase: 2
reviewers: [claude, opencode]
reviewed_at: 2026-07-24T10:53:16+03:00
plans_reviewed:
  - 02-01-PLAN.md
  - 02-02-PLAN.md
  - 02-03-PLAN.md
  - 02-04-PLAN.md
  - 02-05-PLAN.md
  - 02-06-PLAN.md
  - 02-07-PLAN.md
  - 02-08-PLAN.md
---

# Cross-AI Plan Review — Phase 2

## Claude Review

# Cross-AI Plan Review — Phase 2: Core API — Errors & Values (Plans 02-01 … 02-08)

Before writing this review I verified the plans' load-bearing factual claims against the working tree, since almost every `<verify>` command is anchored to exact symbols. All of them check out:

- **12** `log.Printf` sites in `ort/bootstrap.go`, **14** total in production `ort/` code, **3** `logFinalizerWarning` callers (`memory.go:42`, `session.go:118`, `tensor.go:134`) — matches Plans 03/07/08 exactly.
- **Exactly 7** open-coded status blocks (`memory.go:26`, `session.go:82/103/214`, `environment.go:188`, `tensor.go:91/119`) — matches Plans 01/08.
- Every existing test name referenced in the race/native selectors exists verbatim (all 12 concurrency tests, all 16 bootstrap security tests, the 3 real-model tests, the 4 session-test `Value` doubles). The `test-race-ort-concurrency` CI job exists at `ci.yml:111`. `errSharedLibraryNotFound`, `make precommit-lint-new`, `ort/finalizer_log.go`, and the slog spike prototype all exist as described.
- Single Go module (examples/embedders included), `go 1.25.0` baseline — so the `go test -run '^$' ./...` compile gates cover everything.

That level of grounding is rare and materially de-risks execution.

---

## 1. Summary

This is a well-decomposed, exceptionally well-grounded plan set. The wave structure is correct (contracts in Wave 1, disjoint per-resource migrations in Wave 2, convergence/deletion/CI in Wave 3), the Wave-2 `files_modified` sets are fully disjoint so parallel worktree execution cannot conflict, and the plans repeatedly anticipate the subtle traps in this codebase: reading bound session values only under `runMu`, status copy-before-release with exactly one owner, the race/checkptr incompatibility with the `uintptr` FFI boundary, and regex anchoring so `TestAdvancedSessionRunWithValuesRealModel` cannot leak into the race lane. The concerns that remain are second-order: a behavioral change around consumer-handler panics in non-finalizer diagnostic paths, silent-vacuous CI selectors if a test is later renamed, a scope expansion (file-permission hardening) riding inside the bootstrap error-migration plan, and an API-ergonomics gap where `RunWithValues` still requires placeholder bound tensors at construction. None of these should block execution, but two deserve a decision before Wave 2.

`★ Insight ─────────────────────────────────────`
- Name-anchored `go test -run '^(TestX|TestY)$'` selectors have an asymmetric failure mode: a *renamed* test doesn't fail the lane, it silently drops out (Go exits 0 with "no tests to run" for non-matching patterns). Plans that freeze exact names into CI trade drift-resistance for silent-coverage-loss risk.
- The plans' "keep dead helpers until Wave 3" strategy is the correct pattern for parallel-worktree migration: each Wave-2 plan can migrate its own call sites without a cross-worktree edit to `environment.go`, and the still-referenced helper *tests* keep golangci-lint's `unused` checker quiet until Plan 08 deletes both together.
`─────────────────────────────────────────────────`

## 2. Strengths

- **Verified ground truth, not planner folklore.** Call-site counts (7 status blocks, 12+1+1 log sites, 3 finalizer callers), test names, doubles, spike files, and CI job names are all accurate in the live tree. Verify commands will bite on real code.
- **Correct dependency graph.** Plan 04 depends on 01+02+03 (needs `statusToError`, sealed `Value`, `emitFinalizerDiagnostic`); Plans 06/07 correctly depend only on 01+03 (no `Value` dependency). Wave-2 file sets are disjoint (`session.*` / `tensor.*` / `environment.*+memory.*` / `bootstrap.*`), eliminating merge conflicts.
- **The D-23 race/native split is enforced structurally**, not just by convention: `//go:build !windows` on the native test, a Windows cross-compile gate (`GOOS=windows go test -c`), anchored selectors that exclude real-model tests from the race lane, and explicit prohibition of checkptr-disabling flags with a CI grep to prove it.
- **The shared-run-core design (Pattern 2) closes a real race**: reading `s.inputValues`/`s.outputValues` only after `runMu` is held, because `Destroy()` clears them under that lock. The plan calls this out as a hard acceptance criterion rather than leaving it to executor judgment.
- **The Plan 08 audit regexes are precise**: `getErrorMessage\(` does not match `getErrorMessageFunc(` (literal-paren anchoring), so the deletion audit won't false-positive on the retained function pointers the converter still needs. I verified this against the actual identifiers.
- **Anti-scope-creep guardrails are explicit**: no `Unwrap` on `ORTError` (keeping native codes and Go sentinels disjoint per Pitfall 7), no numeric coercion, no ORT-allocated outputs, `go.mod`/`go.sum` diff gates on every plan, and `uses:`-line diff assertions on the workflow edit.
- **Convergence discipline in Plan 08 Task 1**: "stop if a prerequisite site is unmigrated; the fix belongs in its owning plan" prevents the classic Wave-3 failure of re-implementing Wave-2 work in a conflicting way.

## 3. Concerns

**MEDIUM — Consumer-handler panics propagate into library calls on non-finalizer paths (Plan 03/07).**
Only `emitFinalizerDiagnostic` recovers panics. The other ~13 approved sites (bootstrap lock-wait, cache fallback, archive skips, runtime-version warning) call `emitDiagnostic` → `Logger.LogAttrs` with no recovery. Today those are `log.Printf`, which effectively cannot panic; after migration, a buggy consumer handler panics *inside* `EnsureOnnxRuntimeSharedLibrary` or `InitializeEnvironment`. That is a behavioral regression the plans never explicitly accept or reject. slog convention is indeed "handler panics are the consumer's bug," and that's a defensible position — but it should be a recorded decision, not an accident.

**MEDIUM — CI selectors fail silently on rename (Plan 08).**
`go test -run '^(TestDiagnostic|…)$'` exits 0 when nothing matches. If any of the ~28 frozen names is later renamed, that test silently vanishes from the race lane while CI stays green. Plan 08 verifies the selector *text* exists in `ci.yml` but nothing verifies the selected tests *exist*. One `go test -list` assertion (or a grep that each name in the selector appears in `ort/*_test.go`) in the same CI step would close this.

**MEDIUM — `RunWithValues` ergonomics are constrained by the constructor (Plans 02-CONTEXT/04).**
I confirmed `NewAdvancedSession` (session.go:26-42) *requires* bound input/output values matching the name counts. So a consumer who wants purely per-call values must still create and bind placeholder tensors at construction, then ignore them. This follows from locked D-01/D-02 and is not a plan defect, but no plan or test covers the "session used exclusively via RunWithValues" pattern, and it partially undercuts issue #6's polymorphic-usage goal. Worth a conscious call: accept for Phase 2 (documenting the placeholder pattern), or note it as a candidate constructor variant for a later phase.

**MEDIUM — File-permission hardening is scope expansion inside an error-migration plan (Plan 07, T-02-10).**
`TestBootstrapCreatedFilePermissions` plus the conditional production change ("if the red test exposes archive-derived group/other write bits, clamp them in tar and ZIP creation") is genuine behavior hardening, not error-contract work, and its production diff is unpredictable until the test runs. It's well-bounded ("clamp only those unsafe bits… do not chmod unrelated paths"), but it means Plan 07's diff may include extraction-logic changes reviewed under an "errors and diagnostics" heading. Given the stated preference for surgical scope, consider splitting it into its own commit within the plan (it already is a distinct test name) or a separate micro-plan so the security change is reviewable in isolation.

**LOW — The real native ABI proof runs only on Linux.**
The native round trip is gated on `ONNXRUNTIME_LIB_PATH`, which is unset locally and exported only by the Linux integration job (downloading a `.so`). The build tag excludes Windows entirely, and macOS never exercises it in CI. Acceptable per D-23, but the phrase "real ABI proof exists" is effectively "exists on linux/amd64." A one-time local macOS run against the cached 1.23.1 dylib during Plan 01 execution would cheaply extend the evidence.

**LOW — Retained dead helpers vs. `precommit-lint-new` between Wave 2 and Wave 3.**
After Wave 2, all seven production call sites of `getErrorMessage`/`releaseStatus` are gone but the helpers remain until Plan 08. golangci-lint's `unused` includes test usage by default and the legacy helper tests survive until Plan 08, so this should stay quiet — but if any Wave-2 plan happens to remove a helper test early, the wave-gate lint could flag the newly-unused helper. Executors should know the intended fix is "wait for Plan 08," not "delete the helper now."

**LOW — Plan 03's zero-emission test is initially vacuous.** Testing that "a returned error emits nothing" in a package with no returned-error hook proves the absence of a mechanism that doesn't exist yet. The plans acknowledge this and put the real teeth in the per-flow call-site audits (Plans 04–07), so this is fine as scaffolding — just don't count it as D-22 evidence on its own.

**LOW — Stale context in PROJECT.md.** It states "CI runs on Go 1.24.x," while `go.mod` declares 1.25.0 and the research correctly uses the 1.25 baseline. The plans follow the correct number; the doc should be fixed at the next transition so a future planner doesn't inherit the stale claim.

## 4. Suggestions

- **Record a decision on handler-panic policy** (accept propagation as consumer responsibility, or add recovery to `emitDiagnostic` wholesale). One sentence in 02-03's summary or CONTEXT addendum is enough; the tests in Plan 03 should then match whichever is chosen.
- **Add a selector-liveness guard to the CI race lane** in Plan 08: after the `go test -race` step, assert the selector matched a nonzero test count (e.g., `go test -list '<same regex>' ./ort | grep -c '^Test'` ≥ expected), so a future rename fails loudly instead of silently shrinking coverage.
- **Add one test or doc note for the "RunWithValues-only session" pattern** in Plan 04 — even just a test constructing a session with minimal bound tensors and only ever calling `RunWithValues` — so the placeholder-binding requirement is documented behavior rather than a surprise.
- **Isolate the permission clamp into its own commit** within Plan 07 Task 1 (test + production clamp together, separate from sentinel migration) so the security-relevant diff is independently revertable and reviewable.
- **Have the Plan 01 executor run the native round trip once locally** with `ONNXRUNTIME_LIB_PATH` pointed at the cached macOS dylib, and record the result in the summary — it upgrades the ABI evidence from single-platform to two platforms at near-zero cost.
- Minor: in `AsTensor[T]`, return an explicit `nil` (not the typed-nil pointer) alongside `false` for the typed-nil case, so callers who ignore `ok` can't accidentally hold a typed-nil `*Tensor[T]`.

## 5. Risk Assessment

**Overall: LOW-to-MEDIUM.**

The plan set achieves all four phase success criteria by construction: comprehensive `errors.Is`/`errors.As` inspection across environment/tensor/session/memory/bootstrap (Plans 01, 04–07), the sealed `Value` surface with exact extraction (Plan 02), additive `RunWithValues` through a shared core that leaves `Run()` and all embedder hot paths untouched (Plan 04), and compatibility gates that compile every consumer unchanged (Plans 02/04/08). The riskiest work — status lifetime ownership and the shared run core's lock ordering — is exactly where the plans are most prescriptive and where both spikes provide validated prototypes. Wave-2 parallelism is safe because file ownership is disjoint and the dead-helper retention strategy decouples the migrations. The residual risk is concentrated in the four MEDIUM items above, all of which are decision-or-one-line-guard fixes rather than structural problems, and none of which threaten the core value proposition (no CGO, no dependency changes, no lock-hierarchy changes). I'd approve execution with the handler-panic decision and the CI selector-liveness guard addressed, and the other items tracked as executor notes.

---

## OpenCode Review

## Summary
The plans are unusually thorough and mostly well-aligned with the phase goals: additive `Value` support, inspectable errors, native-status lifetime safety, and silent structured diagnostics. The strongest aspect is the explicit sequencing: Wave 1 establishes primitives, Wave 2 migrates resource areas, Wave 3 removes compatibility scaffolding and wires CI. Main risks are over-specified test/workflow selectors, parallel Wave 2 coupling around shared files, and some scope creep in bootstrap permissions/CI hardening that may be valuable but could slow delivery or create brittle plan execution.

## Strengths
- Clear phase boundary: `Run()` remains unchanged, `RunWithValues` is additive, outputs remain caller-owned, and runtime-allocated outputs are explicitly excluded.
- Good no-CGO discipline: native ABI testing is separated from `-race`, and plans avoid disabling `checkptr`.
- Strong error model: `errors.Is` for local categories and `errors.As` for native `*ORTError` keeps local validation distinct from ONNX Runtime failures.
- Correct native status ownership focus: single converter, copy-before-release, exact one-release accounting, fake race tests plus real ABI test.
- Good compatibility posture: examples and embedders must compile unchanged; existing hot-path session behavior is preserved.
- Diagnostics contract is appropriately narrow: `slog.Handler`, silent default, no third-party logger dependency, and no returned-error double logging.
- Security concerns are explicitly tracked: URL redaction, permission hardening, no credential attrs, archive containment, checksum behavior, and safe diagnostics.

## Concerns

### HIGH
- Wave 2 plans modify overlapping files concurrently: `session.go`, `tensor.go`, `environment.go`, `memory.go`, `bootstrap.go`, tests, and shared diagnostics/error contracts all depend on exact prerequisite behavior. Even with declared dependencies, if executed by multiple agents the merge risk is high, especially around shared test helpers and global reset state.
- Plan 02-08 relies on brittle regex/source audits and exact CI selectors. This can fail because helper names, formatting, or test names evolve during implementation, even if behavior is correct.
- Plan 02-01 says `statusToError` should acquire no new lock and use registered function pointers. That assumes all production callers already hold valid lifecycle protection. If one call site later invokes it outside `ortCallMu`/`mu` protection, it could read nil/stale function pointers.

### MEDIUM
- The public `Value` interface sealing is technically a breaking source change for any external custom implementation. The context says external custom implementations were not found, but the plan should explicitly treat this as an intentional compatibility tradeoff.
- `ErrNotInitialized` for “initializing without configured library path” may be semantically questionable if bootstrap can download a runtime. The distinction between “runtime unavailable,” “not initialized,” and “library not found” needs careful wording to avoid misleading callers.
- `ErrSharedLibraryNotFound` is defined in `errors.go` but used heavily by bootstrap. That is fine, but the migration must avoid circular conceptual ownership where bootstrap-specific categories become too general.
- `AsTensor[T any]` allows any `T`, but `Tensor[T]` likely supports only specific element types. This is okay for exact assertion, but tests should include unsupported generic types only if they can be instantiated safely.
- Native test plan for `errors_native_test.go` is Unix-only. Windows native ABI coverage is skipped, even though Windows is a supported target. That is acceptable for this phase if deliberate, but it should be called out as residual risk.
- Bootstrap permission hardening in 02-07 expands beyond error/Value API scope. It addresses a valid threat, but it may become a detour if it exposes platform-specific archive mode behavior.
- `emitFinalizerDiagnostic` recovering panics only on finalizer path is sensible, but a panicking consumer handler in non-finalizer diagnostics can still crash the caller. That should be explicitly accepted as normal `slog.Handler` behavior.

### LOW
- Plans repeatedly require exact top-level test names and selectors. This helps CI but makes implementation less flexible.
- The desired exported identifiers are “exact” in some places. That is useful for contract review, but could force awkward naming if existing package conventions differ.
- `go test -run '^$' ./...` as a compile check can still run package init and may be slower/flakier than expected in packages with environment-sensitive init.
- Some acceptance criteria mix behavioral verification with `rg` implementation checks. Source audits are useful, but behavioral tests should remain the primary gate.

## Suggestions
- Add an explicit “shared-file merge protocol” before Wave 2: execute 02-04 through 02-07 sequentially unless using isolated branches with careful rebases.
- In 02-01, document `statusToError` preconditions: either callers must hold lifecycle protection or the helper snapshots function pointers under a safe lock. Prefer a minimal assertion/test for nil function pointers returning `ErrNotInitialized` if feasible.
- Add a short compatibility note to 02-02: sealing `Value` intentionally prevents external implementations; this is acceptable because raw native handles were never a supported extension point.
- Relax 02-08 CI selector checks slightly: verify required test names are included and native tests are excluded from race, rather than requiring an exact full regex string.
- Keep bootstrap permission hardening in 02-07 only if it is already part of `02-VALIDATION.md`; otherwise split it into a follow-up security cleanup to reduce Phase 2 scope.
- Add one Windows-specific compile or lightweight native-status registration test if feasible, even if the actual ABI round trip remains Unix-only.
- Make sentinel usage guidance explicit in docs/tests: native ORT invalid-argument codes should not satisfy `errors.Is(err, ErrInvalidArgument)` unless a Go validation wrapper intentionally adds that sentinel.
- Ensure all diagnostic tests reset global handler state with `t.Cleanup`; avoid `t.Parallel` in any test touching process-global diagnostics or ORT globals.
- Add a final API surface check in 02-08, such as `go doc` or a small compile-only external package test, to confirm external users can call `RunWithValues`, `AsTensor`, and `SetDiagnosticHandler`.

## Risk Assessment
Overall risk: **MEDIUM**.

The architecture is sound and the plans directly address the phase goals. The highest risk is execution complexity, not design correctness: many shared globals, FFI lifetimes, lock ordering rules, and tests are changing across multiple plans. If implemented sequentially with disciplined verification, risk drops toward LOW. If implemented in parallel by separate agents without strict integration control, conflict and regression risk is HIGH.

---

## Consensus Summary

Both reviewers consider the Phase 2 architecture sound, well-grounded, and aligned with API-02/API-03. They agree that the remaining risk is mainly in execution details rather than the core design. The highest-value planning changes are to make diagnostic-handler panic behavior explicit, add a liveness check for name-anchored CI selectors, and keep bootstrap permission hardening isolated so it cannot expand the phase unpredictably.

### Agreed Strengths

- The wave order is coherent: establish contracts first, migrate resource areas second, then remove scaffolding and wire final gates.
- `RunWithValues` is additive, preserves the existing `Run` hot path, retains caller-owned outputs, and avoids scope creep into runtime-allocated values.
- The error model cleanly separates local sentinels (`errors.Is`) from native ONNX Runtime errors (`errors.As`) and gives native status conversion one clear owner.
- The plans take the no-CGO boundary seriously, especially by separating race/checkptr tests from the real native ABI lane.
- Diagnostics have a deliberately small, silent-by-default standard-library contract, and compatibility checks cover existing examples and embedders.

### Agreed Concerns

- **MEDIUM — Diagnostic handler panic policy is implicit.** Both reviewers noted that non-finalizer diagnostics can propagate a consumer handler panic into a library call. The plan should explicitly accept that as normal `slog.Handler` behavior or recover consistently, with tests matching the decision.
- **MEDIUM — Exact CI selectors and source audits are brittle.** Both reviewers flagged reliance on exact names. In particular, a renamed test can silently disappear from a `go test -run` lane. Add a selector-liveness assertion while keeping behavioral tests as the primary gate.
- **MEDIUM — Bootstrap permission hardening can widen Plan 07.** Both reviewers consider the security check valuable but outside the central errors/values contract. Keep it in a distinct commit or move it to a focused follow-up if the red test reveals a non-trivial platform-specific fix.
- **LOW — Native ABI coverage is platform-limited.** Both reviews call out that the real round trip is Unix/Linux-focused and does not exercise Windows. Treat this as an explicit residual risk and add another platform check if it is cheap.

### Divergent Views

- **Wave 2 parallelism:** Claude verified that the declared production/test file sets are disjoint and judged parallel execution safe. OpenCode rated parallel integration risk high because the plans still share contracts, lifecycle assumptions, and process-global test state. The practical conclusion is that file-level merge risk is low, but wave integration still needs prerequisite verification and careful handling of global-state tests.
- **Overall risk:** Claude rated the plan set LOW-to-MEDIUM and ready with two small decisions; OpenCode rated it MEDIUM, rising to HIGH only if parallel execution lacks integration discipline.
- **Claude-only concern:** A session used exclusively through `RunWithValues` still needs placeholder bound tensors at construction. Decide whether to document that Phase 2 limitation or add a later constructor variant.
- **OpenCode-only concerns:** Document `statusToError` lifecycle/lock preconditions and explicitly record the compatibility tradeoff of sealing `Value` against external implementations.
