---
phase: 3
reviewers: [claude, opencode]
reviewed_at: 2026-08-01T13:52:33+03:00
plans_reviewed: [03-01-PLAN.md]
---

# Cross-AI Plan Review — Phase 3

## Claude Review

## Summary

This is a tightly-scoped, evidence-backed plan for a genuinely small change (3 files: a zero-import generic interface, two OpenCLIP forwarding methods, and a compatibility test). It is fully grounded in Spike 003, which already compiled the exact production shape against the real package graph and ran a negative control proving OpenCLIP's precise conformance gap. The decision log (D-01 through D-16), source-coverage audit, and threat register are unusually thorough for a change this size, and the plan correctly resists scope creep (no factories, no runtime registries, no result-type migration). The main residual risks are process risks (the checkpoint's dependency on an external native/golden-dataset environment) rather than design risks, plus one small bug in the plan's own verification tooling.

## Strengths

- **Spike-grounded, not speculative**: every file shape, method signature, and conformance assertion in the plan traces back to a build-overlay proof that already compiled against the real `minilm`/`splade`/`openclip` packages (Spike 003), including a negative control confirming OpenCLIP's exact gap. This eliminates the usual risk of a generics-interface plan that "should compile" but hasn't been tried against real method sets.
- **Correctly minimal surface**: `embeddings/embedder.go` stays a zero-import leaf; the only production behavior change is two one-line OpenCLIP forwarders. This matches D-08/D-12/D-14 and avoids the common trap of a "generalized API" phase growing into a facade/factory.
- **Compatibility protected by compile-time pins, not runtime checks**: the plan carries forward Spike 003's exact function-type assignments for every existing constructor and method, so a signature change becomes a compile failure rather than a silently-passing behavior change.
- **Explicit non-goals**: D-03 (no heterogeneous registry) and D-10 (no root-owned sparse type) are called out precisely where a less disciplined plan would have been tempted to unify dense/sparse under `any` or a tagged union.
- **PASS-vs-SKIP discipline**: the plan explicitly treats a green-but-skipped native suite as a failed gate (Pitfall 5), which is the right instinct for this codebase's `t.Skip`-on-missing-env pattern.

## Concerns

- **MEDIUM — Phase completion is gated on an environment outside the plan's control.** Task 03-01-03 blocks on making a configured native environment (ONNX Runtime + models + golden datasets) available, and the plan-level gate requires exact PASS counts across 12 named tests with zero SKIPs. This is the correct call given D-16, but it means Phase 3 can be code-complete and still stuck indefinitely if CI/hosted-asset access isn't readily obtainable. Worth confirming before execution starts that the `integration-real-model` CI target (or equivalent) is actually reachable, rather than discovering that at the checkpoint.
- **LOW — Bug in the plan's own verification script reduces failure diagnosability.** In the final `<automated>` block, `phase03_run_gate` computes `phase03_pass_count="$(grep -Ec ... "$phase03_log")"` under `set -eu`. `grep -c` exits 1 when it finds zero matches, and under `set -e` a failing command substitution used in an assignment aborts the script immediately — before the intended `test "$phase03_pass_count" -eq "$phase03_expected"` comparison ever runs. The gate still fails safely, but the zero-PASS/all-skipped case produces an opaque grep-related exit instead of the intended diagnostic. Fix: `grep -Ec ... "$phase03_log" || true`.
- **LOW — No contingency for pre-existing flaky/broken native tests.** Because D-11 forbids touching existing integration/golden tests, if any of the 12 named native tests is already flaky or broken for reasons unrelated to this phase, Phase 3 has no defined path to green other than fixing an out-of-scope issue. Not a plan defect, but worth flagging as a realistic way this specific phase gate could stall on something Phase 3 didn't cause.

## Suggestions

- Before dispatching execution, verify the named CI/environment target for 03-01-03 is actually accessible (not just "exists in principle") so the checkpoint is not a late blocker.
- Fix the `grep -Ec ... || true` (or equivalent) in the `phase03_run_gate` helper so a zero-match case produces the intended `test -eq` failure message.
- Minor: the acceptance-criteria wording `go doc ./embeddings.Embedder` is a slightly unusual invocation form. Confirm it resolves in the target Go version or use the conventional `go doc ./embeddings Embedder` form.

## Risk Assessment

**LOW.** The implementation delta is small, spike-validated against the real package graph, and heavily pinned by compile-time assertions rather than runtime checks. The identified concerns are about the surrounding process (external environment availability) and a cosmetic bug in the verification script's failure-reporting path, not about the design or the production code shape.

---

## OpenCode Review

## Summary

The plan is strong and appropriately narrow: it promotes the spike-proven generic contract with minimal production changes, preserves existing concrete APIs, and uses compile-time assertions plus existing native parity suites to prove API-01. Main risks are in verification mechanics rather than implementation design.

## Strengths

- Scope is tightly constrained to three files and matches the validated spike.
- Good dependency direction: root `embeddings` package remains a zero-import contract instead of becoming a facade.
- OpenCLIP conformance is additive and low-risk through direct forwarding only.
- Compile-time signature pins are a strong compatibility guard for existing constructors and methods.
- Native parity tests explicitly treat `SKIP` as failure, which is important for this repository's environment-sensitive tests.
- The plan correctly avoids factories, registries, result unions, aliases, adapters, and result migration.

## Concerns

- **MEDIUM:** The acceptance command `go doc ./embeddings.Embedder` may be invalid for a relative package path. `go doc ./embeddings Embedder` is safer.
- **MEDIUM:** The final gate says JSON logs must be preserved, but the shell uses `mktemp` without copying paths into durable evidence. Those logs may be lost.
- **MEDIUM:** The native checkpoint makes the phase dependent on an external configured environment. That is valid, but it should clearly define where evidence is stored and how the executor verifies it belongs to the current revision.
- **LOW:** The final JSON grep checks are pragmatic but brittle. They depend on exact `go test -json` event shape and do not distinguish package-level pass from test-level pass except via `"Test"` presence.
- **LOW:** `go doc` is only listed in acceptance criteria, not in the automated gate. If documentation shape matters, it should be automated or dropped as a manual check.
- **LOW:** The plan asks to create `.planning/phases/03-generalized-embedder-api/03-01-SUMMARY.md` but the declared `files_modified` list excludes it. That mismatch could confuse execution tracking.

## Suggestions

- Replace `go doc ./embeddings.Embedder` with `go doc ./embeddings Embedder`.
- Persist native JSON logs under a known path, for example `.planning/phases/03-generalized-embedder-api/evidence/03-01-native-*.jsonl`.
- Add a current-revision check before native verification, such as recording `git rev-parse HEAD`.
- Include the summary file in expected outputs or explicitly mark it as planning metadata outside the implementation diff.
- Consider replacing grep-based JSON checks with a small `go test -json` parser only if this pattern will be reused; for this one phase, grep is acceptable.

## Risk Assessment

**LOW.** The implementation is small, additive, and already spike-validated. The main risks are operational: native environment availability and evidence capture. The plan achieves the phase goal if the verification mechanics are tightened slightly.

---

## Consensus Summary

Both reviewers assess the Phase 3 design as **LOW risk**. The plan is intentionally minimal and spike-proven; its remaining risks are in the native-test gate and how its evidence is recorded, not in the proposed API.

### Agreed Strengths

- The change is small and stays within the validated three-file scope.
- The dependency-free generic contract avoids runtime registries, factories, and dense/sparse type coercion.
- Direct OpenCLIP forwarders preserve its concrete API while satisfying the shared interface.
- Compile-time signature/conformance assertions give strong regression protection.
- Native `SKIP` results are correctly treated as a failed phase gate.

### Agreed Concerns

1. **MEDIUM — Native verification is externally dependent.** Confirm the required models, runtime, golden dataset, and CI/local target are available before execution; otherwise an implementation-complete phase can remain blocked.
2. **MEDIUM — Verification evidence needs a durable, revision-bound home.** Preserve native JSON logs under the phase directory and record the tested commit, rather than leaving evidence only in temporary files.
3. **LOW — Tighten the gate scripts.** Handle a zero-match PASS count without an early `set -e` exit, and correct or automate the `go doc` command.

### Divergent Views

- Claude specifically identified the `grep -Ec` zero-match failure path and the lack of an out-of-scope flaky-test contingency.
- OpenCode additionally flagged brittle JSON-event matching and the mismatch between `files_modified` and the expected summary artifact.
