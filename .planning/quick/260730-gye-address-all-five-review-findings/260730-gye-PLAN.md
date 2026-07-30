---
phase: quick-260730-gye-address-all-five-review-findings
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - Makefile
  - embeddings/minilm/embedder.go
  - embeddings/openclip/bootstrap.go
  - embeddings/openclip/embedder.go
  - embeddings/splade/embedder.go
  - examples/openclip/main.go
  - tools/gen_ortapi.go
  - ort/bootstrap.go
  - ort/bootstrap_lock_unix.go
  - ort/bootstrap_test.go
  - ort/bootstrap_trust_unix.go
  - ort/environment_test.go
  - ort/public_api_compat_test.go
  - ort/session.go
  - ort/session_test.go
  - ort/shape_test.go
  - ort/types.go
autonomous: true
requirements: [RF-1, RF-2, RF-3, RF-4, RF-5]
must_haves:
  truths:
    - "The pinned gosec v2.25.0 scan passes with all 15 known findings resolved or justified at the exact line, while the CI security job remains enforcing."
    - "The PR new-issues lint gate accepts the two intentional purego callback out-parameter casts without disabling govet/unsafeptr elsewhere."
    - "Caller-selected ONNXRUNTIME_LIB_PATH and WithBootstrapLibraryPath soname symlinks resolve to validated regular targets, while cache-managed symlink candidates remain rejected."
    - "Downstream code can still compile zero-value composite literals for Status, Environment, and Session."
    - "A once-live destroyed SessionOptions matches ErrDestroyed, while a never-initialized zero value continues to match ErrInvalidArgument."
  artifacts:
    - path: "ort/bootstrap.go"
      provides: "Separate explicit-path symlink resolution and strict cache-path validation"
      contains: "validateExplicitLibraryFile"
    - path: "ort/types.go"
      provides: "Backward-compatible exported handle structs and explicit SessionOptions lifecycle state"
      contains: "destroyed"
    - path: "ort/public_api_compat_test.go"
      provides: "External-package compile regression for exported handle composite literals"
    - path: "ort/session_test.go"
      provides: "ErrInvalidArgument versus ErrDestroyed SessionOptions regression coverage"
    - path: "Makefile"
      provides: "Local gosec version aligned with the enforcing CI version"
      contains: "GOSEC_VERSION ?= v2.25.0"
  key_links:
    - from: "ort/bootstrap.go"
      to: "filepath.EvalSymlinks"
      via: "explicit cfg.libraryPath validation only"
      pattern: "libraryPath.*validateExplicitLibraryFile|validateExplicitLibraryFile.*EvalSymlinks"
    - from: "ort/bootstrap.go"
      to: "validateLibraryFile"
      via: "cache-managed primary and glob candidate validation remains strict"
      pattern: "resolveExtractedLibraryPath"
    - from: "ort/session.go"
      to: "ort/errors.go"
      via: "NewAdvancedSession classifies zero SessionOptions handles using destroyed state"
      pattern: "ErrDestroyed|ErrInvalidArgument"
    - from: "ort/public_api_compat_test.go"
      to: "ort/types.go"
      via: "external package imports and instantiates Status{}, Environment{}, and Session{}"
      pattern: "Status\\{\\}|Environment\\{\\}|Session\\{\\}"
---

<objective>
Close all five reviewer findings without weakening repository-wide security, lint, bootstrap trust, or public API contracts.

Purpose: Make the Phase 2 branch safe to review and merge by restoring downstream compatibility, preserving inspectable lifecycle errors, accepting normal explicitly trusted soname symlinks, and making the enforcing CI gates honestly green.
Output: Focused production fixes, regression tests for each behavioral contract, and narrowly justified scanner annotations for the exact known false positives.
</objective>

<execution_context>
@/Users/tazarov/.codex/get-shit-done/workflows/execute-plan.md
@/Users/tazarov/.codex/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/02-core-api-errors-values/02-04-SUMMARY.md
@.planning/phases/02-core-api-errors-values/02-07-SUMMARY.md
@.planning/phases/02-core-api-errors-values/02-08-SUMMARY.md
@.github/workflows/ci.yml
@.golangci.yml
@Makefile
@ort/types.go
@ort/session.go
@ort/session_test.go
@ort/bootstrap.go
@ort/bootstrap_test.go
@ort/bootstrap_lock_unix.go
@ort/bootstrap_trust_unix.go
@ort/environment_test.go

<interfaces>
Current contracts the executor must preserve or repair:

- `NewSessionOptions() (*SessionOptions, error)`, `(*SessionOptions).Destroy() error`, and `(*SessionOptions).IsValid() bool` own the native options handle under `handleMu`.
- `NewAdvancedSession(..., options *SessionOptions) (*AdvancedSession, error)` holds `options.handleMu.RLock` while borrowing a supplied options handle.
- `ErrInvalidArgument` identifies an argument that was never valid; `ErrDestroyed` identifies a once-live resource whose native handle is gone.
- `EnsureOnnxRuntimeSharedLibrary(opts ...BootstrapOption) (string, error)` receives explicit paths through either `ONNXRUNTIME_LIB_PATH` or `WithBootstrapLibraryPath`.
- `validateLibraryFile(path string) (string, error)` currently performs strict `Lstat`, regular-file, and non-empty checks and rejects symlinks; cache resolution relies on that strict behavior.
- Before this branch, `Status`, `Environment`, and `Session` were exported structs with private fields. Downstream `ort.Status{}`, `ort.Environment{}`, and `ort.Session{}` therefore compiled even though callers could not forge non-zero handles.
</interfaces>

Project constraints:

- Add no dependency and do not broaden the task beyond the five findings.
- Keep the CI gosec job blocking and keep govet/unsafeptr enabled globally.
- Do not place internal repository information in commits, pull requests, or generated artifacts.
- No merge operation is part of this plan; if one is later requested, use squash merge.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Restore public handle structs and distinguish destroyed session options</name>
  <files>ort/types.go, ort/shape_test.go, ort/public_api_compat_test.go, ort/session.go, ort/session_test.go</files>
  <behavior>
    - External-package code can instantiate `ort.Status{}`, `ort.Environment{}`, and `ort.Session{}` as zero-value composite literals.
    - Status zero/non-zero behavior and native error-code/message access continue to work through the restored private handle field.
    - Passing `&SessionOptions{}` to `NewAdvancedSession` returns an error matching `ErrInvalidArgument` and not `ErrDestroyed`.
    - Destroying a live `SessionOptions` and then passing it to `NewAdvancedSession` returns an error matching `ErrDestroyed`; repeated `Destroy` calls still release the native handle exactly once.
  </behavior>
  <action>
Write the compatibility and lifecycle tests first. Add an external-package regression file that imports `ort` and compiles zero-value composite literals for all three affected exported types; replace the current uintptr-convertibility assertion because that assertion protects the breaking representation instead of the established API.

Restore `Status`, `Environment`, and `Session` to their pre-change exported struct definitions with private fields. Keep the improved Status native accessors, but read `s.handle` and retain the established pointer-receiver method set. The previous `Environment` fields are `handle`, `loggingLevel`, and `logID`; the previous `Session` fields are `handle`, `inputNames`, `outputNames`, `inputCount`, and `outputCount`. Do not add a new public raw-handle type: production internals already use private `uintptr` values.

Add a private `destroyed bool` to `SessionOptions`, guarded by `handleMu`. In `Destroy`, mark it destroyed only when the object had a non-zero native handle (and preserve the flag on repeated calls), then clear the handle and retain the existing exact-once release/finalizer behavior. In `NewAdvancedSession`, inspect `handle` and `destroyed` under the existing read lock: a zero handle with `destroyed=true` wraps `ErrDestroyed` with destroyed-session-options context; a never-live zero value keeps the current `ErrInvalidArgument` classification. Preserve the existing lock order and keep the options read lock held through native session creation so concurrent destruction cannot invalidate the borrowed handle.
  </action>
  <verify>
    <automated>go test -count=1 ./ort -run '^(TestStatus|TestExportedHandleStructCompositeLiteralsCompile|TestNewAdvancedSessionWithUninitializedSessionOptions|TestNewAdvancedSessionWithDestroyedSessionOptions|TestSessionOptionsLifecycle)$'</automated>
  </verify>
  <done>Status, Environment, and Session retain their established struct-shaped public API; status accessors still pass; zero and destroyed SessionOptions states have distinct errors.Is classifications; a live options handle is released once.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Resolve explicitly trusted library symlinks without relaxing cache trust</name>
  <files>ort/bootstrap.go, ort/bootstrap_test.go</files>
  <behavior>
    - An explicit `WithBootstrapLibraryPath` soname symlink returns the fully resolved absolute regular-file target.
    - An `ONNXRUNTIME_LIB_PATH` soname symlink follows the same path and target validation.
    - A dangling explicit symlink remains an `ErrSharedLibraryNotFound` failure, and a resolved directory or empty file remains invalid.
    - Cache-managed primary/glob candidates, install directories, manifests, archive entries, and lock paths continue to reject symlinks.
  </behavior>
  <action>
Add table-driven tests first for both explicit path sources, using a non-empty real file plus an ordinary symlink and asserting the returned path is the absolute resolved target. Cover a dangling explicit link and retain the existing `TestEnsureOnnxRuntimeSharedLibraryRejectsCachedSymlink` and strict `TestValidateLibraryFile` expectations.

Introduce `validateExplicitLibraryFile` for the explicit `cfg.libraryPath` branch only. Trim and absolutize the caller-selected path, resolve its symlink chain with `filepath.EvalSymlinks`, translate a missing or dangling target into an error that preserves `ErrSharedLibraryNotFound`, then pass the resolved target through the existing strict regular/non-empty `validateLibraryFile` checks. Return the resolved absolute target so `InitializeEnvironmentWithBootstrap` loads what was validated. Leave `resolveExtractedLibraryPath`, `validateCachedRuntimeInstall`, directory/manifest collection, archive extraction, and lock validation on their current symlink-rejecting paths; do not change `validateLibraryFile` into a generally symlink-following helper.
  </action>
  <verify>
    <automated>go test -count=1 ./ort -run '^(TestEnsureOnnxRuntimeSharedLibraryWithExplicitPath|TestEnsureOnnxRuntimeSharedLibraryExplicitSymlink|TestValidateLibraryFile|TestEnsureOnnxRuntimeSharedLibraryRejectsCachedSymlink)$'</automated>
  </verify>
  <done>Both explicit path entry points accept a valid soname symlink and return its validated target, invalid targets retain useful error categories, and every cache-managed symlink regression remains rejected.</done>
</task>

<task type="auto">
  <name>Task 3: Clear the exact gosec and PR-lint baselines with local rationales</name>
  <files>Makefile, ort/bootstrap_lock_unix.go, ort/bootstrap_trust_unix.go, embeddings/minilm/embedder.go, embeddings/openclip/embedder.go, embeddings/splade/embedder.go, embeddings/openclip/bootstrap.go, tools/gen_ortapi.go, examples/openclip/main.go, ort/environment_test.go, ort/bootstrap_test.go</files>
  <action>
Align `GOSEC_VERSION` in the Makefile to `v2.25.0`, matching the already pinned enforcing CI action. Clear the exact 15 v2.25.0 findings with rule-specific inline `#nosec` annotations and concrete safety rationales: two Unix `os.File.Fd()` to `int` conversions required by the `unix.Flock` ABI; the effective-UID conversion used to compare `syscall.Stat_t.Uid`; the six positive, option-validated sequence-length conversions used by tokenizer truncation/padding in MiniLM, OpenCLIP, and SPLADE; the public `tokenizer.json` filename false-positive for G101; G304 plus G703 on the two OpenCLIP example reads intentionally selected by the local CLI/dataset; and G703 alongside the existing G304 rationale on the generator's caller-supplied header path. List multiple rules in one annotation where the same statement triggers both, and keep every annotation at the reported statement with a two-dash justification.

Add one line-scoped `//nolint:govet` annotation at each intentional purego callback out-parameter cast in `ort/environment_test.go` and `ort/bootstrap_test.go`. Explain that the callback ABI supplies the native output address as `uintptr` and the test must write the fake `OrtEnv` handle through it. Do not add file-wide exclusions, do not alter `.golangci.yml`, do not disable `unsafeptr`, and do not restore gosec `-no-fail` or `continue-on-error`.

Run the exact scanner version after all behavioral changes so new findings introduced by Tasks 1-2 are caught as well. Every suppression must correspond to a reviewed false positive; fix any genuinely unsafe new finding rather than suppressing it.
  </action>
  <verify>
    <automated>go run github.com/securego/gosec/v2/cmd/gosec@v2.25.0 -exclude-dir=examples/experimental ./... &amp;&amp; make precommit-lint-new PRECOMMIT_BASE_REF=main &amp;&amp; go vet -unsafeptr=false ./ort/...</automated>
  </verify>
  <done>The exact gosec v2.25.0 command reports zero findings, the PR-equivalent new-issues lint reports zero findings, plain project vet remains green, and neither enforcing gate is globally weakened.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| repository code → CI scanners | Inline suppressions can accidentally hide a real issue if they are broad or unexplained. |
| explicit caller path → local filesystem target | A caller-selected soname link may cross a filesystem path boundary, but it is an explicit trusted input rather than cache-controlled state. |
| bootstrap cache → dynamic loader | Cache contents remain untrusted until directory, manifest, file, and symlink integrity checks pass. |
| public Go value → native resource handle | Zero-value and once-live destroyed SessionOptions must not be treated as the same lifecycle state. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-Q-01 | Tampering | gosec/golangci suppressions | mitigate | Permit only exact rule IDs at exact statements with rationales, then require gosec v2.25.0 and PR new-issues lint to report zero. |
| T-Q-02 | Tampering | explicit versus cached library paths | mitigate | Resolve links only in `validateExplicitLibraryFile`; keep every cache-managed validator on `Lstat`-based rejection and retain the planted-cache regression. |
| T-Q-03 | Denial of Service | exported handle representation | mitigate | Restore the prior structs and add an external-package compile regression so downstream composite literals cannot silently break again. |
| T-Q-04 | Repudiation | SessionOptions lifecycle classification | mitigate | Record the once-live destroyed state under the handle mutex and assert distinct `errors.Is` results for zero versus destroyed options. |
| T-Q-SC | Tampering | tool/dependency supply chain | mitigate | Add no dependency; keep existing action pins and align the local gosec pin to the already reviewed CI version. |
</threat_model>

<verification>
Run all focused task checks, then run the combined repository gates:

1. `gofmt -l` on every modified Go file prints no paths.
2. `go test -count=1 -short ./...` passes.
3. `GOOS=windows GOARCH=amd64 go test -c -o /dev/null ./ort` passes.
4. `go run github.com/securego/gosec/v2/cmd/gosec@v2.25.0 -exclude-dir=examples/experimental ./...` exits zero with no findings.
5. `make precommit-lint-new PRECOMMIT_BASE_REF=main` and `go vet -unsafeptr=false ./ort/...` exit zero.
6. `git diff -- .github/workflows/ci.yml .golangci.yml` is empty, proving the enforcing workflow and global lint policy were not weakened.
</verification>

<source_audit>

| Source | ID | Feature / Requirement | Task | Status | Notes |
|--------|----|-----------------------|------|--------|-------|
| GOAL | — | Address all five review findings atomically | 1-3 | COVERED | One self-contained quick plan |
| REQ | RF-1 | Resolve the 15-finding gosec baseline before enforcement | 3 | COVERED | Exact v2.25.0 zero-finding gate |
| REQ | RF-2 | Handle the two intentional govet/unsafeptr callback casts | 3 | COVERED | Two line-scoped rationales |
| REQ | RF-3 | Accept explicit soname symlinks without accepting cached symlinks | 2 | COVERED | Separate validation paths and regressions |
| REQ | RF-4 | Preserve exported handle struct compatibility | 1 | COVERED | Restored structs plus downstream compile test |
| REQ | RF-5 | Distinguish destroyed from never-initialized SessionOptions | 1 | COVERED | Mutex-protected state plus errors.Is tests |
| RESEARCH | — | No research phase requested | — | EXCLUDED | Existing stdlib and project patterns are sufficient |
| CONTEXT | — | No quick-task CONTEXT.md supplied | — | EXCLUDED | Project instructions are recorded in plan context |

</source_audit>

<success_criteria>
- All five reviewer findings have a regression test or an exact automated scanner gate.
- Public exported type shape is compatible with downstream zero-value composite literals.
- Explicit trusted symlinks work while cache-managed symlink rejection is unchanged.
- `errors.Is` reliably distinguishes invalid zero SessionOptions from destroyed live SessionOptions.
- CI security and lint enforcement remain enabled and pass without broad exclusions.
- No dependency, unrelated cleanup, workflow weakening, merge, or internal repository reference is introduced.
</success_criteria>

<output>
Create `.planning/quick/260730-gye-address-all-five-review-findings/260730-gye-SUMMARY.md` when done.
</output>
