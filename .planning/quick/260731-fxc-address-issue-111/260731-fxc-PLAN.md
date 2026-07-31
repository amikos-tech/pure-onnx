---
phase: 260731-fxc-address-issue-111
plan: 01
type: tdd
wave: 1
depends_on: []
files_modified:
  - ort/bootstrap.go
  - ort/bootstrap_test.go
autonomous: true
requirements: [ISSUE-111]

must_haves:
  truths:
    - "A caller can reuse the same ORT BootstrapOption values in concurrent bootstrap calls without a race."
    - "Bootstrap-option whitespace normalization, lowercase checksum handling, validation errors, and configured values remain unchanged."
    - "Every analogous bootstrap option constructor is audited; the already race-safe OpenCLIP constructors remain behaviorally unchanged."
  artifacts:
    - path: "ort/bootstrap.go"
      provides: "Per-invocation normalization for every ORT BootstrapOption closure that currently assigns a captured string."
      contains: "WithBootstrapCacheDir"
    - path: "ort/bootstrap_test.go"
      provides: "Race-detector regression coverage for reusing one shared set of normalized ORT bootstrap options."
      contains: "TestBootstrapOptionsReusableConcurrently"
  key_links:
    - from: "ort/bootstrap_test.go TestBootstrapOptionsReusableConcurrently"
      to: "ort/bootstrap.go BootstrapOption constructors"
      via: "one shared option slice applied to fresh bootstrapConfig values from concurrent goroutines"
      pattern: "WithBootstrapCacheDir|WithBootstrapVersion"
    - from: "ort/bootstrap.go resolveBootstrapConfig"
      to: "BootstrapOption"
      via: "each bootstrap invocation applies supplied options to a new bootstrapConfig"
      pattern: "for _, opt := range opts"
---

<objective>
Remove the data race caused by reusable ORT bootstrap-option closures mutating their
captured input strings, and pin that concurrency contract with a focused race regression.

Purpose: Bootstrap callers may retain and concurrently reuse an option slice. Each call
must normalize its own input into its own configuration without changing existing values,
validation, cache behavior, or download behavior.

Output: A focused concurrent-option regression in `ort/bootstrap_test.go`, race-safe
normalization in `ort/bootstrap.go`, and two atomic conventional commits.
</objective>

<execution_context>
@~/.codex/get-shit-done/workflows/execute-plan.md
@~/.codex/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@ort/bootstrap.go
@ort/bootstrap_test.go
@embeddings/openclip/bootstrap.go
@embeddings/openclip/bootstrap_test.go

Issue #111 is scoped to option-closure safety. Preserve the public API and every current
validation/error message. Do not add dependencies, change cache locking or download logic,
or alter `BootstrapOption`'s type.

Use per-invocation local normalization inside the closures, not eager factory-time
normalization: this retains the current timing of validation errors (when the option is
applied) and avoids mutating state captured by a closure shared across calls.

The OpenCLIP audit is read-only. Its constructors do not reassign captured parameters at
plan time; notably, `WithBootstrapChecksum` already stores normalization in a local
`normalized` variable.

Keep commits atomic and conventional. If this branch is merged through a pull request, use
squash merge as required by project instructions.
</context>

<interfaces>
<!-- Verified against the tree at plan time. Reuse these contracts directly. -->

From `ort/bootstrap.go`:
```go
type BootstrapOption func(*bootstrapConfig) error

func WithBootstrapLibraryPath(path string) BootstrapOption
func WithBootstrapCacheDir(dir string) BootstrapOption
func WithBootstrapVersion(version string) BootstrapOption
func WithBootstrapExpectedSHA256(checksum string) BootstrapOption
func withBootstrapBaseURL(baseURL string) BootstrapOption

func resolveBootstrapConfig(opts ...BootstrapOption) (bootstrapConfig, error)
```

`resolveBootstrapConfig` creates a fresh `bootstrapConfig` and invokes every supplied
option in a loop. The regression must therefore share only the option closures, never one
configuration instance, between goroutines.

From `embeddings/openclip/bootstrap.go`:
```go
type BootstrapOption func(*bootstrapConfig) error

func WithBootstrapCacheDir(path string) BootstrapOption
func WithBootstrapRepoID(repoID string) BootstrapOption
func WithBootstrapRevision(revision string) BootstrapOption
func WithBootstrapToken(token string) BootstrapOption
func WithBootstrapChecksum(fileName string, checksum string) BootstrapOption
```
</interfaces>

## Source Coverage Audit

| Source | ID | Required outcome | Plan | Status |
|---|---|---|---|---|
| GOAL | — | Address issue #111: shared bootstrap options must not race | 01 | COVERED |
| REQ | ISSUE-111 | Normalize captured values per invocation or immutably; audit ORT and OpenCLIP analogues; add/adjust race regression | 01 | COVERED |
| RESEARCH | — | No research artifact supplied for this quick task | — | N/A |
| CONTEXT | — | No locked decision artifact supplied for this quick task | — | N/A |

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add a focused concurrent-reuse regression for ORT bootstrap options</name>
  <files>ort/bootstrap_test.go</files>
  <behavior>
    - A single shared slice containing the five string-normalizing ORT options can be applied from many goroutines to separate bootstrapConfig values.
    - Every invocation preserves current normalization: trimmed library path, cache directory, version and base URL, plus lowercase SHA256.
    - On the unfixed baseline, the race detector reports the closure-capture race; after Task 2, the same test passes cleanly under -race.
  </behavior>
  <action>
Add `TestBootstrapOptionsReusableConcurrently` near the existing constructor validation
tests in `ort/bootstrap_test.go`. Construct exactly one shared `[]BootstrapOption` using
whitespace-padded inputs for `WithBootstrapLibraryPath`, `WithBootstrapCacheDir`,
`WithBootstrapVersion`, `WithBootstrapExpectedSHA256` (uppercase 64-character SHA256),
and `withBootstrapBaseURL`. Use a valid HTTPS URL accepted by the existing base-URL
validator.

Release a fixed worker count (at least 16) from a start channel. Each worker should apply
the shared options repeatedly to a newly allocated `bootstrapConfig`, send errors through
a channel, and verify the resulting fields have the exact current normalized values. Keep
all `t.Fatal`/`t.Errorf` calls in the test goroutine after `sync.WaitGroup` completion;
do not make the test depend on a server, filesystem cache, download, or timing assertion.

Before changing production code, run the new test under `-race` and confirm the current
closure assignments produce the expected RED race report. Commit only this regression as
`test(ort): cover concurrent bootstrap option reuse`.
  </action>
  <verify>
    <automated>race_log="$(mktemp)"; go test ./ort -race -run '^TestBootstrapOptionsReusableConcurrently$' -count=1 &gt;"$race_log" 2&gt;&amp;1; test_status=$?; rg -q 'DATA RACE' "$race_log"; race_reported=$?; rm -f "$race_log"; test "$test_status" -ne 0 &amp;&amp; test "$race_reported" -eq 0</automated>
  </verify>
  <done>
The committed test uses a shared option slice and independent configurations, asserts every
existing normalized value, and reliably exposes the current race only when run with the
race detector. No network or runtime archive is involved.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Make ORT option normalization invocation-local and complete the constructor audit</name>
  <files>ort/bootstrap.go</files>
  <behavior>
    - Reusing any affected ORT BootstrapOption concurrently performs no write to its captured input.
    - Empty input continues to return the same validation error when the option is applied.
    - Valid input continues to populate bootstrapConfig with the same normalized value as before.
    - OpenCLIP option constructors remain unchanged because their captured values are read-only or normalized into invocation-local values already.
  </behavior>
  <action>
In `ort/bootstrap.go`, audit every `BootstrapOption` constructor before editing. Replace
only assignments to captured string parameters inside returned closures with a local value
created on each invocation: `WithBootstrapLibraryPath`, `WithBootstrapCacheDir`,
`WithBootstrapVersion`, `WithBootstrapExpectedSHA256`, and the test helper
`withBootstrapBaseURL`. Use clear local names such as `normalizedPath`,
`normalizedCacheDir`, `normalizedVersion`, `normalizedChecksum`, and
`normalizedBaseURL`; validate and assign those locals exactly where the captured parameter
is currently used.

Leave `WithBootstrapDisableDownload`, `WithBootstrapAllowSharedCache`, and
`withBootstrapHTTPClient` unchanged because they do not assign captured input. Preserve
the return type, error wrapping, validation order, whitespace/lowercase rules, and all
downstream configuration handling. Do not move validation to factory construction.

Audit the analogous constructors in `embeddings/openclip/bootstrap.go` and do not edit
that file: its cache-dir, repo-ID, revision, token, checksum, expected-size, download-cap,
and checksum-toggle options already avoid captured writes when each call receives its own
configuration. This audit prevents fixing only the two reported symptoms while leaving
another ORT closure racy.

Run the regression GREEN, then run the existing concurrent single-download reproduction
under `-race` to keep the end-to-end shared-option path covered. Commit the production
change separately as `fix(ort): make bootstrap option normalization concurrent-safe`.
  </action>
  <verify>
    <automated>go test ./ort -race -run '^(TestBootstrapOptionsReusableConcurrently|TestEnsureOnnxRuntimeSharedLibraryConcurrentLockSingleDownload)$' -count=1</automated>
    <automated>go test ./ort -run '^(TestWithBootstrapVersionRejectsEmpty|TestWithBootstrapLibraryPathAndCacheDirRejectEmpty|TestWithBootstrapExpectedSHA256Validation|TestWithBootstrapBaseURLValidation)$' -count=1</automated>
    <automated>go test ./embeddings/openclip -run '^(TestEnsureDefaultAssetsValidation|TestEnsureDefaultAssetsCustomRepoRequiresChecksums)$' -count=1</automated>
  </verify>
  <done>
Every ORT option closure that normalizes a captured string now uses an invocation-local
variable; the direct regression and the existing eight-worker download test pass with the
race detector; existing validation tests pass; and OpenCLIP has no unnecessary diff.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|---|---|
| Caller to `BootstrapOption` | Caller-controlled option values are validated before they configure a bootstrap operation. |
| Concurrent callers to shared option closures | Separate bootstrap operations may reuse a closure, but must not share mutable closure state. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|---|---|---|---|---|
| T-260731-fxc-01 | D | ORT `BootstrapOption` closure normalization | mitigate | Normalize captured strings into a local variable for each invocation and prove concurrent reuse with `go test -race`. |
| T-260731-fxc-02 | T | Bootstrap validation and configuration values | mitigate | Retain current validation location, error wrapping, and normalized values; run the existing focused constructor tests. |
| T-260731-fxc-SC | Tampering | Package installs | accept | This task installs no packages and introduces no supply-chain dependency. |
</threat_model>

<verification>
Run the three automated commands from Task 2 after both commits. Confirm the race detector
is clean for the focused direct regression and the existing concurrent-download test, and
confirm `git diff --name-only` contains only `ort/bootstrap.go` and
`ort/bootstrap_test.go` before summary generation.
</verification>

<success_criteria>
- The exact issue reproduction command exits 0 under `-race` with the shared `opts` slice.
- The new focused test makes concurrent option reuse a direct, network-free contract.
- No ORT closure writes a captured input string during option application.
- OpenCLIP receives no behavior change because the audit confirms its constructor closures are already safe.
- The work is represented by the two specified atomic commits and any PR integration uses squash merge.
</success_criteria>

<output>
Create `.planning/quick/260731-fxc-address-issue-111/260731-fxc-SUMMARY.md` when execution is complete.
</output>
