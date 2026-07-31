---
phase: 260730-qjz-address-pr-105-review-feedback
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - Makefile
  - ort/session.go
  - ort/environment.go
  - ort/errors.go
  - ort/diagnostics.go
  - ort/errors_test.go
  - .planning/phases/02-core-api-errors-values/*.md
autonomous: true
requirements: [FIX-1, FIX-2, FIX-3, FIX-4, FIX-5, FIX-6, FIX-7, ISSUES-1-5]

must_haves:
  truths:
    - "`make gosec` exits non-zero when gosec reports findings"
    - "`make gosec` still exits zero on the current clean tree"
    - "The lock-order sort in acquireValueLeases carries a comment explaining why it is load-bearing"
    - "The environment.go lock hierarchy block states that multiple Tensor.runMu leases DO nest within one Run, ordered by pointer identity"
    - "ort/errors_test.go covers a panicking `release` and pins the leak + discarded-return behavior"
    - "emitEmergencyDiagnostic renders an arbitrary panic value readably (panic(42) does not render as a rune)"
    - "clearORTGlobalsLocked documents the co-registration/co-clearing invariant statusToError depends on"
    - "godoc on ORTError tells a public caller to use errors.As + Code, and that sentinels deliberately do not match native errors"
    - "Stale phase-02 planning records are marked superseded by c7e58011"
    - ".planning/spikes/ is untouched"
    - "Five GitHub issues exist with [BUG]/[PERF]/[CLN] prefixes, each carrying file:line evidence and a PR #105 reference"
  artifacts:
    - path: "Makefile"
      provides: "gosec target that propagates failure"
      contains: "gosec"
    - path: "ort/errors_test.go"
      provides: "release-panic subtest case"
      contains: "release"
    - path: "ort/errors.go"
      provides: "ORTError godoc contract + statusToError invariant note"
      contains: "errors.As"
  key_links:
    - from: "ort/errors.go statusToError"
      to: "ort/environment.go clearORTGlobalsLocked"
      via: "documented co-clearing invariant replacing a nil guard"
      pattern: "clearORTGlobalsLocked"
    - from: "ort/session.go acquireValueLeases sort"
      to: "ort/environment.go lock hierarchy block"
      via: "pointer-identity total order documented in both places"
      pattern: "orderKey|pointer identity"
---

<objective>
Land seven verified fixes from PR #105 review feedback and file five follow-up issues for
work that is out of scope for this branch.

Purpose: One real build-gate bug (`make gosec` can never fail) plus six documentation/test
gaps where load-bearing invariants exist only in `.planning/` or in nobody's head. The
issues capture genuine findings that are too large or too risky to fold into this PR.

Output: 7-8 atomic conventional commits on `gsd/phase-2-full`, and 5 GitHub issues on
`amikos-tech/pure-onnx`.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@.planning/STATE.md

Scope is ALREADY TRIAGED AND LOCKED. Do not re-litigate which items are valid, do not add
items, do not expand scope. Every reviewer claim not listed below was investigated and
found FALSE or already-addressed — see `<out_of_scope>`.

**Line numbers below were accurate at verification time. Re-locate every target by content
(grep for the quoted identifier/string) before editing. Do not trust a line number.**

Comments must be TIGHT. CLAUDE.md forbids verbose or numerous comments — every comment
added here must be load-bearing. Aim for 1-4 lines each.

NO CGO. Never introduce `import "C"` or CGO types into `ort/`.

Work stays on `gsd/phase-2-full`. Never push to main.
</context>

<interfaces>
<!-- Verified against the tree at plan time. Executor should not need to explore. -->

`Makefile` gosec target (current, buggy — trailing `echo` masks gosec's exit status):
```
gosec:
	@echo "$(YELLOW)Running gosec...$(NC)"
	@if command -v gosec &> /dev/null; then \
		gosec -exclude-dir=examples/experimental ./...; \
		echo "$(GREEN)✓ gosec complete$(NC)"; \
	else \
		echo "$(RED)✗ gosec not installed. Run 'make install-tools' first$(NC)"; \
		exit 1; \
	fi
```
`precommit` already invokes `$(MAKE) gosec` (guarded by `SKIP_GOSEC=1`), so fixing the
target genuinely arms the local gate.

`ort/errors.go` — `defer ops.release(status)` is registered BEFORE the `&ORTError{}`
literal is constructed:
```go
type statusOps struct {
	getCode     func(uintptr) ErrorCode
	copyMessage func(uintptr) string
	release     func(uintptr)
}

func statusToErrorWithOps(status uintptr, operation string, ops statusOps) error {
	if status == 0 {
		return nil
	}
	defer ops.release(status)

	return &ORTError{
		Operation: operation,
		Code:      ops.getCode(status),
		Message:   ops.copyMessage(status),
	}
}
```

`ort/environment.go` lock hierarchy doc block (top of the `var (` group):
```
// Lock relationships across ORT lifecycle and calls form a partial order;
// an arrow means the lock on the left is acquired before the lock on the right:
// AdvancedSession.runMu -> ortCallMu.
// ortCallMu -> SessionOptions.handleMu, mu, Tensor.runMu, MemoryInfo.handleMu.
// SessionOptions.handleMu -> mu only when both are held.
// mu is released before Tensor.runMu or MemoryInfo.handleMu.
// Tensor.runMu and MemoryInfo.handleMu are never nested with each other.
// Not every listed lock is held in one operation.
```

`ort/session.go` `acquireValueLeases` sort (currently zero comment):
```go
sort.Slice(candidates, func(i, j int) bool {
	if candidates[i].orderKey != candidates[j].orderKey {
		return candidates[i].orderKey < candidates[j].orderKey
	}
	return reflect.TypeOf(candidates[i].lockable).String() <
		reflect.TypeOf(candidates[j].lockable).String()
})
```
The loop immediately after acquires `lockForRun()` per candidate in sorted order — so all
leases held by a single `Run()` nest, and the sort is the only thing preventing AB-BA
deadlock between concurrent `Run()` calls sharing tensors.

`ort/diagnostics.go` — `handlerPanic` is a raw `recover()` value typed `any`:
```go
func emitEmergencyDiagnostic(message, resource string, cleanupFailure, handlerPanic any) {
	...
	"onnx-purego emergency diagnostic: %s resource=%q cleanup_failure=%q handler_panic=%q\n",
```
Verified: nothing in `ort/` asserts on this format string, so changing the verb is safe.

`ort/environment.go` — `clearORTGlobalsLocked` nils all func globals together
(`getErrorCodeFunc`, `getErrorMessageFunc`, `releaseStatusFunc`, `createSessionFunc`,
`runSessionFunc`, ...); they are registered together in one `purego.RegisterFunc` block.
This is why `ort/session.go` `NewAdvancedSession` (~:185, ~:213) and `run` (~:358) can call
`statusToError` while guarding only `ortAPI`/`createSessionFunc`/`runSessionFunc`.

Shipped diagnostics default (`ort/diagnostics.go`), changed in commit `c7e58011`:
```go
func newDefaultDiagnosticHandler() slog.Handler {
	return slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn})
}
```
`README.md` already matches this. Only the phase-02 planning records are stale.

Repo for `gh`: `amikos-tech/pure-onnx`. Existing title convention confirmed via
`gh issue list` (e.g. `[TST] Replace timing-based concurrency assertions...`).
</interfaces>

<tasks>

<task type="auto">
  <name>Task 1: Make the gosec target actually blocking</name>
  <files>Makefile</files>
  <action>
Re-locate the `gosec:` target (grep `^gosec:`). The recipe's last command is
`echo "✓ gosec complete"`, so the recipe exit status is always 0 even when gosec reports
findings — despite the target's `(blocking)` label and `precommit` invoking it.

Fix so gosec's non-zero exit propagates. Simplest form that preserves the existing
success message and the not-installed branch: append `|| exit 1;` to the gosec invocation.
Do not restructure the `command -v` guard, do not touch the `&> /dev/null` bashism, do not
change the exclude-dir flag.
  </action>
  <verify>
    <automated>make gosec; echo "clean-tree-exit=$?"</automated>
    <automated>sh -c 'if command -v false >/dev/null 2>&1; then false || exit 1; echo unreachable; else exit 1; fi'; test $? -ne 0 &amp;&amp; echo "propagation-shape-ok"</automated>
  </verify>
  <done>
`make gosec` exits 0 on the current clean tree AND the recipe is structured so a non-zero
gosec exit aborts before the success echo. No test scaffolding committed.
Commit: `fix(build): propagate gosec failures from make gosec`
  </done>
</task>

<task type="auto">
  <name>Task 2: Document the lock-order sort in acquireValueLeases</name>
  <files>ort/session.go, ort/environment.go</files>
  <action>
FIX 2. Two edits, one commit.

(a) `ort/session.go` — re-locate the `sort.Slice(candidates, ...)` call inside
`acquireValueLeases` (grep `sort.Slice(candidates`). Add a tight comment directly above it
explaining WHY it exists: the loop below acquires every candidate's `lockForRun()` while
holding the earlier ones, so leases nest within one Run; sorting by `orderKey` (pointer
identity) with a reflect type-name tiebreak establishes a process-wide total order that
prevents AB-BA deadlock between concurrent `Run()` calls sharing tensors. State that
removing or reordering the sort reintroduces that deadlock. 3-4 lines maximum.

(b) `ort/environment.go` — re-locate the lock hierarchy comment block at the top of the
package `var (` group (grep `Lock relationships across ORT lifecycle`). The line
"Tensor.runMu and MemoryInfo.handleMu are never nested with each other" currently reads as
if no Tensor.runMu nesting occurs at all. Add one line noting that multiple Tensor.runMu
leases DO nest with one another within a single Run, ordered by pointer identity in
acquireValueLeases. Keep it to one line and preserve the existing block's terse style.

CLAUDE.md:141 mandates documenting lock ordering; CLAUDE.md also forbids verbosity. Do not
change any behavior — comments only.
  </action>
  <verify>
    <automated>go build ./... &amp;&amp; go vet ./ort/ &amp;&amp; grep -A2 -B4 'sort.Slice(candidates' ort/session.go | grep -q '//' &amp;&amp; echo "sort-comment-present"</automated>
    <automated>grep -c 'pointer identity' ort/session.go ort/environment.go</automated>
  </verify>
  <done>
The sort has a comment stating the deadlock it prevents; the environment.go hierarchy block
mentions intra-Run Tensor.runMu nesting and its pointer-identity ordering. Zero behavior
change (`git diff` shows only comment lines).
Commit: `docs(ort): explain the value-lease lock ordering`
  </done>
</task>

<task type="auto">
  <name>Task 3: Document the error-path invariants and fix the panic-value verb</name>
  <files>ort/errors.go, ort/environment.go, ort/diagnostics.go</files>
  <action>
FIX 4, FIX 5, FIX 6. Three separate atomic commits.

**Commit A — FIX 4 (`ort/diagnostics.go`).** Re-locate `emitEmergencyDiagnostic` (grep
`handler_panic=`). Change the `handler_panic=%q` verb to `%v`: `handlerPanic` is a raw
`recover()` value of type `any`, so `panic(42)` currently renders as the rune literal `'*'`
and a struct renders as `%!q(...)`. Leave `resource=%q` as-is (it is a `string`).
`cleanupFailure` is always an `error` at its call sites so `%q` reads fine — leave it
unless changing it makes the line more consistent, in which case use your judgment and
say so in the SUMMARY. Verified: no test asserts on this format string.
Commit: `fix(diagnostics): render arbitrary panic values with %v`

**Commit B — FIX 5 (`ort/errors.go` + `ort/environment.go`).** DOCUMENTATION ONLY — do NOT
add runtime nil checks.
`statusToError` reads `getErrorCodeFunc`, `getErrorMessageFunc` and `releaseStatusFunc`
with no nil guard. Most callers pre-check them, but `NewAdvancedSession` (~:185, ~:213) and
`run` (~:358) in `ort/session.go` guard only `ortAPI`/`createSessionFunc`/`runSessionFunc`.
Those sites are safe solely because all the func globals are registered together in one
`purego.RegisterFunc` block and cleared together in `clearORTGlobalsLocked` — so
`runSessionFunc != nil` implies `getErrorCodeFunc != nil`. That coupling is unwritten.
- At `clearORTGlobalsLocked` in `ort/environment.go`: add a comment stating the
  co-registration/co-clearing invariant explicitly — these globals are registered as a set
  and must be cleared as a set, because callers rely on one being non-nil implying the rest.
- At `statusToError` in `ort/errors.go`: extend the existing doc comment with a brief note
  that it has no nil guard by design and relies on that invariant.
2-3 lines each.
Commit: `docs(ort): record the ORT global co-clearing invariant`

**Commit C — FIX 6 (`ort/errors.go`).** Re-locate the `ORTError` type (grep
`type ORTError struct`). Extend its godoc to 3-4 lines total covering: native ORT failures
are reached via `errors.As(err, &ortErr)` and inspected through the `Code` field; the
package sentinels (`ErrInvalidArgument`, `ErrNotInitialized`, `ErrDestroyed`, ...) cover
Go-side validation only and deliberately do NOT match native status-derived errors via
`errors.Is`. This asymmetry is intentional and test-enforced — the point is that a caller
reading the public API can discover it without reading `.planning/`. Do not change the
struct or any behavior.
Commit: `docs(ort): document the ORTError matching contract`
  </action>
  <verify>
    <automated>go build ./... &amp;&amp; go vet ./ort/</automated>
    <automated>go test ./ort/ -run 'TestStatusToError|TestSentinel|TestDiagnostic|TestEmergency' -count=1</automated>
    <automated>grep -q 'handler_panic=%v' ort/diagnostics.go &amp;&amp; grep -q 'errors.As' ort/errors.go &amp;&amp; echo "fix-4-and-6-present"</automated>
  </verify>
  <done>
`%q` on `handlerPanic` is gone; `clearORTGlobalsLocked` and `statusToError` both state the
co-clearing invariant; `ORTError` godoc explains errors.As + the deliberate sentinel
non-match. No runtime nil checks added. Three separate commits.
  </done>
</task>

<task type="auto">
  <name>Task 4: Cover a panicking release in errors_test.go</name>
  <files>ort/errors_test.go</files>
  <action>
FIX 3. Re-locate the `"releases when an accessor panics"` subtest table in
`ort/errors_test.go` (grep that string). It has two cases — `code accessor` and
`message accessor` — both of which assert the panic escapes AND `releases == 1`.

Add a third case for `release` itself panicking. The semantics differ and must be pinned
exactly as they are today:
- `defer ops.release(status)` is registered BEFORE the `&ORTError{...}` literal is built,
  so `getCode` and `copyMessage` DO run first and the return value IS computed —
  then discarded when the deferred `release` panics.
- The native status is LEAKED: nothing else frees it.
- The panic escapes; there is no recover in `statusToError`/`statusToErrorWithOps`.

The existing table shape assumes a `releases *atomic.Int32` counter and a shared loop body
that asserts `releases == 1`. A panicking `release` cannot satisfy that shared assertion,
so restructure minimally — e.g. give the table case an expected-release count and/or its
own assertion — rather than bolting the new case onto an assertion it contradicts. Keep the
table style; do not rewrite the surrounding subtests.

Assert what makes the semantics legible: getCode and copyMessage were invoked (accessors
ran), the panic propagated, and the release attempt is observable. Prefer asserting on the
recovered panic value so a future change to recover-and-swallow fails the test.

PIN THE CURRENT BEHAVIOR — do NOT change `ort/errors.go`.
`.planning/phases/02-core-api-errors-values/02-01-PLAN.md:176` deliberately forbids a nil
fallback for `release` to preserve exactly-one-release semantics; this test must not
pressure that design.
  </action>
  <verify>
    <automated>go test ./ort/ -run 'TestStatusToError' -count=1 -v 2>&amp;1 | grep -i 'release'</automated>
    <automated>go test ./ort/ -race -run 'TestStatusToError' -count=1</automated>
    <automated>git diff --name-only | grep -qx 'ort/errors_test.go' &amp;&amp; ! git diff --name-only | grep -qx 'ort/errors.go' &amp;&amp; echo "test-only-change"</automated>
  </verify>
  <done>
A third subtest case covers a panicking `release`, passes, and documents (via assertions or
a one-line comment) that the return value is discarded and the status leaks.
`ort/errors.go` is unmodified.
Commit: `test(ort): cover a panicking status release`
  </done>
</task>

<task type="auto">
  <name>Task 5: Mark the stale phase-02 diagnostics records as superseded</name>
  <files>.planning/phases/02-core-api-errors-values/*.md</files>
  <action>
FIX 7. Several phase-02 records still describe the diagnostics default as "silent" via
`slog.DiscardHandler`. The SHIPPED default is
`slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn})`
(`ort/diagnostics.go`, `newDefaultDiagnosticHandler`), changed in commit `c7e58011`
("fix(260730-ink): restore fail-safe diagnostics"). `README.md` already matches shipped
behavior and must not be touched.

Re-locate affected files by content:
```
grep -rln 'DiscardHandler\|silent by default\|silent default' .planning/phases/02-core-api-errors-values/
```
At plan time this matched more files than the review listed (including `02-VALIDATION.md`,
`02-REVIEWS.md`, `02-UAT.md`, `02-03-PLAN.md`, `02-03-SUMMARY.md`, `02-07-SUMMARY.md`).
Include a file only if its statement is genuinely contradicted by the shipped default —
skip incidental mentions (e.g. a nil-handler test description that is still accurate).

For each genuinely stale file, add ONE short note near the top:
`> Superseded by c7e58011: the shipped diagnostics default is a stderr TextHandler at
> LevelWarn, not a silent DiscardHandler.`
These are historical records, not living docs — prefer the header note over rewriting every
occurrence. Do not restate the change per-passage unless a header note would be misleading
for that file.

DO NOT TOUCH `.planning/spikes/` — those throwaway prototypes legitimately still use
`DiscardHandler`. Do not touch `README.md` or any `ort/` source.
  </action>
  <verify>
    <automated>git diff --name-only | grep -v '^.planning/phases/02-core-api-errors-values/' | grep -q . &amp;&amp; echo "LEAKED-OUTSIDE-PHASE-02" || echo "scoped-correctly"</automated>
    <automated>git status --porcelain .planning/spikes/ README.md ort/ | grep -q . &amp;&amp; echo "FORBIDDEN-PATH-TOUCHED" || echo "forbidden-paths-clean"</automated>
    <automated>grep -rl 'Superseded by c7e58011' .planning/phases/02-core-api-errors-values/ | wc -l</automated>
  </verify>
  <done>
Every genuinely stale phase-02 record carries a superseded-by-c7e58011 note.
`.planning/spikes/`, `README.md` and `ort/` are untouched.
Commit: `docs(planning): mark stale phase-02 diagnostics records superseded`
  </done>
</task>

<task type="auto">
  <name>Task 6: File five follow-up issues on amikos-tech/pure-onnx</name>
  <files>(no files — gh issue create only)</files>
  <action>
Create five issues via `gh issue create` against `amikos-tech/pure-onnx`. Do NOT fix any of
these in code. Each body must carry the file:line evidence so the issue is actionable
standalone, and must reference PR #105 as its origin. Re-verify each cited path/line by
content before writing the body; if a line has drifted, cite the current one.

1. `[BUG] Bootstrap fingerprint memo bypasses content verification and per-file trust checks`
   `ort/bootstrap.go` `bootstrapInstallFingerprint` (~:1659-1687) keys on relative path +
   mode + size + mtime only — no content hash. On a memo hit (~:1573-1580) BOTH the manifest
   hash comparison AND the per-file ownership/mode/symlink walk (~:1751-1780) are skipped.
   The repo's own passing test `ort/bootstrap_test.go:3234`
   (`TestEnsureOnnxRuntimeSharedLibraryMemoizesVerifiedInstall`) demonstrates the bypass: it
   rewrites the manifest with same-length garbage, restores mtime via `os.Chtimes`, and
   asserts the next call succeeds. The memo is process-scoped (`sync.Map` ~:73) and is
   consulted on all cache-managed paths — `allowSharedCache` is a key field, not a gate.
   Suggested direction: gate the memo on `!allowSharedCache`. Threat model is narrow
   (same-process, requires write access to the cache dir) but this deliberately narrows the
   exact integrity property the manifest exists to provide.

2. `[BUG] TOCTOU: Lstat-then-open without O_NOFOLLOW in bootstrap`
   No `O_NOFOLLOW` anywhere in the repo. Three sites: `ort/bootstrap.go:2050` (lock file
   `os.OpenFile`), `:1727` (manifest `os.ReadFile`), `:1807` (`sha256File` -> `os.Open`).
   The usual mitigation argument — a swapped-in symlink reports mode 0777 and trips the
   world-writable check — is LINUX-ONLY. macOS/BSD symlinks are `lrwxr-xr-x` (0755) and pass
   both the `&0o002` and `&0o020` checks at `ort/bootstrap_trust_unix.go:17,20`. With
   `allowSharedCache` the uid check is skipped (`:28-30`), so on **darwin + shared cache** a
   symlink swapped between check and use passes. The post-open recheck at `:2054-2060` calls
   only `validateBootstrapPathOwnershipAndMode` and does NOT re-test `os.ModeSymlink`.
   Scoped to opt-in `WithBootstrapAllowSharedCache` plus a co-group attacker.
   Suggested: add `O_NOFOLLOW` at those opens and re-check `ModeSymlink` post-open.

3. `[BUG] openclip bootstrap leaks presigned URLs into returned errors`
   `embeddings/openclip/bootstrap.go:377` formats `req.URL.String()` raw in the
   HTTPS-downgrade redirect error. Hugging Face `resolve/` endpoints redirect to CDN URLs
   carrying `X-Amz-Signature`, so a blocked redirect embeds a time-limited download
   credential into an error callers typically log.
   Related: `ort/bootstrap.go:2185-2191` `redactedBootstrapURL` delegates to
   `net/url.(*URL).Redacted()`, which replaces ONLY the userinfo password — query params,
   path and fragment pass through untouched. The name implies far broader coverage than it
   delivers. Current usage is 1 helper call site versus ~33 raw URL interpolations across
   `ort/bootstrap.go` (21) and `embeddings/openclip/bootstrap.go` (12).
   No static-token leak exists today: `GITHUB_TOKEN`/`GH_TOKEN`/`HF_TOKEN` all go into
   `Authorization` headers, and both base URLs are compile-time constants with only
   unexported test hooks. This is defense-in-depth plus one concrete capability leak.
   Suggested: rename the helper to reflect what it actually does (or strip query params) and
   apply it at `:377`. Secondary note to include: `ort/bootstrap.go:1128` and
   `embeddings/openclip/bootstrap.go:652-657` splice up to 512 B / 2 KB of raw HTTP response
   body into errors.

4. `[PERF] Bootstrap cache-hit path SHA-256s the whole install tree on first call per process`
   `ort/bootstrap.go:363` performs cache validation UNLOCKED (before the interprocess lock
   at `:387`), reaching `collectBootstrapInstallFiles` (~:1629) -> `sha256File` (~:1805), a
   full `io.Copy` over every file in the install dir including the 100+ MB shared library.
   The `sync.Map` memo added in commit `120dafd` only short-circuits REPEAT calls within the
   same process, so serverless / short-lived processes pay the full hash on every cold start.
   Suggested direction: an mtime-gated on-disk validation stamp, or document the cold-start
   cost in the bootstrap godoc. Note this trades against issue #1 — any change here must not
   further weaken integrity verification. Cross-link the two issues once both numbers exist.

5. `[CLN] CI runtime-version match step is formatting-brittle`
   `.github/workflows/ci.yml:409-416` extracts the ORT version from three sources with
   `sed -nE` regexes. It breaks on formatting-only changes that leave the versions genuinely
   in agreement: a backtick or explicitly-typed Go literal
   (`DefaultOnnxRuntimeVersion string = "..."`); a double-quoted or bare YAML value; a second
   job introducing its own `ORT_VERSION:` key (two lines printed); or `ORT_VERSION ?= 1.24.1`
   in the Makefile — the style ALREADY used one line away at `Makefile:13` for
   `GOSEC_VERSION`. Failures are loud rather than silent, so severity is low. Also track here:
   `ORT_ARCHIVE_SHA256` at `ci.yml:185` must be bumped in lockstep with `ORT_VERSION` but is
   not covered by this step (already noted as a known gap in the PR body).

Write each body to a scratch file and pass it with `gh issue create --title ... --body-file`
to keep formatting intact. Do NOT commit any scratch files. Record the created issue numbers
in the SUMMARY.
  </action>
  <verify>
    <automated>gh issue list --repo amikos-tech/pure-onnx --limit 10 --json number,title -q '.[] | "\(.number) \(.title)"' | grep -cE '^\S+ \[(BUG|PERF|CLN)\]'</automated>
    <automated>git status --porcelain | grep -q . &amp;&amp; echo "UNCOMMITTED-SCRATCH-PRESENT" || echo "tree-clean"</automated>
  </verify>
  <done>
Five issues exist with `[BUG]`/`[PERF]`/`[CLN]` prefixes, each containing file:line evidence
and a PR #105 origin reference. Issues #4 and #1 cross-link. No code changed, no scratch
files committed, working tree clean.
  </done>
</task>

</tasks>

<out_of_scope>
Investigated and found FALSE or already-addressed. DO NOT change these.

- `#nosec G304 G703` at `tools/gen_ortapi.go:27` and `examples/openclip/main.go:150,236`.
  **G703 IS a real gosec rule** ("Path traversal via taint analysis"), default-enabled in the
  pinned v2.25.0 (`analyzers/analyzerslist.go:158`, `RULES.md:97`); gosec's own
  `cmd/gosecutil/tools.go:77` uses the identical idiom. G7xx rules live in `analyzers/`, not
  `rules/rulelist.go`, which is why rule tables appear to stop at G602.
  This is the ONE out-of-scope case where a brief comment recording the rationale has clear
  value, since a future reader may "fix" the perceived typo. If you add one, keep it to a
  single line at ONE of the three sites and mention it in the SUMMARY. Do not sprawl.
- CI hardcoded test counts (`ci.yml:151-154`, `:256-259`). Explanatory comments were already
  added in commit `2f11f89` at `:142-145` and `:248-250`; the selectors are `^(...)$`-anchored
  so adding a test cannot shift the count.
- The gosec blocking change itself — already documented under "## CI Gate Changes" in the PR
  body with a before/after table. (Task 1 fixes the Makefile bug, not this.)
- The ORT 1.24.1 "bump". CI was already on 1.24.1 on `main` (adopted in `42883e5`); this PR's
  `a47bb66` (+14/-5) was a drift FIX traceable to review finding WR-12.
- Windows trust no-op (`ort/bootstrap_trust_other.go`). Documented at `README.md:78-86`, in
  code comments, and pinned by `ort/bootstrap_test.go:1150`.
- `Value` sealing doc (`ort/types.go:78-88`). Already says "unsupported" rather than claiming
  impossibility; foreign implementations are rejected at `ort/session.go:422-435` before any
  native call, tested at `ort/session_test.go:1172`.
- `RunWithValues` teardown-race coverage and the `IsTensor`/`AsTensor` disagreement quadrant.
  Both verified non-issues — `Run` and `RunWithValues` share an identical locking core, and
  `AsTensor` uses concrete type assertion so the disagreement is by design.
</out_of_scope>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| local build gate -> CI | `make gosec` result is cited as evidence that the blocking CI gosec gate is safe |
| GitHub issue bodies -> public | Issue text is world-readable on a public repo |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-qjz-01 | Repudiation | `make gosec` target | mitigate | Task 1 makes the target propagate gosec's exit status, so a green local run is real evidence rather than an unconditional success |
| T-qjz-02 | Information Disclosure | Issue bodies (Task 6) | mitigate | Bodies cite only public repo paths/line numbers and public upstream URLs; no tokens, no presigned URLs, no internal hostnames. Never paste a real `HF_TOKEN`/`GITHUB_TOKEN` or a live signed URL as an example |
| T-qjz-03 | Tampering | Source edits | accept | No dependency changes, no `go.mod`/`go.sum` edits, no package-manager installs — nothing to audit for supply chain |
| T-qjz-04 | Denial of Service | `acquireValueLeases` sort | mitigate | Task 2 documents the sort as the AB-BA deadlock guard so a future refactor cannot silently remove it |
| T-qjz-SC | Tampering | npm/pip/cargo installs | n/a | No package-manager installs in this plan |
</threat_model>

<verification>
Run the full local gate after Tasks 1-5 (Task 6 touches no files):

```bash
make precommit
```

Note that Task 1 changes this gate's behavior — `make gosec` becomes genuinely blocking, so
`make precommit` may now surface pre-existing gosec findings that were previously masked.
If it does, STOP and report them rather than suppressing them with `#nosec` or
`SKIP_GOSEC=1`; deciding how to handle real findings is out of scope for this plan.

Also confirm no behavior regressions in the packages touched:

```bash
go test ./ort/ -race -count=1
git log --oneline gsd/phase-2-full ^origin/main | head -10
```
</verification>

<success_criteria>
- [ ] `make gosec` exits non-zero on gosec findings and 0 on the clean tree
- [ ] `acquireValueLeases` sort carries a WHY comment; `environment.go` hierarchy block notes intra-Run Tensor.runMu nesting
- [ ] `ort/errors_test.go` covers a panicking `release`; `ort/errors.go` behavior unchanged
- [ ] `handler_panic` uses `%v`; no test asserts broken
- [ ] `clearORTGlobalsLocked` + `statusToError` document the co-clearing invariant; no runtime nil checks added
- [ ] `ORTError` godoc explains `errors.As` + Code and the deliberate sentinel non-match
- [ ] Stale phase-02 records marked superseded; `.planning/spikes/` and `README.md` untouched
- [ ] Five issues filed with correct prefixes, file:line evidence, and PR #105 references
- [ ] 7-8 atomic conventional commits on `gsd/phase-2-full`; nothing pushed to `main`
- [ ] `make precommit` passes
- [ ] No `import "C"` or CGO types anywhere in the diff
</success_criteria>

<output>
Create `.planning/quick/260730-qjz-address-pr-105-review-feedback/260730-qjz-SUMMARY.md` when done.
Include: the created issue numbers, whether `cleanupFailure`'s verb was changed and why,
which phase-02 files were marked superseded (and which matches were skipped as incidental),
and whether a G703 rationale comment was added.
</output>
