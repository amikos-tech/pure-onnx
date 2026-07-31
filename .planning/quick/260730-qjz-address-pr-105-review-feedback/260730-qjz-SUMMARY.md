---
phase: 260730-qjz-address-pr-105-review-feedback
plan: 01
subsystem: build+api-docs
tags: [go, gosec, makefile, diagnostics, errors, locking, planning-records, github-issues]
requires: []
provides: ["blocking make gosec", "documented lock-order and error-path invariants", "release-panic test coverage"]
affects: [Makefile, ort/session.go, ort/environment.go, ort/errors.go, ort/diagnostics.go, ort/errors_test.go, tools/gen_ortapi.go, .planning/phases/02-core-api-errors-values]
tech-stack:
  added: []
  patterns: ["comment-documented lock ordering", "table-driven panic-semantics tests"]
key-files:
  created: []
  modified:
    - Makefile
    - ort/session.go
    - ort/environment.go
    - ort/errors.go
    - ort/diagnostics.go
    - ort/errors_test.go
    - tools/gen_ortapi.go
    - .planning/phases/02-core-api-errors-values/*.md (12 files)
decisions:
  - "Changed cleanup_failure from %q to %v alongside handler_panic: one call site passes a raw recover() value, not an error"
  - "Marked 12 phase-02 records superseded; skipped 6 files whose 'silent' mentions remain accurate"
  - "Added the single allowed G703 rationale comment at tools/gen_ortapi.go"
metrics:
  duration: ~35 min
  completed: 2026-07-30
---

# Quick Task 260730-qjz: Address PR #105 Review Feedback Summary

Landed seven verified PR #105 fixes — one real build-gate bug plus six documentation/test gaps — and filed five follow-up issues (#106-#110) for work too large for this branch.

## Commits

| Commit | Task | Description |
|--------|------|-------------|
| `1a13f51` | 1 | `fix(build): propagate gosec failures from make gosec` |
| `43d2b9a` | 2 | `docs(ort): explain the value-lease lock ordering` |
| `2cfe12a` | 3A | `fix(diagnostics): render arbitrary panic values with %v` |
| `2c86517` | 3B | `docs(ort): record the ORT global co-clearing invariant` |
| `4ea5f87` | 3C | `docs(ort): document the ORTError matching contract` |
| `610be71` | 4 | `test(ort): cover a panicking status release` |
| `4bc770a` | 5 | `docs(planning): mark stale phase-02 diagnostics records superseded` |
| `80dd73f` | — | `docs(tools): record why the G703 nosec suppression is not a typo` |

## What Was Done

**Task 1 — gosec gate (`Makefile:197`).** Appended `|| exit 1` to the gosec invocation so a non-zero exit aborts before the `✓ gosec complete` echo. The `command -v` guard, the `&> /dev/null` bashism, and the exclude-dir flag are untouched. `make gosec` on the clean tree: **0 issues, exit 0**. Arming the gate surfaced **no previously-hidden findings** — nothing was suppressed and `SKIP_GOSEC` was never set.

**Task 2 — lock ordering.** `ort/session.go` now carries a 4-line comment above `sort.Slice(candidates, ...)` stating that the loop below holds each lease while taking the next, that pointer-identity ordering gives every `Run` the same process-wide total order, and that removing the sort reintroduces AB-BA deadlock. `ort/environment.go`'s hierarchy block gained one line: multiple `Tensor.runMu` leases *do* nest within one `Run`, ordered by pointer identity in `acquireValueLeases`. Comments only — `git diff` showed zero behavior change.

**Task 3A — panic-value verb (`ort/diagnostics.go:78`).** Changed **both** `cleanup_failure` and `handler_panic` from `%q` to `%v`, not just `handlerPanic` as the plan's default suggested. Reason: `ort/environment.go:186` passes `rollbackPanic` — a raw `recover()` value typed `any` — as the `cleanupFailure` argument, so the plan's premise ("always an `error` at its call sites") does not hold. `%q` on a non-string mangles it (`panic(42)` renders as a rune literal, a struct as `%!q(...)`). `resource=%q` is unchanged; it is a real `string`. No test asserts on this format string (verified by grep).

**Task 3B — co-clearing invariant.** `clearORTGlobalsLocked` (`ort/environment.go`) now states that the ORT func globals are registered as one set and must be cleared as one set, because callers treat one non-nil global as implying the rest. `statusToError`'s doc comment (`ort/errors.go`) records that it omits nil guards by design and relies on that invariant. **No runtime nil checks added.**

**Task 3C — ORTError contract.** Godoc extended: reach native failures via `errors.As(err, &ortErr)` and branch on `Code`; the package sentinels report Go-side validation only and deliberately never match a native status via `errors.Is`.

**Task 4 — release-panic coverage.** Restructured the `"releases when an accessor panics"` table rather than appending to it, because the shared `releases == 1` assertion cannot express the new case's semantics. The table now carries per-case `wantPanic`, `wantCodes`, `wantMessages`, `wantReleases`, backed by a `statusOpCalls` counter struct. The new `release` case pins current behavior exactly: both accessors run (`wantCodes: 1`, `wantMessages: 1`) because `defer ops.release(status)` is registered before the `&ORTError{}` literal is built; the computed error is discarded (asserted via `got == nil`); the native status leaks (recorded in a comment). Asserting the recovered panic value means a future recover-and-swallow change fails the test. `ort/errors.go` is unmodified — the `.planning/phases/02-core-api-errors-values/02-01-PLAN.md:176` prohibition on a nil `release` fallback is not pressured.

**Task 5 — stale phase-02 records.** Added the one-line superseded note to **12 files**: `02-CONTEXT.md`, `02-RESEARCH.md`, `02-PATTERNS.md`, `02-DISCUSSION-LOG.md`, `02-VALIDATION.md`, `02-VERIFICATION.md`, `02-UAT.md`, `02-REVIEWS.md`, `02-03-PLAN.md`, `02-03-SUMMARY.md`, `02-06-PLAN.md`, `02-07-SUMMARY.md`. Each of these asserts a *default* that the shipped code contradicts — D-17/D-20's `slog.DiscardHandler` initialization, "silent by default", or "nil restores silence" (shipped: `SetDiagnosticHandler(nil)` installs the stderr `TextHandler` at `LevelWarn`, `ort/diagnostics.go:47-52`).

*Skipped as incidental — statement still accurate:*
- `02-04-SUMMARY.md:36,62` and `02-05-SUMMARY.md:66` — "returned failures remain silent" is about not double-logging returned errors, still true.
- `02-08-SUMMARY.md:93`, `02-REVIEWS.md`-style "silently reducing coverage" phrasing, `02-REVIEW.md`, `02-REVIEW-FIX.md`, `02-07-PLAN.md:258` — "silently" used in unrelated senses (integer wrap, CI selectors, panic recovery).

*Judgment call:* `02-06-PLAN.md` was included despite the planning note flagging its `:118` "nil handler is silent" as still accurate — because `:259` ("Diagnostics are silent by default") is a genuine contradiction. The header note is general and does not misrepresent `:118`.

`.planning/spikes/`, `README.md`, and `ort/` were verified untouched by that commit.

**G703 rationale comment (bounded allowance).** Added — one line, one site: `tools/gen_ortapi.go:27`, stating that G703 is gosec's taint-analysis path-traversal analyzer listed under `analyzers/` rather than `rules/`. The `#nosec G304 G703` annotations themselves are unchanged, and the two other sites in `examples/openclip/main.go` were left alone.

**Task 6 — issues filed on `amikos-tech/pure-onnx`:**

| Issue | Title |
|-------|-------|
| [#106](https://github.com/amikos-tech/pure-onnx/issues/106) | `[BUG] Bootstrap fingerprint memo bypasses content verification and per-file trust checks` |
| [#107](https://github.com/amikos-tech/pure-onnx/issues/107) | `[BUG] TOCTOU: Lstat-then-open without O_NOFOLLOW in bootstrap` |
| [#108](https://github.com/amikos-tech/pure-onnx/issues/108) | `[BUG] openclip bootstrap leaks presigned URLs into returned errors` |
| [#109](https://github.com/amikos-tech/pure-onnx/issues/109) | `[PERF] Bootstrap cache-hit path SHA-256s the whole install tree on first call per process` |
| [#110](https://github.com/amikos-tech/pure-onnx/issues/110) | `[CLN] CI runtime-version match step is formatting-brittle` |

Every cited path/line was re-verified by content before writing; several drifted from the plan's figures and were corrected (memo hit `1573-1580`, fingerprint `1659-1687`, per-file walk `1741-1795`, `sha256File` `1805-1807`, manifest read `1727`, lock open `2050` w/ post-open recheck `2054-2060`, openclip `377` and `652-657`, `ort/bootstrap.go:1124-1128`, `ci.yml:409-416` / `:183` / `:185`, `Makefile:13` / `:21`, `ort/bootstrap.go:33`). #106 and #109 are cross-linked in both directions via comments. No tokens, presigned URLs, or internal hostnames appear in any body; no Claude attribution. Scratch bodies were written to the session scratchpad, never to the repo — the working tree is clean.

## Deviations from Plan

**1. [Rule 1 - Bug] `cleanup_failure` verb changed too (Task 3A)**
- **Found during:** Task 3A
- **Issue:** The plan assumed `cleanupFailure` is always an `error`. `ort/environment.go:186` passes `rollbackPanic` (a raw `recover()` value) into that parameter, so `%q` mangles it identically to the `handlerPanic` case.
- **Fix:** Both verbs changed to `%v`; a one-line comment above the format string records why.
- **Commit:** `2cfe12a`

**2. Discretionary G703 comment exercised.** The plan left this optional; added at one site as permitted (`80dd73f`).

No other deviations. No architectural changes, no dependency changes, no CGO introduced anywhere in the diff.

## Verification Results

**`make precommit` — PASSED (full gate, honestly run):**
- `gofmt`/`goimports`, `go vet -unsafeptr=false ./ort/...`, `golangci-lint` (new-issues mode): clean
- `gosec`: 28 files, 9962 lines, **0 issues** — the now-blocking gate is genuinely green
- `go test -short ./...`: `ort`, `minilm`, `splade`, `openclip`, examples all `ok`
- `go mod tidy` check: clean; `govulncheck`: no vulnerabilities

Note: local `go vet` **without** `-unsafeptr=false` reports "possible misuse of unsafe.Pointer" in `ort/`. That is expected and pre-existing — the project's own `vet` target passes `-unsafeptr=false` for `ort/` precisely because the FFI boundary uses `uintptr` deliberately.

**Targeted tests:** `go test ./ort/ -run 'TestStatusToError' -count=1 -v` — all three subtest cases (`code_accessor`, `message_accessor`, `release`) PASS. Same under `-race`.

## Deferred Issues (out of scope — pre-existing, NOT introduced here)

**`go test ./ort/ -race -count=1` (full package race lane) FAILS on this machine.** This is *not* caused by any change in this task and is not covered by `make precommit` (which runs `go test -short ./...` without `-race`).

- **Symptom:** `WARNING: DATA RACE` at `ort/bootstrap.go:262` (`WithBootstrapCacheDir`) and `:274` (`WithBootstrapVersion`), followed by a hang/abort in `TestInitializeEnvironmentWithBootstrapLoadsSelectedPathAtomically`.
- **Cause:** those option closures mutate their own captured parameter (`dir = strings.TrimSpace(dir)`, `version = strings.TrimSpace(version)`), and `TestEnsureOnnxRuntimeSharedLibraryConcurrentLockSingleDownload` (`ort/bootstrap_test.go:497-499`) shares one `opts...` slice across 8 goroutines.
- **Proof it is pre-existing:** reproduced identically from a clean `git archive` export of base commit `52038e3` (before this task's first commit) in a scratch directory. Local toolchain is Go 1.26.5.
- **Not fixed** per the scope boundary; not filed as an issue either, since Task 6 was fenced to exactly five. Flagging here for the orchestrator/user to decide.

## Self-Check: PASSED

- All 8 commits present in `git log a1bbff4..HEAD`.
- All modified files exist and are committed; `git status --porcelain` is clean.
- All 5 issues confirmed present via `gh issue list` with correct `[BUG]`/`[PERF]`/`[CLN]` prefixes.
- `grep -rl 'Superseded by c7e58011' .planning/phases/02-core-api-errors-values/` returns 12.
- No `import "C"` or CGO types anywhere in the diff.
