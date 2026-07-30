---
phase: 2
slug: core-api-errors-values
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-23
last_audited: 2026-07-29
---

> Superseded by c7e58011: the shipped diagnostics default is a stderr TextHandler at
> LevelWarn, not a silent DiscardHandler.

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Go standard-library `testing` with the module's Go 1.25 baseline |
| **Config file** | none — commands and CI selection live in `Makefile` and `.github/workflows/ci.yml` |
| **Quick run command** | `go test -short ./ort -run 'Test(ORTError|StatusToError|ParseShape|ShapeElementCount|Value|AdvancedSessionRunWithValues|Diagnostic)'` |
| **Full suite command** | `go test -short ./...` |
| **Estimated runtime** | under 30 seconds for task-level focused selector-liveness/static assertions, ~2 minutes for the full phase gate; native integration runtime varies by environment |

---

## Sampling Rate

- **After every task commit:** Run `go test -short ./ort -run 'Test(ORTError|StatusToError|ParseShape|ShapeElementCount|Value|AdvancedSessionRunWithValues|Diagnostic)'`
- **After every plan wave:** Run `go test -short ./...`
- **After race-sensitive changes:** Run `go test -race ./ort -run '^(TestStatusToError|TestORTError|TestErrorSentinel|TestDiagnostic|TestAdvancedSessionRunWithValues|TestAdvancedSessionErrorContracts|TestAdvancedSessionDiagnosticPolicy|TestValuesToHandlesDeduplicatesRepeatedLockableValue|TestValuesToHandlesReleasesPriorLeasesOnError|TestValuesToHandlesRejectsNonComparableLockable|TestAdvancedSessionRunConcurrent|TestAdvancedSessionRunConcurrentAcrossSessionsSharingTensor|TestAdvancedSessionRunAndDestroyConcurrent|TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun|TestTensorDestroyWaitsForInFlightRun|TestTensorDestroyDoesNotBlockUnrelatedRun|TestTensorDestroyConcurrentCallsReleaseOnce|TestTensorStatusConversion|TestTensorDiagnosticPolicy|TestEnvironmentErrorChains|TestEnvironmentStatusConversion|TestConcurrentInitialization|TestConcurrentDestroy|TestDiagnosticRuntimeVersion|TestMemoryInfoStatusConversion|TestCreateMemoryInfoBlocksEnvironmentTeardown|TestDiagnosticMemoryInfo|TestDiagnosticCallSites|TestReturnedErrorsDoNotEmit)$'`
- **After Wave 3 / before `$gsd-verify-work`:** Prove the race selector resolves exactly 29 live top-level tests and the native selector resolves exactly 4, then run the full short suite, the targeted race selector above, `go test -run '^$' ./...`, exported-surface `go doc` checks (including `ParseShape` and `ShapeElementCount`), `go test -short ./ort -run '^TestBootstrapCreatedFilePermissions$'`, configured native non-race tests, `go vet -unsafeptr=false ./ort/...`, and `make precommit-lint-new`
- **Phase boundary:** Phase 2 verifies only changed-code/new-issues lint. Historical full-tree lint cleanup, lint `continue-on-error` removal, and the enforcing full-lint outcome remain Phase 5 / CLN-01.
- **Max feedback latency:** 30 seconds for task-level selector-liveness/static assertions; comprehensive suites run at wave/phase gates

---

## Canonical Threat Register

This is the single Phase 2 threat-ID authority. Every PLAN.md `<threat_model>` scopes a subset of these IDs but must retain the category, severity, and disposition defined here. The task and command columns are the canonical automated trace; native tests appear only in non-race commands per D-23.

| Threat ID | STRIDE Category | Severity | Canonical Scope | Disposition | Implementing / Verifying Tasks | Automated Evidence |
|-----------|-----------------|----------|-----------------|-------------|-------------------------------|--------------------|
| T-02-01 | Tampering / Elevation of Privilege | HIGH | Caller-controlled values, parsed/programmatic shapes, and counts crossing the FFI boundary | mitigate | 02-02-01, 02-04-01, 02-05-01 | `go test -short ./ort -run '^(TestValue|TestAdvancedSessionRunWithValues|TestParseShape|TestShapeElementCountExported|TestNewTensorValidationErrorsWithoutORT)$'` |
| T-02-02 | Denial of Service | HIGH | Run/resource lifetime, lock order, leases, pinning, and concurrent teardown | mitigate | 02-02-01, 02-04-01, 02-05-01, 02-06-01, 02-06-02, 02-08-01, 02-08-02 | `go test -race ./ort -run '^(TestAdvancedSessionRunWithValues|TestAdvancedSessionRunConcurrent|TestAdvancedSessionRunConcurrentAcrossSessionsSharingTensor|TestAdvancedSessionRunAndDestroyConcurrent|TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun|TestTensorDestroyWaitsForInFlightRun|TestTensorDestroyDoesNotBlockUnrelatedRun|TestTensorDestroyConcurrentCallsReleaseOnce|TestConcurrentInitialization|TestConcurrentDestroy|TestCreateMemoryInfoBlocksEnvironmentTeardown)$'` |
| T-02-03 | Tampering / Denial of Service | HIGH | Native status copy-before-release, exact one-release ownership, and protected accessor lifetime | mitigate | 02-01-01, 02-01-02, 02-04-02, 02-05-01, 02-06-01, 02-06-02, 02-08-01, 02-08-02 | `go test -race ./ort -run '^(TestStatusToError|TestORTError|TestAdvancedSessionErrorContracts|TestTensorStatusConversion|TestEnvironmentStatusConversion|TestMemoryInfoStatusConversion|TestCreateMemoryInfoBlocksEnvironmentTeardown)$'`, final call-site lifecycle audit, and `ONNXRUNTIME_LIB_PATH="$ONNXRUNTIME_LIB_PATH" go test ./ort -run '^TestNativeORTStatusRoundTrip$'` |
| T-02-04 | Repudiation | MEDIUM | Inspectable local categories and preserved OS/filesystem/network/cleanup causes | mitigate | 02-01-01, 02-04-02, 02-06-01, 02-06-02, 02-07-01 | `go test -short ./ort -run '^(TestErrorSentinel|TestEnvironmentErrorChains|TestBootstrapErrorChains)$'` |
| T-02-06 | Denial of Service | MEDIUM | A consumer-installed synchronous `slog.Handler` panics during a non-finalizer diagnostic callback | accept | 02-03-01, 02-06-01, 02-07-03, 02-08-01, 02-08-02 | The handler is an explicit trusted callback boundary selected by the consumer; `go test -race ./ort -run '^(TestDiagnostic|TestDiagnosticRuntimeVersion|TestDiagnosticCallSites)$'` documents normal synchronous panic propagation, while library-owned finalizer containment is tracked separately by T-02-11. |
| T-02-07 | Information Disclosure | HIGH | Diagnostic attribute allowlisting and URL/credential redaction | mitigate | 02-03-01, 02-06-01, 02-07-03, 02-08-01, 02-08-02 | `go test -race ./ort -run '^(TestDiagnostic|TestDiagnosticRuntimeVersion|TestDiagnosticCallSites)$'` |
| T-02-08 | Repudiation / Denial of Service | MEDIUM | Prohibition on diagnostic duplication for returned errors | mitigate | 02-03-01, 02-04-02, 02-05-01, 02-06-01, 02-06-02, 02-07-03, 02-08-01, 02-08-02 | `go test -race ./ort -run '^(TestDiagnostic|TestAdvancedSessionDiagnosticPolicy|TestTensorDiagnosticPolicy|TestDiagnosticRuntimeVersion|TestDiagnosticMemoryInfo|TestReturnedErrorsDoNotEmit)$'` |
| T-02-09 | Denial of Service | MEDIUM | Go 1.25/platform compatibility, live CI selectors, exported API visibility, and unchanged consumer compilation | mitigate | 02-01-01, 02-01-02, 02-02-01, 02-08-02 | `GOOS=windows GOARCH=amd64 go test -c -o /dev/null ./ort`, 29/4 selector-liveness assertions, exported-surface `go doc`, and `go test -run '^$' ./...` |
| T-02-10 | Tampering / Elevation of Privilege | HIGH | Existing bootstrap HTTPS, checksum, archive containment, size, Unix-safe directory/installed-library/lock-file permissions, and lock integrity | mitigate | 02-07-01, 02-07-02, 02-08-02 | `go test -short ./ort -run '^(TestBootstrapCreatedFilePermissions|TestEnsureOnnxRuntimeSharedLibraryChecksumMismatch|TestRejectHTTPSDowngradeRedirect|TestDownloadRuntimeArchiveRejectsOversize|TestSecureArchiveJoin)$' && go test -short ./...` (`TestBootstrapCreatedFilePermissions` compiles and skips its POSIX-mode assertions on Windows) |
| T-02-11 | Denial of Service | HIGH | Concurrent diagnostic reconfiguration and panics crossing the library-owned best-effort finalizer boundary | mitigate | 02-03-01, 02-04-02, 02-05-01, 02-06-01, 02-06-02, 02-07-03, 02-08-01, 02-08-02 | `go test -race ./ort -run '^(TestDiagnostic|TestAdvancedSessionDiagnosticPolicy|TestTensorDiagnosticPolicy|TestDiagnosticRuntimeVersion|TestDiagnosticMemoryInfo|TestDiagnosticCallSites)$'` proves atomic reconfiguration and finalizer-only panic containment. |
| T-02-SC | Tampering | LOW | Package/module and CI action supply-chain inputs | mitigate | 02-01-01 through 02-08-02 | `git diff --exit-code -- go.mod go.sum && test -z "$(git diff HEAD --unified=0 -- .github/workflows/ci.yml | sed -n '/^[+-][[:space:]]*uses:/p')"` |

No canonical HIGH threat is accepted or unresolved.
T-02-06 is explicitly accepted at MEDIUM severity because `SetDiagnosticHandler` installs consumer-owned synchronous code as a trusted callback; the package still contains that code at the finalizer boundary under T-02-11.

---

## Per-Task Verification Map

Final task IDs below map every missing test/CI need to the executor plan that creates it. Threat refs resolve only through the canonical register above, and commands remain the required sampling contract.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 02-01 | 1 | API-02 | T-02-03 / T-02-04 | Nonzero native status is copied before release and released exactly once; zero status returns nil | unit/race | `go test -race ./ort -run '^(TestStatusToError|TestORTError|TestErrorSentinel)$'` | ✅ yes | ✅ green |
| 02-01-02 | 02-01 | 1 | API-02 | T-02-03 | Real ORT status preserves code and message through the native ABI without race/checkptr mixing | native integration | `ONNXRUNTIME_LIB_PATH="$ONNXRUNTIME_LIB_PATH" go test ./ort -run '^TestNativeORTStatusRoundTrip$'` | ✅ yes | ✅ green in configured CI; local skip verified |
| 02-04-02 / 02-05-01 / 02-06-01 / 02-06-02 / 02-07-01 | 02-04–02-07 | 2 | API-02 | T-02-04 | Validation and lifecycle categories remain inspectable and lower-level causes remain reachable across shape, session, tensor, environment, memory, and bootstrap flows | unit | `go test -short ./ort -run 'Test(ParseShape|ShapeElementCount|ErrorSentinel|Bootstrap.*Error|.*Destroyed|.*NotInitialized)'` | ✅ yes | ✅ green |
| 02-05-01 | 02-05 | 2 | API-02 | T-02-01 / T-02-04 | Exported shape parsing/counting rejects invalid input with ErrInvalidArgument while preserving the strconv parsing cause | unit/race | `go test -race ./ort -run '^(TestParseShape|TestShapeElementCountExported|TestShapeElementCount)$'` plus separate `go doc ./ort.ParseShape` and `go doc ./ort.ShapeElementCount` | ✅ yes | ✅ green |
| 02-06-02 | 02-06 | 2 | API-02 | T-02-02 / T-02-03 | CreateMemoryInfo keeps teardown excluded through the native callback and status conversion/release | race/lifecycle | `go test -race ./ort -run '^(TestMemoryInfoStatusConversion|TestCreateMemoryInfoBlocksEnvironmentTeardown)$'` | ✅ yes | ✅ green |
| 02-03-01 / 02-04-02 / 02-05-01 / 02-06-01 / 02-06-02 / 02-07-03 / 02-08-01 | 02-03–02-08 | 1–3 | API-02 | T-02-06 / T-02-07 / T-02-08 / T-02-11 | Diagnostics default to silent, reconfigure safely, omit sensitive data, cover every approved call site, accept propagation only at the trusted general-handler boundary, contain finalizer panics, and never duplicate returned errors | unit/race/audit | `go test -race ./ort -run '^(TestDiagnostic|TestAdvancedSessionDiagnosticPolicy|TestTensorDiagnosticPolicy|TestDiagnosticRuntimeVersion|TestDiagnosticMemoryInfo|TestDiagnosticCallSites|TestReturnedErrorsDoNotEmit)$'` | ✅ yes | ✅ green |
| 02-02-01 | 02-02 | 1 | API-03 | T-02-01 | Only package-created values cross the FFI boundary; tensor extraction is exact and never coerces | compile/unit | `go test -short ./ort -run '^TestValue$'` | ✅ yes | ✅ green |
| 02-04-01 | 02-04 | 2 | API-03 | T-02-01 / T-02-02 | Per-call values validate counts and preserve ownership, handle leases, serialization, KeepAlive, and lock order | unit/race | `go test -race ./ort -run '^(TestAdvancedSessionRunWithValues|TestAdvancedSessionRunConcurrent|TestAdvancedSessionRunConcurrentAcrossSessionsSharingTensor|TestAdvancedSessionRunAndDestroyConcurrent|TestTensorDestroyWaitsForInFlightRun|TestValuesToHandlesDeduplicatesRepeatedLockableValue|TestValuesToHandlesReleasesPriorLeasesOnError)$'` | ✅ yes | ✅ green |
| 02-04-02 | 02-04 | 2 | API-03 | T-02-01 / T-02-02 | Caller-preallocated per-call tensors produce the expected output against a real model | native integration | `ONNXRUNTIME_LIB_PATH="$ONNXRUNTIME_LIB_PATH" go test ./ort -run '^TestAdvancedSessionRunWithValuesRealModel$'` | ✅ yes | ✅ green in configured CI; local skip verified |
| 02-07-01 / 02-07-02 / 02-08-02 | 02-07–02-08 | 2–3 | API-02 | T-02-10 | Bootstrap HTTPS, checksum, archive containment, size, lock, and Unix-safe directory/installed-library/lock-file permissions remain intact through the error/diagnostic migration and final gate | security regression/full gate | `go test -short ./ort -run '^(TestBootstrapCreatedFilePermissions|TestEnsureOnnxRuntimeSharedLibraryChecksumMismatch|TestRejectHTTPSDowngradeRedirect|TestDownloadRuntimeArchiveRejectsOversize|TestSecureArchiveJoin)$' && go test -short ./...` | ✅ yes | ✅ green |
| 02-01-01 / 02-01-02 / 02-02-01 / 02-08-02 | 02-01, 02-02, 02-08 | 1–3 | API-02, API-03 | T-02-09 | Go 1.25-compatible APIs and supported Windows builds compile while existing consumers remain unchanged | platform/compile | `GOOS=windows GOARCH=amd64 go test -c -o /dev/null ./ort && go test -run '^$' ./...` | ✅ yes | ✅ green |
| 02-01-01 / 02-01-02 / 02-02-01 / 02-03-01 / 02-04-01 / 02-04-02 / 02-05-01 / 02-06-01 / 02-06-02 / 02-07-01 / 02-07-02 / 02-07-03 / 02-08-01 / 02-08-02 | 02-01–02-08 | 1–3 | API-02, API-03 | T-02-SC | No package/module requirement or CI action `uses:` reference changes enter Phase 2 | source integrity | `git diff --exit-code -- go.mod go.sum && test -z "$(git diff HEAD --unified=0 -- .github/workflows/ci.yml | sed -n '/^[+-][[:space:]]*uses:/p')"` | ✅ yes | ✅ green |
| 02-08-02 | 02-08 | 3 | API-02, API-03 | T-02-02 / T-02-03 / T-02-06 / T-02-07 / T-02-08 / T-02-09 / T-02-10 / T-02-11 / T-02-SC | CI keeps fake ownership/concurrency proofs under race and real ABI/model proofs in the configured non-race lane; each selector proves 29/4 live top-level tests before execution, and exported APIs plus unchanged action references are checked while comprehensive suites run at wave/phase scope | CI/liveness + wave/phase gate | `go test -list '<race-selector>' ./ort` resolves 29 tests, `go test -list '<native-selector>' ./ort` resolves 4 tests, and `test -z "$(git diff HEAD --unified=0 -- .github/workflows/ci.yml | sed -n '/^[+-][[:space:]]*uses:/p')"` | ✅ yes | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `ort/errors_test.go` (02-01-01) — fake status store; zero/nonzero conversion; accessor order; message copy; exact release; concurrent conversion; `errors.As`; sentinel wrapping for API-02
- [x] `ort/errors_native_test.go` (02-01-02) — `//go:build !windows` Unix-loader constraint plus real `CreateStatus` ABI round trip gated by `ONNXRUNTIME_LIB_PATH` for API-02
- [x] `ort/diagnostics_test.go` (02-03-01 plus call-site extensions in 02-04-02/02-05-01/02-06-01/02-06-02/02-07-03) — silent default; standard attributes/levels; nil reset; concurrent emit/reconfigure; explicit non-finalizer panic propagation/finalizer containment; returned-error zero-emission proof for API-02
- [x] `ort/value_test.go` (02-02-01) — kind check and exact generic extraction matrix for API-03
- [x] `ort/shape_test.go` additions (02-05-01) — exported ParseShape/ShapeElementCount `errors.Is` coverage plus preserved `*strconv.NumError` `errors.As` coverage for API-02
- [x] `ort/session_test.go` additions (02-04-01/02-04-02) — count validation; supplied handle arrays; bound-path compatibility; borrow/Destroy synchronization; per-call concurrency for API-03
- [x] Flow-test additions (02-04-02/02-05-01/02-06-01/02-06-02/02-07-01/02-07-03) in `shape_test.go`, `environment_test.go`, `memory_test.go`, `tensor_test.go`, `session_test.go`, and `bootstrap_test.go` — `errors.Is`/`errors.As`, preserved causes, lifecycle-lock protection, and approved diagnostic call sites for API-02
- [x] `ort/bootstrap_test.go` (02-07-02) — exact `TestBootstrapCreatedFilePermissions` Unix regression and isolated hardening commit for bootstrap-created directories, installed TGZ/ZIP library files, and lock files, with a Windows-safe POSIX-mode skip
- [x] `.github/workflows/ci.yml` (02-08-02) — run fake-status and diagnostic concurrency tests in the targeted race job; run native-status and `RunWithValues` real-model tests in the existing integration job
- [x] No framework installation or dependency change is required (enforced by every plan and the 02-08 compatibility gate)

---

## Manual-Only Verifications

*None — all phase behaviors have automated verification. Native cases may skip locally when `ONNXRUNTIME_LIB_PATH` is unavailable, but the existing integration CI environment must run them.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verification or create their missing test scaffold before/with implementation
- [x] Sampling continuity: no 3 consecutive tasks without automated verification
- [x] Wave 0 identifies all currently missing test files and cases
- [x] No watch-mode flags
- [x] Feedback latency target is under 30 seconds
- [x] Provisional task IDs replaced with final PLAN.md task IDs
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-07-23 (gsd-plan-checker verified Phase 2 plans and Nyquist coverage)

## Validation Audit 2026-07-29

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Requirements covered | 2/2 |

| Requirement | Status | Live evidence |
|-------------|--------|---------------|
| API-02 | COVERED | Fresh full short suite and exact 29-test race selector passed; error, lifecycle, diagnostic, bootstrap, and review-added session-options/environment paths are exercised. |
| API-03 | COVERED | Value exact-type tests, shared `RunWithValues` ownership/concurrency tests, all-package compilation, and the configured native selector are present and green. |

Audit notes:

- The exact race and native selectors still resolve 29 and 4 top-level tests respectively.
- The four native tests skip locally when `ONNXRUNTIME_LIB_PATH` is unset; the configured CI lane supplies the runtime and executes them without `-race`.
- Review remediation expanded the live source beyond the original plan snapshots to eight `statusToError` call sites and four finalizer resource paths. The full short suite, focused review-remediation tests, generic finalizer containment tests, and source/security audits cover those additions.
- `go test -count=1 -short ./...`, the exact race selector with `-count=1`, all-package compilation, exported API checks, `go vet -unsafeptr=false ./ort/...`, bootstrap permission coverage, module integrity, and CI action-reference integrity passed during this audit.
- `make precommit-lint-new` currently reports two test-only `govet/unsafeptr` findings in review-added native callback fixtures (`ort/bootstrap_test.go` and `ort/environment_test.go`). Their behavioral tests pass; this is a quality-gate warning rather than a missing Nyquist verification path.
