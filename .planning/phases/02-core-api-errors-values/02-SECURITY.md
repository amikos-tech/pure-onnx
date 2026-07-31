---
phase: 2
slug: core-api-errors-values
status: secured
threats_open: 0
asvs_level: 1
created: 2026-07-29
---

# Phase 2 Security Verification

## Result

**SECURED**

All eleven declared threats are closed. T-02-08 was re-verified after remediation: every environment initialization error now returns before runtime-version diagnostic emission.

## Scope

This audit verifies only the threats declared by the Phase 2 plans and the attack-surface flags recorded by the Phase 2 summaries. Documentation and prior verification reports were used to identify the required controls, but only implementation, tests, workflow configuration, and live verification results were accepted as mitigation evidence.

Implementation files were treated as read-only.

## Trust Boundaries

| Boundary | Data or capability crossing it | Relevant threats |
|---|---|---|
| Go caller → value/session API → native runtime | Tensor values, shapes, element counts, native handles, and output slots | T-02-01, T-02-02 |
| Native runtime → Go error API | Status handles, error codes, and native message memory | T-02-03, T-02-04 |
| Library → caller-provided diagnostic handler | Structured diagnostic records and finalizer warnings | T-02-06, T-02-07, T-02-08, T-02-11 |
| Network/archive → local runtime cache | Release metadata, archive bytes, checksums, paths, modes, locks, and manifests | T-02-07, T-02-10 |
| Repository/CI → downstream consumers | Go version, exported API, dependencies, and workflow action references | T-02-09, T-02-SC |

## Threat Verification

| Threat ID | Category | Severity | Disposition | Status | Implementation evidence |
|---|---|---:|---|---|---|
| T-02-01 | Tampering / Elevation of Privilege | HIGH | mitigate | CLOSED | `Value` is sealed by the private `ortValue` method (`ort/types.go:68-74`, `ort/tensor.go:20`). Input/output counts and value leases are validated before the native run call (`ort/session.go:281-328`). Shape parsing and overflow-safe element counting are implemented in `ort/shape_parse.go:12-45` and `ort/tensor.go:297-345`. Focused Value, shape, tensor-validation, and `RunWithValues` tests passed. |
| T-02-02 | Denial of Service | HIGH | mitigate | CLOSED | Session/native-call locks, deterministic value leases, pinning, and `runtime.KeepAlive` protect native handles (`ort/session.go:252-350,498-537`, `ort/tensor.go:84-171,221-253`, `ort/memory.go:26-60`). The exact 29-test race selector covering concurrent runs and teardown passed with `-race`. |
| T-02-03 | Tampering / Denial of Service | HIGH | mitigate | CLOSED | `statusToError` copies the message and defers exactly one release before exposing the Go error (`ort/errors.go:51-75`). All current owning status paths use this helper: `ort/memory.go:59`, `ort/tensor.go:114,150`, `ort/session.go:42,178,206,350`, and `ort/environment.go:130`. Helper, panic-release, concurrency, resource-path, and real-ABI test code exists in `ort/errors_test.go:91-263`, `ort/errors_native_test.go:15-100`, `ort/session_test.go:1264-1418`, `ort/tensor_test.go:368-529`, `ort/environment_test.go:283-357`, and `ort/memory_test.go:349-455`. |
| T-02-04 | Repudiation | MEDIUM | mitigate | CLOSED | Inspectable sentinels and `ORTError` are implemented in `ort/errors.go:9-34`. Environment and bootstrap boundaries preserve lower causes with `%w` or `errors.Join` rather than flattening them (`ort/environment.go:207-227,320-324`; bootstrap error-chain coverage in `ort/bootstrap_test.go:158-420`). Sentinel and chain tests passed. |
| T-02-06 | Denial of Service | MEDIUM | accept | CLOSED — accepted risk | The general synchronous handler deliberately propagates handler panics, while the finalizer-only path contains them (`ort/diagnostics.go:19-50`). This boundary is exercised by `ort/diagnostics_test.go:166-190`, `ort/environment_test.go:373-476`, and `ort/bootstrap_test.go:2192-2264`. The accepted risk is recorded below as AR-02-06. |
| T-02-07 | Information Disclosure | HIGH | mitigate | CLOSED | Diagnostic emission is private (`ort/diagnostics.go:35-45`). Environment diagnostics use allowlisted version attributes (`ort/environment.go:84-116`); bootstrap call sites use fixed attributes, redact download URLs before emission (`ort/bootstrap.go:597-636,1839-1844`), and use `URL.Redacted()` across redirects (`ort/bootstrap.go:493-516`). Credential-redaction tests exist at `ort/bootstrap_test.go:2232-2264` and passed in the focused suite. |
| T-02-08 | Repudiation / Denial of Service | MEDIUM | mitigate | CLOSED | `initializeEnvironmentAt` delegates the complete initialization tuple to `completeEnvironmentInitialization` (`ort/environment.go:151-156`). The helper returns immediately on any non-nil error before the sole version-warning call (`ort/environment.go:156-168`), including the old-runtime plus `CreateEnv` failure tuple returned at `ort/environment.go:271-283`. The regression at `ort/environment_test.go:220-232` preserves the exact error and asserts zero diagnostic records; focused and race-enabled policy suites passed. |
| T-02-09 | Denial of Service | MEDIUM | mitigate | CLOSED | The module and CI use Go 1.25 (`go.mod:3`, `.github/workflows/ci.yml:11`). Exported Value, shape, diagnostic, and session APIs resolve through `go doc`; `go test -run '^$' ./...` and a Windows/amd64 compile gate passed. The live race and native selectors are present in `.github/workflows/ci.yml:129-137,231-239`; neither disables checkptr. |
| T-02-10 | Tampering / Elevation of Privilege | HIGH | mitigate | CLOSED | Bootstrap enforces HTTPS and redirect policy (`ort/bootstrap.go:417-516`), checksum verification (`ort/bootstrap.go:597-636`), archive size limits (`ort/bootstrap.go:1036-1058`), safe modes (`ort/bootstrap.go:43-53,1149-1161,1292-1301`), cache-manifest integrity (`ort/bootstrap.go:1349-1433`), lock integrity (`ort/bootstrap.go:1688-1733`), and archive containment (`ort/bootstrap.go:1777-1812`). Focused checksum, redirect, oversize, permissions, lock, and containment tests passed. |
| T-02-11 | Denial of Service | HIGH | mitigate | CLOSED | Handler publication uses an atomic pointer and structured emission (`ort/diagnostics.go:28-45`); the finalizer-only emitter recovers handler panics (`ort/diagnostics.go:47-50`). Session options, sessions, tensors, and memory objects all use this path for finalizers (`ort/session.go:56,231`, `ort/tensor.go:178`, `ort/memory.go:86`). Concurrency, panic-containment, and general panic-propagation tests in `ort/diagnostics_test.go:14-190` passed under `-race`. |
| T-02-SC | Tampering | LOW | mitigate | CLOSED | The Phase 2 commit range changes neither `go.mod` nor `go.sum`, and adds or changes no workflow `uses:` references. The current worktree also has no dependency-file or action-reference diff. |

## Resolved Findings

### O-02-08 — Environment failure can follow a diagnostic

- **Threat:** T-02-08
- **Original classification:** BLOCKER
- **Resolution:** CLOSED
- **Declared requirement:** Every environment failure returned to the caller emits zero diagnostic records.
- **Files searched:** `ort/environment.go`, `ort/environment_test.go`, `ort/diagnostics.go`, `ort/diagnostics_test.go`
- **Original evidence:** `initializeEnvironmentAtLocked` could return a non-empty runtime version together with a creation error, while the outer entry point emitted the version warning before returning that error.
- **Remediation evidence:** `initializeEnvironmentAt` now delegates its complete result tuple to `completeEnvironmentInitialization` (`ort/environment.go:151-155`). That helper checks `err != nil` and returns it at `ort/environment.go:156-158`; the only call to `emitRuntimeVersionWarning` follows at `ort/environment.go:168`. The native creation failure continues to return the captured runtime version with `newlyInitialized=false` and the exact error (`ort/environment.go:271-283`), so it necessarily takes the no-emission branch.
- **Regression evidence:** `TestEnvironmentErrorChains/initialization_failure_with_old_runtime_emits_nothing` supplies version `1.21.4` with a `CreateEnv` error, verifies the same error remains reachable, and requires an exact diagnostic count of zero (`ort/environment_test.go:220-232`).
- **Fresh verification:** The exact regression subtest passed, and the focused race-enabled error/diagnostic policy suite passed.
- **Resolved:** 2026-07-29

## Accepted Risks

### AR-02-06 — Trusted synchronous diagnostic handler panic

- **Threat:** T-02-06
- **Disposition:** accept
- **Accepted boundary:** A deliberately installed, trusted `slog.Handler` may panic during an ordinary synchronous diagnostic call. That panic propagates to the caller.
- **Rationale:** The handler is application-controlled and synchronous; masking its panic would hide application failure. Library finalizers use a separate containment path because panics there would be process-fatal and occur outside a caller-controlled stack.
- **Constraints:** The handler API remains opt-in, diagnostics do not originate on hidden background goroutines, and every finalizer emission must use the recovery-protected finalizer emitter.
- **Accepted by:** Phase 2 plan-time threat register
- **Acceptance date:** 2026-07-23

No other accepted risks are authorized by the Phase 2 threat register.

## Threat Flags

No unregistered threat flags were found. All eight Phase 2 summaries were examined; the only explicit `## Threat Flags` section, in `02-04-SUMMARY.md`, records no flags.

## Verification Performed

- The exact 29-test Phase 2 race selector passed with `go test -count=1 -race`.
- `go test -count=1 -short ./...` passed.
- Focused Value, `RunWithValues`, shape, error-chain, bootstrap-integrity, and URL-redaction tests passed.
- The exact T-02-08 regression subtest passed with `go test -count=1 -v ./ort -run '^TestEnvironmentErrorChains$/^initialization_failure_with_old_runtime_emits_nothing$'`.
- The focused T-02-08 error/diagnostic policy suite, including `TestEnvironmentErrorChains`, passed with `go test -count=1 -race`.
- `go test -run '^$' ./...` passed as a consumer compile gate.
- `GOOS=windows GOARCH=amd64 go test -c -o /dev/null ./ort` passed.
- The native status test is implemented and selected by CI, but was not executed locally because no native runtime was configured.
- `ort/finalizer_log.go` is absent as intended; Phase 2 removed the former finalizer logging implementation and routed finalizers through `ort/diagnostics.go`.

## Audit Trail

| Date | ASVS level | Threats closed | Threats open | Result |
|---|---:|---:|---:|---|
| 2026-07-29 | 1 | 10/11 | 1/11 | OPEN_THREATS |
| 2026-07-29 (re-audit) | 1 | 11/11 | 0/11 | SECURED |

## Sign-off

- [x] Every declared threat has a disposition and verification result.
- [x] The accepted risk is documented with its scope and constraints.
- [x] Summary threat flags were incorporated.
- [x] All declared mitigations are present at every relevant entry point.
- [x] `threats_open` is zero.

**Approval:** Secured on 2026-07-29 after remediation and re-verification of T-02-08.
