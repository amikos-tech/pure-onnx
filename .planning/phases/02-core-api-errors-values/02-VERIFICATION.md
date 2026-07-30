---
phase: 02-core-api-errors-values
verified: 2026-07-24T15:07:10Z
status: passed
score: 44/44 must-haves verified
overrides_applied: 0
---

> Superseded by c7e58011: the shipped diagnostics default is a stderr TextHandler at
> LevelWarn, not a silent DiscardHandler.

# Phase 2: Core API — Errors & Values Verification Report

**Phase Goal:** The `ort` core returns comprehensive wrapped errors and exposes a `Value` interface for polymorphic tensor handling.
**Verified:** 2026-07-24T15:07:10Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

The score covers four roadmap success criteria and all 40 PLAN frontmatter truths. Overlapping checks are retained so every plan must-have remains auditable.

### Roadmap Success Criteria

| # | Truth | Status | Evidence |
|---|---|---|---|
| R1 | Environment, tensor, session, and bootstrap failures carry actionable wrapped context and support `errors.Is`/`errors.As`. | ✓ VERIFIED | `ORTError` and the public sentinels are defined in `ort/errors.go:8-35`; all seven native status sites call the shared converter; focused environment, tensor, session, memory, shape, and bootstrap error-contract tests passed under the exact race/short selectors. |
| R2 | A `Value` interface supports heterogeneous tensor types for session inputs and outputs. | ✓ VERIFIED | `Value`, `IsTensor`, and exact `AsTensor[T]` are implemented in `ort/types.go:60-84`; `*Tensor[T]` supplies the package-private marker at `ort/tensor.go:19`; heterogeneous and exact-type tests pass in `ort/value_test.go:15-145`. |
| R3 | `AdvancedSession.Run` and polymorphic per-call values work without breaking typed `Tensor[T]` use. | ✓ VERIFIED | Both methods enter `AdvancedSession.run` at `ort/session.go:129-239`. A real all-MiniLM model test executed `RunWithValues`, produced finite nonzero caller-owned output, left bound output unchanged, and allowed every value to be destroyed afterward. Existing bound `Run` real-model tests also executed successfully. |
| R4 | Existing `ort` unit and integration tests pass against the new surfaces. | ✓ VERIFIED | Fresh `go test -count=1 -short ./...`, the exact 29-test race lane, Go 1.25 compile-only checks, and the four-test real native/model lane passed. The full real-runtime `ort` package passed three consecutive runs after one non-reproduced lock-test timeout; the affected subtest then passed 20/20 isolated runs. |

### Every PLAN Must-Have Truth

Rows follow each PLAN frontmatter in order.

| Plan / Truth | Observable truth | Status | Evidence |
|---|---|---|---|
| 02-01 T1 | Native failures are inspectable as Go-owned `*ORTError` values with operation, code, and message intact after release. | ✓ VERIFIED | `ort/errors.go:8-17,43-68`; fake copy-before-release test at `ort/errors_test.go:116-150`; real ABI round trip at `ort/errors_native_test.go:15-100` executed and passed. |
| 02-01 T2 | Zero status is a no-op and every nonzero status is released exactly once, including concurrent conversions. | ✓ VERIFIED | `ort/errors.go:43-53`; zero, panic, exact-release, and 256-worker tests at `ort/errors_test.go:91-249` passed under `-race`. |
| 02-01 T3 | Local invalid, uninitialized, destroyed, missing-library, and unsupported-platform conditions are distinct `errors.Is` categories and do not conflate native codes. | ✓ VERIFIED | Sentinels at `ort/errors.go:20-35` and `ort/bootstrap.go:56-60`; separation test at `ort/errors_test.go:266-295` passed. |
| 02-01 T4 | Every production status conversion is protected by the ORT lifecycle lock. | ✓ VERIFIED | Converter precondition is documented at `ort/errors.go:56-60`; all seven call sites are inside `ortCallMu` read/write scopes. The most subtle path is proven with `TryLock` during native access and conversion in `ort/memory_test.go:357-442`. |
| 02-01 T5 | Useful operation context remains while `errors.Is`/`errors.As`, not complete text, is the compatibility contract. | ✓ VERIFIED | Resource callers add operation context with `%w`; tests assert categories/fields and only selected text fragments, for example `ort/session_test.go:1004-1157` and `ort/environment_test.go:96-288`. |
| 02-02 T1 | Heterogeneous package-created tensors coexist in `[]Value`, while external code cannot forge a native-resource implementation. | ✓ VERIFIED | Sealed method set at `ort/types.go:60-70`, tensor marker at `ort/tensor.go:19`, and heterogeneous test at `ort/value_test.go:137-145`. |
| 02-02 T2 | Sealing is deliberate, while existing package-created typed tensor consumers remain source-compatible. | ✓ VERIFIED | The unexported marker prevents external implementation. Go 1.25 compile-only `./...` passed without consumer edits, including examples and all embedders. |
| 02-02 T3 | `IsTensor` checks kind and `AsTensor[T]` returns only an exact, non-nil `*Tensor[T]` without copy or allocation. | ✓ VERIFIED | Direct assertion at `ort/types.go:72-84`; kind, mismatch, nil, typed-nil, zero-allocation, and no-copy tests at `ort/value_test.go:15-135` passed. |
| 02-02 T4 | The exported `Value` surface stays minimal, with no raw handle or universal wrapper. | ✓ VERIFIED | `go doc` resolves only `Destroy`, `Type`, and the private marker on `Value`; source audit found no exported raw handle, metadata interface, non-tensor implementation, or conversion wrapper. |
| 02-03 T1 | Diagnostics are silent until a consumer installs a standard `slog.Handler`; nil restores silence. | ✓ VERIFIED | `ort/diagnostics.go:13-30`; silence/reset test at `ort/diagnostics_test.go:14-38` passed. |
| 02-03 T2 | Diagnostic configuration/emission is race-safe and stays outside the ORT lock graph. | ✓ VERIFIED | Immutable logger state uses `atomic.Pointer` at `ort/diagnostics.go:9-18,29`; concurrent swap/emission test at `ort/diagnostics_test.go:92-145` passed under `-race`. |
| 02-03 T3 | Emission uses standard `slog.Logger.LogAttrs` levels/attributes without a project-owned logging API. | ✓ VERIFIED | `ort/diagnostics.go:32-42`; JSON level/attribute and pre-bound attribute tests at `ort/diagnostics_test.go:40-90` passed. |
| 02-03 T4 | Finalizer diagnostics contain handler panics, while returned errors have no automatic log path. | ✓ VERIFIED | Recovery is limited to `emitFinalizerDiagnostic` at `ort/diagnostics.go:44-58`; tests at `ort/diagnostics_test.go:147-202` passed. |
| 02-03 T5 | Non-finalizer consumer-handler panics propagate synchronously; only best-effort finalizers recover. | ✓ VERIFIED | Paired propagation/containment tests at `ort/diagnostics_test.go:166-189` and runtime/bootstrap call-site tests passed. |
| 02-04 T1 | Existing `AdvancedSession.Run() error` and bound preallocated tensor use remain source-compatible. | ✓ VERIFIED | Signature remains at `ort/session.go:129-132`; Go 1.25 all-package compilation and real bound-value model execution passed. |
| 02-04 T2 | `RunWithValues` uses fixed names, rejects bad counts before FFI, fills caller outputs, and retains caller ownership. | ✓ VERIFIED | Implementation at `ort/session.go:134-239`; fake behavior matrix at `ort/session_test.go:575-1002`; real-model test at `ort/session_test.go:2014-2116` executed and passed. |
| 02-04 T3 | `Run` and `RunWithValues` share serialization, deduplicated leases, `KeepAlive`, and lock ordering. | ✓ VERIFIED | Both methods call `run`; lock and lifetime core is `ort/session.go:141-239`; lease core is `ort/session.go:329-387`; targeted concurrency tests passed under `-race`. |
| 02-04 T4 | Construction still requires bound values, but callers may subsequently use only `RunWithValues`; no alternate constructor/output allocation path exists. | ✓ VERIFIED | Contract comment at `ort/session.go:134-136`; normally constructed supplied-only test at `ort/session_test.go:637-693`; source audit found no alternate constructor. |
| 02-04 T5 | Session local failures use sentinels, native create/run failures expose `*ORTError`, and returned failures emit nothing. | ✓ VERIFIED | `ort/session.go:28-45,65-67,80-107,165-179,232-237`; contract and diagnostic tests at `ort/session_test.go:1004-1248` passed. |
| 02-05 T1 | `ParseShape` and `ShapeElementCount` reject invalid input with `ErrInvalidArgument`, preserving `*strconv.NumError`. | ✓ VERIFIED | `ort/shape_parse.go:12-47`, `ort/tensor.go:278-330`; public tests at `ort/shape_test.go:103-248` passed. |
| 02-05 T2 | Tensor validation, uninitialized, destroyed, and unavailable-release failures use public sentinels with useful context. | ✓ VERIFIED | `ort/tensor.go:31-77,83-100,191-253,278-385`; focused assertions at `ort/tensor_test.go:189-315` passed. |
| 02-05 T3 | Both tensor-creation status paths use `*ORTError` and exact one-release ownership without losing lifetime barriers. | ✓ VERIFIED | `statusToError` calls at `ort/tensor.go:107-141`; status and cleanup tests at `ort/tensor_test.go:316-408` passed under `-race`. |
| 02-05 T4 | Tensor ownership remains explicit and compatible with `Value`; destroy/lease behavior and exact element types do not regress. | ✓ VERIFIED | `runtime.Pinner`, `KeepAlive`, idempotent destroy, and read leases remain in `ort/tensor.go`; concurrent destroy/lease tests and the real model lane passed. |
| 02-05 T5 | Only finalizer-only tensor cleanup failure emits a structured diagnostic. | ✓ VERIFIED | `ort/tensor.go:157-161`; zero-returned-event and one-finalizer-event tests at `ort/tensor_test.go:410-478` passed. |
| 02-06 T1 | Environment setup preserves OS, loader, symbol, and independent cleanup causes; local/native categories remain inspectable. | ✓ VERIFIED | `%w`/`errors.Join` flow at `ort/environment.go:126-159`; identity-preserving tests at `ort/environment_test.go:96-288` passed. |
| 02-06 T2 | Old-runtime mismatch is an opt-in structured Warn; returned environment failures emit nothing. | ✓ VERIFIED | `ort/environment.go:76-94`; structured, silent, and panic-policy tests at `ort/environment_test.go:290-352` passed. |
| 02-06 T3 | MemoryInfo validation/lifecycle failures use sentinels and native creation uses `ORTError`. | ✓ VERIFIED | `ort/memory.go:10-47,77-111`; tests at `ort/memory_test.go:256-355,444-493` passed. |
| 02-06 T4 | `CreateMemoryInfo` holds `ortCallMu.RLock` across native execution, `KeepAlive`, status access, and release. | ✓ VERIFIED | Lock scope at `ort/memory.go:15-46`; deterministic teardown-exclusion test at `ort/memory_test.go:357-442` passed under `-race`. |
| 02-06 T5 | Environment refcount/locking and MemoryInfo idempotent ownership remain intact; only memory finalizers emit cleanup diagnostics. | ✓ VERIFIED | Existing refcount/concurrency tests are in the 29-test race lane; MemoryInfo destroy and finalizer policy tests passed. |
| 02-07 T1 | Bootstrap validation, unsupported-platform, and missing-library categories are distinguishable with `errors.Is`. | ✓ VERIFIED | `ort/bootstrap.go:56-60,239-325,501-556`; category tests at `ort/bootstrap_test.go:31-169` passed. |
| 02-07 T2 | Filesystem, network, checksum, archive, loader, and independent cleanup causes remain reachable with actionable context. | ✓ VERIFIED | Cause-chain matrix at `ort/bootstrap_test.go:155-294` passed and checks exact underlying error identity/type plus operation paths/URLs. |
| 02-07 T3 | All twelve bootstrap direct notices use private structured Info/Warn diagnostics with redacted sensitive URLs. | ✓ VERIFIED | Exactly 12 `emitDiagnostic` call sites exist in `ort/bootstrap.go`; call-site/URL tests start at `ort/bootstrap_test.go:2010`; no direct production `log.Printf` remains. |
| 02-07 T4 | Non-returnable bootstrap notices may emit once, while returned failures emit zero records. | ✓ VERIFIED | Call-site matrix and explicit validation/network/checksum/archive/lock silence tests at `ort/bootstrap_test.go:2010-2445` passed. |
| 02-07 T5 | Bootstrap-created directories, TGZ/ZIP libraries, and lock files retain least-privilege Unix modes, with Windows-safe compilation. | ✓ VERIFIED | Mode regression at `ort/bootstrap_test.go:1712-1784` passed on Darwin; Windows package compilation passed. |
| 02-08 T1 | All seven production native status sites use the one converter and obsolete message/release wrappers are gone. | ✓ VERIFIED | Source audit found seven calls in environment, memory, tensor, and session code and no production `getErrorMessage(` or `releaseStatus(` wrapper/call. |
| 02-08 T2 | The old finalizer logger is absent; fourteen former direct log paths and three finalizer callers now follow the structured policy without logging returned errors. | ✓ VERIFIED | `ort/finalizer_log.go` is absent; the former 14 direct `log.Printf` paths map to 13 general calls plus the private finalizer emitter; three resource finalizers call the panic-containing wrapper. Production audit found no `log.Printf`/`logFinalizerWarning`. |
| 02-08 T3 | CI separates native-free race tests from real native status/model tests without disabling checkptr. | ✓ VERIFIED | `.github/workflows/ci.yml:111-137,225-239`; exact local race and real native commands passed; no checkptr-disable flag exists. |
| 02-08 T4 | Anchored CI selectors prove 29 race and four native top-level tests are live before execution. | ✓ VERIFIED | Both `go test -list` counts resolved exactly to 29 and 4; the workflow reuses each selector for listing and execution. |
| 02-08 T5 | Final API checks include shape helpers alongside Value/session/diagnostic surfaces and error inspection tests. | ✓ VERIFIED | Separate `go doc` checks resolved `ORTError`, `ParseShape`, `ShapeElementCount`, `Value`, `IsTensor`, `AsTensor`, `RunWithValues`, and `SetDiagnosticHandler`; focused tests passed. |
| 02-08 T6 | Short, race, compile, changed-code lint, and available native suites pass without consumer, dependency, or CI-action regressions. | ✓ VERIFIED | Fresh short suite, 29-test race suite, Go 1.25 compile gate, four executed native tests, vet, and `make precommit-lint-new` all passed. `go.mod`, `go.sum`, workflow action references, lint behavior, and checkptr settings are unchanged from the Phase 2 base. |

**Score:** 44/44 truths verified

## Required Artifacts

All 20 unique PLAN artifacts pass existence, substance, and wiring checks.

| Artifact | Plans | Expected | Status | Details |
|---|---|---|---|---|
| `ort/errors.go` | 01 | Typed errors, sentinels, single status owner | ✓ VERIFIED | 68 substantive lines; imported package-wide through shared identifiers; seven production consumers. |
| `ort/errors_test.go` | 01 | Fake/race ownership and Is/As proof | ✓ VERIFIED | Zero, copy/release, panic, 256-worker, and sentinel matrices execute in race lane. |
| `ort/errors_native_test.go` | 01 | Real native ABI round trip | ✓ VERIFIED | Unix-only real `CreateStatus`/access/release path executed successfully. |
| `ort/environment.go` | 01, 06, 08 | Registered status functions, wrapped environment errors, warning migration | ✓ VERIFIED | Status callbacks register/clear; CreateEnv uses shared converter; lower causes and version warning are wired. |
| `ort/environment_test.go` | 06, 08 | Registration, causes, status, diagnostics, concurrency | ✓ VERIFIED | Anchored tests execute in the race and package suites. |
| `ort/types.go` | 02 | Sealed minimal Value plus inspection helpers | ✓ VERIFIED | Public surface resolves through `go doc`; no raw handle is exported. |
| `ort/tensor.go` | 02, 05 | Sole production Value implementation and tensor ownership/errors | ✓ VERIFIED | Marker, status converter, pinner, leases, sentinels, and finalizer diagnostics are used by sessions/tests. |
| `ort/value_test.go` | 02 | Exact generic extraction matrix | ✓ VERIFIED | Kind, exact type, nil, typed nil, allocation, copy, and heterogeneous tests pass. |
| `ort/diagnostics.go` | 03 | Atomic silent slog bridge | ✓ VERIFIED | `atomic.Pointer` state is read by every private diagnostic emission. |
| `ort/diagnostics_test.go` | 03 | Silence, attrs, races, panic policy | ✓ VERIFIED | Substantive behavior matrix executes under `-race`. |
| `ort/session.go` | 04 | Shared Run/RunWithValues core and session errors | ✓ VERIFIED | Both exported methods enter one used private core; real native output flow passed. |
| `ort/session_test.go` | 02, 04 | Value doubles, run behavior, errors, diagnostics, real model | ✓ VERIFIED | Comprehensive fake/concurrency tests plus real bound/per-call inference execute. |
| `ort/shape_parse.go` | 05 | Inspectable public shape parsing | ✓ VERIFIED | All invalid branches wrap `ErrInvalidArgument`; strconv cause is retained. |
| `ort/shape_test.go` | 05 | Public shape Is/As coverage | ✓ VERIFIED | Parse and exported count matrices pass. |
| `ort/tensor_test.go` | 05 | Status, ownership, race, diagnostic proof | ✓ VERIFIED | Native call-site and lifecycle matrices execute in race lane. |
| `ort/memory.go` | 06 | Lifecycle-safe MemoryInfo status/diagnostic flow | ✓ VERIFIED | Lifecycle lock spans native call and converter; finalizer is wired. |
| `ort/memory_test.go` | 06 | Status release, teardown exclusion, diagnostics | ✓ VERIFIED | Deterministic channel/`TryLock` proof passes under `-race`. |
| `ort/bootstrap.go` | 07 | Categories, wrapped causes, permissions, structured notices | ✓ VERIFIED | Used by public bootstrap entry points; exactly 12 structured notice sites. |
| `ort/bootstrap_test.go` | 07 | Error, security, permission, diagnostic regressions | ✓ VERIFIED | Focused and full short suites pass; one transient timeout is noted below. |
| `.github/workflows/ci.yml` | 08 | Separate live race/native selectors | ✓ VERIFIED | 29/4 liveness counts and execution commands are wired into existing jobs. |
| `ort/finalizer_log.go` | 08 deletion | Legacy helper absent | ✓ VERIFIED | File does not exist; no production references remain. |

## Key Link Verification

| Plan | From | To | Status | Details |
|---|---|---|---|---|
| 01 | `errors.go` | environment status callbacks | ✓ WIRED | `statusToError` reads registered code/message/release functions. |
| 01 | native status test | real ONNX Runtime ABI | ✓ WIRED | Real round trip executed against ONNX Runtime 1.24.1. |
| 02 | `types.go` | `tensor.go` | ✓ WIRED | `Value.ortValue()` is implemented by `*Tensor[T]`. |
| 02 | `value_test.go` | `AsTensor[T]` | ✓ WIRED | Exact generic assertions execute without allocation/copy. |
| 03 | `diagnostics.go` | `log/slog` | ✓ WIRED | `slog.New` and `Logger.LogAttrs` are the only emission path. |
| 03 | `diagnostics.go` | `sync/atomic` | ✓ WIRED | Immutable logger state is stored in `atomic.Pointer[diagnosticState]`. |
| 04 | `Run` / `RunWithValues` | private `run` | ✓ WIRED | Both exported methods directly call the same core. |
| 04 | private `run` | `valuesToHandles` / tensor leases | ✓ WIRED | Both input/output roles acquire and release deduplicated value leases. |
| 04 | `session.go` | `errors.go` | ✓ WIRED | Three native session statuses use `statusToError`. |
| 05 | `shape_parse.go` | public sentinels | ✓ WIRED | Invalid parsing wraps `ErrInvalidArgument` and the strconv cause. |
| 05 | tensor creation | shared status converter | ✓ WIRED | Both tensor native statuses use `statusToError`. |
| 05 | tensor run lease | session handle conversion | ✓ WIRED | `valuesToHandles` invokes `Tensor.lockForRun`; destroy waits for the lease. |
| 06 | `environment.go` | shared errors | ✓ WIRED | CreateEnv and lower causes reach returned errors. |
| 06 | `environment.go` | private diagnostics | ✓ WIRED | Only old-runtime mismatch emits a structured Warn. |
| 06 | `memory.go` | shared errors | ✓ WIRED | Local sentinels and native status conversion reach callers. |
| 06 | `memory.go` | lifecycle lock | ✓ WIRED | `ortCallMu.RLock` spans callback snapshot, native execution, and conversion. |
| 07 | `bootstrap.go` | public error categories | ✓ WIRED | Unsupported, missing-library, and invalid-input paths are distinct. |
| 07 | `bootstrap.go` | private diagnostics | ✓ WIRED | Twelve approved non-returnable notices use the private emitter. |
| 07 | bootstrap tests | URL redaction | ✓ WIRED | Recorded URL equals its redacted form and excludes credentials. |
| 08 | race CI job | fake status/diagnostic/session/resource tests | ✓ WIRED | One anchored selector is listed, counted, then run with `-race`. |
| 08 | native CI job | status ABI and real model tests | ✓ WIRED | Runtime-backed selector is listed, counted, then run without `-race`. |

## Data-Flow Trace (Level 4)

| Artifact | Data | Source | Produces real data | Status |
|---|---|---|---|---|
| `ort/errors.go` | operation, native error code, native message | Real ONNX Runtime `OrtStatus` accessors | Yes — real status message survived `ReleaseStatus` | ✓ FLOWING |
| `ort/session.go` | caller `[]Value` handles and output buffer | Caller-created tensors → `valuesToHandles` → native `Run` | Yes — real all-MiniLM output was finite, nonzero, and written into supplied output | ✓ FLOWING |
| `ort/diagnostics.go` | level, message, structured attrs | Environment/bootstrap/finalizer call sites | Yes — standard JSON and recording handlers received exact records | ✓ FLOWING |
| `ort/bootstrap.go` | lower filesystem/network/archive/loader causes | Real/synthetic OS and HTTP boundaries | Yes — original error identity/type remains reachable through public wrapping | ✓ FLOWING |

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Full short compatibility | `go test -count=1 -short ./...` | All packages passed | ✓ PASS |
| Exact race contract | `go test -count=1 -race ./ort -run "$RACE_SELECTOR"` after a 29-test liveness check | 29 tests selected; package passed | ✓ PASS |
| Real status and inference | Four-test native selector with ONNX Runtime 1.24.1 and cached checksum-verified all-MiniLM model | All four tests executed and passed; none skipped | ✓ PASS |
| Full real-runtime `ort` package | `go test -count=3 ./ort` with native runtime/model configured | Three consecutive package runs passed | ✓ PASS |
| Go baseline and unchanged consumers | `GOTOOLCHAIN=go1.25.12+auto go test -count=1 -run '^$' ./...` | Every package compiled | ✓ PASS |
| Windows compatibility | `GOOS=windows GOARCH=amd64 go test -c -o /dev/null ./ort` | Compiled | ✓ PASS |
| Static gates | `go vet -unsafeptr=false ./ort/... && make precommit-lint-new` | Vet passed; lint reported 0 issues | ✓ PASS |

## Probe Execution

No probe script or probe path is declared by the Phase 2 plans, and no conventional `scripts/**/tests/probe-*.sh` exists. Step 7c is not applicable.

## Requirements Coverage

| Requirement | Source plans | Description | Status | Evidence |
|---|---|---|---|---|
| API-02 | 01, 03, 04, 05, 06, 07, 08 | Public API returns comprehensive wrapped errors with actionable context across environment, tensor, session, and bootstrap flows. | ✓ SATISFIED | Shared typed native converter, public local sentinels, preserved lower causes, seven migrated status sites, structured non-duplicate diagnostics, and passing error-contract/native tests. |
| API-03 | 02, 04, 05, 08 | A `Value` interface enables polymorphic tensor handling for session inputs and outputs. | ✓ SATISFIED | Sealed heterogeneous Value surface, exact inspection helpers, shared `RunWithValues` core, caller-owned output flow, and passing real inference/compatibility tests. |

All requirement IDs declared by the eight plans are accounted for. `.planning/REQUIREMENTS.md` maps only API-02 and API-03 to Phase 2, so there are no orphaned Phase 2 requirements.

## Anti-Patterns and Adversarial Checks

| File | Line | Finding | Severity | Impact |
|---|---:|---|---|---|
| `ort/types.go` | 23, 28, 33, 38 | Pre-existing TODOs on the legacy `Status` compatibility wrapper | ℹ️ Info | Git blame predates Phase 2. The new `ORTError` path does not call these methods, so they do not hollow out API-02. |
| `ort/tensor.go` | 104 | Pre-existing TODO for caller-configurable non-CPU allocation | ℹ️ Info | Predates Phase 2 and is outside the fixed CPU tensor/Value contract. |
| `ort/session.go` | 215 | Pre-existing `RunOptions` placeholder (`0`) | ℹ️ Info | Predates Phase 2; both old and new run methods intentionally share the same existing behavior. |
| `ort/bootstrap_test.go` | 2256-2259 | First full real-runtime package run timed out once waiting for the lock-holder goroutine | ⚠️ Warning | Not reproduced: the full package then passed 3/3 and this subtest passed 20/20 isolated runs. This does not block the phase, but the one-second test safety timeout should be watched for future flakes. |

No `TBD`, `FIXME`, or `XXX` marker exists in Phase 2 modified files. No missing, stub, orphaned, or hollow required artifact was found.

Disconfirmation checks:

- A native selector run without `ONNXRUNTIME_LIB_PATH` exits successfully with four skips, so that command alone is weak evidence. Verification therefore reran it with a real runtime and confirmed all four tests executed.
- `TestAdvancedSessionRunWithValuesRealModel` validates shape, finiteness, nonzero output, bound-output isolation, and ownership, but not a golden numerical vector. Numerical parity is not a Phase 2 requirement and is handled by later embedder work.
- A Windows real DLL status round trip is not present. Phase 2 explicitly uses cross-platform registration/reset plus Windows compilation; the later release phase owns the full supported-platform CI matrix.

## Human Verification Required

None. This phase is a Go API/runtime phase with fully automatable behavior; the real native and model-backed paths were executed during verification.

## Gaps Summary

No goal-blocking gaps were found. All roadmap criteria, all 40 plan truths, all artifacts, all key links, and both Phase 2 requirements are verified.

---

_Verified: 2026-07-24T15:07:10Z_
_Verifier: the agent (gsd-verifier)_
