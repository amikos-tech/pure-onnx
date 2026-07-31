---
phase: 02-core-api-errors-values
fixed_at: 2026-07-24T16:38:08Z
review_path: .planning/phases/02-core-api-errors-values/02-REVIEW.md
iteration: 1
findings_in_scope: 23
fixed: 23
skipped: 0
status: all_fixed
---

# Phase 02: Code Review Fix Report

**Fixed at:** 2026-07-24T16:38:08Z
**Source review:** `.planning/phases/02-core-api-errors-values/02-REVIEW.md`
**Iteration:** 1

**Summary:**

- Findings in scope: 23
- Fixed: 23
- Skipped: 0

## Fixed Issues

### CR-01: Unsupported runtime API versions cause a nil-pointer crash

**Status:** fixed
**Files modified:** `ort/environment.go`, `ort/environment_test.go`, `ort/errors.go`
**Commit:** 119a045
**Applied fix:** Added explicit checks for nil API-base, API-function, and requested-version pointers, returning `ErrUnsupportedRuntime` instead of dereferencing them.
**Verification:** `go test ./ort -run '^TestInitializeEnvironmentRejectsUnsupportedAPIVersion$' -count=1`

### CR-02: Leasing the same value in both roles can deadlock with `Destroy`

**Status:** fixed: requires human verification
**Files modified:** `ort/session.go`, `ort/session_test.go`
**Commit:** 65b79cf
**Applied fix:** Deduplicated input/output value leases, acquired them in stable pointer order, and released them in reverse order.
**Verification:** Focused value-handle, shared-input/output, and `RunWithValues` tests passed.

### CR-03: `NewEmptyTensor` can panic before byte-size overflow is validated

**Status:** fixed: requires human verification
**Files modified:** `ort/tensor.go`, `ort/tensor_test.go`
**Commit:** 5234078
**Applied fix:** Validated the tensor byte size before allocating the backing slice.
**Verification:** `go test ./ort -run '^(TestTensorDataByteSizeOverflow|TestNewEmptyTensorRejectsByteSizeOverflowBeforeAllocation)$' -count=1`

### CR-04: Bootstrap path selection and environment initialization are not atomic

**Status:** fixed: requires human verification
**Files modified:** `ort/bootstrap.go`, `ort/bootstrap_test.go`, `ort/environment.go`
**Commit:** 228d328
**Applied fix:** Added an atomic path-and-initialize lifecycle transition and made bootstrap initialization use it.
**Verification:** Focused bootstrap atomicity, path-conflict, and reference-count tests passed.

### CR-05: Cache hits bypass checksum and filesystem trust checks

**Status:** fixed: requires human verification
**Files modified:** `ort/bootstrap.go`, `ort/bootstrap_test.go`, `ort/bootstrap_trust_other.go`, `ort/bootstrap_trust_unix.go`
**Commit:** 57ae800
**Applied fix:** Validated cache ownership, permissions, symlink safety, and a content manifest while holding the process lock; invalid entries are removed and downloaded again.
**Verification:** Focused bootstrap integrity and trust tests passed, including tampered manifests and cached symlinks; Windows cross-compilation passed.

### WR-01: Device IDs silently wrap before reaching the native API

**Status:** fixed
**Files modified:** `ort/memory.go`, `ort/memory_test.go`
**Commit:** 96cfbbe
**Applied fix:** Rejected device IDs outside the native `int32` range before conversion.
**Verification:** `go test ./ort -run '^TestMemoryInfoBeforeInit$' -count=1`

### WR-02: `MemoryInfo.IsValid` races with `Destroy`

**Status:** fixed: requires human verification
**Files modified:** `ort/memory.go`, `ort/memory_test.go`, `ort/types.go`
**Commit:** 39e07be
**Applied fix:** Protected the memory-info handle with a per-object read/write mutex.
**Verification:** `go test -race ./ort -run '^TestMemoryInfoIsValidConcurrentDestroy$' -count=1`

### WR-03: Zero-sized shapes produce order-dependent overflow results

**Status:** fixed: requires human verification
**Files modified:** `ort/tensor.go`, `ort/tensor_test.go`
**Commit:** b289dcb
**Applied fix:** Made zero-dimension handling independent of dimension order while preserving overflow validation.
**Verification:** `go test ./ort -run '^TestShapeElementCount$' -count=1`

### WR-04: `Tensor.Shape` exposes mutable internal state

**Status:** fixed
**Files modified:** `ort/tensor.go`, `ort/tensor_test.go`
**Commit:** c5de473
**Applied fix:** Returned a cloned shape instead of the tensor's internal slice.
**Verification:** `go test -race ./ort -run '^TestTensorShapeReturnsCopy$' -count=1`

### WR-05: Embedded NULs are handled differently across platforms

**Status:** fixed
**Files modified:** `ort/cstring.go`, `ort/memory.go`, `ort/memory_test.go`, `ort/ortchar_unix.go`, `ort/ortchar_windows.go`, `ort/session.go`, `ort/session_test.go`
**Commit:** 5191c3a
**Applied fix:** Centralized native-string validation and rejected embedded NULs for model paths, allocator names, and session input/output names on every platform.
**Verification:** Focused string/session/memory tests and Windows cross-compilation passed.

### WR-06: Exported `Status` methods return fabricated error information

**Status:** fixed
**Files modified:** `ort/shape_test.go`, `ort/types.go`
**Commit:** 37c1b79
**Applied fix:** Routed status code and message access through the registered native accessors.
**Verification:** Native accessor tests distinguished multiple status codes and messages.

### WR-07: External callers cannot create a usable `SessionOptions`

**Status:** fixed
**Files modified:** `ort/session.go`, `ort/session_test.go`, `ort/types.go`
**Commit:** 2e2d1d0
**Applied fix:** Added a public constructor, validity query, idempotent destroy, finalizer, and safe session-time handle lease.
**Verification:** Session-options lifecycle and provided-options creation tests passed under the race detector.

### WR-08: Public handle types were changed incompatibly

**Status:** fixed
**Files modified:** `ort/shape_test.go`, `ort/types.go`
**Commit:** 7c21361
**Applied fix:** Restored `Status`, `Environment`, and `Session` as defined `uintptr` handle types.
**Verification:** Public handle conversion and status accessor tests passed.

### WR-09: Failed close operations leak downloaded temporary archives

**Status:** fixed: requires human verification
**Files modified:** `ort/bootstrap.go`, `ort/bootstrap_test.go`
**Commit:** 76b9a3a
**Applied fix:** Closed the response and temporary archive before success and based cleanup on the final returned error.
**Verification:** Error, response-close failure, and transient-retry download tests passed.

### WR-10: The compatibility CI job cannot fail its advertised checks

**Status:** fixed
**Files modified:** `.github/workflows/ci.yml`
**Commit:** 8b8f6c8
**Applied fix:** Removed success-forcing fallbacks and made the API-constant check an anchored assertion.
**Verification:** `make -n list-ort-versions`, the anchored grep, and `actionlint -shellcheck=` passed.

### WR-11: Security scanning never gates a change

**Status:** fixed
**Files modified:** `.github/workflows/ci.yml`
**Commit:** 5e702f7
**Applied fix:** Removed `-no-fail` and `continue-on-error` from the gosec scan while keeping SARIF upload availability nonblocking.
**Verification:** Workflow syntax passed `actionlint -shellcheck=`.

### WR-12: The default bootstrap runtime has drifted from the CI runtime

**Status:** fixed
**Files modified:** `.github/workflows/ci.yml`, `Makefile`, `README.md`, `examples/inference/README.md`, `ort/bootstrap.go`
**Commit:** a47bb66
**Applied fix:** Aligned defaults on ONNX Runtime 1.24.1 and added a failing CI assertion that bootstrap, integration, and Makefile versions match.
**Verification:** Local version equality assertion, bootstrap tests, and workflow validation passed.

### WR-13: Runtime-version comparison ignores the major version

**Status:** fixed: requires human verification
**Files modified:** `ort/environment.go`, `ort/environment_test.go`
**Commit:** a299e75
**Applied fix:** Compared parsed `(major, minor)` tuples and emitted a structured diagnostic for malformed nonempty versions.
**Verification:** `go test ./ort -run '^TestDiagnosticRuntimeVersion$' -count=1`

### WR-14: The finalizer test has no observable assertion

**Status:** fixed
**Files modified:** `ort/memory_test.go`
**Commit:** 8dca057
**Applied fix:** Replaced the smoke test with an injected release callback, atomic release count, and bounded finalizer wait.
**Verification:** `go test -race ./ort -run '^TestMemoryInfoFinalizer$' -count=1`

### WR-15: Synchronous diagnostics run while non-reentrant lifecycle locks are held

**Status:** fixed: requires human verification
**Files modified:** `ort/diagnostics.go`, `ort/environment.go`, `ort/environment_test.go`
**Commit:** 336ac71
**Applied fix:** Captured the runtime version under lifecycle locks and emitted its diagnostic after unlocking; documented the remaining bootstrap/lifecycle reentrancy restriction and added a bounded query-reentrancy regression test.
**Verification:** Environment and runtime-diagnostic tests passed.

### WR-16: Successful native calls may return unusable zero handles

**Status:** fixed
**Files modified:** `ort/environment.go`, `ort/environment_test.go`, `ort/errors.go`, `ort/memory.go`, `ort/memory_test.go`, `ort/session.go`, `ort/session_test.go`, `ort/tensor.go`, `ort/tensor_test.go`
**Commit:** 98924c0
**Applied fix:** Added `ErrNativeContract` and validated successful environment, memory-info, tensor, session-options, and session outputs before constructing live Go objects; acquired resources are released or unpinned on failure.
**Verification:** Focused zero-handle constructor tests passed normally and under the race detector.

### WR-17: Negative runtime version segments pass validation

**Status:** fixed
**Files modified:** `ort/bootstrap.go`, `ort/bootstrap_test.go`
**Commit:** db2c069
**Applied fix:** Rejected negative segments and returned a canonical decimal version.
**Verification:** `go test ./ort -run '^TestNormalizeRuntimeVersion$' -count=1`

### WR-18: The session serialization test can pass without serialization

**Status:** fixed: requires human verification
**Files modified:** `ort/session_test.go`
**Commit:** 2d055a6
**Applied fix:** Blocked the first native callback, asserted `runMu.TryLock` fails while it is active, then released it and required all runs to finish.
**Verification:** The test passed 20 consecutive runs and under the race detector.

## Aggregate Verification

- `go test ./ort -count=1`
- Focused concurrency tests under `go test -race`
- `GOOS=windows GOARCH=amd64 go test -exec=true ./ort -run '^$'`
- `actionlint -shellcheck= .github/workflows/ci.yml`
- `git diff --check`

All aggregate checks above passed. The reentrant native-fixture test passed in the normal package run; race/checkptr cannot inspect its deliberate Go-pointer-to-FFI round trip, so that test was excluded from the focused race invocation.

---

_Fixed: 2026-07-24T16:38:08Z_
_Fixer: the agent (gsd-code-fixer)_
_Iteration: 1_
