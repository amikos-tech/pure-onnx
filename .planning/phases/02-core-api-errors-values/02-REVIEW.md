---
phase: 02-core-api-errors-values
reviewed: 2026-07-24T14:42:02Z
depth: standard
files_reviewed: 20
files_reviewed_list:
  - .github/workflows/ci.yml
  - ort/bootstrap.go
  - ort/bootstrap_test.go
  - ort/diagnostics.go
  - ort/diagnostics_test.go
  - ort/environment.go
  - ort/environment_test.go
  - ort/errors.go
  - ort/errors_native_test.go
  - ort/errors_test.go
  - ort/memory.go
  - ort/memory_test.go
  - ort/session.go
  - ort/session_test.go
  - ort/shape_parse.go
  - ort/shape_test.go
  - ort/tensor.go
  - ort/tensor_test.go
  - ort/types.go
  - ort/value_test.go
findings:
  critical: 5
  warning: 18
  info: 0
  total: 23
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-07-24T14:42:02Z
**Depth:** standard
**Files Reviewed:** 20
**Status:** issues_found

## Summary

The reviewed implementation has five ship-blocking correctness and security failures: incompatible runtimes can crash initialization, value leasing can deadlock permanently, large valid-looking tensor shapes can panic instead of returning an error, bootstrap initialization can load a different library than the one it selected, and cache hits bypass bootstrap integrity controls before executable code is loaded. The review also found concurrency races, cross-platform input inconsistencies, incomplete exported APIs, source compatibility breaks, ineffective CI gates, and missing test coverage.

The scoped package tests and static checks pass, but they do not exercise the failure paths described below.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01 [BLOCKER]: Unsupported runtime API versions cause a nil-pointer crash

**File:** `ort/environment.go:164-177`

**Issue:** `InitializeEnvironment` dereferences the result of `OrtGetApiBase` and then casts and dereferences the result of `GetApi(ORT_API_VERSION)` without validating either pointer. The bundled C API contract explicitly permits `GetApi` to return `nullptr` when the loaded runtime does not support the requested API version. Selecting an older shared library therefore crashes the process instead of returning an initialization error. This also makes the later runtime-version warning ineffective for the runtimes that need it most.

**Fix:**

```go
apiBase := ortGetApiBase()
if apiBase == nil {
	return fmt.Errorf("OrtGetApiBase returned nil")
}

apiPtr := getApi(ORT_API_VERSION)
if apiPtr == 0 {
	return fmt.Errorf(
		"runtime does not support ONNX Runtime API version %d",
		ORT_API_VERSION,
	)
}
api := (*ortAPI)(unsafe.Pointer(apiPtr))
```

Add a native fixture whose `GetApi` returns zero and assert that initialization returns a typed error without panicking.

### CR-02 [BLOCKER]: Leasing the same value in both roles can deadlock with `Destroy`

**File:** `ort/session.go:201-211`

**Issue:** Input and output handles are prepared by two independent `valuesToHandles` calls. Each call creates its own deduplication map (`ort/session.go:329-370`), so a tensor used as both input and output receives two read locks. If `Tensor.Destroy` queues for the write lock (`ort/tensor.go:210`) after the first read lock but before the second, Go's writer-preferring `RWMutex` blocks the second read lock. The run waits for its own second lease while `Destroy` waits for its first lease forever. The same partial-acquisition pattern can form cycles when multiple values are supplied in different orders.

**Fix:** Build one unique lease set across both input and output values, acquire each lease exactly once in a stable global order, and only then construct the two role-specific handle arrays. Release the unique leases in reverse order.

```go
leases, err := acquireUniqueValueLeases(inputValues, outputValues)
if err != nil {
	return err
}
defer leases.Release()

inputHandles := handlesFromLeasedValues(inputValues)
outputHandles := handlesFromLeasedValues(outputValues)
```

Add a deterministic regression test that pauses between the two former acquisition points, starts `Destroy`, resumes the run, and requires both goroutines to complete.

### CR-03 [BLOCKER]: `NewEmptyTensor` can panic before byte-size overflow is validated

**File:** `ort/tensor.go:63-75`

**Issue:** `NewEmptyTensor` calculates an element count and immediately executes `make([]T, elementCount)`. The byte-size overflow check only occurs later inside `newTensorFromData`. On a 64-bit platform, a shape such as `Shape{math.MaxInt64}` can pass the element-count check for a multi-byte element type and then panic with `len out of range` or an allocation failure. A public constructor must reject unrepresentable tensors as an error rather than terminating the caller.

**Fix:**

```go
elementCount, err := shapeElementCount(shape)
if err != nil {
	return nil, err
}
_, elementSize, err := tensorElementType[T]()
if err != nil {
	return nil, err
}
if _, err := tensorDataByteSize(elementCount, elementSize); err != nil {
	return nil, err
}

data := make([]T, elementCount)
return newTensorFromData(data, shape)
```

Keep the lower-level validation as defense in depth and add boundary tests for one-byte and multi-byte element types.

### CR-04 [BLOCKER]: Bootstrap path selection and environment initialization are not atomic

**File:** `ort/bootstrap.go:302-326`

**Issue:** `InitializeEnvironmentWithBootstrap` calls `SetSharedLibraryPath(path)` and `InitializeEnvironment()` as separate globally visible operations. `bootstrapInitMu` only serializes bootstrap callers; it does not prevent another goroutine from calling `SetSharedLibraryPath` directly (`ort/environment.go:262-274`) between those calls. The bootstrap call can therefore report success after loading a different library from the one it verified and returned.

**Fix:** Introduce an internal initialization operation that accepts the selected path and performs path assignment plus loading under the same environment state transition. Both public initialization entry points should delegate to that operation; no caller should be able to mutate the configured path in the middle.

```go
func initializeEnvironmentAt(path string) error {
	ortCallMu.Lock()
	defer ortCallMu.Unlock()
	mu.Lock()
	defer mu.Unlock()

	if refCount == 0 {
		libPath = path
	}
	return initializeEnvironmentLocked()
}
```

Add a concurrency test that races direct path configuration against bootstrap initialization and verifies that a successful bootstrap call loaded its own returned path.

### CR-05 [BLOCKER]: Cache hits bypass checksum and filesystem trust checks

**File:** `ort/bootstrap.go:254-259,1309-1380`

**Issue:** `EnsureOnnxRuntimeSharedLibrary` returns a cached, nonempty library before resolving or applying `WithBootstrapExpectedSHA256`, and `validateLibraryFile` uses `os.Stat`, which follows symlinks. Existing cache directories are not checked for ownership or unsafe write permissions. A planted library or symlink in a predictable shared/fallback cache therefore bypasses the checksum path and is later loaded as executable code. This is especially dangerous when the temporary-directory fallback at `ort/bootstrap.go:1482-1506` is used or a caller supplies a shared cache directory.

**Fix:** Treat a cache hit as an integrity-checked state, not merely a nonempty file:

- enter the process lock before accepting the hit;
- reject symlinks with `os.Lstat` and verify cache/install ownership and write permissions on supported platforms;
- persist and verify a manifest of extracted-file hashes produced only after the archive checksum succeeds;
- remove and redownload an invalid cache entry while holding the lock.

Add tests that prepopulate the cache with a nonempty wrong file and a symlink while an expected checksum is configured, then assert that neither candidate is returned.

## Warnings

### WR-01 [WARNING]: Device IDs silently wrap before reaching the native API

**File:** `ort/memory.go:38-40`

**Issue:** `deviceID` is accepted as a Go `int` and converted directly to `int32`. On 64-bit platforms, values outside the signed 32-bit range silently wrap, so native validation sees a different device ID. For example, `1<<32` becomes device zero.

**Fix:** Validate `deviceID` against `math.MinInt32` and `math.MaxInt32` before conversion and return `ErrInvalidArgument` when it is out of range.

### WR-02 [WARNING]: `MemoryInfo.IsValid` races with `Destroy`

**File:** `ort/memory.go:91-96,134-136`

**Issue:** `Destroy` writes `m.handle` while holding the package mutex, but `IsValid` reads the same field without synchronization. Concurrent use is a Go data race and can return a stale validity result.

**Fix:** Protect the handle with a per-object mutex or an atomic integer and use the same synchronization in every getter, `IsValid`, and `Destroy`. Add a race-detector test that repeatedly calls `IsValid` while destroying the object.

### WR-03 [WARNING]: Zero-sized shapes produce order-dependent overflow results

**File:** `ort/tensor.go:281-320`

**Issue:** `shapeElementCount` multiplies dimensions as it iterates and returns zero only when it reaches a zero dimension. Consequently, `Shape{0, math.MaxInt64, 2}` has zero elements, while the mathematically equivalent `Shape{math.MaxInt64, 2, 0}` reports overflow before seeing the zero. Shape validity should not depend on where the zero appears.

**Fix:** First validate every dimension and record whether any dimension is zero. Return zero after validation when one is present; otherwise perform the checked multiplication.

### WR-04 [WARNING]: `Tensor.Shape` exposes mutable internal state

**File:** `ort/tensor.go:175-184`

**Issue:** `Shape` returns `t.shape` directly after releasing the read lock. Callers can mutate the tensor's stored metadata without updating the native `OrtValue`, and concurrent mutation creates a data race with readers. The tensor can then report a shape that does not match its native value or data buffer.

**Fix:** Clone the shape while holding the lock:

```go
return slices.Clone(t.shape)
```

### WR-05 [WARNING]: Embedded NULs are handled differently across platforms

**File:** `ort/session.go:93-99,419-430`

**Issue:** Unix conversions use `purego.GoToCstring`, which silently exposes only the prefix before an embedded NUL to native code. Windows conversion rejects the same string. Model paths, input names, output names, and memory-info names can therefore select a different native resource on Unix than the Go value represents.

**Fix:** Reject `'\x00'` in every Go string before platform-specific conversion and return `ErrInvalidArgument`. Centralize this validation in the string-to-native helper so all call sites have identical behavior.

### WR-06 [WARNING]: Exported `Status` methods return fabricated error information

**File:** `ort/types.go:22-39`

**Issue:** Every nonzero `Status` reports `ErrorCodeFail` and the literal message `"Error occurred"`, regardless of the native status. These exported methods look functional but discard the actual native error code and message; tests currently lock in the placeholder behavior.

**Fix:** Either implement the methods through the loaded native status accessors with explicit lifetime rules, or remove/hide the exported placeholder until it can be correct. Tests should distinguish at least two native error codes and messages.

### WR-07 [WARNING]: External callers cannot create a usable `SessionOptions`

**File:** `ort/types.go:106-121`

**Issue:** `SessionOptions` contains only an unexported native handle, there is no public constructor or configuration API, and `NewAdvancedSession` rejects a zero handle (`ort/session.go:43-45,75-78`). External callers can only pass `nil`; the non-nil options path is unreachable through supported API use.

**Fix:** Provide a constructor and a matching idempotent `Destroy`, or remove the parameter until session options are supported. Do not expose a public type that users cannot put into a valid state.

### WR-08 [WARNING]: Public handle types were changed incompatibly

**File:** `ort/types.go:11-58`

**Issue:** `Status`, `Environment`, and `Session` changed from defined `uintptr` handle types to structs. Existing callers that convert native handles, compare values, or store these values directly no longer compile. No compatibility shim or migration path is present.

**Fix:** Preserve the existing public handle types and add separate internal owner structs, or make the breaking change explicit in a major-version migration with replacement constructors and documented conversions.

### WR-09 [WARNING]: Failed close operations leak downloaded temporary archives

**File:** `ort/bootstrap.go:984-1030`

**Issue:** `success` is set to true before deferred response and temporary-file closes run. A deferred close can set the named return error, but `success` stays true, so the cleanup defer does not remove the temporary archive. Repeated close failures leave one cache-adjacent file per attempt.

**Fix:** Explicitly close the response and temporary file before setting success, and base cleanup on the final returned error rather than a flag set before deferred operations complete.

### WR-10 [WARNING]: The compatibility CI job cannot fail its advertised checks

**File:** `.github/workflows/ci.yml:385-392`

**Issue:** Both compatibility checks append `|| echo ...`. A missing `list-ort-versions` target, a command failure, or a constants mismatch therefore exits successfully. The job reports green without enforcing compatibility.

**Fix:** Remove the success-forcing fallbacks. If a check is intentionally informational, move it to a clearly named informational step and add a separate assertion that exits nonzero on mismatch.

### WR-11 [WARNING]: Security scanning never gates a change

**File:** `.github/workflows/ci.yml:335-354`

**Issue:** `gosec` runs with `-no-fail`, and the step also has `continue-on-error: true`; SARIF upload failures are suppressed as well. Security findings and scanner execution failures cannot fail the job, so the workflow provides no enforcement signal.

**Fix:** Remove `-no-fail` and `continue-on-error` from the scan step. Keep upload failure handling separate if reporting availability must not block builds.

### WR-12 [WARNING]: The default bootstrap runtime has drifted from the CI runtime

**File:** `ort/bootstrap.go:30-36`

**Issue:** The source comment says `DefaultOnnxRuntimeVersion` must track the CI runtime, but the code defaults to `1.23.1` while the integration workflow validates `1.24.1` (`.github/workflows/ci.yml:160-167`). Default bootstrap behavior is therefore not covered by the advertised integration run.

**Fix:** Define the version once and consume it in both bootstrap code and CI, or add a failing CI assertion that the two values match.

### WR-13 [WARNING]: Runtime-version comparison ignores the major version

**File:** `ort/environment.go:76-94`

**Issue:** The compatibility warning parses only the minor component and compares it with `22`. A future `2.0` runtime is incorrectly warned as too old, while a hypothetical `0.99` runtime is treated as new enough. Malformed versions are silently ignored.

**Fix:** Parse and compare the `(major, minor)` tuple against the minimum supported version, and emit a diagnostic when a nonempty version string cannot be parsed.

### WR-14 [WARNING]: The finalizer test has no observable assertion

**File:** `ort/memory_test.go:236-254`

**Issue:** The test drops a `MemoryInfo`, invokes garbage collection twice, and then exits without proving that the finalizer released the native handle. It passes even if finalizer registration is removed or release fails.

**Fix:** Use an injectable release callback or native test fixture with an atomic release count, then wait with a bounded timeout for exactly one release. Keep finalizer timing out of tests that cannot observe an effect.

### WR-15 [WARNING]: Synchronous diagnostics run while non-reentrant lifecycle locks are held

**File:** `ort/environment.go:114-119,190-195`

**Issue:** `InitializeEnvironment` invokes the consumer's synchronous `slog.Handler` while holding both `ortCallMu` and `mu`. A handler that calls a harmless-looking package query such as `IsInitialized` or `GetVersionString` waits on a lock already held by its own call stack and deadlocks. Bootstrap diagnostics can likewise run while the process file lock is held. The handler documentation warns about concurrency and panic behavior but does not prohibit reentrant package calls.

**Fix:** Capture the diagnostic data while state is protected, release lifecycle/file locks, and invoke the handler afterward. If reentrancy is intentionally unsupported, document that restriction explicitly and add a bounded regression test instead of allowing an unexplained deadlock.

### WR-16 [WARNING]: Successful native calls may return unusable zero handles

**File:** `ort/environment.go:102-110,197-207`

**Issue:** Environment, memory-info, tensor, session-options, and session constructors treat a zero native status as complete success without checking that the required output handle is nonzero (`ort/memory.go:37-60`, `ort/tensor.go:103-149`, `ort/session.go:75-120`). An ABI mismatch or faulty runtime can therefore produce a successful Go object that is immediately reported as destroyed, or mark the global environment initialized with no environment handle.

**Fix:** After every successful create call, validate the out handle before committing state. Return a clear initialization/native-contract error and release or unpin any resources already acquired.

### WR-17 [WARNING]: Negative runtime version segments pass validation

**File:** `ort/bootstrap.go:1518-1539`

**Issue:** `normalizeRuntimeVersion` checks segments with `strconv.Atoi` but never checks that the values are nonnegative. Inputs such as `-1.23.1` are accepted and used to construct cache paths and download URLs instead of returning `ErrInvalidArgument`.

**Fix:** Parse every segment, reject values below zero, and return a canonical decimal form or enforce an explicit version grammar.

### WR-18 [WARNING]: The session serialization test can pass without serialization

**File:** `ort/session_test.go:1321-1385`

**Issue:** `TestAdvancedSessionRunConcurrent` relies on a one-millisecond sleep to make native callbacks overlap if serialization is removed. Scheduler timing can still execute callbacks one at a time, leaving `maxInFlight == 1` and allowing the regression to pass.

**Fix:** Block the first callback on a channel, assert deterministically that `session.runMu.TryLock()` fails while it is in flight, then release it and require every run to complete. This removes timing as the proof of correctness.

---

_Reviewed: 2026-07-24T14:42:02Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
