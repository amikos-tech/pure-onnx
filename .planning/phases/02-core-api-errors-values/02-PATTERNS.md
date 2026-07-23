# Phase 2: Core API — Errors & Values - Pattern Map

**Mapped:** 2026-07-23  
**Files analyzed:** 19 new/modified files  
**Analogs found:** 19 / 19  
**Primary analog families:** 5

## Scope Extracted from Phase Inputs

The file set below comes from the recommended project structure, integration points, and Wave 0 gaps in `02-RESEARCH.md`, constrained by D-01 through D-23 in `02-CONTEXT.md`.

### New files

- `ort/errors.go`
- `ort/errors_test.go`
- `ort/errors_native_test.go`
- `ort/diagnostics.go`
- `ort/diagnostics_test.go`
- `ort/value_test.go`

### Existing files to edit

- `ort/types.go`
- `ort/tensor.go`
- `ort/session.go`
- `ort/memory.go`
- `ort/environment.go`
- `ort/bootstrap.go`
- `ort/finalizer_log.go` (edit or remove after preserving its finalizer-safety behavior)
- `ort/session_test.go`
- `ort/environment_test.go`
- `ort/memory_test.go`
- `ort/tensor_test.go`
- `ort/bootstrap_test.go`
- `.github/workflows/ci.yml`

No example or embedder source change is implied. Their existing constructor-bound `AdvancedSession.Run()` use is a compatibility verification target, not a reason to rewrite consumers.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `ort/errors.go` | utility | transform (native status → Go error) | `.planning/spikes/001-ort-status-lifetime/status_lifetime_test.go` | exact prototype |
| `ort/errors_test.go` | test | transform + concurrent callbacks | `.planning/spikes/001-ort-status-lifetime/status_lifetime_test.go` | exact prototype |
| `ort/errors_native_test.go` | test | request-response (real FFI round trip) | `.planning/spikes/001-ort-status-lifetime/native_status_test.go` | exact prototype |
| `ort/diagnostics.go` | provider | event-driven | `.planning/spikes/002-b-slog-handler-sink/diagnostic.go` | exact prototype |
| `ort/diagnostics_test.go` | test | event-driven + concurrent reconfiguration | `.planning/spikes/002-b-slog-handler-sink/diagnostic_test.go` | exact prototype |
| `ort/types.go` | model | transform/type inspection | `ort/types.go` + `ort/tensor.go` | self-extension |
| `ort/value_test.go` | test | transform/type inspection | `ort/session_test.go` in-package value doubles | role/data-flow |
| `ort/tensor.go` | model | CRUD/resource lifecycle | `ort/tensor.go` | self-extension |
| `ort/session.go` | service | request-response | `ort/session.go` | self-extension |
| `ort/memory.go` | model | CRUD/resource lifecycle | `ort/tensor.go` | role/data-flow |
| `ort/environment.go` | provider | CRUD/runtime lifecycle | `ort/environment.go` | self-extension |
| `ort/bootstrap.go` | service | file-I/O + request-response | `ort/bootstrap.go` | self-extension |
| `ort/finalizer_log.go` | utility | event-driven | `.planning/spikes/002-b-slog-handler-sink/diagnostic.go` | role/data-flow |
| `ort/session_test.go` | test | request-response + concurrency | `ort/session_test.go` | self-extension |
| `ort/environment_test.go` | test | CRUD/runtime lifecycle | `ort/environment_test.go` | self-extension |
| `ort/memory_test.go` | test | CRUD/resource lifecycle | `ort/tensor_test.go` | role/data-flow |
| `ort/tensor_test.go` | test | CRUD/resource lifecycle | `ort/tensor_test.go` | self-extension |
| `ort/bootstrap_test.go` | test | file-I/O + event-driven | `ort/bootstrap_test.go` | self-extension |
| `.github/workflows/ci.yml` | config | batch | `.github/workflows/ci.yml` | self-extension |

## Pattern Assignments

### `ort/errors.go` (utility, native-status transform)

**Primary analog:** `.planning/spikes/001-ort-status-lifetime/status_lifetime_test.go`

**Imports pattern** (lines 3-8):

```go
import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"testing"
)
```

Production needs only the standard-library subset actually used, normally `errors` and `fmt`. Do not add an error package.

**Typed error and ownership pattern** (lines 13-45):

```go
type statusError struct {
	Op      string
	Code    ort.ErrorCode
	Message string
}

func (e *statusError) Error() string {
	return fmt.Sprintf("%s: ORT code %d: %s", e.Op, e.Code, e.Message)
}

type statusOps struct {
	getCode     func(uintptr) ort.ErrorCode
	copyMessage func(uintptr) string
	release     func(uintptr)
}

func statusToErrorPrototype(status uintptr, op string, ops statusOps) error {
	if status == 0 {
		return nil
	}
	defer ops.release(status)

	return &statusError{
		Op:      op,
		Code:    ops.getCode(status),
		Message: ops.copyMessage(status),
	}
}
```

Copy this ownership sequence exactly into production naming:

1. Return `nil` for status `0`.
2. Install `defer ReleaseStatus(status)` before either accessor.
3. Capture `ErrorCode`.
4. copy the native message with `CstringToGo`.
5. return only Go-owned fields in `*ORTError`.

The helper owns the one and only release. Callers must not call `getErrorMessage` or `releaseStatus` after migration.

**Sentinel pattern** — `ort/bootstrap.go` lines 49-56 and 529:

```go
var errSharedLibraryNotFound = errors.New("ONNX Runtime shared library not found")

var ErrUnsupportedPlatform = errors.New("unsupported platform for ONNX Runtime bootstrap")

func IsUnsupportedPlatformError(err error) bool {
	return errors.Is(err, ErrUnsupportedPlatform)
}

// ...
return runtimeArtifact{}, fmt.Errorf("%w: GOOS=%s GOARCH=%s", ErrUnsupportedPlatform, goos, goarch)
```

Use the same lean pattern for `ErrInvalidArgument`, `ErrNotInitialized`, `ErrDestroyed`, and the promoted shared-library-not-found category. Put operation and identifiers in `%w` wrappers at call sites. Do not make `ORTError` unwrap to a local sentinel: native codes are inspected with `errors.As`, local validation categories with `errors.Is`.

**Error-chain pattern** — `ort/environment.go` lines 107-129:

```go
defer func() {
	if cleanupNeeded {
		if ortLib != 0 {
			if closeErr := closeLibrary(ortLib); closeErr != nil {
				closeErr = fmt.Errorf(
					"failed to close ONNX Runtime library during initialization cleanup: %w",
					closeErr,
				)
				if err == nil {
					err = closeErr
				} else {
					err = errors.Join(err, closeErr)
				}
			}
			ortLib = 0
		}
		clearORTGlobalsLocked()
	}
}()

ortLib, err = loadLibrary(libPath)
if err != nil {
	return fmt.Errorf("failed to load ONNX Runtime library: %w", err)
}
```

Use `%w` for one causal chain and `errors.Join` only when a primary and cleanup failure are independently useful.

**Guard:** `InitializeEnvironment` already holds global `mu` for its whole body (`ort/environment.go` lines 91-97). The production converter must not blindly reacquire `mu` and deadlock. Either pass a snapshotted `statusOps` into the ownership helper or use pointers already protected by the caller's `ortCallMu`/`mu` scope.

---

### `ort/errors_test.go` (test, fake native status)

**Analog:** `.planning/spikes/001-ort-status-lifetime/status_lifetime_test.go`

**Fake callback store** (lines 47-116):

```go
type fakeStatusStore struct {
	mu           sync.Mutex
	next         uintptr
	statuses     map[uintptr]*fakeStatus
	releaseCount map[uintptr]int
}

func (s *fakeStatusStore) copyMessage(handle uintptr) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	message := s.statuses[handle].message
	return string(message[:len(message)-1])
}

func (s *fakeStatusStore) release(handle uintptr) {
	s.mu.Lock()
	defer s.mu.Unlock()

	status := s.statuses[handle]
	for i := range status.message {
		status.message[i] = 'x'
	}
	s.releaseCount[handle]++
}
```

Mutating the fake native buffer during release proves the returned message was copied rather than retained.

**Assertions pattern** (lines 118-151):

```go
err := statusToErrorPrototype(handle, "run inference", store.ops())

var got *statusError
if !errors.As(err, &got) {
	t.Fatalf("errors.As did not find *statusError in %T", err)
}
if got.Code != ort.ErrorCodeInvalidArgument {
	t.Fatalf("code mismatch: got %d", got.Code)
}
if got.Message != "shape mismatch" {
	t.Fatalf("message changed after release: got %q", got.Message)
}
if releases := store.releases(handle); releases != 1 {
	t.Fatalf("release count: got %d, want 1", releases)
}
```

Retain the zero-status/no-release case and the 256-worker exact-release test at lines 144-195. Run these callback tests under `-race`; they contain no real `uintptr`-to-native ABI crossing.

---

### `ort/errors_native_test.go` (test, real FFI round trip)

**Analog:** `.planning/spikes/001-ort-status-lifetime/native_status_test.go`

**Platform constraint:** The analog begins with `//go:build !windows` because its `purego.Dlopen`/`Dlsym` loader path is Unix-only. Preserve that first-line constraint in `ort/errors_native_test.go`, and prove the Windows package/test matrix still compiles with `GOOS=windows GOARCH=amd64 go test -c -o /dev/null ./ort`.

**Real ABI registration pattern** (lines 49-79):

```go
var getAPI func(uint32) uintptr
purego.RegisterFunc(&getAPI, apiBase.GetApi)
apiPointer := getAPI(ort.ORT_API_VERSION)
api := (*ort.OrtApi)(unsafe.Pointer(apiPointer))

var createStatus func(ort.ErrorCode, uintptr) uintptr
var getErrorCode func(uintptr) ort.ErrorCode
var getErrorMessage func(uintptr) uintptr
var releaseStatus func(uintptr)
purego.RegisterFunc(&createStatus, api.CreateStatus)
purego.RegisterFunc(&getErrorCode, api.GetErrorCode)
purego.RegisterFunc(&getErrorMessage, api.GetErrorMessage)
purego.RegisterFunc(&releaseStatus, api.ReleaseStatus)

messageBacking, messagePointer := ort.GoToCstring("native status survives release")
status := createStatus(ort.ErrorCodeInvalidGraph, messagePointer)
runtime.KeepAlive(messageBacking)
```

Adapt the gate to the research decision: skip unless `ONNXRUNTIME_LIB_PATH` is set, use that path, and run this test only in the existing non-race integration job. Do not disable checkptr and do not fold this test into the race callback suite.

---

### `ort/diagnostics.go` (provider, event-driven)

**Analog:** `.planning/spikes/002-b-slog-handler-sink/diagnostic.go`

**Copy nearly verbatim, changing only package-local names** (lines 3-37):

```go
import (
	"context"
	"log/slog"
	"sync/atomic"
)

type handlerState struct {
	logger *slog.Logger
}

var configuredHandler = newHandlerStore()

func newHandlerStore() *atomic.Pointer[handlerState] {
	store := &atomic.Pointer[handlerState]{}
	store.Store(&handlerState{logger: slog.New(slog.DiscardHandler)})
	return store
}

func SetDiagnosticHandler(handler slog.Handler) {
	if handler == nil {
		handler = slog.DiscardHandler
	}
	configuredHandler.Store(&handlerState{logger: slog.New(handler)})
}

func emitDiagnostic(ctx context.Context, level slog.Level, message string, attrs ...slog.Attr) {
	if ctx == nil {
		ctx = context.Background()
	}
	configuredHandler.Load().logger.LogAttrs(ctx, level, message, attrs...)
}
```

This is the complete concurrency/configuration pattern. Keep emission private and accept only standard `slog.Handler`, `slog.Level`, and `slog.Attr`. Do not create package-owned logger, field, or level types.

No authentication pattern applies. The relevant guards are atomic configuration, a silent default, and call-site policy.

---

### `ort/diagnostics_test.go` (test, event-driven/concurrent)

**Analog:** `.planning/spikes/002-b-slog-handler-sink/diagnostic_test.go`

**Opt-in and no-double-log pattern** (lines 19-47):

```go
var output bytes.Buffer
handler := slog.NewJSONHandler(&output, nil)
SetDiagnosticHandler(handler)
t.Cleanup(func() { SetDiagnosticHandler(nil) })

if err := operationThatReturnsError(); err == nil {
	t.Fatal("operationThatReturnsError returned nil")
}
if output.Len() != 0 {
	t.Fatalf("returned error produced diagnostic output: %q", output.String())
}

emitDiagnostic(
	context.Background(),
	slog.LevelWarn,
	"finalizer cleanup failed",
	slog.String("resource", "tensor"),
	slog.Any("error", errors.New("release failed")),
)
```

**Race pattern** (lines 66-110):

```go
first := &countingHandler{}
second := &countingHandler{}
SetDiagnosticHandler(first)
t.Cleanup(func() { SetDiagnosticHandler(nil) })

var wg sync.WaitGroup
// Writers call emitDiagnostic while other goroutines alternate handlers.
// ...
wg.Wait()

got := first.count.Load() + second.count.Load()
want := int64(writers * events)
if got != want {
	t.Fatalf("captured events: got %d, want %d", got, want)
}
```

Also retain nil-reset silence and consumer `logger.Handler()` attribute preservation. Add call-site tests that prove returned failures emit zero records; the handler type cannot enforce that rule.

---

### `ort/types.go` (model, type inspection)

**Analog:** existing interface in `ort/types.go` lines 60-67:

```go
type Value interface {
	// Destroy releases the underlying resources
	Destroy() error
	// Type returns the type of the value
	Type() ValueType
}
```

Add one unexported marker to this existing interface; do not replace it with a second wrapper/interface. Keep the public method set minimal.

**Tensor discriminator analog:** `ort/tensor.go` lines 237-240:

```go
func (t *Tensor[T]) Type() ValueType {
	return ValueTypeTensor
}
```

Place `IsTensor(Value) bool` and exact `AsTensor[T](Value)` next to the public value surface. The semantic delta has no production analog, so follow the research's direct type-assertion shape: no reflection, allocation, copying, or numeric coercion.

---

### `ort/value_test.go` (test, exact type inspection)

**Analog:** in-package value doubles in `ort/session_test.go` lines 15-34:

```go
type fakeValue struct {
	handle uintptr
}

func (f *fakeValue) Destroy() error          { return nil }
func (f *fakeValue) Type() ValueType         { return ValueTypeTensor }
func (f *fakeValue) ortValueHandle() uintptr { return f.handle }

type unsupportedValue struct{}

func (u *unsupportedValue) Destroy() error  { return nil }
func (u *unsupportedValue) Type() ValueType { return ValueTypeTensor }
```

All in-package doubles that remain `Value` implementations must gain the private marker. Prefer real zero-value `*Tensor[T]` pointers for the `AsTensor` table because extraction does not need ORT initialization.

Cover:

- matching `*Tensor[float32]`,
- mismatched `*Tensor[int64]`,
- nil `Value`,
- typed-nil `*Tensor[T]`,
- `IsTensor` kind check independent from exact element type.

Assert the boolean/result, not exact error text. There is no conversion behavior to test.

---

### `ort/tensor.go` (model, resource CRUD)

**Analog:** existing tensor implementation.

**Marker placement and private handle pattern** — lines 10-27:

```go
type Tensor[T any] struct {
	shape  Shape
	data   []T
	handle uintptr
	pinner *runtime.Pinner
	runMu  sync.RWMutex
}

func (t *Tensor[T]) ortValueHandle() uintptr {
	if t == nil {
		return 0
	}
	t.runMu.RLock()
	handle := t.handle
	t.runMu.RUnlock()
	return handle
}
```

Add the no-op private `Value` marker near this private handle method.

**Run lease pattern** — lines 215-235:

```go
func (t *Tensor[T]) lockForRun() (uintptr, error) {
	if t == nil {
		return 0, errValueNil
	}

	t.runMu.RLock()
	handle := t.handle
	if handle == 0 {
		t.runMu.RUnlock()
		return 0, errValueDestroyed
	}

	return handle, nil
}

func (t *Tensor[T]) unlockForRun() {
	if t == nil {
		return
	}
	t.runMu.RUnlock()
}
```

Preserve this lease exactly. Migrate the two status blocks at lines 88-94 and 111-122 to `statusToError`, while retaining:

- `runtime.KeepAlive` after the native call,
- `pinner.Unpin()` on tensor-creation failure,
- `defer releaseMemoryInfo(memInfo)`,
- the existing `ortCallMu → mu → Tensor.runMu` order.

Wrap local validation and lifecycle failures with the shared sentinels.

---

### `ort/session.go` (service, request-response)

**Primary analog:** current `Run` and handle leasing in the same file.

**Lock entry to preserve** — lines 125-140:

```go
func (s *AdvancedSession) Run() error {
	if s == nil {
		return fmt.Errorf("session is nil")
	}

	// Lock order here is runMu -> ortCallMu -> mu.
	s.runMu.Lock()
	defer s.runMu.Unlock()

	// Holding ortCallMu RLock keeps DestroyEnvironment() from closing the runtime
	// while raw pointers are passed into ORT.
	ortCallMu.RLock()
	defer ortCallMu.RUnlock()
```

**Lease, FFI, and lifetime portion** — lines 181-219:

```go
inputNameBackings, inputNamePtrs := makeCStringPointerArray(inputNames)
outputNameBackings, outputNamePtrs := makeCStringPointerArray(outputNames)

inputValueHandles, releaseInputValueHandles, err := valuesToHandles(inputValues, "input")
if err != nil {
	return err
}
defer releaseInputValueHandles()

outputValueHandles, releaseOutputValueHandles, err := valuesToHandles(outputValues, "output")
if err != nil {
	return err
}
defer releaseOutputValueHandles()

status := run(
	sessionHandle,
	0,
	uintptrSlicePtr(inputNamePtrs),
	uintptrSlicePtr(inputValueHandles),
	uintptr(len(inputValueHandles)),
	uintptrSlicePtr(outputNamePtrs),
	uintptr(len(outputValueHandles)),
	uintptrSlicePtr(outputValueHandles),
)
runtime.KeepAlive(inputNameBackings)
runtime.KeepAlive(outputNameBackings)
runtime.KeepAlive(inputNamePtrs)
runtime.KeepAlive(outputNamePtrs)
runtime.KeepAlive(inputValueHandles)
runtime.KeepAlive(outputValueHandles)
if status != 0 {
	errMsg := getErrorMessage(status)
	releaseStatus(status)
	return fmt.Errorf("failed to run inference: %s", errMsg)
}

return nil
```

Create thin public entries for `Run()` and `RunWithValues(inputs, outputs []Value)`, then move this body into one private core.

Critical adaptation:

- acquire `s.runMu` before reading `s.inputValues`/`s.outputValues`,
- for `Run`, select constructor-bound values under that lock,
- for `RunWithValues`, validate supplied counts against fixed names under the same core,
- borrow values for the call only,
- retain every `runtime.KeepAlive` at lines 206-212,
- preserve `runMu → ortCallMu → mu → Tensor.runMu`.

**Deduplicated lease pattern** — lines 309-367:

```go
handles := make([]uintptr, len(values))
unlockFns := make([]func(), 0, len(values))
leasedLockables := make(map[any]int, len(values))
release := func() {
	for i := len(unlockFns) - 1; i >= 0; i-- {
		unlockFns[i]()
	}
}

for i, v := range values {
	if lockable, ok := v.(valueRunLockable); ok {
		key, keyOk := comparableIdentityKey(lockable)
		if !keyOk {
			release()
			return nil, noOpRelease, fmt.Errorf(
				"%s value at index %d is invalid: lockable value type %T must be comparable",
				role, i, v,
			)
		}
		if leasedIndex, exists := leasedLockables[key]; exists {
			handles[i] = handles[leasedIndex]
			continue
		}
		// acquire and remember lease...
	}
}
```

Reuse this function; do not implement handle conversion inside `RunWithValues`. After `Value` is sealed, remove only compatibility branches proven unnecessary by tests—do not change the lease protocol merely because external implementations are no longer possible.

Replace the three status blocks at lines 80-85, 98-106, and 196-217 with the central converter.

---

### `ort/session_test.go` (test, request-response/concurrency)

**Analog:** current session tests.

**Runtime callback seam** — lines 531-598:

```go
resetEnvironmentState()
defer resetEnvironmentState()

mu.Lock()
ortAPI = &OrtApi{}
runSessionFunc = func(
	session uintptr,
	runOptions uintptr,
	inputNames *uintptr,
	inputValues *uintptr,
	inputLen uintptr,
	outputNames *uintptr,
	outputLen uintptr,
	outputValues *uintptr,
) uintptr {
	atomic.AddInt32(&calls, 1)
	current := atomic.AddInt32(&inFlight, 1)
	// record max concurrency
	atomic.AddInt32(&inFlight, -1)
	return 0
}
mu.Unlock()
```

Use this seam to inspect that `RunWithValues` passes the supplied handles while `Run` still passes bound handles.

**Deterministic lifetime test pattern** — lines 680-780 and 864-956:

```go
runStarted := make(chan struct{})
allowRunReturn := make(chan struct{})

runSessionFunc = func(/* ... */) uintptr {
	close(runStarted)
	<-allowRunReturn
	record("run-returned")
	return 0
}

go func() { runErrCh <- session.Run() }()
<-runStarted

if session.runMu.TryLock() {
	session.runMu.Unlock()
	t.Fatal("expected session.runMu to be held")
}

go func() { destroyErrCh <- session.Destroy() }()
// prove blocking before allowing Run to return
close(allowRunReturn)
```

Clone this structure for `RunWithValues`:

- count mismatch rejects before the callback,
- supplied input/output handles reach the callback,
- `Destroy` waits for a borrowed tensor lease,
- same-session calls remain serialized,
- unrelated sessions/tensors remain independent,
- repeated values retain deduplication,
- bound `Run` behavior stays unchanged.

Convert lifecycle assertions from full strings/substrings to `errors.Is`; native failures use `errors.As` and field checks.

---

### `ort/memory.go` (model, resource CRUD)

**Closest analog:** `ort/tensor.go` resource construction/destruction.

**Current construction shape** — `ort/memory.go` lines 10-46:

```go
func CreateMemoryInfo(
	name string,
	allocatorType AllocatorType,
	deviceID int,
	memType MemType,
) (*MemoryInfo, error) {
	mu.Lock()
	defer mu.Unlock()

	if createMemoryInfoFunc == nil {
		return nil, fmt.Errorf("ONNX Runtime not initialized")
	}

	nameBytes, namePtr := GoToCstring(name)
	defer runtime.KeepAlive(nameBytes)

	var handle uintptr
	status := createMemoryInfoFunc(namePtr, allocatorType, int32(deviceID), memType, &handle)
	// status conversion...
}
```

Replace the status block at lines 24-29 with the central converter and wrap uninitialized/local validation states with sentinels. Preserve the existing mutex scope and `KeepAlive`.

**Destroy pattern** — lines 57-86:

```go
ortCallMu.RLock()
defer ortCallMu.RUnlock()

mu.Lock()
handle = m.handle
releaseMemoryInfo = releaseMemoryInfoFunc
m.handle = 0
runtime.SetFinalizer(m, nil)
mu.Unlock()

if handle == 0 {
	return nil
}
```

Keep destroy idempotent. A resource already cleared remains a no-op; an attempted operation on a destroyed resource should match `ErrDestroyed`.

---

### `ort/environment.go` (provider, runtime lifecycle)

**Analog:** current global registration/cleanup block.

**Function-pointer lifecycle pattern** — lines 39-68 and 150-161:

```go
getVersionStringFunc func() uintptr
getErrorMessageFunc  func(uintptr) uintptr
releaseStatusFunc    func(uintptr)

func clearORTGlobalsLocked() {
	ortAPI = nil
	ortEnv = 0
	getVersionStringFunc = nil
	getErrorMessageFunc = nil
	releaseStatusFunc = nil
	// ...
}

purego.RegisterFunc(&getErrorMessageFunc, ortAPI.GetErrorMessage)
purego.RegisterFunc(&releaseStatusFunc, ortAPI.ReleaseStatus)
```

Add `getErrorCodeFunc func(uintptr) ErrorCode` beside message/release, register it from `ortAPI.GetErrorCode`, and clear it in the same lifecycle block. Update `resetEnvironmentState` in lockstep.

Replace the environment status block at lines 185-191 with the central converter. Preserve cleanup joining at lines 107-124 and underlying load/symbol causes at lines 126-134.

Replace the runtime-version `log.Printf` at lines 163-176 with one warn diagnostic. It is a notice that cannot be returned, so it is an approved emitter call.

---

### `ort/environment_test.go` (test, runtime lifecycle)

**Analog:** `resetEnvironmentState`, lines 10-32:

```go
func resetEnvironmentState() {
	mu.Lock()
	defer mu.Unlock()
	refCount = 0
	ortLib = 0
	ortAPI = nil
	ortEnv = 0
	libPath = ""
	logLevel = LoggingLevelWarning
	getVersionStringFunc = nil
	getErrorMessageFunc = nil
	releaseStatusFunc = nil
	// remaining registered functions...
}
```

Add `getErrorCodeFunc = nil`. New status tests should install code/message/release callbacks together and restore all state through this helper.

Replace message-only integration assertions with:

```go
var ortErr *ORTError
if !errors.As(err, &ortErr) { /* fail */ }
```

then check operation, `ErrorCode`, copied message, and release count. Keep filesystem/dynamic-library causes reachable through `errors.Is`/`errors.As`.

---

### `ort/memory_test.go` (test, resource CRUD)

**Closest analog:** `ort/tensor_test.go` destroy accounting.

**Current memory lifecycle case** — `ort/memory_test.go` lines 187-209:

```go
resetEnvironmentState()
defer resetEnvironmentState()

memInfo := &MemoryInfo{handle: 123, name: "Cpu", memType: MemTypeCPU}

err := memInfo.Destroy()
if memInfo.handle != 0 {
	t.Fatalf("expected handle to be reset even on release failure")
}

if err := memInfo.Destroy(); err != nil {
	t.Fatalf("second destroy should be no-op, got: %v", err)
}
```

Keep the state-transition assertions but replace the substring-only category check with `errors.Is(err, ErrNotInitialized)` or `errors.Is(err, ErrDestroyed)` as appropriate. Add a fake nonzero status case asserting `errors.As(err, *ORTError)` and exactly one release.

---

### `ort/tensor_test.go` (test, resource CRUD/concurrency)

**Analog:** existing atomic release tests, lines 223-301:

```go
var releases int32
mu.Lock()
releaseValueFunc = func(handle uintptr) {
	atomic.AddInt32(&releases, 1)
}
mu.Unlock()

tensor := &Tensor[float32]{handle: 777, data: []float32{1, 2, 3}, shape: Shape{3}}

const workers = 16
start := make(chan struct{})
errCh := make(chan error, workers)
var wg sync.WaitGroup

for i := 0; i < workers; i++ {
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		errCh <- tensor.Destroy()
	}()
}
```

Reuse this exact accounting style for status release tests and preserve concurrent-destroy coverage. Validation tests at lines 152-177 currently use substrings; migrate category assertions to `errors.Is` while retaining only actionable text-fragment checks.

---

### `ort/bootstrap.go` (service, file-I/O/request-response)

**Analog:** current sentinel, wrapping, cleanup, and call-site behavior.

**Preserved-cause pattern** — lines 258-279:

```go
if err := os.MkdirAll(cfg.cacheDir, secureDirectoryPermission); err != nil {
	return "", fmt.Errorf(
		"failed to create bootstrap cache directory %q: %w",
		cfg.cacheDir,
		err,
	)
}

// ...
if resolveErr != nil {
	return fmt.Errorf(
		"bootstrap completed but shared library could not be resolved: %w",
		resolveErr,
	)
}
```

**Multiple cleanup causes** — lines 997-1020:

```go
defer func() {
	if closeErr := archiveFile.Close(); closeErr != nil {
		closeErr = fmt.Errorf("failed to close archive %q: %w", archivePath, closeErr)
		if err == nil {
			err = closeErr
		} else {
			err = errors.Join(err, closeErr)
		}
	}
}()
```

Extend these established patterns rather than introducing a bootstrap-specific error framework. Promote `errSharedLibraryNotFound` only if callers need the category; keep retry-control sentinels private.

**Diagnostic migration audit:**

| Existing call site | Existing lines | Level | Structured intent |
|---|---:|---|---|
| Temporary archive removal failed | 554-557 | Warn | `operation`, `path`, `error` |
| Download proceeded without checksum verification | 563-570 | Warn | redacted `url`, observed checksum |
| Staging cleanup failed | 580-583 | Warn | `operation`, `path`, `error` |
| Metadata lookup failed; pinned checksum fallback used | 657-667 | Warn | `operation`, `error` |
| Archive glob matching failed | 1082-1085, 1143-1147 | Warn | `archive_entry`, `library_glob`, `error` |
| Unsafe/unsupported archive entry skipped | 1092-1096, 1154 | Warn | `archive_entry`, `entry_type` |
| Bootstrap lock still waiting | 1327-1330 | Info | `path`, `wait_duration` |
| User-cache fallback selected | 1383-1390 | Warn | `path`, optional `error` |

Use `URL.Redacted()` or an already redacted string for URL attributes. Never attach tokens, authorization headers, or raw credential environment values.

Do not emit on paths that return the same error. The existing non-returnable cleanup/fallback notices above are approved; returned failures stay in the error chain only.

---

### `ort/finalizer_log.go` (utility, event-driven)

**Current safety analog** — lines 3-14:

```go
func logFinalizerWarning(format string, args ...any) {
	defer func() {
		if r := recover(); r != nil {
			_ = r
		}
	}()
	log.Printf(format, args...)
}
```

Preserve the recovery boundary because finalizers run during best-effort teardown and a consumer handler can be arbitrary code. Replace formatted logging with a warn-level call to `emitDiagnostic`, using structured `resource` and `error` attributes.

Minimal-change options:

1. keep this helper, change it to structured arguments, and route it through `emitDiagnostic`; or
2. move the recovery guard into a finalizer-specific diagnostic helper and remove this file.

Do not make the general emitter a formatting facade.

Apply the helper to finalizer callers in:

- `ort/session.go` lines 116-120,
- `ort/tensor.go` lines 131-136,
- `ort/memory.go` lines 39-44.

---

### `ort/bootstrap_test.go` (test, file-I/O/event-driven)

**Sentinel assertion pattern** — lines 110-121:

```go
got, err := resolveRuntimeArtifact(tc.goos, tc.goarch)
if tc.wantErr {
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if tc.wantUnsupportedPlatform {
		if !errors.Is(err, ErrUnsupportedPlatform) {
			t.Fatalf("expected ErrUnsupportedPlatform, got: %v", err)
		}
	}
	return
}
```

Use this style for invalid argument/shared-library categories. Do not compare complete error strings.

**Timing-state cleanup pattern** — lines 1477-1520:

```go
oldTimeout := bootstrapLockAcquireTimeout
oldRetry := bootstrapLockRetryInterval
oldLogInterval := bootstrapLockLogInterval
bootstrapLockAcquireTimeout = 80 * time.Millisecond
bootstrapLockRetryInterval = 5 * time.Millisecond
bootstrapLockLogInterval = 15 * time.Millisecond
t.Cleanup(func() {
	bootstrapLockAcquireTimeout = oldTimeout
	bootstrapLockRetryInterval = oldRetry
	bootstrapLockLogInterval = oldLogInterval
})
```

Install a recording `slog.Handler` with `t.Cleanup(func() { SetDiagnosticHandler(nil) })` around approved call sites. Assert:

- the lock wait emits Info with a duration/path,
- fallback and skipped-entry notices emit Warn,
- returned bootstrap errors emit zero diagnostic records,
- URL attributes are redacted,
- global timing/logger state is restored.

**Permission regression pattern:** Add the exact top-level `TestBootstrapCreatedFilePermissions` in `ort/bootstrap_test.go`. Keep it cross-platform-compilable and immediately `t.Skip` on `runtime.GOOS == "windows"` because Windows ACLs do not provide portable POSIX mode semantics. Unix subtests must exercise production helpers with fresh paths and synthetic TGZ/ZIP library entries, then inspect `os.FileMode.Perm()` for:

- bootstrap-created cache/install/lock directories: owner-accessible, no group-write or other-user bits;
- final installed library files: required owner read/execute retained, no group/other write bits, including when the archive supplies permissive write bits;
- lock files: owner read/write, no group/other bits.

If permissive archive input exposes unsafe write bits, clamp only group/other write permissions in both tar and ZIP regular-file creation while retaining executable bits. Do not apply recursive chmod or broaden an existing mode. The anchored verification command is `go test -short ./ort -run '^TestBootstrapCreatedFilePermissions$'`.

---

### `.github/workflows/ci.yml` (config, batch)

**Targeted race lane analog** — lines 111-130:

```yaml
test-race-ort-concurrency:
  name: Test Race (ORT concurrency subset)
  runs-on: ubuntu-latest
  # ...
  - name: Run race detector on ORT concurrency tests
    run: |
      go test -race ./ort -run '^(TestValuesToHandlesDeduplicatesRepeatedLockableValue|TestValuesToHandlesReleasesPriorLeasesOnError|TestAdvancedSessionRunConcurrent|TestAdvancedSessionRunConcurrentAcrossSessionsSharingTensor|TestAdvancedSessionRunAndDestroyConcurrent|TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun|TestTensorDestroyWaitsForInFlightRun|TestTensorDestroyDoesNotBlockUnrelatedRun|TestTensorDestroyConcurrentCallsReleaseOnce)$'
```

Append fake-status, diagnostic reconfiguration, and `RunWithValues` concurrency tests to this lane.

**Non-race native lane analog** — lines 205-223:

```yaml
- name: Download ONNX Runtime shared library
  run: |
    # ...
    echo "ONNXRUNTIME_LIB_PATH=${RUNNER_TEMP}/onnxruntime/lib/libonnxruntime.so" >> "$GITHUB_ENV"

- name: Run ort real-model integration tests
  run: |
    go test -v ./ort/... -run '^(TestAdvancedSessionRunWithAllMiniLML6V2|TestAdvancedSessionRunWithAllMiniLML6V2MemoryStability)$'
```

Add `TestNativeORTStatusRoundTrip` and the real-model `RunWithValues` case here without `-race`. Keep the fake callback ownership proof in the race lane. Never add checkptr-disabling flags.

**Fast workflow and supply-chain assertions:** Task 02-08-02 should verify only the anchored selector text and workflow-input invariants during its under-30-second feedback loop. Before committing the workflow edit, require:

```bash
test -z "$(git diff HEAD --unified=0 -- .github/workflows/ci.yml |
  sed -n '/^[+-][[:space:]]*uses:/p')"
```

This audits only added/removed `uses:` lines, so the intended `run:` selector edits remain allowed while action references cannot drift. Apply the same focused diff check to `continue-on-error` so Phase 2 does not take ownership of Phase 5's enforcing lint change.

Run the full short, targeted race, compile-only, configured native, and vet commands at wave/phase scope. For Phase 2 lint feedback, use the existing `make precommit-lint-new` target from `Makefile`; historical full-tree lint debt cleanup and removal of lint `continue-on-error` remain Phase 5 / CLN-01.

## Shared Patterns

### Lock order and ownership

**Source:** `ort/environment.go` lines 23-30  
**Apply to:** `ort/session.go`, `ort/tensor.go`, `ort/memory.go`, `ort/environment.go`

```go
// Lock hierarchy across ORT lifecycle and calls:
// 1) AdvancedSession.runMu
// 2) ortCallMu
// 3) mu
// 4) Tensor.runMu
//
// Keep this order to avoid deadlocks.
```

`RunWithValues` changes which values are selected, not the concurrency protocol. Values are borrowed only until the native call and all `KeepAlive` barriers complete.

### Native status conversion

**Source:** `.planning/spikes/001-ort-status-lifetime/status_lifetime_test.go` lines 31-45  
**Apply to:** all seven current native status blocks

| File | Existing blocks |
|---|---|
| `ort/environment.go` | lines 185-191 |
| `ort/memory.go` | lines 24-29 |
| `ort/tensor.go` | lines 88-94 and 111-122 |
| `ort/session.go` | lines 80-85, 98-106, and 196-217 |

After migration, production call sites should not mention `getErrorMessage(status)` or `releaseStatus(status)`.

### Local category vs native detail

**Sources:** `ort/bootstrap.go` lines 49-56, `.planning/spikes/001-ort-status-lifetime/status_lifetime_test.go` lines 13-23  
**Apply to:** all production and test files

- Local validation/lifecycle failures: wrap a lean public sentinel and inspect with `errors.Is`.
- Native ONNX Runtime failures: return/wrap `*ORTError` and inspect with `errors.As`.
- OS/filesystem/network causes: preserve with `%w`.
- Primary plus independent cleanup failure: preserve with `errors.Join`.
- Exact English text is not a compatibility contract.

### Diagnostics policy

**Source:** `.planning/spikes/002-b-slog-handler-sink/diagnostic.go` lines 21-37  
**Apply to:** `ort/environment.go`, `ort/bootstrap.go`, finalizer callers

- Silent by default via `slog.DiscardHandler`.
- Configuration works before ORT initialization.
- Emit through `*slog.Logger.LogAttrs`.
- Use Info and Warn unless a concrete need proves another level.
- Emit only notices/failures that cannot be returned.
- Never emit an error that is also returned.

### Test state isolation

**Source:** `ort/environment_test.go` lines 10-32  
**Apply to:** all tests that replace ORT callbacks or process-global diagnostic state

Reset every global function pointer, including new `getErrorCodeFunc`, and restore the diagnostic handler to nil with `t.Cleanup`. Tests that mutate global state must not use `t.Parallel`.

### No authentication/authorization pattern

This is an in-process library API. There is no auth middleware or guard analog. The relevant boundaries are sealed values, validation before FFI, resource ownership, and lock/lifetime guards.

## No Exact Semantic Analog

Every planned file has a structural analog, but two additive behaviors have no existing production implementation to copy verbatim:

| File | Delta without exact production analog | Planner source |
|---|---|---|
| `ort/types.go` / `ort/value_test.go` | Generic exact-type `AsTensor[T]` and `IsTensor` behavior | `02-RESEARCH.md` Pattern 1; use a direct type assertion only |
| `ort/session.go` / `ort/session_test.go` | Public per-call `RunWithValues` entry and bound-vs-supplied selection | `02-RESEARCH.md` Pattern 2; reuse the current `Run` core and lease machinery |

These are partial semantic gaps, not invitations for a new abstraction. The research-provided delta should be applied to the existing analogs above.

## Planner Guardrails

1. Keep `Run()` source-compatible and route it through the same private core as `RunWithValues`.
2. Hold `runMu` before selecting constructor-bound slices; `Destroy` clears them under that lock.
3. Seal `Value`, but keep raw handles and leases package-private.
4. Do not add runtime-allocated outputs, numeric coercion, sequence/map/optional implementations, or a logging framework.
5. Centralize status ownership once and prove exactly one release with both fake/race and real/non-race layers.
6. Audit all 14 direct `log.Printf` sites plus three finalizer callers; migrate only non-returnable notices.
7. Preserve existing examples and all three embedder hot paths as compatibility tests rather than editing them.
8. No dependency or `go.mod` change is needed.
9. Anchor `TestBootstrapCreatedFilePermissions` in Plan 07 and canonical T-02-10 evidence, with Unix mode assertions and a Windows-safe skip.
10. Keep Task 02-08-02 feedback static/focused; run comprehensive suites at wave/phase scope, use `make precommit-lint-new`, and leave the full-tree lint gate to Phase 5.
11. Treat unchanged `.github/workflows/ci.yml` `uses:` lines as T-02-SC evidence alongside unchanged `go.mod`/`go.sum`.

## Metadata

**Analog search scope:** `ort/`, `.planning/spikes/`, `.github/workflows/`  
**Candidates indexed:** 58 Go/workflow files  
**Code/test/workflow files inspected:** 18  
**Primary analog families inspected:** 5 (session/value leases, resource lifecycle, bootstrap errors, status spike, diagnostics spike/CI lanes)  
**Pattern extraction date:** 2026-07-23
