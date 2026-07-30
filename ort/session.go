package ort

import (
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"sort"
	"sync"
	"unsafe"
)

// AdvancedSession represents an ONNX Runtime inference session
type AdvancedSession struct {
	handle       uintptr
	inputNames   []string
	outputNames  []string
	inputValues  []Value
	outputValues []Value
	runMu        sync.Mutex // Guards all fields above and serializes Run with Destroy.
}

// NewSessionOptions creates a native ONNX Runtime session-options object.
func NewSessionOptions() (*SessionOptions, error) {
	ortCallMu.RLock()
	defer ortCallMu.RUnlock()

	mu.Lock()
	if ortAPI == nil ||
		createSessionOptionsFunc == nil ||
		releaseSessionOptionsFunc == nil ||
		getErrorCodeFunc == nil ||
		getErrorMessageFunc == nil ||
		releaseStatusFunc == nil {
		mu.Unlock()
		return nil, fmt.Errorf("create session options: required ONNX Runtime functions are unavailable: %w", ErrNotInitialized)
	}
	createSessionOptions := createSessionOptionsFunc
	mu.Unlock()

	var handle uintptr
	if err := statusToError(createSessionOptions(&handle), "create session options"); err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	if handle == 0 {
		return nil, fmt.Errorf("create session options returned a nil handle: %w", ErrNativeContract)
	}

	options := &SessionOptions{handle: handle}
	runtime.SetFinalizer(options, finalizeSessionOptions)
	return options, nil
}

func finalizeSessionOptions(options *SessionOptions) {
	if err := options.Destroy(); err != nil {
		emitFinalizerDiagnostic("session_options", err)
	}
}

// Destroy releases the native session-options object. Repeated calls are safe.
func (o *SessionOptions) Destroy() error {
	if o == nil {
		return nil
	}

	ortCallMu.RLock()
	defer ortCallMu.RUnlock()

	mu.Lock()
	releaseSessionOptions := releaseSessionOptionsFunc
	mu.Unlock()

	o.handleMu.Lock()
	handle := o.handle
	if handle != 0 {
		o.destroyed = true
	}
	o.handle = 0
	runtime.SetFinalizer(o, nil)
	o.handleMu.Unlock()

	if handle == 0 {
		return nil
	}
	if releaseSessionOptions == nil {
		return fmt.Errorf("cannot destroy session options: ONNX Runtime release function unavailable: %w", ErrNotInitialized)
	}
	releaseSessionOptions(handle)
	return nil
}

// IsValid reports whether the session options still own a native handle.
func (o *SessionOptions) IsValid() bool {
	if o == nil {
		return false
	}
	o.handleMu.RLock()
	valid := o.handle != 0
	o.handleMu.RUnlock()
	return valid
}

// NewAdvancedSession creates a new session with specified inputs and outputs.
// Callers retain ownership of input/output values and must keep them alive.
// Values must not be Destroy()'d while this session may still Run().
// If a value is destroyed early, Run() returns a "...value at index N has been destroyed" error.
func NewAdvancedSession(modelPath string, inputNames []string, outputNames []string,
	inputValues []Value, outputValues []Value, options *SessionOptions) (*AdvancedSession, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path cannot be empty: %w", ErrInvalidArgument)
	}
	if len(inputNames) == 0 {
		return nil, fmt.Errorf("at least one input name is required: %w", ErrInvalidArgument)
	}
	if len(outputNames) == 0 {
		return nil, fmt.Errorf("at least one output name is required: %w", ErrInvalidArgument)
	}
	if err := validateNativeString(modelPath, "model path"); err != nil {
		return nil, err
	}
	for i, name := range inputNames {
		if err := validateNativeString(name, fmt.Sprintf("input name at index %d", i)); err != nil {
			return nil, err
		}
	}
	for i, name := range outputNames {
		if err := validateNativeString(name, fmt.Sprintf("output name at index %d", i)); err != nil {
			return nil, err
		}
	}
	if len(inputNames) != len(inputValues) {
		return nil, fmt.Errorf("input names/values count mismatch: got %d names and %d values: %w", len(inputNames), len(inputValues), ErrInvalidArgument)
	}
	if len(outputNames) != len(outputValues) {
		return nil, fmt.Errorf("output names/values count mismatch: got %d names and %d values: %w", len(outputNames), len(outputValues), ErrInvalidArgument)
	}
	ortCallMu.RLock()
	defer ortCallMu.RUnlock()

	sessionOptionsHandle := uintptr(0)
	if options != nil {
		options.handleMu.RLock()
		defer options.handleMu.RUnlock()
		sessionOptionsHandle = options.handle
		if sessionOptionsHandle == 0 {
			if options.destroyed {
				return nil, fmt.Errorf("session options have been destroyed: %w", ErrDestroyed)
			}
			return nil, fmt.Errorf("session options handle is not initialized: %w", ErrInvalidArgument)
		}
	}

	// validateSessionValue is a value-local synchronized check; it does not lease the handle.
	// Run obtains the native-use leases after it has finished all call setup.
	for i, v := range inputValues {
		if err := validateSessionValue(v, "input", i); err != nil {
			return nil, err
		}
	}
	for i, v := range outputValues {
		if err := validateSessionValue(v, "output", i); err != nil {
			return nil, err
		}
	}

	mu.Lock()
	// Safe to snapshot under mu here because ortCallMu.RLock is already held.
	// DestroyEnvironment takes ortCallMu.Lock before it can nil these globals.
	if ortAPI == nil || ortEnv == 0 || createSessionOptionsFunc == nil || releaseSessionOptionsFunc == nil || createSessionFunc == nil {
		mu.Unlock()
		return nil, fmt.Errorf("ONNX Runtime not initialized: %w", ErrNotInitialized)
	}
	envHandle := ortEnv
	createSessionOptions := createSessionOptionsFunc
	releaseSessionOptions := releaseSessionOptionsFunc
	createSession := createSessionFunc
	mu.Unlock()

	releaseCreatedOptions := false
	if options == nil {
		status := createSessionOptions(&sessionOptionsHandle)
		if status != 0 {
			return nil, fmt.Errorf(
				"failed to create session options: %w",
				statusToError(status, "create session options"),
			)
		}
		if sessionOptionsHandle == 0 {
			return nil, fmt.Errorf(
				"create session options returned a nil handle: %w",
				ErrNativeContract,
			)
		}
		releaseCreatedOptions = true
	}
	if releaseCreatedOptions {
		defer releaseSessionOptions(sessionOptionsHandle)
	}

	modelPathPtr, modelPathBacking, err := goStringToORTChar(modelPath)
	if err != nil {
		return nil, err
	}

	var sessionHandle uintptr
	status := createSession(envHandle, modelPathPtr, sessionOptionsHandle, &sessionHandle)
	// modelPathBacking owns the native char buffer returned by goStringToORTChar.
	// Keep it alive until createSession returns.
	runtime.KeepAlive(modelPathBacking)
	if status != 0 {
		return nil, fmt.Errorf(
			"failed to create session: %w",
			statusToError(status, "create session"),
		)
	}
	if sessionHandle == 0 {
		return nil, fmt.Errorf(
			"create session returned a nil handle: %w",
			ErrNativeContract,
		)
	}

	session := &AdvancedSession{
		handle:       sessionHandle,
		inputNames:   cloneStringSlice(inputNames),
		outputNames:  cloneStringSlice(outputNames),
		inputValues:  cloneValueSlice(inputValues),
		outputValues: cloneValueSlice(outputValues),
	}

	runtime.SetFinalizer(session, finalizeAdvancedSession)

	return session, nil
}

func finalizeAdvancedSession(session *AdvancedSession) {
	if err := session.Destroy(); err != nil {
		emitFinalizerDiagnostic("session", err)
	}
}

// Run executes inference with the input and output values bound at construction.
func (s *AdvancedSession) Run() error {
	return s.run(nil, nil, true)
}

// RunWithValues executes inference with caller-supplied input and output values.
// NewAdvancedSession still requires bound values at construction; this method leaves
// those values unchanged so later Run calls continue to use the original bindings.
func (s *AdvancedSession) RunWithValues(inputs, outputs []Value) error {
	return s.run(inputs, outputs, false)
}

func (s *AdvancedSession) run(inputs, outputs []Value, useBoundValues bool) error {
	if s == nil {
		return fmt.Errorf("session is nil: %w", ErrInvalidArgument)
	}

	// Lock order here is runMu -> ortCallMu -> mu.
	s.runMu.Lock()
	defer s.runMu.Unlock()

	// Holding ortCallMu RLock keeps DestroyEnvironment() from closing the runtime
	// while raw pointers are passed into ORT.
	ortCallMu.RLock()
	defer ortCallMu.RUnlock()

	var (
		sessionHandle uintptr
		inputNames    []string
		outputNames   []string
		inputValues   []Value
		outputValues  []Value
		run           func(session uintptr, runOptions uintptr, inputNames *uintptr, inputValues *uintptr, inputLen uintptr, outputNames *uintptr, outputLen uintptr, outputValues *uintptr) uintptr
	)

	// Session-owned fields are guarded by runMu.
	if s.handle == 0 {
		return fmt.Errorf("session has been destroyed: %w", ErrDestroyed)
	}
	if len(s.inputNames) == 0 || len(s.outputNames) == 0 {
		return fmt.Errorf("session is missing input/output names: %w", ErrInvalidArgument)
	}
	if useBoundValues {
		inputs = s.inputValues
		outputs = s.outputValues
	}
	if len(s.inputNames) != len(inputs) {
		return fmt.Errorf("session input names/values count mismatch: got %d names and %d values: %w", len(s.inputNames), len(inputs), ErrInvalidArgument)
	}
	if len(s.outputNames) != len(outputs) {
		return fmt.Errorf("session output names/values count mismatch: got %d names and %d values: %w", len(s.outputNames), len(outputs), ErrInvalidArgument)
	}
	sessionHandle = s.handle
	inputNames = s.inputNames
	outputNames = s.outputNames
	inputValues = inputs
	outputValues = outputs

	// Global runtime pointers/functions are guarded by mu.
	// Safe to snapshot under mu here because ortCallMu.RLock is already held.
	// DestroyEnvironment takes ortCallMu.Lock before it can nil these globals.
	mu.Lock()
	if ortAPI == nil || runSessionFunc == nil {
		mu.Unlock()
		return fmt.Errorf("ONNX Runtime not initialized: %w", ErrNotInitialized)
	}
	run = runSessionFunc
	mu.Unlock()

	inputNameBackings, inputNamePtrs, err := makeCStringPointerArray(inputNames, "input name")
	if err != nil {
		return err
	}
	outputNameBackings, outputNamePtrs, err := makeCStringPointerArray(outputNames, "output name")
	if err != nil {
		return err
	}

	// acquireUniqueValueLeases prevents handle release during native Run.
	valueLeases, err := acquireUniqueValueLeases(inputValues, outputValues)
	if err != nil {
		return err
	}
	defer valueLeases.Release()

	inputValueHandles, err := handlesFromLeasedValues(inputValues, "input", valueLeases)
	if err != nil {
		return err
	}
	outputValueHandles, err := handlesFromLeasedValues(outputValues, "output", valueLeases)
	if err != nil {
		return err
	}

	status := run(
		sessionHandle,
		0, // RunOptions not yet implemented
		uintptrSlicePtr(inputNamePtrs),
		uintptrSlicePtr(inputValueHandles),
		uintptr(len(inputValueHandles)),
		uintptrSlicePtr(outputNamePtrs),
		uintptr(len(outputValueHandles)),
		uintptrSlicePtr(outputValueHandles),
	)
	// Keep backing slices alive until ORT returns because runSessionFunc receives raw pointers into them.
	runtime.KeepAlive(inputNameBackings)
	runtime.KeepAlive(outputNameBackings)
	runtime.KeepAlive(inputNamePtrs)
	runtime.KeepAlive(outputNamePtrs)
	runtime.KeepAlive(inputValueHandles)
	runtime.KeepAlive(outputValueHandles)
	runtime.KeepAlive(inputValues)
	runtime.KeepAlive(outputValues)
	if status != 0 {
		return fmt.Errorf(
			"failed to run inference: %w",
			statusToError(status, "run inference"),
		)
	}

	return nil
}

// Destroy releases the session resources
func (s *AdvancedSession) Destroy() error {
	if s == nil {
		return nil
	}

	// Lock order here is runMu -> ortCallMu -> mu.
	// runMu prevents overlap with Run() on this same session.
	// ortCallMu.RLock keeps environment teardown from closing the runtime while
	// this release call is in flight, without stalling unrelated session runs.
	s.runMu.Lock()
	defer s.runMu.Unlock()

	ortCallMu.RLock()
	defer ortCallMu.RUnlock()

	mu.Lock()
	releaseSession := releaseSessionFunc
	mu.Unlock()

	handle := s.handle
	s.handle = 0
	s.inputNames = nil
	s.outputNames = nil
	s.inputValues = nil
	s.outputValues = nil
	runtime.SetFinalizer(s, nil)

	if handle != 0 && releaseSession != nil {
		releaseSession(handle)
	} else if handle != 0 {
		return fmt.Errorf("cannot destroy session: ONNX Runtime release function unavailable (environment may already be destroyed); ensure all tensors and sessions are destroyed before calling DestroyEnvironment(): %w", ErrNotInitialized)
	}

	return nil
}

// valueWithORTHandle is intentionally package-private.
// Today, sessions only support Value implementations created by this package.
type valueWithORTHandle interface {
	ortValueHandle() uintptr
}

// valueRunLockable values can provide a stable handle lease for the whole Run() call.
// This lets Destroy() wait only on sessions currently using that specific value.
// Implementations must be comparable so repeated values can be deduplicated safely.
type valueRunLockable interface {
	lockForRun() (uintptr, error)
	unlockForRun()
}

var (
	errValueNil         = errors.New("value is nil")
	errValueDestroyed   = errors.New("value has been destroyed")
	errValueUnsupported = errors.New("unsupported value implementation")
)

func valueHandle(v Value) (uintptr, error) {
	if v == nil {
		return 0, errValueNil
	}
	handleProvider, ok := v.(valueWithORTHandle)
	if !ok {
		return 0, fmt.Errorf("%w %T", errValueUnsupported, v)
	}
	handle := handleProvider.ortValueHandle()
	if handle == 0 {
		return 0, errValueDestroyed
	}
	return handle, nil
}

func validateSessionValue(v Value, role string, index int) error {
	_, err := valueHandle(v)
	if err == nil {
		return nil
	}
	if errors.Is(err, errValueDestroyed) {
		return fmt.Errorf("%s value at index %d has been destroyed: %w", role, index, ErrDestroyed)
	}
	return fmt.Errorf("invalid %s value at index %d: %v: %w", role, index, err, ErrInvalidArgument)
}

type valueRoleValues struct {
	role   string
	values []Value
}

type valueLeaseCandidate struct {
	key       any
	orderKey  uintptr
	lockable  valueRunLockable
	role      string
	roleIndex int
}

type valueLeaseSet struct {
	handles   map[any]uintptr
	unlockFns []func()
}

func (l *valueLeaseSet) Release() {
	if l == nil {
		return
	}
	for i := len(l.unlockFns) - 1; i >= 0; i-- {
		l.unlockFns[i]()
	}
	l.unlockFns = nil
}

func acquireUniqueValueLeases(inputValues, outputValues []Value) (*valueLeaseSet, error) {
	return acquireValueLeases(
		valueRoleValues{role: "input", values: inputValues},
		valueRoleValues{role: "output", values: outputValues},
	)
}

func acquireValueLeases(groups ...valueRoleValues) (*valueLeaseSet, error) {
	leases := &valueLeaseSet{handles: make(map[any]uintptr)}
	candidates := make([]valueLeaseCandidate, 0)
	seen := make(map[any]struct{})

	for _, group := range groups {
		for i, v := range group.values {
			lockable, ok := v.(valueRunLockable)
			if !ok {
				if _, err := valueHandle(v); err != nil {
					return nil, sessionValueLeaseError(group.role, i, err)
				}
				continue
			}

			key, keyOk := comparableIdentityKey(lockable)
			if !keyOk {
				return nil, fmt.Errorf("%s value at index %d is invalid: lockable value type %T must be comparable: %w", group.role, i, v, ErrInvalidArgument)
			}
			if _, exists := seen[key]; exists {
				continue
			}
			orderKey, orderKeyOK := valueLeaseOrderKey(lockable)
			if !orderKeyOK {
				return nil, fmt.Errorf("%s value at index %d is invalid: lockable value type %T must have pointer identity: %w", group.role, i, v, ErrInvalidArgument)
			}
			seen[key] = struct{}{}
			candidates = append(candidates, valueLeaseCandidate{
				key:       key,
				orderKey:  orderKey,
				lockable:  lockable,
				role:      group.role,
				roleIndex: i,
			})
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].orderKey != candidates[j].orderKey {
			return candidates[i].orderKey < candidates[j].orderKey
		}
		return reflect.TypeOf(candidates[i].lockable).String() <
			reflect.TypeOf(candidates[j].lockable).String()
	})

	for _, candidate := range candidates {
		handle, err := candidate.lockable.lockForRun()
		if err != nil {
			leases.Release()
			return nil, sessionValueLeaseError(candidate.role, candidate.roleIndex, err)
		}
		leases.handles[candidate.key] = handle
		leases.unlockFns = append(leases.unlockFns, candidate.lockable.unlockForRun)
	}

	return leases, nil
}

func handlesFromLeasedValues(values []Value, role string, leases *valueLeaseSet) ([]uintptr, error) {
	if len(values) == 0 {
		return nil, nil
	}

	handles := make([]uintptr, len(values))
	for i, v := range values {
		if lockable, ok := v.(valueRunLockable); ok {
			key, keyOK := comparableIdentityKey(lockable)
			if !keyOK {
				return nil, fmt.Errorf("%s value at index %d is invalid: lockable value type %T must be comparable: %w", role, i, v, ErrInvalidArgument)
			}
			handle, exists := leases.handles[key]
			if !exists {
				return nil, fmt.Errorf("%s value at index %d has no active run lease: %w", role, i, ErrInvalidArgument)
			}
			handles[i] = handle
			continue
		}

		handle, err := valueHandle(v)
		if err != nil {
			return nil, sessionValueLeaseError(role, i, err)
		}
		handles[i] = handle
	}
	return handles, nil
}

func sessionValueLeaseError(role string, index int, err error) error {
	if errors.Is(err, errValueDestroyed) {
		return fmt.Errorf("%s value at index %d has been destroyed: %w", role, index, ErrDestroyed)
	}
	return fmt.Errorf("%s value at index %d is invalid: %v: %w", role, index, err, ErrInvalidArgument)
}

func comparableIdentityKey(v any) (any, bool) {
	if v == nil {
		return nil, false
	}
	t := reflect.TypeOf(v)
	if !t.Comparable() {
		return nil, false
	}
	return v, true
}

func valueLeaseOrderKey(v any) (uintptr, bool) {
	value := reflect.ValueOf(v)
	switch value.Kind() {
	case reflect.Chan, reflect.Pointer, reflect.UnsafePointer:
		return value.Pointer(), true
	default:
		return 0, false
	}
}

func cloneStringSlice(input []string) []string {
	if len(input) == 0 {
		// Use nil for optional string collections when there are no entries.
		return nil
	}
	out := make([]string, len(input))
	copy(out, input)
	return out
}

func cloneValueSlice(input []Value) []Value {
	if len(input) == 0 {
		return nil
	}
	out := make([]Value, len(input))
	copy(out, input)
	return out
}

func makeCStringPointerArray(values []string, field string) ([][]byte, []uintptr, error) {
	if len(values) == 0 {
		return nil, nil, nil
	}

	backings := make([][]byte, len(values))
	ptrs := make([]uintptr, len(values))
	for i, value := range values {
		bytes, ptr, err := goStringToCString(value, fmt.Sprintf("%s at index %d", field, i))
		if err != nil {
			return nil, nil, err
		}
		backings[i] = bytes
		ptrs[i] = ptr
	}
	return backings, ptrs, nil
}

func uintptrSlicePtr(values []uintptr) *uintptr {
	if len(values) == 0 {
		return nil
	}
	// The returned pointer aliases values' backing array.
	// Callers must KeepAlive(values) until ORT returns.
	// #nosec G103 -- Required for CGO-free FFI to pass pointer arrays to ONNX Runtime C API.
	return (*uintptr)(unsafe.Pointer(unsafe.SliceData(values)))
}
