package ort

import (
	"errors"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type fakeValue struct {
	handle uintptr
}

func (f *fakeValue) Destroy() error          { return nil }
func (f *fakeValue) Type() ValueType         { return ValueTypeTensor }
func (f *fakeValue) ortValue()               {}
func (f *fakeValue) ortValueHandle() uintptr { return f.handle }
func (f *fakeValue) lockForRun() (uintptr, error) {
	if f.handle == 0 {
		return 0, errValueDestroyed
	}
	return f.handle, nil
}
func (f *fakeValue) unlockForRun() {}

type unsupportedValue struct{}

func (u *unsupportedValue) Destroy() error  { return nil }
func (u *unsupportedValue) Type() ValueType { return ValueTypeTensor }
func (u *unsupportedValue) ortValue()       {}

type blockingLeaseValue struct {
	handle uintptr
	runMu  sync.RWMutex

	firstLeaseAcquired    chan struct{}
	allowFirstLeaseReturn chan struct{}
	firstLeaseOnce        sync.Once
}

func newBlockingLeaseValue(handle uintptr) *blockingLeaseValue {
	return &blockingLeaseValue{
		handle:                handle,
		firstLeaseAcquired:    make(chan struct{}),
		allowFirstLeaseReturn: make(chan struct{}),
	}
}

func (v *blockingLeaseValue) Destroy() error {
	v.runMu.Lock()
	v.handle = 0
	v.runMu.Unlock()
	return nil
}

func (v *blockingLeaseValue) Type() ValueType { return ValueTypeTensor }
func (v *blockingLeaseValue) ortValue()       {}

func (v *blockingLeaseValue) ortValueHandle() uintptr {
	v.runMu.RLock()
	handle := v.handle
	v.runMu.RUnlock()
	return handle
}

func (v *blockingLeaseValue) lockForRun() (uintptr, error) {
	v.runMu.RLock()
	handle := v.handle
	if handle == 0 {
		v.runMu.RUnlock()
		return 0, errValueDestroyed
	}

	blockFirstLease := false
	v.firstLeaseOnce.Do(func() {
		blockFirstLease = true
		close(v.firstLeaseAcquired)
	})
	if blockFirstLease {
		<-v.allowFirstLeaseReturn
	}

	return handle, nil
}

func (v *blockingLeaseValue) unlockForRun() {
	v.runMu.RUnlock()
}

type nonComparableLeaseValue struct {
	handle  uintptr
	payload []int
}

func (v nonComparableLeaseValue) Destroy() error          { return nil }
func (v nonComparableLeaseValue) Type() ValueType         { return ValueTypeTensor }
func (v nonComparableLeaseValue) ortValue()               {}
func (v nonComparableLeaseValue) ortValueHandle() uintptr { return v.handle }
func (v nonComparableLeaseValue) lockForRun() (uintptr, error) {
	if v.handle == 0 {
		return 0, errValueDestroyed
	}
	return v.handle, nil
}
func (v nonComparableLeaseValue) unlockForRun() {}

type countingLeaseValue struct {
	handle      uintptr
	runMu       sync.RWMutex
	lockCalls   atomic.Int32
	unlockCalls atomic.Int32
	released    chan<- uintptr
}

func (v *countingLeaseValue) Destroy() error {
	v.runMu.Lock()
	v.handle = 0
	v.runMu.Unlock()
	return nil
}

func (v *countingLeaseValue) Type() ValueType { return ValueTypeTensor }
func (v *countingLeaseValue) ortValue()       {}
func (v *countingLeaseValue) ortValueHandle() uintptr {
	v.runMu.RLock()
	defer v.runMu.RUnlock()
	return v.handle
}
func (v *countingLeaseValue) lockForRun() (uintptr, error) {
	v.runMu.RLock()
	if v.handle == 0 {
		v.runMu.RUnlock()
		return 0, errValueDestroyed
	}
	v.lockCalls.Add(1)
	return v.handle, nil
}
func (v *countingLeaseValue) unlockForRun() {
	v.unlockCalls.Add(1)
	if v.released != nil {
		v.released <- v.handle
	}
	v.runMu.RUnlock()
}

func TestValuesToHandlesDeduplicatesRepeatedLockableValue(t *testing.T) {
	value := newBlockingLeaseValue(42)

	type valuesToHandlesResult struct {
		handles []uintptr
		release func()
		err     error
	}

	resultCh := make(chan valuesToHandlesResult, 1)
	go func() {
		handles, release, err := valuesToHandles([]Value{value, value}, "input")
		resultCh <- valuesToHandlesResult{
			handles: handles,
			release: release,
			err:     err,
		}
	}()

	<-value.firstLeaseAcquired

	destroyDone := make(chan struct{})
	go func() {
		_ = value.Destroy()
		close(destroyDone)
	}()

	close(value.allowFirstLeaseReturn)

	var result valuesToHandlesResult
	require.Eventually(t, func() bool {
		select {
		case result = <-resultCh:
			return true
		default:
			return false
		}
	}, 2*time.Second, 10*time.Millisecond, "valuesToHandles blocked while acquiring repeated lockable value")

	if result.err != nil {
		t.Fatalf("valuesToHandles failed: %v", result.err)
	}
	if got := len(result.handles); got != 2 {
		t.Fatalf("expected two handles, got %d", got)
	}
	if result.handles[0] != 42 || result.handles[1] != 42 {
		t.Fatalf("expected both handles to reuse 42, got %v", result.handles)
	}

	select {
	case <-destroyDone:
		t.Fatalf("destroy should block until release() unlocks leases")
	default:
	}

	result.release()

	require.Eventually(t, func() bool {
		select {
		case <-destroyDone:
			return true
		default:
			return false
		}
	}, 2*time.Second, 10*time.Millisecond, "destroy did not complete after release()")
}

func TestValuesToHandlesReleasesPriorLeasesOnError(t *testing.T) {
	value := newBlockingLeaseValue(42)
	close(value.allowFirstLeaseReturn)

	_, release, err := valuesToHandles([]Value{value, &fakeValue{handle: 0}}, "input")
	if err == nil || !strings.Contains(err.Error(), "input value at index 1 has been destroyed") {
		t.Fatalf("expected destroyed-value error at index 1, got: %v", err)
	}
	if release == nil {
		t.Fatalf("expected non-nil release callback on error")
	}
	release()

	destroyDone := make(chan struct{})
	go func() {
		_ = value.Destroy()
		close(destroyDone)
	}()

	require.Eventually(t, func() bool {
		select {
		case <-destroyDone:
			return true
		default:
			return false
		}
	}, 2*time.Second, 10*time.Millisecond, "destroy should not block; prior leases should have been released on error")
}

func TestValuesToHandlesRejectsNonComparableLockable(t *testing.T) {
	value := nonComparableLeaseValue{
		handle:  7,
		payload: []int{1, 2, 3},
	}

	_, release, err := valuesToHandles([]Value{value}, "input")
	if err == nil || !strings.Contains(err.Error(), "must be comparable") {
		t.Fatalf("expected non-comparable lockable error, got: %v", err)
	}
	if release == nil {
		t.Fatalf("expected non-nil release callback on error")
	}
	release()
}

func TestNewAdvancedSessionValidation(t *testing.T) {
	validValue := &fakeValue{handle: 1}

	tests := []struct {
		name         string
		modelPath    string
		inputNames   []string
		outputNames  []string
		inputValues  []Value
		outputValues []Value
		wantErr      string
	}{
		{
			name:         "empty model path",
			modelPath:    "",
			inputNames:   []string{"input"},
			outputNames:  []string{"output"},
			inputValues:  []Value{validValue},
			outputValues: []Value{validValue},
			wantErr:      "model path cannot be empty",
		},
		{
			name:         "missing input names",
			modelPath:    "model.onnx",
			inputNames:   nil,
			outputNames:  []string{"output"},
			inputValues:  []Value{validValue},
			outputValues: []Value{validValue},
			wantErr:      "at least one input name is required",
		},
		{
			name:         "missing output names",
			modelPath:    "model.onnx",
			inputNames:   []string{"input"},
			outputNames:  nil,
			inputValues:  []Value{validValue},
			outputValues: []Value{validValue},
			wantErr:      "at least one output name is required",
		},
		{
			name:         "input name/value mismatch",
			modelPath:    "model.onnx",
			inputNames:   []string{"input1", "input2"},
			outputNames:  []string{"output"},
			inputValues:  []Value{validValue},
			outputValues: []Value{validValue},
			wantErr:      "input names/values count mismatch",
		},
		{
			name:         "output name/value mismatch",
			modelPath:    "model.onnx",
			inputNames:   []string{"input"},
			outputNames:  []string{"output1", "output2"},
			inputValues:  []Value{validValue},
			outputValues: []Value{validValue},
			wantErr:      "output names/values count mismatch",
		},
		{
			name:         "unsupported input value implementation",
			modelPath:    "model.onnx",
			inputNames:   []string{"input"},
			outputNames:  []string{"output"},
			inputValues:  []Value{&unsupportedValue{}},
			outputValues: []Value{validValue},
			wantErr:      "unsupported value implementation",
		},
		{
			name:         "nil input value",
			modelPath:    "model.onnx",
			inputNames:   []string{"input"},
			outputNames:  []string{"output"},
			inputValues:  []Value{nil},
			outputValues: []Value{validValue},
			wantErr:      "invalid input value at index 0: value is nil",
		},
		{
			name:         "nil output value",
			modelPath:    "model.onnx",
			inputNames:   []string{"input"},
			outputNames:  []string{"output"},
			inputValues:  []Value{validValue},
			outputValues: []Value{nil},
			wantErr:      "invalid output value at index 0: value is nil",
		},
		{
			name:         "zero handle output value",
			modelPath:    "model.onnx",
			inputNames:   []string{"input"},
			outputNames:  []string{"output"},
			inputValues:  []Value{validValue},
			outputValues: []Value{&fakeValue{handle: 0}},
			wantErr:      "output value at index 0 has been destroyed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewAdvancedSession(tt.modelPath, tt.inputNames, tt.outputNames, tt.inputValues, tt.outputValues, nil)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %q", tt.wantErr, err.Error())
			}
		})
	}
}

func TestNewAdvancedSessionWithoutORT(t *testing.T) {
	resetEnvironmentState()

	_, err := NewAdvancedSession(
		"model.onnx",
		[]string{"input"},
		[]string{"output"},
		[]Value{&fakeValue{handle: 1}},
		[]Value{&fakeValue{handle: 2}},
		nil,
	)
	if err == nil || !strings.Contains(err.Error(), "ONNX Runtime not initialized") {
		t.Fatalf("expected not initialized error, got: %v", err)
	}
}

func TestNewAdvancedSessionWithUninitializedSessionOptions(t *testing.T) {
	resetEnvironmentState()

	_, err := NewAdvancedSession(
		"model.onnx",
		[]string{"input"},
		[]string{"output"},
		[]Value{&fakeValue{handle: 1}},
		[]Value{&fakeValue{handle: 2}},
		&SessionOptions{},
	)
	if err == nil || !strings.Contains(err.Error(), "session options handle is not initialized") {
		t.Fatalf("expected session options error, got: %v", err)
	}
}

func TestNewAdvancedSessionWithProvidedSessionOptionsHandle(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	var (
		createSessionOptionsCalls  int32
		releaseSessionOptionsCalls int32
		createSessionCalls         int32
		receivedSessionOptions     uintptr
	)

	mu.Lock()
	ortAPI = &OrtApi{}
	ortEnv = 99
	createSessionOptionsFunc = func(out *uintptr) uintptr {
		atomic.AddInt32(&createSessionOptionsCalls, 1)
		if out != nil {
			*out = 111
		}
		return 0
	}
	releaseSessionOptionsFunc = func(handle uintptr) {
		atomic.AddInt32(&releaseSessionOptionsCalls, 1)
	}
	createSessionFunc = func(env uintptr, modelPath uintptr, sessionOptions uintptr, out *uintptr) uintptr {
		atomic.AddInt32(&createSessionCalls, 1)
		receivedSessionOptions = sessionOptions
		if out != nil {
			*out = 123
		}
		return 0
	}
	releaseSessionFunc = func(handle uintptr) {}
	mu.Unlock()

	options := &SessionOptions{handle: 777}
	session, err := NewAdvancedSession(
		"model.onnx",
		[]string{"input"},
		[]string{"output"},
		[]Value{&fakeValue{handle: 1}},
		[]Value{&fakeValue{handle: 2}},
		options,
	)
	if err != nil {
		t.Fatalf("expected session creation to succeed with provided options handle, got: %v", err)
	}
	defer func() {
		if destroyErr := session.Destroy(); destroyErr != nil {
			t.Errorf("session destroy failed: %v", destroyErr)
		}
	}()

	if got := atomic.LoadInt32(&createSessionCalls); got != 1 {
		t.Fatalf("expected createSession to be called once, got %d", got)
	}
	if got := atomic.LoadInt32(&createSessionOptionsCalls); got != 0 {
		t.Fatalf("expected createSessionOptions not to be called, got %d", got)
	}
	if got := atomic.LoadInt32(&releaseSessionOptionsCalls); got != 0 {
		t.Fatalf("expected releaseSessionOptions not to be called, got %d", got)
	}
	if receivedSessionOptions != options.handle {
		t.Fatalf("expected createSession to receive options handle %d, got %d", options.handle, receivedSessionOptions)
	}
}

func TestAdvancedSessionRunNil(t *testing.T) {
	var session *AdvancedSession
	err := session.Run()
	if err == nil || !strings.Contains(err.Error(), "session is nil") {
		t.Fatalf("expected nil session error, got: %v", err)
	}
}

func TestAdvancedSessionRunDestroyed(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	mu.Lock()
	ortAPI = &OrtApi{}
	runSessionFunc = func(session uintptr, runOptions uintptr, inputNames *uintptr, inputValues *uintptr, inputLen uintptr, outputNames *uintptr, outputLen uintptr, outputValues *uintptr) uintptr {
		return 0
	}
	mu.Unlock()

	session := &AdvancedSession{
		handle:       0,
		inputNames:   []string{"input"},
		outputNames:  []string{"output"},
		inputValues:  []Value{&fakeValue{handle: 1}},
		outputValues: []Value{&fakeValue{handle: 2}},
	}

	err := session.Run()
	if err == nil || !strings.Contains(err.Error(), "session has been destroyed") {
		t.Fatalf("expected destroyed session error, got: %v", err)
	}

}

func TestAdvancedSessionRunWithValues(t *testing.T) {
	t.Run("uses supplied handles and preserves bound handles for Run", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		type runCall struct {
			inputName    string
			outputName   string
			inputHandle  uintptr
			outputHandle uintptr
		}
		calls := make([]runCall, 0, 2)

		mu.Lock()
		ortAPI = &OrtApi{}
		runSessionFunc = func(_ uintptr, _ uintptr, inputNames *uintptr, inputValues *uintptr, _ uintptr, outputNames *uintptr, _ uintptr, outputValues *uintptr) uintptr {
			calls = append(calls, runCall{
				inputName:    CstringToGo(*inputNames),
				outputName:   CstringToGo(*outputNames),
				inputHandle:  *inputValues,
				outputHandle: *outputValues,
			})
			return 0
		}
		mu.Unlock()

		boundInput := &fakeValue{handle: 11}
		boundOutput := &fakeValue{handle: 12}
		session := &AdvancedSession{
			handle:       100,
			inputNames:   []string{"fixed_input"},
			outputNames:  []string{"fixed_output"},
			inputValues:  []Value{boundInput},
			outputValues: []Value{boundOutput},
		}

		if err := session.RunWithValues(
			[]Value{&fakeValue{handle: 21}},
			[]Value{&fakeValue{handle: 22}},
		); err != nil {
			t.Fatalf("RunWithValues failed: %v", err)
		}
		if err := session.Run(); err != nil {
			t.Fatalf("Run failed after RunWithValues: %v", err)
		}

		want := []runCall{
			{inputName: "fixed_input", outputName: "fixed_output", inputHandle: 21, outputHandle: 22},
			{inputName: "fixed_input", outputName: "fixed_output", inputHandle: 11, outputHandle: 12},
		}
		if !slices.Equal(calls, want) {
			t.Fatalf("run calls = %#v, want %#v", calls, want)
		}
		if session.inputValues[0] != boundInput || session.outputValues[0] != boundOutput {
			t.Fatal("RunWithValues changed the session's bound values")
		}
	})

	t.Run("normally constructed session may use only supplied values", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		var gotInput, gotOutput uintptr
		mu.Lock()
		ortAPI = &OrtApi{}
		ortEnv = 99
		createSessionOptionsFunc = func(out *uintptr) uintptr {
			*out = 200
			return 0
		}
		releaseSessionOptionsFunc = func(uintptr) {}
		createSessionFunc = func(_ uintptr, _ uintptr, _ uintptr, out *uintptr) uintptr {
			*out = 201
			return 0
		}
		runSessionFunc = func(_ uintptr, _ uintptr, _ *uintptr, inputValues *uintptr, _ uintptr, _ *uintptr, _ uintptr, outputValues *uintptr) uintptr {
			gotInput = *inputValues
			gotOutput = *outputValues
			return 0
		}
		releaseSessionFunc = func(uintptr) {}
		mu.Unlock()

		boundInput := &fakeValue{handle: 31}
		boundOutput := &fakeValue{handle: 32}
		session, err := NewAdvancedSession(
			"model.onnx",
			[]string{"input"},
			[]string{"output"},
			[]Value{boundInput},
			[]Value{boundOutput},
			nil,
		)
		if err != nil {
			t.Fatalf("NewAdvancedSession failed: %v", err)
		}
		t.Cleanup(func() {
			if destroyErr := session.Destroy(); destroyErr != nil {
				t.Errorf("session destroy failed: %v", destroyErr)
			}
		})

		if err := session.RunWithValues(
			[]Value{&fakeValue{handle: 41}},
			[]Value{&fakeValue{handle: 42}},
		); err != nil {
			t.Fatalf("RunWithValues failed: %v", err)
		}
		if gotInput != 41 || gotOutput != 42 {
			t.Fatalf("runtime received handles (%d, %d), want (41, 42)", gotInput, gotOutput)
		}
		if session.inputValues[0] != boundInput || session.outputValues[0] != boundOutput {
			t.Fatal("RunWithValues changed constructor-bound values")
		}
	})

	t.Run("rejects invalid counts and values before FFI", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		var runCalls atomic.Int32
		mu.Lock()
		ortAPI = &OrtApi{}
		runSessionFunc = func(_ uintptr, _ uintptr, _ *uintptr, _ *uintptr, _ uintptr, _ *uintptr, _ uintptr, _ *uintptr) uintptr {
			runCalls.Add(1)
			return 0
		}
		mu.Unlock()

		session := &AdvancedSession{
			handle:       300,
			inputNames:   []string{"input"},
			outputNames:  []string{"output"},
			inputValues:  []Value{&fakeValue{handle: 1}},
			outputValues: []Value{&fakeValue{handle: 2}},
		}
		validInput := []Value{&fakeValue{handle: 51}}
		validOutput := []Value{&fakeValue{handle: 52}}
		tests := []struct {
			name    string
			inputs  []Value
			outputs []Value
			want    error
		}{
			{name: "input count mismatch", inputs: nil, outputs: validOutput, want: ErrInvalidArgument},
			{name: "output count mismatch", inputs: validInput, outputs: nil, want: ErrInvalidArgument},
			{name: "nil input", inputs: []Value{nil}, outputs: validOutput, want: ErrInvalidArgument},
			{name: "unsupported input", inputs: []Value{&unsupportedValue{}}, outputs: validOutput, want: ErrInvalidArgument},
			{name: "destroyed input", inputs: []Value{&fakeValue{}}, outputs: validOutput, want: ErrDestroyed},
			{name: "nil output", inputs: validInput, outputs: []Value{nil}, want: ErrInvalidArgument},
			{name: "destroyed output", inputs: validInput, outputs: []Value{&fakeValue{}}, want: ErrDestroyed},
		}
		for _, tt := range tests {
			err := session.RunWithValues(tt.inputs, tt.outputs)
			if !errors.Is(err, tt.want) {
				t.Errorf("%s: error = %v, want errors.Is(..., %v)", tt.name, err, tt.want)
			}
		}
		if got := runCalls.Load(); got != 0 {
			t.Fatalf("runtime called %d times for locally invalid values", got)
		}
	})

	t.Run("serializes same session and leaves unrelated sessions independent", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		var inFlight, maxInFlight atomic.Int32
		entered := make(chan uintptr, 2)
		allowReturn := make(chan struct{})
		mu.Lock()
		ortAPI = &OrtApi{}
		runSessionFunc = func(session uintptr, _ uintptr, _ *uintptr, _ *uintptr, _ uintptr, _ *uintptr, _ uintptr, _ *uintptr) uintptr {
			current := inFlight.Add(1)
			for {
				seen := maxInFlight.Load()
				if current <= seen || maxInFlight.CompareAndSwap(seen, current) {
					break
				}
			}
			entered <- session
			<-allowReturn
			inFlight.Add(-1)
			return 0
		}
		mu.Unlock()

		newSession := func(handle uintptr) *AdvancedSession {
			return &AdvancedSession{
				handle:       handle,
				inputNames:   []string{"input"},
				outputNames:  []string{"output"},
				inputValues:  []Value{&fakeValue{handle: handle + 10}},
				outputValues: []Value{&fakeValue{handle: handle + 20}},
			}
		}
		sameSession := newSession(400)
		firstErr := make(chan error, 1)
		secondErr := make(chan error, 1)
		go func() {
			firstErr <- sameSession.RunWithValues(
				[]Value{&fakeValue{handle: 61}},
				[]Value{&fakeValue{handle: 62}},
			)
		}()
		<-entered
		go func() {
			secondErr <- sameSession.Run()
		}()
		if sameSession.runMu.TryLock() {
			sameSession.runMu.Unlock()
			t.Fatal("same-session RunWithValues did not hold runMu")
		}
		close(allowReturn)
		if err := <-firstErr; err != nil {
			t.Fatalf("first same-session run failed: %v", err)
		}
		if err := <-secondErr; err != nil {
			t.Fatalf("second same-session run failed: %v", err)
		}
		if got := maxInFlight.Load(); got != 1 {
			t.Fatalf("same-session max in-flight = %d, want 1", got)
		}

		inFlight.Store(0)
		maxInFlight.Store(0)
		entered = make(chan uintptr, 2)
		allowReturn = make(chan struct{})
		sharedInput := &Tensor[float32]{handle: 71}
		firstSession := newSession(401)
		secondSession := newSession(402)
		firstErr = make(chan error, 1)
		secondErr = make(chan error, 1)
		go func() {
			firstErr <- firstSession.RunWithValues(
				[]Value{sharedInput},
				[]Value{&fakeValue{handle: 72}},
			)
		}()
		go func() {
			secondErr <- secondSession.RunWithValues(
				[]Value{sharedInput},
				[]Value{&fakeValue{handle: 73}},
			)
		}()
		for i := 0; i < 2; i++ {
			select {
			case <-entered:
			case <-time.After(2 * time.Second):
				t.Fatal("unrelated sessions did not reach the runtime concurrently")
			}
		}
		if got := maxInFlight.Load(); got != 2 {
			t.Fatalf("unrelated-session max in-flight = %d, want 2", got)
		}
		close(allowReturn)
		if err := <-firstErr; err != nil {
			t.Fatalf("first unrelated-session run failed: %v", err)
		}
		if err := <-secondErr; err != nil {
			t.Fatalf("second unrelated-session run failed: %v", err)
		}
	})

	t.Run("holds supplied tensor lease until the call returns", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		runEntered := make(chan struct{})
		allowReturn := make(chan struct{})
		var enterOnce sync.Once
		var eventsMu sync.Mutex
		var events []string
		record := func(event string) {
			eventsMu.Lock()
			events = append(events, event)
			eventsMu.Unlock()
		}

		mu.Lock()
		ortAPI = &OrtApi{}
		runSessionFunc = func(_ uintptr, _ uintptr, _ *uintptr, _ *uintptr, _ uintptr, _ *uintptr, _ uintptr, _ *uintptr) uintptr {
			enterOnce.Do(func() { close(runEntered) })
			<-allowReturn
			record("run returned")
			return 0
		}
		releaseValueFunc = func(uintptr) {
			record("tensor released")
		}
		mu.Unlock()

		input := &Tensor[float32]{handle: 81}
		session := &AdvancedSession{
			handle:       500,
			inputNames:   []string{"input"},
			outputNames:  []string{"output"},
			inputValues:  []Value{&fakeValue{handle: 1}},
			outputValues: []Value{&fakeValue{handle: 2}},
		}
		runErr := make(chan error, 1)
		go func() {
			runErr <- session.RunWithValues(
				[]Value{input},
				[]Value{&fakeValue{handle: 82}},
			)
		}()
		<-runEntered
		if input.runMu.TryLock() {
			input.runMu.Unlock()
			t.Fatal("RunWithValues did not retain the supplied tensor lease")
		}

		destroyErr := make(chan error, 1)
		go func() {
			destroyErr <- input.Destroy()
		}()
		close(allowReturn)
		if err := <-runErr; err != nil {
			t.Fatalf("RunWithValues failed: %v", err)
		}
		if err := <-destroyErr; err != nil {
			t.Fatalf("tensor destroy failed: %v", err)
		}
		eventsMu.Lock()
		got := append([]string(nil), events...)
		eventsMu.Unlock()
		if want := []string{"run returned", "tensor released"}; !slices.Equal(got, want) {
			t.Fatalf("events = %v, want %v", got, want)
		}
	})

	t.Run("deduplicates within each role and releases in reverse order", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		mu.Lock()
		ortAPI = &OrtApi{}
		runSessionFunc = func(_ uintptr, _ uintptr, _ *uintptr, _ *uintptr, _ uintptr, _ *uintptr, _ uintptr, _ *uintptr) uintptr {
			return 0
		}
		mu.Unlock()

		released := make(chan uintptr, 4)
		shared := &countingLeaseValue{handle: 91, released: released}
		inputOnly := &countingLeaseValue{handle: 92, released: released}
		outputOnly := &countingLeaseValue{handle: 93, released: released}
		session := &AdvancedSession{
			handle:       600,
			inputNames:   []string{"input_a", "input_b", "input_c"},
			outputNames:  []string{"output_a", "output_b", "output_c"},
			inputValues:  []Value{shared, inputOnly, shared},
			outputValues: []Value{shared, outputOnly, shared},
		}
		if err := session.RunWithValues(
			[]Value{shared, inputOnly, shared},
			[]Value{shared, outputOnly, shared},
		); err != nil {
			t.Fatalf("RunWithValues failed: %v", err)
		}

		if got := shared.lockCalls.Load(); got != 2 {
			t.Fatalf("shared value lease count = %d, want one per role", got)
		}
		if got := shared.unlockCalls.Load(); got != 2 {
			t.Fatalf("shared value unlock count = %d, want one per role", got)
		}
		if got := inputOnly.lockCalls.Load(); got != 1 {
			t.Fatalf("input-only lease count = %d, want 1", got)
		}
		if got := outputOnly.lockCalls.Load(); got != 1 {
			t.Fatalf("output-only lease count = %d, want 1", got)
		}
		gotReleaseOrder := []uintptr{<-released, <-released, <-released, <-released}
		wantReleaseOrder := []uintptr{93, 91, 92, 91}
		if !slices.Equal(gotReleaseOrder, wantReleaseOrder) {
			t.Fatalf("release order = %v, want %v", gotReleaseOrder, wantReleaseOrder)
		}
	})

	t.Run("fills caller output in place and leaves it caller destroyable", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		var releasedHandle atomic.Uintptr
		output := &Tensor[float32]{handle: 102, data: []float32{0}}
		mu.Lock()
		ortAPI = &OrtApi{}
		runSessionFunc = func(_ uintptr, _ uintptr, _ *uintptr, _ *uintptr, _ uintptr, _ *uintptr, _ uintptr, outputValues *uintptr) uintptr {
			if got := *outputValues; got != output.handle {
				t.Errorf("runtime output handle = %d, want caller handle %d", got, output.handle)
			}
			output.data[0] = 42
			return 0
		}
		releaseValueFunc = func(handle uintptr) {
			releasedHandle.Store(handle)
		}
		mu.Unlock()

		session := &AdvancedSession{
			handle:       700,
			inputNames:   []string{"input"},
			outputNames:  []string{"output"},
			inputValues:  []Value{&fakeValue{handle: 1}},
			outputValues: []Value{&fakeValue{handle: 2}},
		}
		if err := session.RunWithValues(
			[]Value{&fakeValue{handle: 101}},
			[]Value{output},
		); err != nil {
			t.Fatalf("RunWithValues failed: %v", err)
		}
		if got := output.GetData()[0]; got != 42 {
			t.Fatalf("caller output data = %v, want 42", got)
		}
		if err := output.Destroy(); err != nil {
			t.Fatalf("caller output destroy failed: %v", err)
		}
		if got := releasedHandle.Load(); got != 102 {
			t.Fatalf("released output handle = %d, want 102", got)
		}
	})
}

func TestAdvancedSessionDestroy(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	releasedCount := 0
	releasedHandle := uintptr(0)
	mu.Lock()
	releaseSessionFunc = func(handle uintptr) {
		releasedCount++
		releasedHandle = handle
	}
	mu.Unlock()

	session := &AdvancedSession{
		handle:       123,
		inputNames:   []string{"input"},
		outputNames:  []string{"output"},
		inputValues:  []Value{&fakeValue{handle: 1}},
		outputValues: []Value{&fakeValue{handle: 2}},
	}

	if err := session.Destroy(); err != nil {
		t.Fatalf("destroy failed: %v", err)
	}
	if session.handle != 0 {
		t.Fatalf("expected handle to be reset")
	}
	if session.inputNames != nil || session.outputNames != nil || session.inputValues != nil || session.outputValues != nil {
		t.Fatalf("expected session fields to be cleared")
	}
	if releasedCount != 1 {
		t.Fatalf("expected release callback to be called once, got %d", releasedCount)
	}
	if releasedHandle != 123 {
		t.Fatalf("expected release callback to receive handle 123, got %d", releasedHandle)
	}

	if err := session.Destroy(); err != nil {
		t.Fatalf("second destroy should be no-op, got: %v", err)
	}
	if releasedCount != 1 {
		t.Fatalf("expected second destroy to not release again, got %d releases", releasedCount)
	}

}

func TestAdvancedSessionDestroyReleaseUnavailable(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	session := &AdvancedSession{
		handle:       123,
		inputNames:   []string{"input"},
		outputNames:  []string{"output"},
		inputValues:  []Value{&fakeValue{handle: 1}},
		outputValues: []Value{&fakeValue{handle: 2}},
	}

	err := session.Destroy()
	if err == nil || !strings.Contains(err.Error(), "release function unavailable") {
		t.Fatalf("expected release-unavailable destroy error, got: %v", err)
	}
	if session.handle != 0 {
		t.Fatalf("expected handle to be reset even on release failure")
	}
	if session.inputNames != nil || session.outputNames != nil || session.inputValues != nil || session.outputValues != nil {
		t.Fatalf("expected session fields to be cleared even on release failure")
	}
}

func TestAdvancedSessionRunConcurrent(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	const runCalls = 32

	var (
		calls       int32
		inFlight    int32
		maxInFlight int32
	)

	mu.Lock()
	ortAPI = &OrtApi{}
	runSessionFunc = func(session uintptr, runOptions uintptr, inputNames *uintptr, inputValues *uintptr, inputLen uintptr, outputNames *uintptr, outputLen uintptr, outputValues *uintptr) uintptr {
		atomic.AddInt32(&calls, 1)
		current := atomic.AddInt32(&inFlight, 1)
		for {
			seen := atomic.LoadInt32(&maxInFlight)
			if current <= seen {
				break
			}
			if atomic.CompareAndSwapInt32(&maxInFlight, seen, current) {
				break
			}
		}
		time.Sleep(1 * time.Millisecond)
		atomic.AddInt32(&inFlight, -1)
		return 0
	}
	mu.Unlock()

	session := &AdvancedSession{
		handle:       123,
		inputNames:   []string{"input"},
		outputNames:  []string{"output"},
		inputValues:  []Value{&fakeValue{handle: 1}},
		outputValues: []Value{&fakeValue{handle: 2}},
	}

	start := make(chan struct{})
	errCh := make(chan error, runCalls)
	var wg sync.WaitGroup
	for i := 0; i < runCalls; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			errCh <- session.Run()
		}()
	}
	close(start)
	wg.Wait()
	close(errCh)

	for err := range errCh {
		if err != nil {
			t.Fatalf("concurrent run failed: %v", err)
		}
	}

	if got := atomic.LoadInt32(&calls); got != runCalls {
		t.Fatalf("expected %d Run() calls to reach runtime, got %d", runCalls, got)
	}
	if got := atomic.LoadInt32(&maxInFlight); got != 1 {
		t.Fatalf("expected Run() calls to be serialized per session, max in-flight=%d", got)
	}
}

func TestAdvancedSessionRunConcurrentAcrossSessionsSharingTensor(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	var (
		inFlight    int32
		maxInFlight int32
	)
	enterRun := make(chan struct{}, 2)
	allowRunReturn := make(chan struct{})

	mu.Lock()
	ortAPI = &OrtApi{}
	runSessionFunc = func(session uintptr, runOptions uintptr, inputNames *uintptr, inputValues *uintptr, inputLen uintptr, outputNames *uintptr, outputLen uintptr, outputValues *uintptr) uintptr {
		current := atomic.AddInt32(&inFlight, 1)
		for {
			seen := atomic.LoadInt32(&maxInFlight)
			if current <= seen {
				break
			}
			if atomic.CompareAndSwapInt32(&maxInFlight, seen, current) {
				break
			}
		}
		enterRun <- struct{}{}
		<-allowRunReturn
		atomic.AddInt32(&inFlight, -1)
		return 0
	}
	mu.Unlock()

	sharedInputTensor := &Tensor[float32]{handle: 1}
	firstSession := &AdvancedSession{
		handle:       101,
		inputNames:   []string{"input"},
		outputNames:  []string{"output"},
		inputValues:  []Value{sharedInputTensor},
		outputValues: []Value{&fakeValue{handle: 2}},
	}
	secondSession := &AdvancedSession{
		handle:       102,
		inputNames:   []string{"input"},
		outputNames:  []string{"output"},
		inputValues:  []Value{sharedInputTensor},
		outputValues: []Value{&fakeValue{handle: 3}},
	}

	firstRunErrCh := make(chan error, 1)
	secondRunErrCh := make(chan error, 1)
	go func() {
		firstRunErrCh <- firstSession.Run()
	}()
	go func() {
		secondRunErrCh <- secondSession.Run()
	}()

	received := 0
	require.Eventually(t, func() bool {
		select {
		case <-enterRun:
			received++
		default:
		}
		return received >= 2
	}, 2*time.Second, 10*time.Millisecond, "expected both sessions to reach runtime concurrently")

	if got := atomic.LoadInt32(&maxInFlight); got < 2 {
		t.Fatalf("expected shared-tensor runs across sessions to overlap, max in-flight=%d", got)
	}

	close(allowRunReturn)

	if err := <-firstRunErrCh; err != nil {
		t.Fatalf("first session run failed: %v", err)
	}
	if err := <-secondRunErrCh; err != nil {
		t.Fatalf("second session run failed: %v", err)
	}
}

func TestAdvancedSessionRunAndDestroyConcurrent(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	runStarted := make(chan struct{})
	allowRunReturn := make(chan struct{})
	var closeRunStarted sync.Once

	releasedCount := int32(0)
	var releasedHandle atomic.Uintptr

	var (
		eventsMu sync.Mutex
		events   []string
	)
	record := func(e string) {
		eventsMu.Lock()
		events = append(events, e)
		eventsMu.Unlock()
	}

	mu.Lock()
	ortAPI = &OrtApi{}
	runSessionFunc = func(session uintptr, runOptions uintptr, inputNames *uintptr, inputValues *uintptr, inputLen uintptr, outputNames *uintptr, outputLen uintptr, outputValues *uintptr) uintptr {
		closeRunStarted.Do(func() { close(runStarted) })
		<-allowRunReturn
		record("run-returned")
		return 0
	}
	releaseSessionFunc = func(handle uintptr) {
		record("destroy-released")
		atomic.AddInt32(&releasedCount, 1)
		releasedHandle.Store(handle)
	}
	mu.Unlock()

	session := &AdvancedSession{
		handle:       456,
		inputNames:   []string{"input"},
		outputNames:  []string{"output"},
		inputValues:  []Value{&fakeValue{handle: 1}},
		outputValues: []Value{&fakeValue{handle: 2}},
	}

	runErrCh := make(chan error, 1)
	go func() {
		runErrCh <- session.Run()
	}()

	<-runStarted

	// Deterministic lock-contention probe: Run() locks session.runMu before invoking
	// the mock runSessionFunc, and <-runStarted only unblocks after that mock is entered,
	// so runMu is guaranteed held here. TryLock() must fail, proving Destroy() WILL block
	// independent of goroutine scheduling.
	if session.runMu.TryLock() {
		session.runMu.Unlock()
		t.Fatalf("expected session.runMu to be held by the in-flight Run(), TryLock unexpectedly succeeded")
	}

	destroyErrCh := make(chan error, 1)
	go func() {
		destroyErrCh <- session.Destroy()
	}()

	// Deadlock safety net only -- the TryLock probe above already proved blocking.
	// This watchdog runs on the main test goroutine (testing.T.FailNow contract) and
	// fires only on a genuine regression where destroy returns before Run unblocks.
	select {
	case err := <-destroyErrCh:
		t.Fatalf("destroy returned before run completed (err=%v) -- deadlock-safety-net fired unexpectedly early", err)
	case <-time.After(500 * time.Millisecond):
	}

	close(allowRunReturn)

	if err := <-runErrCh; err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if err := <-destroyErrCh; err != nil {
		t.Fatalf("destroy failed: %v", err)
	}

	eventsMu.Lock()
	got := append([]string(nil), events...)
	eventsMu.Unlock()
	want := []string{"run-returned", "destroy-released"}
	if !slices.Equal(got, want) {
		t.Fatalf("expected event order %v, got %v", want, got)
	}

	if got := atomic.LoadInt32(&releasedCount); got != 1 {
		t.Fatalf("expected release callback once, got %d", got)
	}
	if got := releasedHandle.Load(); got != 456 {
		t.Fatalf("expected release callback handle 456, got %d", got)
	}

	if err := session.Run(); err == nil || !strings.Contains(err.Error(), "session has been destroyed") {
		t.Fatalf("expected destroyed session error after concurrent destroy, got: %v", err)
	}
}

func TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	runStarted := make(chan struct{})
	allowRunReturn := make(chan struct{})
	var closeRunStarted sync.Once

	otherDestroyed := int32(0)

	mu.Lock()
	ortAPI = &OrtApi{}
	runSessionFunc = func(session uintptr, runOptions uintptr, inputNames *uintptr, inputValues *uintptr, inputLen uintptr, outputNames *uintptr, outputLen uintptr, outputValues *uintptr) uintptr {
		closeRunStarted.Do(func() { close(runStarted) })
		<-allowRunReturn
		return 0
	}
	releaseSessionFunc = func(handle uintptr) {
		if handle == 222 {
			atomic.StoreInt32(&otherDestroyed, 1)
		}
	}
	mu.Unlock()

	runningSession := &AdvancedSession{
		handle:       111,
		inputNames:   []string{"input"},
		outputNames:  []string{"output"},
		inputValues:  []Value{&fakeValue{handle: 1}},
		outputValues: []Value{&fakeValue{handle: 2}},
	}
	otherSession := &AdvancedSession{
		handle:       222,
		inputNames:   []string{"input"},
		outputNames:  []string{"output"},
		inputValues:  []Value{&fakeValue{handle: 3}},
		outputValues: []Value{&fakeValue{handle: 4}},
	}

	runErrCh := make(chan error, 1)
	go func() {
		runErrCh <- runningSession.Run()
	}()

	<-runStarted

	destroyErrCh := make(chan error, 1)
	go func() {
		destroyErrCh <- otherSession.Destroy()
	}()

	// Deliberate mirror-image of Task 1's watchdog: here the receive branch is the
	// expected/passing path and the timeout is the FAILURE condition, since this test
	// proves the ABSENCE of blocking on an unrelated in-flight Run. The timeout only
	// elapses on a genuine regression, so a generous budget costs passing runs nothing.
	var destroyErr error
	select {
	case destroyErr = <-destroyErrCh:
		// expected: destroy is not blocked by the unrelated in-flight Run
	case <-time.After(2 * time.Second):
		t.Fatal("destroy on unrelated session did not return within 2s -- appears blocked by in-flight Run")
	}
	if destroyErr != nil {
		t.Fatalf("destroy failed: %v", destroyErr)
	}

	close(allowRunReturn)

	if err := <-runErrCh; err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if got := atomic.LoadInt32(&otherDestroyed); got != 1 {
		t.Fatalf("expected unrelated session to be released once, got flag=%d", got)
	}

	if err := otherSession.Run(); err == nil || !strings.Contains(err.Error(), "session has been destroyed") {
		t.Fatalf("expected destroyed session error for other session, got: %v", err)
	}
}

func TestTensorDestroyWaitsForInFlightRun(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	runStarted := make(chan struct{})
	allowRunReturn := make(chan struct{})
	var closeRunStarted sync.Once

	releasedTensor := int32(0)

	var (
		eventsMu sync.Mutex
		events   []string
	)
	record := func(e string) {
		eventsMu.Lock()
		events = append(events, e)
		eventsMu.Unlock()
	}

	mu.Lock()
	ortAPI = &OrtApi{}
	runSessionFunc = func(session uintptr, runOptions uintptr, inputNames *uintptr, inputValues *uintptr, inputLen uintptr, outputNames *uintptr, outputLen uintptr, outputValues *uintptr) uintptr {
		closeRunStarted.Do(func() { close(runStarted) })
		<-allowRunReturn
		record("run-returned")
		return 0
	}
	releaseValueFunc = func(handle uintptr) {
		record("destroy-released")
		atomic.AddInt32(&releasedTensor, 1)
	}
	mu.Unlock()

	inputTensor := &Tensor[float32]{handle: 1}
	session := &AdvancedSession{
		handle:       333,
		inputNames:   []string{"input"},
		outputNames:  []string{"output"},
		inputValues:  []Value{inputTensor},
		outputValues: []Value{&fakeValue{handle: 2}},
	}

	runErrCh := make(chan error, 1)
	go func() {
		runErrCh <- session.Run()
	}()

	<-runStarted

	// Deterministic lock-contention probe: Run()'s value-locking path takes
	// inputTensor.runMu.RLock() (via lockForRun) before invoking the mock, and
	// <-runStarted only unblocks after that mock is entered. A write-lock TryLock()
	// against the RWMutex must fail while any RLock is held, proving Tensor.Destroy()
	// WILL block independent of goroutine scheduling.
	if inputTensor.runMu.TryLock() {
		inputTensor.runMu.Unlock()
		t.Fatalf("expected inputTensor.runMu to be held (RLock) by the in-flight Run(), TryLock unexpectedly succeeded")
	}

	tensorDestroyErrCh := make(chan error, 1)
	go func() {
		tensorDestroyErrCh <- inputTensor.Destroy()
	}()

	// Deadlock safety net only -- the TryLock probe above already proved blocking.
	select {
	case err := <-tensorDestroyErrCh:
		t.Fatalf("tensor destroy returned before run completed (err=%v) -- deadlock-safety-net fired unexpectedly early", err)
	case <-time.After(500 * time.Millisecond):
	}

	close(allowRunReturn)

	if err := <-runErrCh; err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if err := <-tensorDestroyErrCh; err != nil {
		t.Fatalf("tensor destroy failed: %v", err)
	}

	eventsMu.Lock()
	got := append([]string(nil), events...)
	eventsMu.Unlock()
	want := []string{"run-returned", "destroy-released"}
	if !slices.Equal(got, want) {
		t.Fatalf("expected event order %v, got %v", want, got)
	}

	if got := atomic.LoadInt32(&releasedTensor); got != 1 {
		t.Fatalf("expected tensor release callback once, got %d", got)
	}
}

func TestTensorDestroyDoesNotBlockUnrelatedRun(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	runStarted := make(chan struct{})
	allowRunReturn := make(chan struct{})
	var closeRunStarted sync.Once

	releasedTensor := int32(0)
	var releasedHandle atomic.Uintptr

	mu.Lock()
	ortAPI = &OrtApi{}
	runSessionFunc = func(session uintptr, runOptions uintptr, inputNames *uintptr, inputValues *uintptr, inputLen uintptr, outputNames *uintptr, outputLen uintptr, outputValues *uintptr) uintptr {
		closeRunStarted.Do(func() { close(runStarted) })
		<-allowRunReturn
		return 0
	}
	releaseValueFunc = func(handle uintptr) {
		atomic.AddInt32(&releasedTensor, 1)
		releasedHandle.Store(handle)
	}
	mu.Unlock()

	runningInputTensor := &Tensor[float32]{handle: 1}
	unrelatedTensor := &Tensor[float32]{handle: 99}
	session := &AdvancedSession{
		handle:       333,
		inputNames:   []string{"input"},
		outputNames:  []string{"output"},
		inputValues:  []Value{runningInputTensor},
		outputValues: []Value{&fakeValue{handle: 2}},
	}

	runErrCh := make(chan error, 1)
	go func() {
		runErrCh <- session.Run()
	}()

	<-runStarted

	tensorDestroyErrCh := make(chan error, 1)
	go func() {
		tensorDestroyErrCh <- unrelatedTensor.Destroy()
	}()

	var tensorDestroyErr error
	require.Eventually(t, func() bool {
		select {
		case tensorDestroyErr = <-tensorDestroyErrCh:
			return true
		default:
			return false
		}
	}, 2*time.Second, 10*time.Millisecond, "destroy should not block on unrelated in-flight Run")
	if tensorDestroyErr != nil {
		t.Fatalf("unrelated tensor destroy failed: %v", tensorDestroyErr)
	}

	if got := atomic.LoadInt32(&releasedTensor); got != 1 {
		t.Fatalf("expected unrelated tensor release callback once, got %d", got)
	}
	if got := releasedHandle.Load(); got != 99 {
		t.Fatalf("expected release callback handle 99, got %d", got)
	}

	close(allowRunReturn)
	if err := <-runErrCh; err != nil {
		t.Fatalf("run failed: %v", err)
	}
}

func TestAdvancedSessionRunDestroyedInputValue(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	runCalled := false
	mu.Lock()
	ortAPI = &OrtApi{}
	runSessionFunc = func(session uintptr, runOptions uintptr, inputNames *uintptr, inputValues *uintptr, inputLen uintptr, outputNames *uintptr, outputLen uintptr, outputValues *uintptr) uintptr {
		runCalled = true
		return 0
	}
	mu.Unlock()

	session := &AdvancedSession{
		handle:       123,
		inputNames:   []string{"input"},
		outputNames:  []string{"output"},
		inputValues:  []Value{&fakeValue{handle: 0}},
		outputValues: []Value{&fakeValue{handle: 2}},
	}

	err := session.Run()
	if err == nil || !strings.Contains(err.Error(), "input value at index 0 has been destroyed") {
		t.Fatalf("expected destroyed input value error, got: %v", err)
	}
	if runCalled {
		t.Fatalf("expected runSessionFunc not to be called when input value is destroyed")
	}

}

func TestAdvancedSessionRunDestroyedInputTensor(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	runCalled := false
	mu.Lock()
	ortAPI = &OrtApi{}
	runSessionFunc = func(session uintptr, runOptions uintptr, inputNames *uintptr, inputValues *uintptr, inputLen uintptr, outputNames *uintptr, outputLen uintptr, outputValues *uintptr) uintptr {
		runCalled = true
		return 0
	}
	mu.Unlock()

	session := &AdvancedSession{
		handle:       123,
		inputNames:   []string{"input"},
		outputNames:  []string{"output"},
		inputValues:  []Value{&Tensor[float32]{handle: 0}},
		outputValues: []Value{&fakeValue{handle: 2}},
	}

	err := session.Run()
	if err == nil || !strings.Contains(err.Error(), "input value at index 0 has been destroyed") {
		t.Fatalf("expected destroyed input tensor error, got: %v", err)
	}
	if runCalled {
		t.Fatalf("expected runSessionFunc not to be called when input tensor is destroyed")
	}

}

func TestMakeCStringPointerArrayEmpty(t *testing.T) {
	backings, ptrs := makeCStringPointerArray(nil)
	if backings != nil {
		t.Fatalf("expected nil backings for empty input")
	}
	if ptrs != nil {
		t.Fatalf("expected nil ptrs for empty input")
	}

	backings, ptrs = makeCStringPointerArray([]string{})
	if backings != nil {
		t.Fatalf("expected nil backings for empty slice")
	}
	if ptrs != nil {
		t.Fatalf("expected nil ptrs for empty slice")
	}
}

func TestNewAdvancedSessionInvalidModelPath(t *testing.T) {
	cleanup := setupTestEnvironment(t)
	defer cleanup()

	inputTensor, err := NewTensor[float32](Shape{1}, []float32{1.0})
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}
	defer func() {
		if destroyErr := inputTensor.Destroy(); destroyErr != nil {
			t.Errorf("input tensor destroy failed: %v", destroyErr)
		}
	}()

	outputTensor, err := NewEmptyTensor[float32](Shape{1})
	if err != nil {
		t.Fatalf("failed to create output tensor: %v", err)
	}
	defer func() {
		if destroyErr := outputTensor.Destroy(); destroyErr != nil {
			t.Errorf("output tensor destroy failed: %v", destroyErr)
		}
	}()

	_, err = NewAdvancedSession(
		"/this/path/does/not/exist/model.onnx",
		[]string{"input"},
		[]string{"output"},
		[]Value{inputTensor},
		[]Value{outputTensor},
		nil,
	)
	if err == nil {
		t.Fatalf("expected session creation to fail for invalid model path")
	}
	if !strings.Contains(err.Error(), "failed to create session") {
		t.Fatalf("unexpected error for invalid model path: %v", err)
	}
}

func TestAdvancedSessionRunWithRealModel(t *testing.T) {
	modelPath := os.Getenv("ONNXRUNTIME_TEST_MODEL_PATH")
	inputName := os.Getenv("ONNXRUNTIME_TEST_INPUT_NAME")
	outputName := os.Getenv("ONNXRUNTIME_TEST_OUTPUT_NAME")
	inputShapeRaw := os.Getenv("ONNXRUNTIME_TEST_INPUT_SHAPE")
	outputShapeRaw := os.Getenv("ONNXRUNTIME_TEST_OUTPUT_SHAPE")

	if modelPath == "" || inputName == "" || outputName == "" || inputShapeRaw == "" || outputShapeRaw == "" {
		t.Skip("set ONNXRUNTIME_TEST_MODEL_PATH, ONNXRUNTIME_TEST_INPUT_NAME, ONNXRUNTIME_TEST_OUTPUT_NAME, ONNXRUNTIME_TEST_INPUT_SHAPE, ONNXRUNTIME_TEST_OUTPUT_SHAPE for real model run test")
	}

	inputShape, err := ParseShape(inputShapeRaw)
	if err != nil {
		t.Fatalf("invalid ONNXRUNTIME_TEST_INPUT_SHAPE: %v", err)
	}
	outputShape, err := ParseShape(outputShapeRaw)
	if err != nil {
		t.Fatalf("invalid ONNXRUNTIME_TEST_OUTPUT_SHAPE: %v", err)
	}

	cleanup := setupTestEnvironment(t)
	defer cleanup()

	inputCount, err := shapeElementCount(inputShape)
	if err != nil {
		t.Fatalf("invalid input shape: %v", err)
	}
	inputData := make([]float32, inputCount)
	for i := range inputData {
		inputData[i] = 1
	}

	inputTensor, err := NewTensor[float32](inputShape, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}
	defer func() {
		if destroyErr := inputTensor.Destroy(); destroyErr != nil {
			t.Errorf("input tensor destroy failed: %v", destroyErr)
		}
	}()

	outputTensor, err := NewEmptyTensor[float32](outputShape)
	if err != nil {
		t.Fatalf("failed to create output tensor: %v", err)
	}
	defer func() {
		if destroyErr := outputTensor.Destroy(); destroyErr != nil {
			t.Errorf("output tensor destroy failed: %v", destroyErr)
		}
	}()

	session, err := NewAdvancedSession(
		modelPath,
		[]string{inputName},
		[]string{outputName},
		[]Value{inputTensor},
		[]Value{outputTensor},
		nil,
	)
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}
	defer func() {
		if destroyErr := session.Destroy(); destroyErr != nil {
			t.Errorf("session destroy failed: %v", destroyErr)
		}
	}()

	if err := session.Run(); err != nil {
		t.Fatalf("session run failed: %v", err)
	}
}

func TestAdvancedSessionRunWithAllMiniLML6V2(t *testing.T) {
	cleanup := setupTestEnvironment(t)
	defer cleanup()

	modelPath := resolveAllMiniLMModelPath(t)
	sequenceLength := allMiniLMSequenceLength(t)
	output := runAllMiniLMInference(t, modelPath, sequenceLength)
	requireFiniteFloat32Slice(t, "all-MiniLM output", output)
}
