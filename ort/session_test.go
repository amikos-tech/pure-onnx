package ort

import (
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type fakeValue struct {
	handle uintptr
}

func (f *fakeValue) Destroy() error          { return nil }
func (f *fakeValue) Type() ValueType         { return ValueTypeTensor }
func (f *fakeValue) ortValueHandle() uintptr { return f.handle }

type unsupportedValue struct{}

func (u *unsupportedValue) Destroy() error  { return nil }
func (u *unsupportedValue) Type() ValueType { return ValueTypeTensor }

const (
	allMiniLMModelURL           = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
	allMiniLMOutputEmbeddingDim = int64(384)
)

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
		_ = session.Destroy()
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

	resetEnvironmentState()
}

func TestAdvancedSessionDestroy(t *testing.T) {
	resetEnvironmentState()

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

	resetEnvironmentState()
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

func TestAdvancedSessionRunAndDestroyConcurrent(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	runStarted := make(chan struct{})
	allowRunReturn := make(chan struct{})
	var closeRunStarted sync.Once

	releasedCount := int32(0)
	var releasedHandle atomic.Uintptr

	mu.Lock()
	ortAPI = &OrtApi{}
	runSessionFunc = func(session uintptr, runOptions uintptr, inputNames *uintptr, inputValues *uintptr, inputLen uintptr, outputNames *uintptr, outputLen uintptr, outputValues *uintptr) uintptr {
		closeRunStarted.Do(func() { close(runStarted) })
		<-allowRunReturn
		return 0
	}
	releaseSessionFunc = func(handle uintptr) {
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

	destroyErrCh := make(chan error, 1)
	go func() {
		destroyErrCh <- session.Destroy()
	}()

	select {
	case err := <-destroyErrCh:
		t.Fatalf("destroy returned before run completed: %v", err)
	case <-time.After(500 * time.Millisecond):
	}

	close(allowRunReturn)

	if err := <-runErrCh; err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if err := <-destroyErrCh; err != nil {
		t.Fatalf("destroy failed: %v", err)
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

	select {
	case err := <-destroyErrCh:
		if err != nil {
			t.Fatalf("destroy failed: %v", err)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("destroy should not block on unrelated in-flight Run")
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

	mu.Lock()
	ortAPI = &OrtApi{}
	runSessionFunc = func(session uintptr, runOptions uintptr, inputNames *uintptr, inputValues *uintptr, inputLen uintptr, outputNames *uintptr, outputLen uintptr, outputValues *uintptr) uintptr {
		closeRunStarted.Do(func() { close(runStarted) })
		<-allowRunReturn
		return 0
	}
	releaseValueFunc = func(handle uintptr) {
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

	tensorDestroyErrCh := make(chan error, 1)
	go func() {
		tensorDestroyErrCh <- inputTensor.Destroy()
	}()

	select {
	case err := <-tensorDestroyErrCh:
		t.Fatalf("tensor destroy returned before run completed: %v", err)
	case <-time.After(500 * time.Millisecond):
	}

	close(allowRunReturn)

	if err := <-runErrCh; err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if err := <-tensorDestroyErrCh; err != nil {
		t.Fatalf("tensor destroy failed: %v", err)
	}

	if got := atomic.LoadInt32(&releasedTensor); got != 1 {
		t.Fatalf("expected tensor release callback once, got %d", got)
	}
}

func TestAdvancedSessionRunDestroyedInputValue(t *testing.T) {
	resetEnvironmentState()

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

	resetEnvironmentState()
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
		_ = inputTensor.Destroy()
	}()

	outputTensor, err := NewEmptyTensor[float32](Shape{1})
	if err != nil {
		t.Fatalf("failed to create output tensor: %v", err)
	}
	defer func() {
		_ = outputTensor.Destroy()
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
		_ = inputTensor.Destroy()
	}()

	outputTensor, err := NewEmptyTensor[float32](outputShape)
	if err != nil {
		t.Fatalf("failed to create output tensor: %v", err)
	}
	defer func() {
		_ = outputTensor.Destroy()
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
		_ = session.Destroy()
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

	inputShape := Shape{1, int64(sequenceLength)}
	outputShape := Shape{1, int64(sequenceLength), allMiniLMOutputEmbeddingDim}

	inputIDs, attentionMask, tokenTypeIDs := makeAllMiniLMInputs(sequenceLength)

	inputIDsTensor, err := NewTensor[int64](inputShape, inputIDs)
	if err != nil {
		t.Fatalf("failed to create input_ids tensor: %v", err)
	}
	defer func() {
		_ = inputIDsTensor.Destroy()
	}()

	attentionMaskTensor, err := NewTensor[int64](inputShape, attentionMask)
	if err != nil {
		t.Fatalf("failed to create attention_mask tensor: %v", err)
	}
	defer func() {
		_ = attentionMaskTensor.Destroy()
	}()

	tokenTypeIDsTensor, err := NewTensor[int64](inputShape, tokenTypeIDs)
	if err != nil {
		t.Fatalf("failed to create token_type_ids tensor: %v", err)
	}
	defer func() {
		_ = tokenTypeIDsTensor.Destroy()
	}()

	outputTensor, err := NewEmptyTensor[float32](outputShape)
	if err != nil {
		t.Fatalf("failed to create output tensor: %v", err)
	}
	defer func() {
		_ = outputTensor.Destroy()
	}()

	session, err := NewAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		[]Value{inputIDsTensor, attentionMaskTensor, tokenTypeIDsTensor},
		[]Value{outputTensor},
		nil,
	)
	if err != nil {
		t.Fatalf("failed to create all-MiniLM session: %v", err)
	}
	defer func() {
		_ = session.Destroy()
	}()

	if err := session.Run(); err != nil {
		t.Fatalf("all-MiniLM inference failed: %v", err)
	}

	output := outputTensor.GetData()
	expected := sequenceLength * int(allMiniLMOutputEmbeddingDim)
	if len(output) != expected {
		t.Fatalf("unexpected output length: got %d want %d", len(output), expected)
	}
	for i, value := range output {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("output contains non-finite value at index %d: %v", i, value)
		}
	}
}

func allMiniLMSequenceLength(tb testing.TB) int {
	tb.Helper()

	raw := os.Getenv("ONNXRUNTIME_TEST_ALL_MINILM_SEQUENCE_LENGTH")
	if raw == "" {
		return 8
	}

	sequenceLength, err := strconv.Atoi(raw)
	if err != nil {
		tb.Fatalf("invalid ONNXRUNTIME_TEST_ALL_MINILM_SEQUENCE_LENGTH %q: %v", raw, err)
	}
	if sequenceLength < 2 {
		tb.Fatalf("ONNXRUNTIME_TEST_ALL_MINILM_SEQUENCE_LENGTH must be >= 2, got %d", sequenceLength)
	}

	return sequenceLength
}

func resolveAllMiniLMModelPath(tb testing.TB) string {
	tb.Helper()

	if modelPath := os.Getenv("ONNXRUNTIME_TEST_ALL_MINILM_MODEL_PATH"); modelPath != "" {
		if _, err := os.Stat(modelPath); err != nil {
			tb.Fatalf("ONNXRUNTIME_TEST_ALL_MINILM_MODEL_PATH %q is not usable: %v", modelPath, err)
		}
		return modelPath
	}

	cacheRoot := os.Getenv("ONNXRUNTIME_TEST_MODEL_CACHE_DIR")
	if cacheRoot == "" {
		userCacheDir, err := os.UserCacheDir()
		if err != nil {
			tb.Skipf("cannot determine user cache directory: %v; set ONNXRUNTIME_TEST_ALL_MINILM_MODEL_PATH", err)
		}
		cacheRoot = filepath.Join(userCacheDir, "onnx-purego", "models")
	}

	modelPath := filepath.Join(cacheRoot, "all-MiniLM-L6-v2.onnx")
	if info, err := os.Stat(modelPath); err == nil && info.Size() > 0 {
		return modelPath
	}

	if err := os.MkdirAll(filepath.Dir(modelPath), 0o755); err != nil {
		tb.Fatalf("failed to create model cache directory: %v", err)
	}

	url := os.Getenv("ONNXRUNTIME_TEST_ALL_MINILM_MODEL_URL")
	if url == "" {
		url = allMiniLMModelURL
	}

	tb.Logf("downloading all-MiniLM model from %s", url)
	if err := downloadModelFile(modelPath, url); err != nil {
		tb.Skipf("unable to download all-MiniLM model: %v", err)
	}

	return modelPath
}

func downloadModelFile(destinationPath string, modelURL string) error {
	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		if attempt > 1 {
			time.Sleep(time.Duration(attempt) * time.Second)
		}
		if err := downloadModelFileOnce(destinationPath, modelURL); err == nil {
			return nil
		} else {
			lastErr = err
		}
	}
	return lastErr
}

func downloadModelFileOnce(destinationPath string, modelURL string) error {
	client := &http.Client{Timeout: 3 * time.Minute}
	response, err := client.Get(modelURL)
	if err != nil {
		return err
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected HTTP status %d", response.StatusCode)
	}

	tempPath := destinationPath + ".tmp"
	file, err := os.Create(tempPath)
	if err != nil {
		return err
	}

	copyErr := error(nil)
	if _, err := io.Copy(file, response.Body); err != nil {
		copyErr = err
	}
	closeErr := file.Close()
	if copyErr != nil {
		_ = os.Remove(tempPath)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tempPath)
		return closeErr
	}

	if err := os.Rename(tempPath, destinationPath); err != nil {
		_ = os.Remove(tempPath)
		return err
	}

	return nil
}

func makeAllMiniLMInputs(sequenceLength int) ([]int64, []int64, []int64) {
	// [CLS] this is a test [SEP] + padding.
	templateTokenIDs := []int64{101, 2023, 2003, 1037, 3231, 102}

	inputIDs := make([]int64, sequenceLength)
	attentionMask := make([]int64, sequenceLength)
	tokenTypeIDs := make([]int64, sequenceLength)

	nonPaddingCount := sequenceLength
	if nonPaddingCount > len(templateTokenIDs) {
		nonPaddingCount = len(templateTokenIDs)
	}

	copy(inputIDs, templateTokenIDs[:nonPaddingCount])
	for i := 0; i < nonPaddingCount; i++ {
		attentionMask[i] = 1
	}

	return inputIDs, attentionMask, tokenTypeIDs
}
