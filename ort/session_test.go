package ort

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
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
			name:         "zero handle output value",
			modelPath:    "model.onnx",
			inputNames:   []string{"input"},
			outputNames:  []string{"output"},
			inputValues:  []Value{validValue},
			outputValues: []Value{&fakeValue{handle: 0}},
			wantErr:      "value handle is not initialized",
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

	if err := session.Destroy(); err != nil {
		t.Fatalf("second destroy should be no-op, got: %v", err)
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

	inputShape, err := parseShape(inputShapeRaw)
	if err != nil {
		t.Fatalf("invalid ONNXRUNTIME_TEST_INPUT_SHAPE: %v", err)
	}
	outputShape, err := parseShape(outputShapeRaw)
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

func parseShape(raw string) (Shape, error) {
	parts := strings.Split(raw, ",")
	out := make(Shape, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil, fmt.Errorf("empty dimension")
		}
		dim, err := strconv.ParseInt(part, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse dimension %q: %w", part, err)
		}
		out = append(out, dim)
	}
	return out, nil
}
