package ort

import (
	"fmt"
	"runtime"
	"unsafe"
)

// AdvancedSession represents an ONNX Runtime inference session
type AdvancedSession struct {
	handle       uintptr
	inputNames   []string
	outputNames  []string
	inputValues  []Value
	outputValues []Value
}

// NewAdvancedSession creates a new session with specified inputs and outputs
func NewAdvancedSession(modelPath string, inputNames []string, outputNames []string,
	inputValues []Value, outputValues []Value, options *SessionOptions) (*AdvancedSession, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path cannot be empty")
	}
	if len(inputNames) == 0 {
		return nil, fmt.Errorf("at least one input name is required")
	}
	if len(outputNames) == 0 {
		return nil, fmt.Errorf("at least one output name is required")
	}
	if len(inputNames) != len(inputValues) {
		return nil, fmt.Errorf("input names/values count mismatch: got %d names and %d values", len(inputNames), len(inputValues))
	}
	if len(outputNames) != len(outputValues) {
		return nil, fmt.Errorf("output names/values count mismatch: got %d names and %d values", len(outputNames), len(outputValues))
	}

	for i, v := range inputValues {
		_, err := valueHandle(v)
		if err != nil {
			return nil, fmt.Errorf("invalid input value at index %d: %w", i, err)
		}
	}
	for i, v := range outputValues {
		_, err := valueHandle(v)
		if err != nil {
			return nil, fmt.Errorf("invalid output value at index %d: %w", i, err)
		}
	}
	if options != nil && options.handle == 0 {
		return nil, fmt.Errorf("session options handle is not initialized")
	}

	mu.Lock()
	defer mu.Unlock()

	if ortAPI == nil || ortEnv == 0 || createSessionOptionsFunc == nil || releaseSessionOptionsFunc == nil || createSessionFunc == nil {
		return nil, fmt.Errorf("ONNX Runtime not initialized")
	}

	sessionOptionsHandle := uintptr(0)
	releaseCreatedOptions := false
	if options != nil {
		sessionOptionsHandle = options.handle
	} else {
		status := createSessionOptionsFunc(&sessionOptionsHandle)
		if status != 0 {
			errMsg := getErrorMessage(status)
			releaseStatus(status)
			return nil, fmt.Errorf("failed to create session options: %s", errMsg)
		}
		releaseCreatedOptions = true
	}
	if releaseCreatedOptions {
		defer releaseSessionOptionsFunc(sessionOptionsHandle)
	}

	modelPathPtr, modelPathBacking, err := goStringToORTChar(modelPath)
	if err != nil {
		return nil, err
	}

	var sessionHandle uintptr
	status := createSessionFunc(ortEnv, modelPathPtr, sessionOptionsHandle, &sessionHandle)
	runtime.KeepAlive(modelPathBacking)
	if status != 0 {
		errMsg := getErrorMessage(status)
		releaseStatus(status)
		return nil, fmt.Errorf("failed to create session: %s", errMsg)
	}

	session := &AdvancedSession{
		handle:       sessionHandle,
		inputNames:   cloneStringSlice(inputNames),
		outputNames:  cloneStringSlice(outputNames),
		inputValues:  cloneValueSlice(inputValues),
		outputValues: cloneValueSlice(outputValues),
	}

	runtime.SetFinalizer(session, func(s *AdvancedSession) {
		_ = s.Destroy()
	})

	return session, nil
}

// Run executes inference on the session
func (s *AdvancedSession) Run() error {
	if s == nil {
		return fmt.Errorf("session is nil")
	}

	mu.Lock()
	defer mu.Unlock()

	if ortAPI == nil || runSessionFunc == nil {
		return fmt.Errorf("ONNX Runtime not initialized")
	}
	if s.handle == 0 {
		return fmt.Errorf("session has been destroyed")
	}
	if len(s.inputNames) == 0 || len(s.outputNames) == 0 {
		return fmt.Errorf("session is missing input/output names")
	}
	if len(s.inputNames) != len(s.inputValues) {
		return fmt.Errorf("session input names/values count mismatch: got %d names and %d values", len(s.inputNames), len(s.inputValues))
	}
	if len(s.outputNames) != len(s.outputValues) {
		return fmt.Errorf("session output names/values count mismatch: got %d names and %d values", len(s.outputNames), len(s.outputValues))
	}

	inputNameBackings, inputNamePtrs := makeCStringPointerArray(s.inputNames)
	outputNameBackings, outputNamePtrs := makeCStringPointerArray(s.outputNames)

	inputValueHandles, err := valuesToHandles(s.inputValues)
	if err != nil {
		return fmt.Errorf("failed to prepare input values: %w", err)
	}
	outputValueHandles, err := valuesToHandles(s.outputValues)
	if err != nil {
		return fmt.Errorf("failed to prepare output values: %w", err)
	}

	status := runSessionFunc(
		s.handle,
		0, // RunOptions not yet implemented
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
}

// Destroy releases the session resources
func (s *AdvancedSession) Destroy() error {
	if s == nil {
		return nil
	}

	mu.Lock()
	defer mu.Unlock()

	if s.handle != 0 && releaseSessionFunc != nil {
		releaseSessionFunc(s.handle)
	}

	s.handle = 0
	s.inputNames = nil
	s.outputNames = nil
	s.inputValues = nil
	s.outputValues = nil
	runtime.SetFinalizer(s, nil)

	return nil
}

type valueWithORTHandle interface {
	ortValueHandle() uintptr
}

func valueHandle(v Value) (uintptr, error) {
	if v == nil {
		return 0, fmt.Errorf("value is nil")
	}
	handleProvider, ok := v.(valueWithORTHandle)
	if !ok {
		return 0, fmt.Errorf("unsupported value implementation %T", v)
	}
	handle := handleProvider.ortValueHandle()
	if handle == 0 {
		return 0, fmt.Errorf("value handle is not initialized")
	}
	return handle, nil
}

func valuesToHandles(values []Value) ([]uintptr, error) {
	if len(values) == 0 {
		return nil, nil
	}
	handles := make([]uintptr, len(values))
	for i, v := range values {
		handle, err := valueHandle(v)
		if err != nil {
			return nil, fmt.Errorf("value at index %d is invalid: %w", i, err)
		}
		handles[i] = handle
	}
	return handles, nil
}

func cloneStringSlice(input []string) []string {
	if len(input) == 0 {
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

func makeCStringPointerArray(values []string) ([][]byte, []uintptr) {
	if len(values) == 0 {
		return nil, nil
	}

	backings := make([][]byte, len(values))
	ptrs := make([]uintptr, len(values))
	for i, value := range values {
		bytes, ptr := GoToCstring(value)
		backings[i] = bytes
		ptrs[i] = ptr
	}
	return backings, ptrs
}

func uintptrSlicePtr(values []uintptr) *uintptr {
	if len(values) == 0 {
		return nil
	}
	// #nosec G103 -- Required for CGO-free FFI to pass pointer arrays to ONNX Runtime C API.
	return (*uintptr)(unsafe.Pointer(unsafe.SliceData(values)))
}
