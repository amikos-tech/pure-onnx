package ort

import "sync"

// OrtApiBase represents the base API structure
type OrtApiBase struct {
	GetApi           uintptr
	GetVersionString uintptr
}

// OrtApi is defined in ortapi_generated.go (auto-generated from C header)

// Status represents a borrowed native OrtStatus handle.
// Thread-safe: Status can be shared across goroutines for read operations.
type Status uintptr

// IsOK returns true if the status represents success
func (s Status) IsOK() bool {
	return s == 0
}

// GetErrorCode returns the error code from the status
func (s Status) GetErrorCode() ErrorCode {
	if s.IsOK() {
		return ErrorCodeOK
	}

	ortCallMu.RLock()
	defer ortCallMu.RUnlock()

	mu.Lock()
	getErrorCode := getErrorCodeFunc
	mu.Unlock()
	if getErrorCode == nil {
		return ErrorCodeFail
	}
	return getErrorCode(uintptr(s))
}

// GetErrorMessage returns the error message from the status
func (s Status) GetErrorMessage() string {
	if s.IsOK() {
		return ""
	}

	ortCallMu.RLock()
	defer ortCallMu.RUnlock()

	mu.Lock()
	getErrorMessage := getErrorMessageFunc
	mu.Unlock()
	if getErrorMessage == nil {
		return ""
	}
	return CstringToGo(getErrorMessage(uintptr(s)))
}

// Environment represents a borrowed native OrtEnv handle.
type Environment uintptr

// Session represents a borrowed native OrtSession handle.
type Session uintptr

// Value represents an ONNX Runtime value created by this package.
//
// Value is intentionally sealed: external implementations are unsupported because
// native handles and their run-lifetime protocol remain package-owned.
type Value interface {
	// Destroy releases the underlying resources
	Destroy() error
	// Type returns the type of the value
	Type() ValueType
	ortValue()
}

// IsTensor reports whether value has the ONNX tensor kind.
func IsTensor(value Value) bool {
	return value != nil && value.Type() == ValueTypeTensor
}

// AsTensor returns value as an exact, non-nil *Tensor[T].
func AsTensor[T any](value Value) (*Tensor[T], bool) {
	tensor, ok := value.(*Tensor[T])
	if !ok || tensor == nil {
		return nil, false
	}
	return tensor, true
}

// ValueType represents the type of an ONNX Runtime value
type ValueType int

const (
	ValueTypeUnknown ValueType = iota
	ValueTypeTensor
	ValueTypeSequence
	ValueTypeMap
	ValueTypeOpaque
	ValueTypeOptional
)

// Shape represents the shape of a tensor
type Shape []int64

// NewShape creates a new shape from dimensions
func NewShape(dims ...int64) Shape {
	return Shape(dims)
}

// SessionOptions represents options for creating a session.
// It is not safe to mutate a SessionOptions instance concurrently with session creation.
type SessionOptions struct {
	handle                 uintptr // Pointer to OrtSessionOptions
	handleMu               sync.RWMutex
	graphOptimizationLevel GraphOptimizationLevel
	executionMode          ExecutionMode
	interOpNumThreads      int
	intraOpNumThreads      int
	logSeverityLevel       LoggingLevel
	logVerbosityLevel      int
	logID                  string
	enableCPUMemArena      bool
	enableMemPattern       bool
	enableProfiling        bool
	optimizedModelFilePath string
}

// MemoryInfo represents memory allocation information
type MemoryInfo struct {
	handle        uintptr // Pointer to OrtMemoryInfo
	handleMu      sync.RWMutex
	name          string
	memType       MemType
	allocatorType AllocatorType
	deviceID      int
}

// TypeInfo represents type information for an ONNX value
type TypeInfo struct {
	handle uintptr // Pointer to OrtTypeInfo
}

// TensorTypeAndShapeInfo represents tensor type and shape information
type TensorTypeAndShapeInfo struct {
	handle      uintptr // Pointer to OrtTensorTypeAndShapeInfo
	elementType TensorElementDataType
	shape       Shape
}

// RunOptions represents options for running inference
type RunOptions struct {
	handle            uintptr // Pointer to OrtRunOptions
	logVerbosityLevel int
	logSeverityLevel  LoggingLevel
	runTag            string
	terminate         bool
}

// CustomOpDomain represents a custom operator domain
type CustomOpDomain struct {
	handle uintptr // Pointer to OrtCustomOpDomain
	domain string
}
