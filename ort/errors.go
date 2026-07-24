package ort

import (
	"errors"
	"fmt"
)

// ORTError describes a native ONNX Runtime failure using Go-owned data.
type ORTError struct {
	Operation string
	Code      ErrorCode
	Message   string
}

// Error returns the failed operation, native error code, and native message.
func (e *ORTError) Error() string {
	return fmt.Sprintf("%s: ONNX Runtime error %d: %s", e.Operation, e.Code, e.Message)
}

var (
	// ErrInvalidArgument identifies an invalid explicit argument or option.
	ErrInvalidArgument = errors.New("invalid argument")

	// ErrNotInitialized identifies missing runtime, function, or configuration
	// state required for an operation, including an unset shared-library path
	// during direct environment initialization.
	ErrNotInitialized = errors.New("ONNX Runtime is not initialized")

	// ErrDestroyed identifies a once-live resource whose native handle is gone.
	ErrDestroyed = errors.New("ONNX Runtime resource is destroyed")

	// ErrSharedLibraryNotFound identifies a supported-platform lookup that
	// completed without finding a usable ONNX Runtime shared library.
	ErrSharedLibraryNotFound = errors.New("ONNX Runtime shared library not found")
)

type statusOps struct {
	getCode     func(uintptr) ErrorCode
	copyMessage func(uintptr) string
	release     func(uintptr)
}

func statusToErrorWithOps(status uintptr, operation string, ops statusOps) error {
	if status == 0 {
		return nil
	}
	defer ops.release(status)

	return &ORTError{
		Operation: operation,
		Code:      ops.getCode(status),
		Message:   ops.copyMessage(status),
	}
}
