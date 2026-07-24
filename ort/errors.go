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

	// ErrUnsupportedRuntime identifies an ONNX Runtime library that does not
	// expose the API version required by this package.
	ErrUnsupportedRuntime = errors.New("unsupported ONNX Runtime API")

	// ErrNativeContract identifies a successful native call that did not
	// populate a required output handle.
	ErrNativeContract = errors.New("ONNX Runtime native contract violation")
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

// statusToError copies a native status into a Go-owned error and releases it.
// Every production caller must hold ortCallMu for the complete native call and
// conversion. InitializeEnvironment instead holds ortCallMu.Lock plus mu. Those
// scopes prevent runtime reset from clearing these accessors during conversion.
func statusToError(status uintptr, operation string) error {
	return statusToErrorWithOps(status, operation, statusOps{
		getCode: getErrorCodeFunc,
		copyMessage: func(status uintptr) string {
			return CstringToGo(getErrorMessageFunc(status))
		},
		release: releaseStatusFunc,
	})
}
