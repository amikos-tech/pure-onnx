package ort

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Tensor represents a tensor with data of type T
type Tensor[T any] struct {
	shape  Shape
	data   []T
	handle uintptr // Pointer to OrtValue
}

func (t *Tensor[T]) ortValueHandle() uintptr {
	if t == nil {
		return 0
	}
	return t.handle
}

// NewTensor creates a new tensor with the given shape and data
func NewTensor[T any](shape Shape, data []T) (*Tensor[T], error) {
	elementType, elementSize, err := tensorElementType[T]()
	if err != nil {
		return nil, err
	}

	shapeCopy := cloneShape(shape)
	elementCount, err := shapeElementCount(shapeCopy)
	if err != nil {
		return nil, err
	}
	if len(data) != elementCount {
		return nil, fmt.Errorf("data length mismatch: got %d elements, expected %d for shape %v", len(data), elementCount, shapeCopy)
	}

	dataBytes, err := tensorDataByteSize(len(data), elementSize)
	if err != nil {
		return nil, err
	}

	mu.Lock()
	defer mu.Unlock()

	if ortAPI == nil || createMemoryInfoFunc == nil || releaseMemoryInfoFunc == nil {
		return nil, fmt.Errorf("ONNX Runtime not initialized")
	}

	var createTensorWithDataAsOrtValue func(info uintptr, pData uintptr, pDataLen uintptr, shape *int64, shapeLen uintptr, dataType TensorElementDataType, out *uintptr) uintptr
	purego.RegisterFunc(&createTensorWithDataAsOrtValue, ortAPI.CreateTensorWithDataAsOrtValue)

	nameBytes, namePtr := GoToCstring("Cpu")
	var memInfo uintptr
	status := createMemoryInfoFunc(namePtr, AllocatorTypeArena, 0, MemTypeCPU, &memInfo)
	runtime.KeepAlive(nameBytes)
	if status != 0 {
		errMsg := getErrorMessage(status)
		releaseStatus(status)
		return nil, fmt.Errorf("failed to create CPU memory info: %s", errMsg)
	}
	defer releaseMemoryInfoFunc(memInfo)

	var dataPtr uintptr
	if len(data) > 0 {
		// #nosec G103 -- Required for CGO-free FFI; pointer remains valid via runtime.KeepAlive(data)
		dataPtr = uintptr(unsafe.Pointer(unsafe.SliceData(data)))
	}

	var valueHandle uintptr
	status = createTensorWithDataAsOrtValue(memInfo, dataPtr, dataBytes, shapePtr(shapeCopy), uintptr(len(shapeCopy)), elementType, &valueHandle)
	runtime.KeepAlive(data)
	runtime.KeepAlive(shapeCopy)
	if status != 0 {
		errMsg := getErrorMessage(status)
		releaseStatus(status)
		return nil, fmt.Errorf("failed to create tensor: %s", errMsg)
	}

	tensor := &Tensor[T]{
		shape:  shapeCopy,
		data:   data,
		handle: valueHandle,
	}

	// Finalizer is a safety net to avoid leaking OrtValue if callers forget Destroy().
	runtime.SetFinalizer(tensor, func(t *Tensor[T]) {
		_ = t.Destroy()
	})

	return tensor, nil
}

// NewEmptyTensor creates a new empty tensor with the given shape
func NewEmptyTensor[T any](shape Shape) (*Tensor[T], error) {
	elementType, _, err := tensorElementType[T]()
	if err != nil {
		return nil, err
	}

	shapeCopy := cloneShape(shape)
	elementCount, err := shapeElementCount(shapeCopy)
	if err != nil {
		return nil, err
	}

	mu.Lock()
	defer mu.Unlock()

	if ortAPI == nil {
		return nil, fmt.Errorf("ONNX Runtime not initialized")
	}

	var getAllocatorWithDefaultOptions func(out *uintptr) uintptr
	var createTensorAsOrtValue func(allocator uintptr, shape *int64, shapeLen uintptr, dataType TensorElementDataType, out *uintptr) uintptr
	var getTensorMutableData func(value uintptr, out *uintptr) uintptr
	var releaseValue func(value uintptr)

	purego.RegisterFunc(&getAllocatorWithDefaultOptions, ortAPI.GetAllocatorWithDefaultOptions)
	purego.RegisterFunc(&createTensorAsOrtValue, ortAPI.CreateTensorAsOrtValue)
	purego.RegisterFunc(&getTensorMutableData, ortAPI.GetTensorMutableData)
	purego.RegisterFunc(&releaseValue, ortAPI.ReleaseValue)

	var allocator uintptr
	status := getAllocatorWithDefaultOptions(&allocator)
	if status != 0 {
		errMsg := getErrorMessage(status)
		releaseStatus(status)
		return nil, fmt.Errorf("failed to get default allocator: %s", errMsg)
	}

	var valueHandle uintptr
	status = createTensorAsOrtValue(allocator, shapePtr(shapeCopy), uintptr(len(shapeCopy)), elementType, &valueHandle)
	runtime.KeepAlive(shapeCopy)
	if status != 0 {
		errMsg := getErrorMessage(status)
		releaseStatus(status)
		return nil, fmt.Errorf("failed to create empty tensor: %s", errMsg)
	}

	var rawData uintptr
	status = getTensorMutableData(valueHandle, &rawData)
	if status != 0 {
		releaseValue(valueHandle)
		errMsg := getErrorMessage(status)
		releaseStatus(status)
		return nil, fmt.Errorf("failed to access tensor data: %s", errMsg)
	}
	if elementCount > 0 && rawData == 0 {
		releaseValue(valueHandle)
		return nil, fmt.Errorf("failed to access tensor data: got null data pointer")
	}

	var data []T
	if elementCount > 0 {
		// #nosec G103 -- Required for CGO-free FFI; rawData points to memory owned by OrtValue.
		data = unsafe.Slice((*T)(unsafe.Pointer(rawData)), elementCount)
	} else {
		data = make([]T, 0)
	}

	tensor := &Tensor[T]{
		shape:  shapeCopy,
		data:   data,
		handle: valueHandle,
	}

	// Finalizer is a safety net to avoid leaking OrtValue if callers forget Destroy().
	runtime.SetFinalizer(tensor, func(t *Tensor[T]) {
		_ = t.Destroy()
	})

	return tensor, nil
}

// GetData returns the tensor data
func (t *Tensor[T]) GetData() []T {
	return t.data
}

// Shape returns the tensor shape
func (t *Tensor[T]) Shape() Shape {
	return t.shape
}

// Destroy releases the tensor resources
func (t *Tensor[T]) Destroy() error {
	if t == nil {
		return nil
	}

	mu.Lock()
	defer mu.Unlock()

	if t.handle != 0 && ortAPI != nil {
		var releaseValue func(value uintptr)
		purego.RegisterFunc(&releaseValue, ortAPI.ReleaseValue)
		releaseValue(t.handle)
	}

	t.handle = 0
	t.data = nil
	t.shape = nil
	runtime.SetFinalizer(t, nil)

	return nil
}

// Type returns the value type (always ValueTypeTensor for tensors)
func (t *Tensor[T]) Type() ValueType {
	return ValueTypeTensor
}

func cloneShape(shape Shape) Shape {
	if len(shape) == 0 {
		return Shape{}
	}

	shapeCopy := make(Shape, len(shape))
	copy(shapeCopy, shape)
	return shapeCopy
}

func shapeElementCount(shape Shape) (int, error) {
	maxInt := int(^uint(0) >> 1)

	count := 1
	for i, dim := range shape {
		if dim < 0 {
			return 0, fmt.Errorf("invalid shape dimension at index %d: %d (must be >= 0)", i, dim)
		}

		if dim == 0 {
			count = 0
			continue
		}

		if count == 0 {
			continue
		}

		if dim > int64(maxInt) {
			return 0, fmt.Errorf("shape dimension at index %d is too large: %d", i, dim)
		}

		dimInt := int(dim)
		if count > maxInt/dimInt {
			return 0, fmt.Errorf("shape %v exceeds maximum supported element count", shape)
		}

		count *= dimInt
	}

	return count, nil
}

func shapePtr(shape Shape) *int64 {
	if len(shape) == 0 {
		return nil
	}
	return unsafe.SliceData(shape)
}

func tensorDataByteSize(elementCount int, elementSize uintptr) (uintptr, error) {
	if elementCount < 0 {
		return 0, fmt.Errorf("element count cannot be negative: %d", elementCount)
	}
	if elementCount == 0 {
		return 0, nil
	}
	if elementSize == 0 {
		return 0, fmt.Errorf("element size cannot be zero")
	}

	count := uintptr(elementCount)
	if count > ^uintptr(0)/elementSize {
		return 0, fmt.Errorf("tensor data size overflow: %d elements with element size %d", elementCount, elementSize)
	}

	return count * elementSize, nil
}

func tensorElementType[T any]() (TensorElementDataType, uintptr, error) {
	var zero T

	switch any(zero).(type) {
	case float32:
		return TensorElementDataTypeFloat, unsafe.Sizeof(zero), nil
	case float64:
		return TensorElementDataTypeDouble, unsafe.Sizeof(zero), nil
	case int32:
		return TensorElementDataTypeInt32, unsafe.Sizeof(zero), nil
	case int64:
		return TensorElementDataTypeInt64, unsafe.Sizeof(zero), nil
	default:
		return TensorElementDataTypeUndefined, 0, fmt.Errorf("unsupported tensor element type %T", zero)
	}
}
