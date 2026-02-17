package ort

import (
	"fmt"
	"runtime"
	"unsafe"
)

// Tensor represents a tensor with data of type T
type Tensor[T any] struct {
	shape  Shape
	data   []T
	handle uintptr         // Pointer to OrtValue
	pinner *runtime.Pinner // Pins data backing array while OrtValue may access it.
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

	return newTensorFromData(shapeCopy, data, elementType, elementSize)
}

// NewEmptyTensor creates a new empty tensor with the given shape
func NewEmptyTensor[T any](shape Shape) (*Tensor[T], error) {
	elementType, elementSize, err := tensorElementType[T]()
	if err != nil {
		return nil, err
	}

	shapeCopy := cloneShape(shape)
	elementCount, err := shapeElementCount(shapeCopy)
	if err != nil {
		return nil, err
	}

	data := make([]T, elementCount)

	return newTensorFromData(shapeCopy, data, elementType, elementSize)
}

func newTensorFromData[T any](shape Shape, data []T, elementType TensorElementDataType, elementSize uintptr) (*Tensor[T], error) {
	dataBytes, err := tensorDataByteSize(len(data), elementSize)
	if err != nil {
		return nil, err
	}

	ortCallMu.RLock()
	defer ortCallMu.RUnlock()

	mu.Lock()
	if ortAPI == nil || createMemoryInfoFunc == nil || releaseMemoryInfoFunc == nil || createTensorWithDataAsOrtValueFunc == nil {
		mu.Unlock()
		return nil, fmt.Errorf("ONNX Runtime not initialized")
	}
	createMemoryInfo := createMemoryInfoFunc
	releaseMemoryInfo := releaseMemoryInfoFunc
	createTensorWithData := createTensorWithDataAsOrtValueFunc
	mu.Unlock()

	nameBytes, namePtr := GoToCstring("Cpu")
	var memInfo uintptr
	status := createMemoryInfo(namePtr, AllocatorTypeArena, 0, MemTypeCPU, &memInfo)
	runtime.KeepAlive(nameBytes)
	if status != 0 {
		errMsg := getErrorMessage(status)
		releaseStatus(status)
		return nil, fmt.Errorf("failed to create CPU memory info: %s", errMsg)
	}
	defer releaseMemoryInfo(memInfo)

	var dataPtr uintptr
	var pinner *runtime.Pinner
	if len(data) > 0 {
		pinner = &runtime.Pinner{}
		pinner.Pin(unsafe.SliceData(data))
		// #nosec G103 -- Required for CGO-free FFI; backing array is pinned for OrtValue lifetime via runtime.Pinner.
		dataPtr = uintptr(unsafe.Pointer(unsafe.SliceData(data)))
	}

	var valueHandle uintptr
	status = createTensorWithData(memInfo, dataPtr, dataBytes, shapePtr(shape), uintptr(len(shape)), elementType, &valueHandle)
	// ORT reads shape dimensions synchronously during CreateTensorWithDataAsOrtValue call.
	// Keep shape alive for the call; tensor data lifetime is guarded by pinner.
	runtime.KeepAlive(shape)
	if status != 0 {
		if pinner != nil {
			pinner.Unpin()
		}
		errMsg := getErrorMessage(status)
		releaseStatus(status)
		return nil, fmt.Errorf("failed to create tensor: %s", errMsg)
	}

	tensor := &Tensor[T]{
		shape:  shape,
		data:   data,
		handle: valueHandle,
		pinner: pinner,
	}

	// Finalizer is a safety net to avoid leaking OrtValue if callers forget Destroy().
	runtime.SetFinalizer(tensor, func(t *Tensor[T]) {
		_ = t.Destroy()
	})

	return tensor, nil
}

// GetData returns the tensor data.
// After Destroy() it returns nil. Calling on a nil receiver also returns nil.
func (t *Tensor[T]) GetData() []T {
	if t == nil {
		return nil
	}
	return t.data
}

// Shape returns the tensor shape
func (t *Tensor[T]) Shape() Shape {
	if t == nil {
		return nil
	}
	return t.shape
}

// Destroy releases the tensor resources
func (t *Tensor[T]) Destroy() error {
	if t == nil {
		return nil
	}

	// Lock order here is ortCallMu -> mu.
	ortCallMu.Lock()
	defer ortCallMu.Unlock()

	var handle uintptr
	var releaseValue func(uintptr)
	var pinner *runtime.Pinner

	mu.Lock()
	handle = t.handle
	releaseValue = releaseValueFunc
	pinner = t.pinner
	t.handle = 0
	t.data = nil
	t.shape = nil
	t.pinner = nil
	runtime.SetFinalizer(t, nil)
	mu.Unlock()

	if handle != 0 && releaseValue != nil {
		releaseValue(handle)
	}
	if pinner != nil {
		pinner.Unpin()
	}

	return nil
}

// Type returns the value type (always ValueTypeTensor for tensors)
func (t *Tensor[T]) Type() ValueType {
	return ValueTypeTensor
}

func cloneShape(shape Shape) Shape {
	if len(shape) == 0 {
		// Keep scalar tensors as non-nil empty shape (rank 0), not nil.
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

// ShapeElementCount returns the total element count for a shape.
// Dimensions must be non-negative; zero dimensions produce a count of zero.
func ShapeElementCount(shape Shape) (int, error) {
	return shapeElementCount(shape)
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

// tensorElementType maps Go generic element type T to ONNX tensor element metadata.
// Supported types in this MVP are float32, float64, int32, and int64.
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
