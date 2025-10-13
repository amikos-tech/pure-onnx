package ort

import "fmt"

// Tensor represents a tensor with data of type T
type Tensor[T any] struct {
	shape Shape
	data  []T
	// TODO: Add OrtValue pointer when implementing actual tensor operations
}

// NewTensor creates a new tensor with the given shape and data
func NewTensor[T any](shape Shape, data []T) (*Tensor[T], error) {
	// TODO: Implement tensor creation as per issue #3
	return nil, fmt.Errorf("not yet implemented")
}

// NewEmptyTensor creates a new empty tensor with the given shape
func NewEmptyTensor[T any](shape Shape) (*Tensor[T], error) {
	// TODO: Implement empty tensor creation as per issue #3
	return nil, fmt.Errorf("not yet implemented")
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
	// TODO: Implement tensor cleanup as per issue #3
	return nil
}

// Type returns the value type (always ValueTypeTensor for tensors)
func (t *Tensor[T]) Type() ValueType {
	return ValueTypeTensor
}
