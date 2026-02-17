package ort

import (
	"reflect"
	"strings"
	"testing"
	"unsafe"
)

func TestTensorElementType(t *testing.T) {
	tests := []struct {
		name      string
		fn        func() (TensorElementDataType, uintptr, error)
		wantType  TensorElementDataType
		wantSize  uintptr
		expectErr bool
	}{
		{
			name: "float32",
			fn: func() (TensorElementDataType, uintptr, error) {
				return tensorElementType[float32]()
			},
			wantType: TensorElementDataTypeFloat,
			wantSize: unsafe.Sizeof(float32(0)),
		},
		{
			name: "float64",
			fn: func() (TensorElementDataType, uintptr, error) {
				return tensorElementType[float64]()
			},
			wantType: TensorElementDataTypeDouble,
			wantSize: unsafe.Sizeof(float64(0)),
		},
		{
			name: "int32",
			fn: func() (TensorElementDataType, uintptr, error) {
				return tensorElementType[int32]()
			},
			wantType: TensorElementDataTypeInt32,
			wantSize: unsafe.Sizeof(int32(0)),
		},
		{
			name: "int64",
			fn: func() (TensorElementDataType, uintptr, error) {
				return tensorElementType[int64]()
			},
			wantType: TensorElementDataTypeInt64,
			wantSize: unsafe.Sizeof(int64(0)),
		},
		{
			name: "unsupported uint16",
			fn: func() (TensorElementDataType, uintptr, error) {
				return tensorElementType[uint16]()
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotType, gotSize, err := tt.fn()
			if tt.expectErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				if !strings.Contains(err.Error(), "unsupported tensor element type") {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if gotType != tt.wantType {
				t.Fatalf("unexpected tensor type: got %v, want %v", gotType, tt.wantType)
			}

			if gotSize != tt.wantSize {
				t.Fatalf("unexpected tensor size: got %d, want %d", gotSize, tt.wantSize)
			}
		})
	}
}

func TestShapeElementCount(t *testing.T) {
	tests := []struct {
		name      string
		shape     Shape
		wantCount int
		wantErr   string
	}{
		{
			name:      "scalar shape",
			shape:     Shape{},
			wantCount: 1,
		},
		{
			name:      "standard shape",
			shape:     Shape{2, 3, 4},
			wantCount: 24,
		},
		{
			name:      "zero dimension",
			shape:     Shape{2, 0, 4},
			wantCount: 0,
		},
		{
			name:    "negative dimension",
			shape:   Shape{2, -1},
			wantErr: "must be >= 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := shapeElementCount(tt.shape)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.wantErr)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("expected error containing %q, got %q", tt.wantErr, err.Error())
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.wantCount {
				t.Fatalf("unexpected element count: got %d, want %d", got, tt.wantCount)
			}
		})
	}
}

func TestTensorDataByteSizeOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	_, err := tensorDataByteSize(maxInt, 3)
	if err == nil {
		t.Fatalf("expected overflow error")
	}
	if !strings.Contains(err.Error(), "overflow") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestNewTensorValidationErrorsWithoutORT(t *testing.T) {
	resetEnvironmentState()

	_, err := NewTensor[float32](Shape{2, 2}, []float32{1, 2, 3})
	if err == nil || !strings.Contains(err.Error(), "data length mismatch") {
		t.Fatalf("expected data length mismatch error, got: %v", err)
	}

	_, err = NewTensor[uint16](Shape{1}, []uint16{1})
	if err == nil || !strings.Contains(err.Error(), "unsupported tensor element type") {
		t.Fatalf("expected unsupported type error, got: %v", err)
	}

	_, err = NewTensor[float32](Shape{1}, []float32{1})
	if err == nil || !strings.Contains(err.Error(), "ONNX Runtime not initialized") {
		t.Fatalf("expected not initialized error, got: %v", err)
	}
}

func TestNewEmptyTensorWithoutORT(t *testing.T) {
	resetEnvironmentState()

	_, err := NewEmptyTensor[float32](Shape{2, 2})
	if err == nil || !strings.Contains(err.Error(), "ONNX Runtime not initialized") {
		t.Fatalf("expected not initialized error, got: %v", err)
	}
}

func TestTensorDestroyNil(t *testing.T) {
	var tns *Tensor[float32]
	if err := tns.Destroy(); err != nil {
		t.Fatalf("destroy on nil tensor should be a no-op, got error: %v", err)
	}
}

func TestTensorDestroyDoubleWithoutORT(t *testing.T) {
	resetEnvironmentState()

	tensor := &Tensor[float32]{
		handle: 123,
		data:   []float32{1, 2, 3},
		shape:  Shape{3},
	}

	if err := tensor.Destroy(); err != nil {
		t.Fatalf("first destroy failed: %v", err)
	}
	if tensor.handle != 0 {
		t.Fatalf("expected handle to be reset")
	}
	if tensor.data != nil || tensor.shape != nil {
		t.Fatalf("expected tensor fields to be cleared")
	}

	// With ORT funcs unset, second destroy should remain a safe no-op.
	if err := tensor.Destroy(); err != nil {
		t.Fatalf("second destroy should be no-op, got: %v", err)
	}
}

func TestNewTensorWithORT(t *testing.T) {
	cleanup := setupTestEnvironment(t)
	defer cleanup()

	input := []float32{1, 2, 3, 4}
	tensor, err := NewTensor[float32](Shape{2, 2}, input)
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	defer func() {
		if err := tensor.Destroy(); err != nil {
			t.Fatalf("tensor destroy failed: %v", err)
		}
	}()

	if tensor.handle == 0 {
		t.Fatal("tensor handle should be non-zero")
	}

	if !reflect.DeepEqual(tensor.Shape(), Shape{2, 2}) {
		t.Fatalf("unexpected shape: got %v, want [2 2]", tensor.Shape())
	}

	if !reflect.DeepEqual(tensor.GetData(), input) {
		t.Fatalf("unexpected data: got %v, want %v", tensor.GetData(), input)
	}
}

func TestNewEmptyTensorWithORT(t *testing.T) {
	cleanup := setupTestEnvironment(t)
	defer cleanup()

	tensor, err := NewEmptyTensor[float32](Shape{2, 3})
	if err != nil {
		t.Fatalf("NewEmptyTensor failed: %v", err)
	}

	if tensor.handle == 0 {
		t.Fatal("tensor handle should be non-zero")
	}

	data := tensor.GetData()
	if len(data) != 6 {
		t.Fatalf("unexpected data length: got %d, want 6", len(data))
	}

	data[0] = 42.5
	if tensor.GetData()[0] != 42.5 {
		t.Fatalf("tensor data mutation was not reflected")
	}

	if err := tensor.Destroy(); err != nil {
		t.Fatalf("first destroy failed: %v", err)
	}
	if err := tensor.Destroy(); err != nil {
		t.Fatalf("second destroy should be no-op, got: %v", err)
	}
}
