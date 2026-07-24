package ort

import (
	"bytes"
	"errors"
	"log/slog"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
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
				if !errors.Is(err, ErrInvalidArgument) {
					t.Fatalf("tensorElementType error = %v, want ErrInvalidArgument", err)
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
				if !errors.Is(err, ErrInvalidArgument) {
					t.Fatalf("shapeElementCount(%v) error = %v, want ErrInvalidArgument", tt.shape, err)
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
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("tensorDataByteSize error = %v, want ErrInvalidArgument", err)
	}
}

func TestNewTensorValidationErrorsWithoutORT(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	_, err := NewTensor[float32](Shape{2, 2}, []float32{1, 2, 3})
	if err == nil || !strings.Contains(err.Error(), "data length mismatch") {
		t.Fatalf("expected data length mismatch error, got: %v", err)
	}
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("data length mismatch error = %v, want ErrInvalidArgument", err)
	}

	_, err = NewTensor[uint16](Shape{1}, []uint16{1})
	if err == nil || !strings.Contains(err.Error(), "unsupported tensor element type") {
		t.Fatalf("expected unsupported type error, got: %v", err)
	}
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("unsupported type error = %v, want ErrInvalidArgument", err)
	}

	_, err = NewTensor[float32](Shape{-1}, nil)
	if err == nil || !strings.Contains(err.Error(), "must be >= 0") {
		t.Fatalf("expected invalid shape error, got: %v", err)
	}
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("invalid shape error = %v, want ErrInvalidArgument", err)
	}

	maxInt := int64(int(^uint(0) >> 1))
	_, err = NewTensor[float32](Shape{maxInt, 2}, nil)
	if err == nil || !strings.Contains(err.Error(), "exceeds maximum supported element count") {
		t.Fatalf("expected shape overflow error, got: %v", err)
	}
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("shape overflow error = %v, want ErrInvalidArgument", err)
	}

	_, err = NewTensor[float32](Shape{1}, []float32{1})
	if err == nil || !strings.Contains(err.Error(), "ONNX Runtime not initialized") {
		t.Fatalf("expected not initialized error, got: %v", err)
	}
	if !errors.Is(err, ErrNotInitialized) {
		t.Fatalf("uninitialized NewTensor error = %v, want ErrNotInitialized", err)
	}

	var nilTensor *Tensor[float32]
	if _, err = nilTensor.lockForRun(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil lockForRun error = %v, want ErrInvalidArgument", err)
	}
	if _, err = (&Tensor[float32]{}).lockForRun(); !errors.Is(err, ErrDestroyed) {
		t.Fatalf("destroyed lockForRun error = %v, want ErrDestroyed", err)
	}
}

func TestNewEmptyTensorWithoutORT(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	_, err := NewEmptyTensor[float32](Shape{2, 2})
	if err == nil || !strings.Contains(err.Error(), "ONNX Runtime not initialized") {
		t.Fatalf("expected not initialized error, got: %v", err)
	}
	if !errors.Is(err, ErrNotInitialized) {
		t.Fatalf("uninitialized NewEmptyTensor error = %v, want ErrNotInitialized", err)
	}
}

func TestTensorDestroyNil(t *testing.T) {
	var tns *Tensor[float32]
	if err := tns.Destroy(); err != nil {
		t.Fatalf("destroy on nil tensor should be a no-op, got error: %v", err)
	}
}

func TestTensorAccessorsNilReceiver(t *testing.T) {
	var tns *Tensor[float32]
	if data := tns.GetData(); data != nil {
		t.Fatalf("expected nil data for nil receiver, got %v", data)
	}
	if shape := tns.Shape(); shape != nil {
		t.Fatalf("expected nil shape for nil receiver, got %v", shape)
	}
}

func TestTensorDestroyDoubleWithoutORT(t *testing.T) {
	resetEnvironmentState()

	tensor := &Tensor[float32]{
		handle: 123,
		data:   []float32{1, 2, 3},
		shape:  Shape{3},
	}

	err := tensor.Destroy()
	if err == nil || !strings.Contains(err.Error(), "release function unavailable") {
		t.Fatalf("expected first destroy to fail with release-unavailable error, got: %v", err)
	}
	if !errors.Is(err, ErrNotInitialized) {
		t.Fatalf("first destroy error = %v, want ErrNotInitialized", err)
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

func TestTensorStatusConversion(t *testing.T) {
	t.Run("CreateMemoryInfo status", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		probe := installSessionStatusProbe(t, ErrorCodeInvalidArgument, "invalid memory info")
		mu.Lock()
		ortAPI = &OrtApi{}
		createMemoryInfoFunc = func(_ uintptr, _ AllocatorType, _ int32, _ MemType, _ *uintptr) uintptr {
			return probe.handle
		}
		releaseMemoryInfoFunc = func(uintptr) {
			t.Fatal("ReleaseMemoryInfo called after CreateMemoryInfo failed")
		}
		createTensorWithDataAsOrtValueFunc = func(_ uintptr, _ uintptr, _ uintptr, _ *int64, _ uintptr, _ TensorElementDataType, _ *uintptr) uintptr {
			t.Fatal("CreateTensorWithDataAsOrtValue called after CreateMemoryInfo failed")
			return 0
		}
		mu.Unlock()

		_, err := NewTensor[float32](Shape{1}, []float32{1})
		requireSessionORTError(
			t,
			err,
			"create CPU memory info",
			ErrorCodeInvalidArgument,
			"invalid memory info",
			&probe.releases,
		)
	})

	t.Run("CreateTensorWithDataAsOrtValue status", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		probe := installSessionStatusProbe(t, ErrorCodeRuntimeException, "tensor creation failed")
		var memoryInfoReleases atomic.Int32
		mu.Lock()
		ortAPI = &OrtApi{}
		createMemoryInfoFunc = func(_ uintptr, _ AllocatorType, _ int32, _ MemType, out *uintptr) uintptr {
			*out = 700
			return 0
		}
		releaseMemoryInfoFunc = func(handle uintptr) {
			if handle != 700 {
				t.Errorf("ReleaseMemoryInfo handle = %d, want 700", handle)
			}
			memoryInfoReleases.Add(1)
		}
		createTensorWithDataAsOrtValueFunc = func(
			memoryInfo uintptr,
			data uintptr,
			dataBytes uintptr,
			shape *int64,
			shapeLen uintptr,
			elementType TensorElementDataType,
			out *uintptr,
		) uintptr {
			if memoryInfo != 700 || data == 0 || dataBytes != unsafe.Sizeof(float32(0)) ||
				shape == nil || shapeLen != 1 || elementType != TensorElementDataTypeFloat || out == nil {
				t.Errorf(
					"unexpected tensor creation args: memoryInfo=%d data=%d bytes=%d shape=%p shapeLen=%d type=%d out=%p",
					memoryInfo,
					data,
					dataBytes,
					shape,
					shapeLen,
					elementType,
					out,
				)
			}
			return probe.handle
		}
		mu.Unlock()

		data := []float32{1}
		_, err := NewTensor[float32](Shape{1}, data)
		requireSessionORTError(
			t,
			err,
			"create tensor with data",
			ErrorCodeRuntimeException,
			"tensor creation failed",
			&probe.releases,
		)
		if got := memoryInfoReleases.Load(); got != 1 {
			t.Fatalf("memory info release count = %d, want 1", got)
		}
		if data[0] != 1 {
			t.Fatalf("tensor data changed after failed creation: %v", data)
		}
	})
}

func TestTensorDiagnosticPolicy(t *testing.T) {
	t.Run("returned failures emit no records", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		handler := &diagnosticCountingHandler{}
		SetDiagnosticHandler(handler)
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		if _, err := NewTensor[float32](Shape{2}, []float32{1}); err == nil {
			t.Fatal("invalid NewTensor returned nil error")
		}
		if _, err := NewEmptyTensor[float32](Shape{1}); err == nil {
			t.Fatal("uninitialized NewEmptyTensor returned nil error")
		}
		if err := (&Tensor[float32]{handle: 801}).Destroy(); err == nil {
			t.Fatal("release-unavailable Destroy returned nil error")
		}

		probe := installSessionStatusProbe(t, ErrorCodeFail, "native memory info failed")
		mu.Lock()
		ortAPI = &OrtApi{}
		createMemoryInfoFunc = func(_ uintptr, _ AllocatorType, _ int32, _ MemType, _ *uintptr) uintptr {
			return probe.handle
		}
		releaseMemoryInfoFunc = func(uintptr) {}
		createTensorWithDataAsOrtValueFunc = func(_ uintptr, _ uintptr, _ uintptr, _ *int64, _ uintptr, _ TensorElementDataType, _ *uintptr) uintptr {
			return 0
		}
		mu.Unlock()
		if _, err := NewTensor[float32](Shape{1}, []float32{1}); err == nil {
			t.Fatal("native NewTensor failure returned nil error")
		}

		if got := handler.count.Load(); got != 0 {
			t.Fatalf("returned tensor failures emitted %d diagnostic records, want 0", got)
		}
	})

	t.Run("finalizer-only failure emits one structured warning", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		var output bytes.Buffer
		SetDiagnosticHandler(slog.NewJSONHandler(&output, nil))
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		tensor := &Tensor[float32]{handle: 802}
		finalizeTensor(tensor)
		finalizeTensor(tensor)

		if got := strings.Count(output.String(), "\n"); got != 1 {
			t.Fatalf("finalizer diagnostic record count = %d, want 1", got)
		}
		record := decodeDiagnosticRecord(t, &output)
		if got := record["level"]; got != "WARN" {
			t.Fatalf("level = %v, want WARN", got)
		}
		if got := record["msg"]; got != "finalizer cleanup failed" {
			t.Fatalf("message = %v, want finalizer cleanup failed", got)
		}
		if got := record["resource"]; got != "tensor" {
			t.Fatalf("resource = %v, want tensor", got)
		}
		if got := record["error"]; got == nil || !strings.Contains(got.(string), "release function unavailable") {
			t.Fatalf("error attr = %v, want release-function failure", got)
		}
	})
}

func TestTensorDestroyDoubleCallsReleaseOnce(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	var releases int32
	mu.Lock()
	releaseValueFunc = func(handle uintptr) {
		atomic.AddInt32(&releases, 1)
	}
	mu.Unlock()

	tensor := &Tensor[float32]{
		handle: 321,
		data:   []float32{1, 2, 3},
		shape:  Shape{3},
	}

	if err := tensor.Destroy(); err != nil {
		t.Fatalf("first destroy failed: %v", err)
	}
	if err := tensor.Destroy(); err != nil {
		t.Fatalf("second destroy should be no-op, got: %v", err)
	}
	if got := atomic.LoadInt32(&releases); got != 1 {
		t.Fatalf("expected one native release, got %d", got)
	}
}

func TestTensorDestroyConcurrentCallsReleaseOnce(t *testing.T) {
	resetEnvironmentState()
	defer resetEnvironmentState()

	var releases int32
	mu.Lock()
	releaseValueFunc = func(handle uintptr) {
		atomic.AddInt32(&releases, 1)
	}
	mu.Unlock()

	tensor := &Tensor[float32]{
		handle: 777,
		data:   []float32{1, 2, 3},
		shape:  Shape{3},
	}

	const workers = 16
	start := make(chan struct{})
	errCh := make(chan error, workers)
	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			errCh <- tensor.Destroy()
		}()
	}

	close(start)
	wg.Wait()
	close(errCh)

	for err := range errCh {
		if err != nil {
			t.Fatalf("concurrent destroy failed: %v", err)
		}
	}

	if got := atomic.LoadInt32(&releases); got != 1 {
		t.Fatalf("expected exactly one native release call, got %d", got)
	}
	if tensor.handle != 0 {
		t.Fatalf("expected tensor handle to be cleared")
	}
	if tensor.data != nil || tensor.shape != nil {
		t.Fatalf("expected tensor fields to be cleared")
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

func TestScalarTensorWithORT(t *testing.T) {
	cleanup := setupTestEnvironment(t)
	defer cleanup()

	tensor, err := NewTensor[float32](Shape{}, []float32{3.14})
	if err != nil {
		t.Fatalf("NewTensor scalar failed: %v", err)
	}
	defer func() {
		_ = tensor.Destroy()
	}()

	if got := tensor.Shape(); !reflect.DeepEqual(got, Shape{}) {
		t.Fatalf("unexpected scalar shape: got %v, want []", got)
	}
	if got := tensor.GetData(); len(got) != 1 || got[0] != float32(3.14) {
		t.Fatalf("unexpected scalar data: %v", got)
	}

	emptyScalar, err := NewEmptyTensor[float32](Shape{})
	if err != nil {
		t.Fatalf("NewEmptyTensor scalar failed: %v", err)
	}
	defer func() {
		_ = emptyScalar.Destroy()
	}()

	if got := emptyScalar.Shape(); !reflect.DeepEqual(got, Shape{}) {
		t.Fatalf("unexpected empty scalar shape: got %v, want []", got)
	}
	if got := emptyScalar.GetData(); len(got) != 1 {
		t.Fatalf("unexpected empty scalar data length: got %d, want 1", len(got))
	}
}
