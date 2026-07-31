package ort

import (
	"bytes"
	"errors"
	"log/slog"
	"math"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

type tensorStatusProbe struct {
	handle   uintptr
	code     ErrorCode
	releases atomic.Int32
}

func installTensorStatusProbe(t *testing.T, code ErrorCode) *tensorStatusProbe {
	t.Helper()

	probe := &tensorStatusProbe{
		handle: 9101,
		code:   code,
	}
	mu.Lock()
	getErrorCodeFunc = func(status uintptr) ErrorCode {
		if status != probe.handle {
			t.Errorf("GetErrorCode status = %d, want %d", status, probe.handle)
		}
		return probe.code
	}
	// A null native message exercises the production conversion without routing
	// a Go heap pointer back through uintptr, which checkptr rejects under -race.
	getErrorMessageFunc = func(status uintptr) uintptr {
		if status != probe.handle {
			t.Errorf("GetErrorMessage status = %d, want %d", status, probe.handle)
		}
		return 0
	}
	releaseStatusFunc = func(status uintptr) {
		if status != probe.handle {
			t.Errorf("ReleaseStatus status = %d, want %d", status, probe.handle)
		}
		probe.releases.Add(1)
	}
	mu.Unlock()

	return probe
}

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
			name:      "zero before otherwise overflowing dimensions",
			shape:     Shape{0, int64(int(^uint(0) >> 1)), 2},
			wantCount: 0,
		},
		{
			name:      "zero after otherwise overflowing dimensions",
			shape:     Shape{int64(int(^uint(0) >> 1)), 2, 0},
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

	got, err := tensorDataByteSize(maxInt, 1)
	if err != nil {
		t.Fatalf("one-byte tensor size at maximum element count: %v", err)
	}
	if got != uintptr(maxInt) {
		t.Fatalf("one-byte tensor size = %d, want %d", got, maxInt)
	}

	_, err = tensorDataByteSize(maxInt, 3)
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

func TestNewEmptyTensorRejectsByteSizeOverflowBeforeAllocation(t *testing.T) {
	if math.MaxInt != math.MaxInt64 {
		t.Skip("test requires a 64-bit int")
	}

	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("NewEmptyTensor panicked before rejecting byte-size overflow: %v", recovered)
		}
	}()

	_, err := NewEmptyTensor[int64](Shape{math.MaxInt64})
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("NewEmptyTensor error = %v, want ErrInvalidArgument", err)
	}
	if !strings.Contains(err.Error(), "tensor data size overflow") {
		t.Fatalf("NewEmptyTensor error = %v, want byte-size overflow", err)
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

func TestTensorShapeReturnsCopy(t *testing.T) {
	tensor := &Tensor[float32]{shape: Shape{2, 3}}

	shape := tensor.Shape()
	shape[0] = 99

	got := tensor.Shape()
	if !reflect.DeepEqual(got, Shape{2, 3}) {
		t.Fatalf("tensor shape mutated through returned slice: %v", got)
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

		probe := installTensorStatusProbe(t, ErrorCodeInvalidArgument)
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
			"",
			&probe.releases,
		)
	})

	t.Run("CreateTensorWithDataAsOrtValue status", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		probe := installTensorStatusProbe(t, ErrorCodeRuntimeException)
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
			"",
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

func TestTensorCreationRejectsZeroHandles(t *testing.T) {
	t.Run("memory info", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		mu.Lock()
		ortAPI = &OrtApi{}
		createMemoryInfoFunc = func(_ uintptr, _ AllocatorType, _ int32, _ MemType, _ *uintptr) uintptr {
			return 0
		}
		releaseMemoryInfoFunc = func(uintptr) {
			t.Fatal("ReleaseMemoryInfo called for a zero handle")
		}
		createTensorWithDataAsOrtValueFunc = func(_ uintptr, _ uintptr, _ uintptr, _ *int64, _ uintptr, _ TensorElementDataType, _ *uintptr) uintptr {
			t.Fatal("CreateTensorWithDataAsOrtValue called after a zero memory-info handle")
			return 0
		}
		getErrorCodeFunc = func(uintptr) ErrorCode { return ErrorCodeFail }
		getErrorMessageFunc = func(uintptr) uintptr { return 0 }
		releaseStatusFunc = func(uintptr) {}
		mu.Unlock()

		tensor, err := NewTensor[float32](Shape{1}, []float32{1})
		if tensor != nil {
			t.Fatalf("tensor = %#v, want nil", tensor)
		}
		if !errors.Is(err, ErrNativeContract) {
			t.Fatalf("NewTensor error = %v, want ErrNativeContract", err)
		}
	})

	t.Run("tensor value", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		var memoryInfoReleases atomic.Int32
		mu.Lock()
		ortAPI = &OrtApi{}
		createMemoryInfoFunc = func(_ uintptr, _ AllocatorType, _ int32, _ MemType, out *uintptr) uintptr {
			*out = 720
			return 0
		}
		releaseMemoryInfoFunc = func(handle uintptr) {
			if handle != 720 {
				t.Errorf("ReleaseMemoryInfo handle = %d, want 720", handle)
			}
			memoryInfoReleases.Add(1)
		}
		createTensorWithDataAsOrtValueFunc = func(_ uintptr, _ uintptr, _ uintptr, _ *int64, _ uintptr, _ TensorElementDataType, _ *uintptr) uintptr {
			return 0
		}
		getErrorCodeFunc = func(uintptr) ErrorCode { return ErrorCodeFail }
		getErrorMessageFunc = func(uintptr) uintptr { return 0 }
		releaseStatusFunc = func(uintptr) {}
		mu.Unlock()

		tensor, err := NewTensor[float32](Shape{1}, []float32{1})
		if tensor != nil {
			t.Fatalf("tensor = %#v, want nil", tensor)
		}
		if !errors.Is(err, ErrNativeContract) {
			t.Fatalf("NewTensor error = %v, want ErrNativeContract", err)
		}
		if got := memoryInfoReleases.Load(); got != 1 {
			t.Fatalf("memory info release count = %d, want 1", got)
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

		probe := installTensorStatusProbe(t, ErrorCodeFail)
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

// TestTensorPinnedBackingSurvivesGC proves the backing array stays reachable
// and unchanged across repeated GC cycles. It does not, and structurally
// cannot, prove the Pinner itself prevents a move: Go's current garbage
// collector never relocates heap objects, so this data-liveness check would
// pass identically even without runtime.Pinner. runtime.Pinner exposes no way
// to confirm Pin() itself ran, so the accompanying tensor.pinner != nil check
// only catches the coarser regression of the whole pinning branch being
// dropped for a non-empty tensor, not a Pin() call quietly disappearing from
// inside it.
func TestTensorPinnedBackingSurvivesGC(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	var (
		capturedData       uintptr
		memoryInfoReleases atomic.Int32
		valueReleases      atomic.Int32
	)
	mu.Lock()
	ortAPI = &OrtApi{}
	createMemoryInfoFunc = func(_ uintptr, _ AllocatorType, _ int32, _ MemType, out *uintptr) uintptr {
		*out = 801
		return 0
	}
	releaseMemoryInfoFunc = func(handle uintptr) {
		if handle != 801 {
			t.Errorf("released memory-info handle = %d, want 801", handle)
		}
		memoryInfoReleases.Add(1)
	}
	createTensorWithDataAsOrtValueFunc = func(
		_ uintptr,
		data uintptr,
		_ uintptr,
		_ *int64,
		_ uintptr,
		_ TensorElementDataType,
		out *uintptr,
	) uintptr {
		capturedData = data
		*out = 802
		return 0
	}
	releaseValueFunc = func(handle uintptr) {
		if handle != 802 {
			t.Errorf("released tensor handle = %d, want 802", handle)
		}
		valueReleases.Add(1)
	}
	getErrorCodeFunc = func(uintptr) ErrorCode { return ErrorCodeFail }
	getErrorMessageFunc = func(uintptr) uintptr { return 0 }
	releaseStatusFunc = func(uintptr) {}
	mu.Unlock()

	want := []int64{11, 22, 33, 44}
	tensor, err := func() (*Tensor[int64], error) {
		callerData := append([]int64(nil), want...)
		return NewTensor[int64](Shape{int64(len(callerData))}, callerData)
	}()
	if err != nil {
		t.Fatalf("NewTensor: %v", err)
	}
	if capturedData == 0 {
		t.Fatal("native tensor constructor received a nil data pointer")
	}
	if got := uintptr(unsafe.Pointer(unsafe.SliceData(tensor.GetData()))); got != capturedData {
		t.Fatalf("tensor data pointer = %#x, native pointer = %#x", got, capturedData)
	}
	if tensor.pinner == nil {
		t.Fatal("NewTensor did not pin the backing array")
	}

	for iteration := 0; iteration < 25; iteration++ {
		pressure := make([][]byte, 32)
		for i := range pressure {
			pressure[i] = make([]byte, 32*1024)
			pressure[i][0] = byte(iteration + i)
		}
		runtime.GC()
		runtime.Gosched()

		liveData := tensor.GetData()
		if got := uintptr(unsafe.Pointer(unsafe.SliceData(liveData))); got != capturedData {
			t.Fatalf("tensor data pointer moved after GC iteration %d: got %#x, want %#x", iteration, got, capturedData)
		}
		if !reflect.DeepEqual(liveData, want) {
			t.Fatalf("pinned data changed after GC iteration %d: got %v, want %v", iteration, liveData, want)
		}
		runtime.KeepAlive(pressure)
	}
	runtime.KeepAlive(tensor)

	if err := tensor.Destroy(); err != nil {
		t.Fatalf("Destroy tensor: %v", err)
	}
	if got := memoryInfoReleases.Load(); got != 1 {
		t.Fatalf("memory-info release count = %d, want 1", got)
	}
	if got := valueReleases.Load(); got != 1 {
		t.Fatalf("tensor release count = %d, want 1", got)
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

func TestTensorSupportedElementTypesWithORT(t *testing.T) {
	cleanup := setupTestEnvironment(t)
	defer cleanup()

	tests := []struct {
		name   string
		create func() (Value, error)
	}{
		{
			name: "float32",
			create: func() (Value, error) {
				return NewTensor[float32](Shape{2}, []float32{1, 2})
			},
		},
		{
			name: "float64",
			create: func() (Value, error) {
				return NewTensor[float64](Shape{2}, []float64{1, 2})
			},
		},
		{
			name: "int32",
			create: func() (Value, error) {
				return NewTensor[int32](Shape{2}, []int32{1, 2})
			},
		},
		{
			name: "int64",
			create: func() (Value, error) {
				return NewTensor[int64](Shape{2}, []int64{1, 2})
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			value, err := test.create()
			if err != nil {
				t.Fatalf("create %s tensor: %v", test.name, err)
			}
			handleProvider, ok := value.(valueWithORTHandle)
			if !ok {
				t.Fatalf("%s tensor does not expose its package-owned native handle", test.name)
			}
			if handle := handleProvider.ortValueHandle(); handle == 0 {
				t.Fatalf("%s tensor returned a zero native handle", test.name)
			}
			if err := value.Destroy(); err != nil {
				t.Fatalf("destroy %s tensor: %v", test.name, err)
			}
		})
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
