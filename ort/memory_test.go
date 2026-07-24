package ort

import (
	"bytes"
	"errors"
	"log/slog"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func setupTestEnvironment(tb testing.TB) func() {
	tb.Helper()

	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		tb.Skip("ONNXRUNTIME_LIB_PATH not set, skipping test")
	}

	if err := SetSharedLibraryPath(libPath); err != nil {
		tb.Fatalf("Failed to set library path: %v", err)
	}

	if err := InitializeEnvironment(); err != nil {
		tb.Fatalf("Failed to initialize environment: %v", err)
	}

	return func() {
		if err := DestroyEnvironment(); err != nil {
			tb.Errorf("Failed to destroy environment: %v", err)
		}
	}
}

func TestCreateCpuMemoryInfo(t *testing.T) {
	cleanup := setupTestEnvironment(t)
	defer cleanup()

	tests := []struct {
		name          string
		allocatorType AllocatorType
		memType       MemType
		wantErr       bool
	}{
		{
			name:          "CPU input memory with arena allocator",
			allocatorType: AllocatorTypeArena,
			memType:       MemTypeCPUInput,
			wantErr:       false,
		},
		{
			name:          "CPU output memory with device allocator",
			allocatorType: AllocatorTypeDevice,
			memType:       MemTypeCPUOutput,
			wantErr:       false,
		},
		{
			name:          "CPU memory with arena allocator",
			allocatorType: AllocatorTypeArena,
			memType:       MemTypeCPU,
			wantErr:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			memInfo, err := CreateCpuMemoryInfo(tt.allocatorType, tt.memType)
			if (err != nil) != tt.wantErr {
				t.Errorf("CreateCpuMemoryInfo() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err == nil {
				if !memInfo.IsValid() {
					t.Error("Created memory info is not valid")
				}

				if memInfo.GetName() != "Cpu" {
					t.Errorf("Expected name 'Cpu', got '%s'", memInfo.GetName())
				}

				if memInfo.GetMemType() != tt.memType {
					t.Errorf("Expected memType %v, got %v", tt.memType, memInfo.GetMemType())
				}

				if memInfo.GetAllocatorType() != tt.allocatorType {
					t.Errorf("Expected allocatorType %v, got %v", tt.allocatorType, memInfo.GetAllocatorType())
				}

				if err := memInfo.Destroy(); err != nil {
					t.Errorf("Failed to destroy memory info: %v", err)
				}

				if memInfo.IsValid() {
					t.Error("Memory info should not be valid after destroy")
				}
			}
		})
	}
}

func TestCreateMemoryInfo(t *testing.T) {
	cleanup := setupTestEnvironment(t)
	defer cleanup()

	tests := []struct {
		name              string
		allocName         string
		allocatorType     AllocatorType
		deviceID          int
		memType           MemType
		wantErr           bool
		allowNotSupported bool
	}{
		{
			name:          "CPU memory info",
			allocName:     "Cpu",
			allocatorType: AllocatorTypeArena,
			deviceID:      0,
			memType:       MemTypeCPU,
			wantErr:       false,
		},
		{
			name:              "Custom allocator",
			allocName:         "CustomAlloc",
			allocatorType:     AllocatorTypeDevice,
			deviceID:          0,
			memType:           MemTypeDefault,
			wantErr:           false,
			allowNotSupported: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			memInfo, err := CreateMemoryInfo(tt.allocName, tt.allocatorType, tt.deviceID, tt.memType)
			if tt.allowNotSupported && err != nil {
				errLower := strings.ToLower(err.Error())
				if strings.Contains(errLower, "not supported") {
					t.Logf("allocator not supported by this ONNX Runtime build, skipping strict assertion: %v", err)
					return
				}
			}
			if (err != nil) != tt.wantErr {
				t.Errorf("CreateMemoryInfo() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err == nil {
				if !memInfo.IsValid() {
					t.Error("Created memory info is not valid")
				}

				if memInfo.GetName() != tt.allocName {
					t.Errorf("Expected name '%s', got '%s'", tt.allocName, memInfo.GetName())
				}

				if memInfo.GetDeviceID() != tt.deviceID {
					t.Errorf("Expected deviceID %d, got %d", tt.deviceID, memInfo.GetDeviceID())
				}

				if err := memInfo.Destroy(); err != nil {
					t.Errorf("Failed to destroy memory info: %v", err)
				}
			}
		})
	}
}

func TestMemoryInfoDoubleDestroy(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	var releases atomic.Int32
	mu.Lock()
	releaseMemoryInfoFunc = func(handle uintptr) {
		if handle != 701 {
			t.Errorf("release handle = %d, want 701", handle)
		}
		releases.Add(1)
	}
	mu.Unlock()

	memInfo := &MemoryInfo{handle: 701, name: "Cpu"}
	if err := memInfo.Destroy(); err != nil {
		t.Fatalf("first destroy failed: %v", err)
	}

	if err := memInfo.Destroy(); err != nil {
		t.Fatalf("second destroy should be a no-op: %v", err)
	}
	if got := releases.Load(); got != 1 {
		t.Fatalf("release count = %d, want 1", got)
	}

	var nilInfo *MemoryInfo
	if err := nilInfo.Destroy(); err != nil {
		t.Fatalf("nil destroy should be a no-op: %v", err)
	}
}

func TestMemoryInfoIsValidConcurrentDestroy(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	mu.Lock()
	releaseMemoryInfoFunc = func(uintptr) {}
	mu.Unlock()

	memInfo := &MemoryInfo{handle: 702, name: "Cpu"}
	stop := make(chan struct{})
	var readers sync.WaitGroup
	for range 8 {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for {
				select {
				case <-stop:
					return
				default:
					_ = memInfo.IsValid()
				}
			}
		}()
	}

	if err := memInfo.Destroy(); err != nil {
		t.Fatalf("destroy memory info: %v", err)
	}
	close(stop)
	readers.Wait()

	if memInfo.IsValid() {
		t.Fatal("memory info remained valid after Destroy")
	}
}

func TestMemoryInfoDestroyReleaseUnavailable(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	handler := &diagnosticCountingHandler{}
	SetDiagnosticHandler(handler)
	t.Cleanup(func() { SetDiagnosticHandler(nil) })

	memInfo := &MemoryInfo{
		handle:   123,
		name:     "Cpu",
		memType:  MemTypeCPU,
		deviceID: 0,
	}

	err := memInfo.Destroy()
	if !errors.Is(err, ErrNotInitialized) {
		t.Fatalf("destroy error = %v, want ErrNotInitialized", err)
	}
	if memInfo.handle != 0 {
		t.Fatalf("expected handle to be reset even on release failure")
	}

	// Second destroy remains a safe no-op once handle has been cleared.
	if err := memInfo.Destroy(); err != nil {
		t.Fatalf("second destroy should be no-op, got: %v", err)
	}
	if got := handler.count.Load(); got != 0 {
		t.Fatalf("returned destroy error emitted %d diagnostics, want 0", got)
	}
}

func TestMemoryInfoFinalizer(t *testing.T) {
	cleanup := setupTestEnvironment(t)
	defer cleanup()

	// Create memory info without explicitly destroying
	func() {
		_, err := CreateCpuMemoryInfo(AllocatorTypeArena, MemTypeCPU)
		if err != nil {
			t.Fatalf("Failed to create memory info: %v", err)
		}
		// Memory info goes out of scope without calling Destroy()
	}()

	// Force GC to run finalizers
	runtime.GC()
	runtime.GC() // Call twice to ensure finalizers run

	// If we get here without crashing, the finalizer worked correctly
}

func TestMemoryInfoBeforeInit(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	handler := &diagnosticCountingHandler{}
	SetDiagnosticHandler(handler)
	t.Cleanup(func() { SetDiagnosticHandler(nil) })

	if _, err := CreateMemoryInfo("", AllocatorTypeArena, 0, MemTypeCPU); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("empty name error = %v, want ErrInvalidArgument", err)
	}
	if _, err := CreateMemoryInfo("Cpu\x00Injected", AllocatorTypeArena, 0, MemTypeCPU); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("embedded-NUL name error = %v, want ErrInvalidArgument", err)
	}
	if strconv.IntSize > 32 {
		tooLarge := int64(math.MaxInt32) + 1
		if _, err := CreateMemoryInfo("Cpu", AllocatorTypeArena, int(tooLarge), MemTypeCPU); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("oversized device ID error = %v, want ErrInvalidArgument", err)
		}
		tooSmall := int64(math.MinInt32) - 1
		if _, err := CreateMemoryInfo("Cpu", AllocatorTypeArena, int(tooSmall), MemTypeCPU); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("undersized device ID error = %v, want ErrInvalidArgument", err)
		}
	}
	if _, err := CreateCpuMemoryInfo(AllocatorTypeArena, MemTypeCPU); !errors.Is(err, ErrNotInitialized) {
		t.Fatalf("before-init error = %v, want ErrNotInitialized", err)
	}
	if got := handler.count.Load(); got != 0 {
		t.Fatalf("returned creation errors emitted %d diagnostics, want 0", got)
	}
}

func TestMemoryInfoStatusConversion(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	handler := &diagnosticCountingHandler{}
	SetDiagnosticHandler(handler)
	t.Cleanup(func() { SetDiagnosticHandler(nil) })

	const statusHandle = uintptr(711)
	var releases atomic.Int32
	mu.Lock()
	ortAPI = &OrtApi{}
	createMemoryInfoFunc = func(
		name uintptr,
		allocatorType AllocatorType,
		deviceID int32,
		memType MemType,
		out *uintptr,
	) uintptr {
		if name == 0 || allocatorType != AllocatorTypeArena || deviceID != 4 ||
			memType != MemTypeCPU || out == nil {
			t.Errorf(
				"unexpected creation args: name=%d allocator=%d device=%d memType=%d out=%p",
				name,
				allocatorType,
				deviceID,
				memType,
				out,
			)
		}
		return statusHandle
	}
	getErrorCodeFunc = func(status uintptr) ErrorCode {
		if status != statusHandle {
			t.Errorf("GetErrorCode status = %d, want %d", status, statusHandle)
		}
		return ErrorCodeInvalidArgument
	}
	getErrorMessageFunc = func(status uintptr) uintptr {
		if status != statusHandle {
			t.Errorf("GetErrorMessage status = %d, want %d", status, statusHandle)
		}
		// TestStatusToError owns the non-empty message copy proof. Keeping this
		// call-site probe null avoids a Go-pointer uintptr round trip under -race.
		return 0
	}
	releaseMemoryInfoFunc = func(uintptr) {
		t.Fatal("ReleaseMemoryInfo called after creation status failure")
	}
	releaseStatusFunc = func(status uintptr) {
		if status != statusHandle {
			t.Errorf("ReleaseStatus status = %d, want %d", status, statusHandle)
		}
		releases.Add(1)
	}
	mu.Unlock()

	memInfo, err := CreateMemoryInfo("Cpu", AllocatorTypeArena, 4, MemTypeCPU)
	if memInfo != nil {
		t.Fatalf("memory info = %#v, want nil on status failure", memInfo)
	}
	var nativeErr *ORTError
	if !errors.As(err, &nativeErr) {
		t.Fatalf("errors.As(%v, *ORTError) = false", err)
	}
	if nativeErr.Operation != "create memory info" {
		t.Fatalf("operation = %q, want create memory info", nativeErr.Operation)
	}
	if nativeErr.Code != ErrorCodeInvalidArgument {
		t.Fatalf("code = %d, want %d", nativeErr.Code, ErrorCodeInvalidArgument)
	}
	if nativeErr.Message != "" {
		t.Fatalf("message = %q, want empty race-safe probe message", nativeErr.Message)
	}
	if got := releases.Load(); got != 1 {
		t.Fatalf("status release count = %d, want 1", got)
	}
	if got := handler.count.Load(); got != 0 {
		t.Fatalf("returned status error emitted %d diagnostics, want 0", got)
	}
}

func TestCreateMemoryInfoBlocksEnvironmentTeardown(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	const statusHandle = uintptr(712)
	nativeStarted := make(chan struct{})
	allowNativeReturn := make(chan struct{})
	accessorStarted := make(chan struct{})
	allowAccessorReturn := make(chan struct{})
	statusReleased := make(chan struct{})

	mu.Lock()
	ortAPI = &OrtApi{}
	createMemoryInfoFunc = func(
		uintptr,
		AllocatorType,
		int32,
		MemType,
		*uintptr,
	) uintptr {
		close(nativeStarted)
		<-allowNativeReturn
		return statusHandle
	}
	getErrorCodeFunc = func(status uintptr) ErrorCode {
		if status != statusHandle {
			t.Errorf("GetErrorCode status = %d, want %d", status, statusHandle)
		}
		close(accessorStarted)
		<-allowAccessorReturn
		return ErrorCodeFail
	}
	getErrorMessageFunc = func(uintptr) uintptr { return 0 }
	releaseMemoryInfoFunc = func(uintptr) {
		t.Fatal("ReleaseMemoryInfo called after creation status failure")
	}
	releaseStatusFunc = func(status uintptr) {
		if status != statusHandle {
			t.Errorf("ReleaseStatus status = %d, want %d", status, statusHandle)
		}
		close(statusReleased)
	}
	mu.Unlock()

	result := make(chan error, 1)
	go func() {
		_, err := CreateMemoryInfo("Cpu", AllocatorTypeArena, 0, MemTypeCPU)
		result <- err
	}()

	<-nativeStarted
	nativeProtected := !ortCallMu.TryLock()
	if !nativeProtected {
		ortCallMu.Unlock()
	}

	close(allowNativeReturn)
	<-accessorStarted
	accessorProtected := !ortCallMu.TryLock()
	if !accessorProtected {
		ortCallMu.Unlock()
	}

	close(allowAccessorReturn)
	err := <-result
	<-statusReleased

	availableAfterConversion := ortCallMu.TryLock()
	if availableAfterConversion {
		ortCallMu.Unlock()
	}

	if !nativeProtected {
		t.Error("exclusive teardown lock was available during native CreateMemoryInfo")
	}
	if !accessorProtected {
		t.Error("exclusive teardown lock was available during status conversion")
	}
	if !availableAfterConversion {
		t.Error("exclusive teardown lock remained unavailable after conversion and release")
	}
	var nativeErr *ORTError
	if !errors.As(err, &nativeErr) {
		t.Fatalf("errors.As(%v, *ORTError) = false", err)
	}
}

func TestDiagnosticMemoryInfo(t *testing.T) {
	t.Run("returned errors stay silent", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		handler := &diagnosticCountingHandler{}
		SetDiagnosticHandler(handler)
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		if _, err := CreateMemoryInfo("", AllocatorTypeArena, 0, MemTypeCPU); err == nil {
			t.Fatal("empty-name creation returned nil error")
		}
		if _, err := CreateCpuMemoryInfo(AllocatorTypeArena, MemTypeCPU); err == nil {
			t.Fatal("before-init creation returned nil error")
		}
		if err := (&MemoryInfo{handle: 801, name: "Cpu"}).Destroy(); err == nil {
			t.Fatal("release-unavailable destroy returned nil error")
		}

		if got := handler.count.Load(); got != 0 {
			t.Fatalf("returned memory failures emitted %d diagnostics, want 0", got)
		}
	})

	t.Run("finalizer-only failure emits one structured warning", func(t *testing.T) {
		resetEnvironmentState()
		t.Cleanup(resetEnvironmentState)

		var output bytes.Buffer
		SetDiagnosticHandler(slog.NewJSONHandler(&output, nil))
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		memInfo := &MemoryInfo{handle: 802, name: "Cpu"}
		finalizeMemoryInfo(memInfo)
		finalizeMemoryInfo(memInfo)

		if got := strings.Count(output.String(), "\n"); got != 1 {
			t.Fatalf("finalizer diagnostic record count = %d, want 1", got)
		}
		record := decodeDiagnosticRecord(t, &output)
		if got := record["level"]; got != "WARN" {
			t.Fatalf("level = %v, want WARN", got)
		}
		if got := record["resource"]; got != "memory_info" {
			t.Fatalf("resource = %v, want memory_info", got)
		}
		if got := record["error"]; got == nil || !strings.Contains(got.(string), "release function unavailable") {
			t.Fatalf("error attr = %v, want release-function failure", got)
		}
	})
}
