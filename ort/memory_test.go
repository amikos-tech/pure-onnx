package ort

import (
	"os"
	"runtime"
	"testing"
)

func setupTestEnvironment(t *testing.T) func() {
	t.Helper()

	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		t.Skip("ONNXRUNTIME_LIB_PATH not set, skipping test")
	}

	if err := SetSharedLibraryPath(libPath); err != nil {
		t.Fatalf("Failed to set library path: %v", err)
	}

	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("Failed to initialize environment: %v", err)
	}

	return func() {
		if err := DestroyEnvironment(); err != nil {
			t.Errorf("Failed to destroy environment: %v", err)
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
		name          string
		allocName     string
		allocatorType AllocatorType
		deviceID      int
		memType       MemType
		wantErr       bool
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
			name:          "Custom allocator",
			allocName:     "CustomAlloc",
			allocatorType: AllocatorTypeDevice,
			deviceID:      0,
			memType:       MemTypeDefault,
			wantErr:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			memInfo, err := CreateMemoryInfo(tt.allocName, tt.allocatorType, tt.deviceID, tt.memType)
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
	cleanup := setupTestEnvironment(t)
	defer cleanup()

	memInfo, err := CreateCpuMemoryInfo(AllocatorTypeArena, MemTypeCPU)
	if err != nil {
		t.Fatalf("Failed to create memory info: %v", err)
	}

	if err := memInfo.Destroy(); err != nil {
		t.Fatalf("First destroy failed: %v", err)
	}

	// Second destroy should be safe (no-op)
	if err := memInfo.Destroy(); err != nil {
		t.Errorf("Second destroy should not return error, got: %v", err)
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
	// Don't initialize environment
	_, err := CreateCpuMemoryInfo(AllocatorTypeArena, MemTypeCPU)
	if err == nil {
		t.Error("Expected error when creating memory info before initialization")
	}
}
