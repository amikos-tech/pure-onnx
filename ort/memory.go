package ort

import (
	"fmt"
	"math"
	"runtime"
)

// CreateMemoryInfo creates a memory info structure with specified parameters.
// Maps to OrtApi::CreateMemoryInfo in the ONNX Runtime C API.
func CreateMemoryInfo(name string, allocatorType AllocatorType, deviceID int, memType MemType) (*MemoryInfo, error) {
	if name == "" {
		return nil, fmt.Errorf("create memory info: allocator name is empty: %w", ErrInvalidArgument)
	}
	if err := validateNativeString(name, "memory info allocator name"); err != nil {
		return nil, err
	}
	if deviceID < math.MinInt32 || deviceID > math.MaxInt32 {
		return nil, fmt.Errorf(
			"create memory info: device ID %d is outside the native int32 range: %w",
			deviceID,
			ErrInvalidArgument,
		)
	}

	ortCallMu.RLock()
	defer ortCallMu.RUnlock()

	mu.Lock()
	if ortAPI == nil ||
		createMemoryInfoFunc == nil ||
		releaseMemoryInfoFunc == nil ||
		getErrorCodeFunc == nil ||
		getErrorMessageFunc == nil ||
		releaseStatusFunc == nil {
		mu.Unlock()
		return nil, fmt.Errorf(
			"create memory info %q: required ONNX Runtime functions are unavailable: %w",
			name,
			ErrNotInitialized,
		)
	}
	createMemoryInfo := createMemoryInfoFunc
	mu.Unlock()

	nameBytes, namePtr, err := goStringToCString(name, "memory info allocator name")
	if err != nil {
		return nil, err
	}

	var handle uintptr
	// #nosec G115 -- deviceID is range-checked above before conversion.
	status := createMemoryInfo(namePtr, allocatorType, int32(deviceID), memType, &handle)
	runtime.KeepAlive(nameBytes)
	if status != 0 {
		return nil, fmt.Errorf(
			"failed to create memory info %q: %w",
			name,
			statusToError(status, "create memory info"),
		)
	}

	memInfo := &MemoryInfo{
		handle:        handle,
		name:          name,
		memType:       memType,
		allocatorType: allocatorType,
		deviceID:      deviceID,
	}

	// Set finalizer to ensure cleanup even if Destroy() is not called
	runtime.SetFinalizer(memInfo, finalizeMemoryInfo)

	return memInfo, nil
}

func finalizeMemoryInfo(memInfo *MemoryInfo) {
	if err := memInfo.Destroy(); err != nil {
		emitFinalizerDiagnostic("memory_info", err)
	}
}

// CreateCpuMemoryInfo creates a memory info structure for CPU memory.
// This is a convenience function for the most common use case.
func CreateCpuMemoryInfo(allocatorType AllocatorType, memType MemType) (*MemoryInfo, error) {
	return CreateMemoryInfo("Cpu", allocatorType, 0, memType)
}

// Destroy releases the memory info resources.
// Maps to OrtApi::ReleaseMemoryInfo in the ONNX Runtime C API.
func (m *MemoryInfo) Destroy() error {
	if m == nil {
		return nil
	}

	// Keep environment teardown from racing the native release call.
	ortCallMu.RLock()
	defer ortCallMu.RUnlock()

	var (
		handle            uintptr
		releaseMemoryInfo func(uintptr)
	)

	mu.Lock()
	releaseMemoryInfo = releaseMemoryInfoFunc
	mu.Unlock()

	m.handleMu.Lock()
	handle = m.handle
	m.handle = 0
	runtime.SetFinalizer(m, nil)
	m.handleMu.Unlock()

	if handle == 0 {
		return nil
	}

	if releaseMemoryInfo == nil {
		return fmt.Errorf(
			"cannot destroy memory info %q: ONNX Runtime release function unavailable; "+
				"destroy all tensors, sessions, and memory infos before DestroyEnvironment: %w",
			m.name,
			ErrNotInitialized,
		)
	}
	releaseMemoryInfo(handle)
	return nil
}

// GetName returns the name of the memory allocator
func (m *MemoryInfo) GetName() string {
	return m.name
}

// GetMemType returns the memory type
func (m *MemoryInfo) GetMemType() MemType {
	return m.memType
}

// GetAllocatorType returns the allocator type
func (m *MemoryInfo) GetAllocatorType() AllocatorType {
	return m.allocatorType
}

// GetDeviceID returns the device ID
func (m *MemoryInfo) GetDeviceID() int {
	return m.deviceID
}

// IsValid returns true if the memory info has a valid handle.
func (m *MemoryInfo) IsValid() bool {
	if m == nil {
		return false
	}
	m.handleMu.RLock()
	valid := m.handle != 0
	m.handleMu.RUnlock()
	return valid
}
