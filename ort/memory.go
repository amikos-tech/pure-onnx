package ort

import (
	"fmt"
	"runtime"
)

// CreateMemoryInfo creates a memory info structure with specified parameters.
// Maps to OrtApi::CreateMemoryInfo in the ONNX Runtime C API.
func CreateMemoryInfo(name string, allocatorType AllocatorType, deviceID int, memType MemType) (*MemoryInfo, error) {
	mu.Lock()
	defer mu.Unlock()

	if createMemoryInfoFunc == nil {
		return nil, fmt.Errorf("ONNX Runtime not initialized")
	}

	// Convert the name string to C string
	nameBytes, namePtr := GoToCstring(name)
	defer runtime.KeepAlive(nameBytes)

	var handle uintptr
	// #nosec G115 -- deviceID is validated by ONNX Runtime, conversion is safe
	status := createMemoryInfoFunc(namePtr, allocatorType, int32(deviceID), memType, &handle)
	if status != 0 {
		errMsg := getErrorMessage(status)
		releaseStatus(status)
		return nil, fmt.Errorf("failed to create memory info: %s", errMsg)
	}

	memInfo := &MemoryInfo{
		handle:        handle,
		name:          name,
		memType:       memType,
		allocatorType: allocatorType,
		deviceID:      deviceID,
	}

	// Set finalizer to ensure cleanup even if Destroy() is not called
	runtime.SetFinalizer(memInfo, func(m *MemoryInfo) {
		_ = m.Destroy()
	})

	return memInfo, nil
}

// CreateCpuMemoryInfo creates a memory info structure for CPU memory.
// This is a convenience function for the most common use case.
func CreateCpuMemoryInfo(allocatorType AllocatorType, memType MemType) (*MemoryInfo, error) {
	return CreateMemoryInfo("Cpu", allocatorType, 0, memType)
}

// Destroy releases the memory info resources.
// Maps to OrtApi::ReleaseMemoryInfo in the ONNX Runtime C API.
func (m *MemoryInfo) Destroy() error {
	mu.Lock()
	defer mu.Unlock()

	if m.handle == 0 {
		return nil
	}

	if releaseMemoryInfoFunc != nil {
		releaseMemoryInfoFunc(m.handle)
	}

	m.handle = 0
	runtime.SetFinalizer(m, nil)
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
	return m.handle != 0
}
