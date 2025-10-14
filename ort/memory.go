package ort

import (
	"fmt"
	"runtime"
)

// CreateMemoryInfo creates a memory info structure with specified parameters.
// NOTE: Due to OrtApi struct layout issues (see issue #20), this function creates
// a lightweight Go-only wrapper that tracks memory info metadata without calling
// the actual ONNX Runtime CreateMemoryInfo function. This is sufficient for tensor
// creation which primarily needs the metadata.
func CreateMemoryInfo(name string, allocatorType AllocatorType, deviceID int, memType MemType) (*MemoryInfo, error) {
	mu.Lock()
	defer mu.Unlock()

	if ortAPI == nil {
		return nil, fmt.Errorf("ONNX Runtime not initialized")
	}

	// Create a Go-only memory info structure
	// This avoids calling the broken CreateMemoryInfo C function
	memInfo := &MemoryInfo{
		handle:        0, // No ORT handle - pure Go object for now
		name:          name,
		id:            deviceID,
		memType:       memType,
		allocatorType: allocatorType,
		deviceID:      deviceID,
	}

	return memInfo, nil
}

// CreateCpuMemoryInfo creates a memory info structure for CPU memory.
// This is a convenience function for the most common use case.
// NOTE: Due to OrtApi struct layout issues (see issue #20), this function creates
// a lightweight Go-only wrapper. This is sufficient for basic tensor operations.
func CreateCpuMemoryInfo(allocatorType AllocatorType, memType MemType) (*MemoryInfo, error) {
	return CreateMemoryInfo("Cpu", allocatorType, 0, memType)
}

// Destroy releases the memory info resources.
// For Go-only MemoryInfo objects (handle == 0), this is a no-op.
func (m *MemoryInfo) Destroy() error {
	// For Go-only MemoryInfo objects, nothing to clean up
	// When issue #20 is fixed and we create actual ORT handles,
	// this will call ReleaseMemoryInfo
	m.handle = 0
	m.name = "" // Mark as destroyed for IsValid check
	runtime.SetFinalizer(m, nil)
	return nil
}

// GetName returns the name of the memory allocator
func (m *MemoryInfo) GetName() string {
	return m.name
}

// GetID returns the device ID
func (m *MemoryInfo) GetID() int {
	return m.id
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

// IsValid returns true if the memory info is valid.
// For Go-only MemoryInfo objects, this checks if the object has been destroyed.
func (m *MemoryInfo) IsValid() bool {
	// For Go-only objects, consider them valid if they haven't been explicitly destroyed
	// We use deviceID as a marker since handle is always 0 for Go-only objects
	return m.name != ""
}
