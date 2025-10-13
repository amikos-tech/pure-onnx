package ort

import "unsafe"

// CstringToGo converts a C null-terminated string pointer to a Go string.
// The pointer must point to a valid null-terminated string in memory.
// Returns empty string if ptr is 0 (null).
func CstringToGo(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}

	// We scan for the null terminator in chunks to avoid potential segfaults
	// from creating large slices that might cross memory page boundaries.
	// ONNX Runtime strings (version, error messages) are typically < 1KB,
	// so we use a conservative chunk size that fits in one memory page (4KB).
	const chunkSize = 4096 // One page size, safe for most systems
	const maxChunks = 256  // Maximum 1MB total (256 * 4KB)

	var totalLen int
	for chunk := 0; chunk < maxChunks; chunk++ {
		// Get pointer to current chunk
		chunkPtr := ptr + uintptr(chunk*chunkSize)
		bytes := unsafe.Slice((*byte)(unsafe.Pointer(chunkPtr)), chunkSize)

		// Scan this chunk for null terminator
		for i := 0; i < chunkSize; i++ {
			if bytes[i] == 0 {
				// Found null terminator - build final string from all chunks
				totalLen += i
				if chunk == 0 {
					// Fast path: string is in first chunk
					return string(bytes[:i])
				}
				// Slow path: concatenate from multiple chunks
				result := make([]byte, totalLen)
				copied := 0
				for c := 0; c < chunk; c++ {
					srcPtr := ptr + uintptr(c*chunkSize)
					srcBytes := unsafe.Slice((*byte)(unsafe.Pointer(srcPtr)), chunkSize)
					copied += copy(result[copied:], srcBytes)
				}
				copy(result[copied:], bytes[:i])
				return string(result)
			}
		}
		totalLen += chunkSize
	}

	// If we reach here, string exceeds maxChunks * chunkSize (1MB)
	// This likely indicates memory corruption or an invalid pointer
	return ""
}

// GoToCstring converts a Go string to a null-terminated byte slice suitable for passing to C functions.
// Returns the byte slice (which must be kept alive by the caller to prevent GC) and a uintptr to its first byte.
//
// IMPORTANT: The caller MUST keep the returned []byte alive for as long as the C function might access it.
// Example usage:
//
//	logIDBytes, logIDPtr := GoToCstring("my-log-id")
//	status := cFunction(logIDPtr)  // logIDBytes must stay in scope here
func GoToCstring(s string) ([]byte, uintptr) {
	// Create a null-terminated byte slice
	b := append([]byte(s), 0)

	// Return both the slice (to keep it alive) and pointer to first byte
	return b, uintptr(unsafe.Pointer(&b[0]))
}
