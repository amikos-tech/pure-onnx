package ort

import "unsafe"

// CstringToGo converts a C null-terminated string pointer to a Go string.
// The pointer must point to a valid null-terminated string in memory.
// Returns empty string if ptr is 0 (null).
func CstringToGo(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}

	// Cast to byte array pointer with maximum possible size (1GB = 1 << 30).
	// This is sufficient for all practical C strings. The actual string length
	// is determined by finding the null terminator, so we never read beyond
	// the valid memory region. On 64-bit systems, strings larger than 1GB are
	// not supported by this implementation, but such strings are extremely
	// rare in practice and not expected from ONNX Runtime APIs.
	p := (*[1 << 30]byte)(unsafe.Pointer(ptr))

	// Find the null terminator
	n := 0
	for p[n] != 0 {
		n++
	}

	// Create Go string from the bytes up to (not including) null terminator
	// Use three-index slice to set capacity = length, preventing append from modifying underlying C memory
	return string(p[:n:n])
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
