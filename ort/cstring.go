package ort

import "unsafe"

// CstringToGo converts a C null-terminated string pointer to a Go string.
// The pointer must point to a valid null-terminated string in memory.
// Returns empty string if ptr is 0 (null).
//
// Safety: This function uses byte-by-byte pointer arithmetic to avoid segfaults
// from creating slices that might cross memory page boundaries into unmapped memory.
// This is the safest approach for reading C-allocated strings with unknown length.
func CstringToGo(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}

	// Find the length by reading byte-by-byte using pointer arithmetic.
	// This is safe because we only dereference one byte at a time,
	// and we trust that the C API provides valid null-terminated strings.
	// We don't create slices until we know the exact length.
	var length int
	const maxStringLen = 1 << 20 // 1MB safety limit
	for length < maxStringLen {
		// Read one byte at a time - safe even near page boundaries
		b := *(*byte)(unsafe.Pointer(ptr + uintptr(length)))
		if b == 0 {
			break // Found null terminator
		}
		length++
	}

	// Check if we hit the limit (likely invalid pointer or corrupted memory)
	if length >= maxStringLen {
		return ""
	}

	// Special case: empty string
	if length == 0 {
		return ""
	}

	// Now that we know the exact length, create a slice and convert to string
	// This is safe because we verified all bytes exist and found the null terminator
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(ptr)), length)
	return string(bytes)
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
