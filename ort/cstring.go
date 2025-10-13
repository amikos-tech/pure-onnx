package ort

import "unsafe"

// CstringToGo converts a C null-terminated string pointer to a Go string.
// The pointer must point to a valid null-terminated string in memory.
// Returns empty string if ptr is 0 (null).
func CstringToGo(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}

	// Find the null terminator using a large but valid slice.
	// We use a conservative max length (1MB) to avoid checkptr issues when scanning
	// C-allocated memory. This is safe because:
	// 1. We only read up to the null terminator, not the entire 1MB
	// 2. ONNX Runtime strings (version, error messages, etc.) are typically < 1KB
	// 3. If a string exceeds 1MB, it likely indicates memory corruption
	const maxStringLen = 1 << 20 // 1MB maximum string length
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(ptr)), maxStringLen)

	// Find null terminator
	var length int
	for i := 0; i < maxStringLen; i++ {
		if bytes[i] == 0 {
			length = i
			break
		}
	}

	// Return string from found length
	return string(bytes[:length])
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
