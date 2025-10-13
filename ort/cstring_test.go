package ort

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestGoToCstring(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"empty string", ""},
		{"simple ascii", "hello"},
		{"with spaces", "hello world"},
		{"with special chars", "hello\tworld\n"},
		{"unicode", "Hello, 世界"},
		{"emoji", "Hello 👋 World 🌍"},
		{"long string", strings.Repeat("a", 1000)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bytes, ptr := GoToCstring(tt.input)

			// Verify the byte slice is correct
			if len(bytes) != len(tt.input)+1 {
				t.Errorf("expected byte slice length %d, got %d", len(tt.input)+1, len(bytes))
			}

			// Verify null terminator
			if bytes[len(bytes)-1] != 0 {
				t.Error("expected null terminator at end of byte slice")
			}

			// Verify pointer is not null for non-empty strings
			if ptr == 0 {
				t.Error("expected non-null pointer")
			}

			// Verify the content before null terminator matches input
			if string(bytes[:len(bytes)-1]) != tt.input {
				t.Errorf("expected content %q, got %q", tt.input, string(bytes[:len(bytes)-1]))
			}
		})
	}
}

func TestCstringToGo(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"empty string", "", ""},
		{"simple ascii", "hello", "hello"},
		{"with spaces", "hello world", "hello world"},
		{"unicode", "Hello, 世界", "Hello, 世界"},
		{"emoji", "Hello 👋 World 🌍", "Hello 👋 World 🌍"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Convert Go to C string
			bytes, ptr := GoToCstring(tt.input)
			_ = bytes // Keep bytes alive

			// Convert back to Go string
			result := CstringToGo(ptr)

			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}

			// Verify UTF-8 validity
			if !utf8.ValidString(result) {
				t.Error("result is not valid UTF-8")
			}
		})
	}
}

func TestCstringToGoNullPointer(t *testing.T) {
	result := CstringToGo(0)
	if result != "" {
		t.Errorf("expected empty string for null pointer, got %q", result)
	}
}

func TestCstringToGoInvalidLowAddresses(t *testing.T) {
	// Test that low addresses (< 4096) are rejected as invalid
	testCases := []struct {
		name string
		ptr  uintptr
	}{
		{"address 1", 1},
		{"address 100", 100},
		{"address 1000", 1000},
		{"address 4095", 4095},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := CstringToGo(tc.ptr)
			if result != "" {
				t.Errorf("expected empty string for invalid low address %d, got %q", tc.ptr, result)
			}
		})
	}
}

func TestCstringToGoValidHighAddress(t *testing.T) {
	// Test that valid high addresses are processed (when they point to valid strings)
	// We can't easily test with arbitrary high addresses without causing segfaults,
	// but we can verify the threshold logic by using valid Go-allocated memory

	// Create a valid C string in Go memory
	testStr := "test"
	bytes, ptr := GoToCstring(testStr)
	defer func() { _ = bytes }() // Keep alive

	// The pointer should be well above 4096 (Go heap addresses are high)
	if ptr < 4096 {
		t.Skip("Go allocated memory at unexpectedly low address")
	}

	result := CstringToGo(ptr)
	if result != testStr {
		t.Errorf("expected %q for valid high address, got %q", testStr, result)
	}
}

func TestRoundTripConversion(t *testing.T) {
	tests := []string{
		"",
		"a",
		"hello",
		"hello world",
		"Hello, 世界",
		"Hello 👋 World 🌍",
		strings.Repeat("x", 100),
		strings.Repeat("y", 1000),
		"special\x00chars", // Note: this will be truncated at \x00
	}

	for _, original := range tests {
		t.Run(original, func(t *testing.T) {
			// For strings with embedded nulls, we expect truncation
			expectedAfterRoundTrip := original
			if idx := strings.IndexByte(original, 0); idx >= 0 {
				expectedAfterRoundTrip = original[:idx]
			}

			// Round trip: Go -> C -> Go
			bytes, ptr := GoToCstring(original)
			result := CstringToGo(ptr)
			_ = bytes // Keep alive

			if result != expectedAfterRoundTrip {
				t.Errorf("round trip failed: expected %q, got %q", expectedAfterRoundTrip, result)
			}
		})
	}
}

func TestGoToCstringPreservesBytes(t *testing.T) {
	input := "test string"
	bytes1, ptr1 := GoToCstring(input)
	bytes2, ptr2 := GoToCstring(input)

	// Different calls should produce different byte slices and pointers
	if ptr1 == ptr2 {
		t.Error("expected different pointers for different calls")
	}

	// But both should have the same content
	result1 := CstringToGo(ptr1)
	result2 := CstringToGo(ptr2)
	_ = bytes1 // Keep alive
	_ = bytes2 // Keep alive

	if result1 != result2 || result1 != input {
		t.Errorf("expected both conversions to produce %q, got %q and %q", input, result1, result2)
	}
}

func BenchmarkGoToCstring(b *testing.B) {
	tests := []struct {
		name  string
		input string
	}{
		{"short", "hello"},
		{"medium", strings.Repeat("a", 100)},
		{"long", strings.Repeat("b", 1000)},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				bytes, _ := GoToCstring(tt.input)
				_ = bytes
			}
		})
	}
}

func BenchmarkCstringToGo(b *testing.B) {
	tests := []struct {
		name  string
		input string
	}{
		{"short", "hello"},
		{"medium", strings.Repeat("a", 100)},
		{"long", strings.Repeat("b", 1000)},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			bytes, ptr := GoToCstring(tt.input)
			_ = bytes // Keep alive
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = CstringToGo(ptr)
			}
		})
	}
}
