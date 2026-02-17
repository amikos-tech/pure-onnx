//go:build !windows

package ort

func goStringToORTChar(s string) (uintptr, any, error) {
	bytes, ptr := GoToCstring(s)
	return ptr, bytes, nil
}
