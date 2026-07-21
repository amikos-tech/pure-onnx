//go:build windows

package ort

import (
	"golang.org/x/sys/windows"
)

func loadLibrary(path string) (uintptr, error) {
	handle, err := windows.LoadLibrary(path)
	if err != nil || handle == 0 {
		return 0, err
	}
	return uintptr(handle), nil
}

func getSymbol(handle uintptr, symbol string) (uintptr, error) {
	// GetProcAddress already returns a uintptr; no unsafe.Pointer round-trip needed.
	return windows.GetProcAddress(windows.Handle(handle), symbol)
}

func closeLibrary(handle uintptr) error {
	if handle == 0 {
		return nil
	}
	return windows.FreeLibrary(windows.Handle(handle))
}
