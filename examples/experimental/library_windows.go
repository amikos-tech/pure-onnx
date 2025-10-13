//go:build windows

package main

import (
	"os"
	"unsafe"

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
	proc, err := windows.GetProcAddress(windows.Handle(handle), symbol)
	if err != nil {
		return 0, err
	}
	// Safe: Converting proc address to uintptr for immediate use with purego.RegisterFunc
	// The library handle ensures the proc address remains valid during usage
	return uintptr(unsafe.Pointer(proc)), nil
}

func closeLibrary(handle uintptr) error {
	if handle == 0 {
		return nil
	}
	return windows.FreeLibrary(windows.Handle(handle))
}

func isLibraryValid(path string) bool {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return false
	}

	// Try to load the library to verify it's valid
	if handle, err := windows.LoadLibrary(path); err == nil {
		_ = windows.FreeLibrary(handle)
		return true
	}
	return false
}
