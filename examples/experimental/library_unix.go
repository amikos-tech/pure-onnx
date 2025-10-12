//go:build !windows

package main

import (
	"os"

	"github.com/ebitengine/purego"
)

func loadLibrary(path string) (uintptr, error) {
	libHandle, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil || libHandle == 0 {
		return 0, err
	}
	return libHandle, nil
}

func getSymbol(handle uintptr, symbol string) (uintptr, error) {
	return purego.Dlsym(handle, symbol)
}

func closeLibrary(handle uintptr) error {
	if handle == 0 {
		return nil
	}
	return purego.Dlclose(handle)
}

func isLibraryValid(path string) bool {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return false
	}

	// Try to load the library to verify it's valid
	if libh, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL); err == nil {
		_ = purego.Dlclose(libh)
		return true
	}
	return false
}