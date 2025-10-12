package main

import (
	"runtime"
)

// getDefaultLibraryPath returns a platform-specific default path for ONNX Runtime
// This is used for demonstration purposes - in production, users should specify their own paths
func getDefaultLibraryPath() string {
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			return "/usr/local/lib/libonnxruntime.dylib"
		}
		return "/usr/local/lib/libonnxruntime.dylib"
	case "linux":
		return "/usr/lib/libonnxruntime.so"
	case "windows":
		return "onnxruntime.dll"
	default:
		return ""
	}
}

// getTestLibraryPaths returns a list of paths to try for testing
// These include common installation locations
func getTestLibraryPaths() []string {
	switch runtime.GOOS {
	case "darwin":
		return []string{
			"/usr/local/lib/libonnxruntime.dylib",
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"./third_party/onnxruntime/lib/libonnxruntime.dylib",
			"/Users/tazarov/Downloads/onnxruntime-osx-arm64-1.21.0/lib/libonnxruntime.1.21.0.dylib", // Dev path
		}
	case "linux":
		return []string{
			"/usr/lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"./third_party/onnxruntime/lib/libonnxruntime.so",
		}
	case "windows":
		return []string{
			"onnxruntime.dll",
			"./third_party/onnxruntime/lib/onnxruntime.dll",
			"C:\\Program Files\\ONNXRuntime\\lib\\onnxruntime.dll",
		}
	default:
		return []string{}
	}
}