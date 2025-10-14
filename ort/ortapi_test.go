package ort

import (
	"os"
	"testing"
)

// TestOrtApiStructLayout verifies that the OrtApi struct layout matches the C API
// by testing that function pointers at various offsets work correctly
func TestOrtApiStructLayout(t *testing.T) {
	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		t.Skip("ONNXRUNTIME_LIB_PATH not set, skipping test")
	}

	if err := SetSharedLibraryPath(libPath); err != nil {
		t.Fatalf("Failed to set library path: %v", err)
	}

	// Test initialization (uses early functions in struct: CreateEnv, etc.)
	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("Failed to initialize environment: %v", err)
	}

	// Verify we can get the version string (function #2 in OrtApiBase)
	version := GetVersionString()
	if version == "" || version == "0.0.0-dev" {
		t.Error("Failed to get version string")
	}
	t.Logf("ONNX Runtime version: %s", version)

	// Test cleanup (uses ReleaseEnv which is function #93 in the struct)
	// This verifies that functions later in the struct layout are accessible
	if err := DestroyEnvironment(); err != nil {
		t.Fatalf("Failed to destroy environment (ReleaseEnv may have failed): %v", err)
	}

	t.Log("Successfully called ReleaseEnv - struct layout is correct!")
}

// TestReleaseEnvMultipleTimes ensures ReleaseEnv doesn't crash when called multiple times
func TestReleaseEnvMultipleTimes(t *testing.T) {
	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		t.Skip("ONNXRUNTIME_LIB_PATH not set, skipping test")
	}

	if err := SetSharedLibraryPath(libPath); err != nil {
		t.Fatalf("Failed to set library path: %v", err)
	}

	// Initialize and destroy multiple times
	for i := 0; i < 3; i++ {
		if err := InitializeEnvironment(); err != nil {
			t.Fatalf("Failed to initialize environment (iteration %d): %v", i, err)
		}

		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("Failed to destroy environment (iteration %d): %v", i, err)
		}
	}

	t.Log("Successfully initialized and destroyed environment 3 times")
}
