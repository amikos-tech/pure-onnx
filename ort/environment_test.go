package ort

import (
	"os"
	"strings"
	"sync"
	"testing"
)

// resetEnvironmentState resets global state for testing
func resetEnvironmentState() {
	mu.Lock()
	defer mu.Unlock()
	refCount = 0
	ortLib = 0
	ortAPI = nil
	ortEnv = 0
	libPath = ""
	logLevel = LoggingLevelWarning
	getVersionStringFunc = nil
	getErrorMessageFunc = nil
	releaseStatusFunc = nil
	createMemoryInfoFunc = nil
	releaseMemoryInfoFunc = nil
	createTensorWithDataAsOrtValueFunc = nil
	releaseValueFunc = nil
	createSessionOptionsFunc = nil
	releaseSessionOptionsFunc = nil
	createSessionFunc = nil
	runSessionFunc = nil
	releaseSessionFunc = nil
}

func TestIsInitialized(t *testing.T) {
	resetEnvironmentState()

	if IsInitialized() {
		t.Error("expected environment to not be initialized")
	}

	// Manually set refCount to simulate initialization
	mu.Lock()
	refCount = 1
	mu.Unlock()

	if !IsInitialized() {
		t.Error("expected environment to be initialized")
	}

	// Reset
	resetEnvironmentState()
}

func TestSetSharedLibraryPath(t *testing.T) {
	resetEnvironmentState()

	path := "/test/path/libonnxruntime.so"
	err := SetSharedLibraryPath(path)
	if err != nil {
		t.Errorf("unexpected error setting library path: %v", err)
	}

	mu.Lock()
	if libPath != path {
		t.Errorf("expected libPath to be %q, got %q", path, libPath)
	}
	mu.Unlock()

	// Test that changing path after init returns an error
	mu.Lock()
	refCount = 1
	mu.Unlock()

	newPath := "/different/path.so"
	err = SetSharedLibraryPath(newPath)
	if err == nil {
		t.Error("expected error when setting library path after initialization")
	}

	mu.Lock()
	if libPath != path {
		t.Errorf("expected libPath to remain %q after init, got %q", path, libPath)
	}
	mu.Unlock()

	resetEnvironmentState()
}

func TestSetLogLevel(t *testing.T) {
	resetEnvironmentState()

	tests := []LoggingLevel{
		LoggingLevelVerbose,
		LoggingLevelInfo,
		LoggingLevelWarning,
		LoggingLevelError,
		LoggingLevelFatal,
	}

	for _, level := range tests {
		err := SetLogLevel(level)
		if err != nil {
			t.Errorf("unexpected error setting log level: %v", err)
		}

		mu.Lock()
		if logLevel != level {
			t.Errorf("expected logLevel to be %d, got %d", level, logLevel)
		}
		mu.Unlock()
	}

	// Test that changing level after init returns an error
	err := SetLogLevel(LoggingLevelWarning)
	if err != nil {
		t.Errorf("unexpected error setting log level: %v", err)
	}
	mu.Lock()
	refCount = 1
	mu.Unlock()

	err = SetLogLevel(LoggingLevelError)
	if err == nil {
		t.Error("expected error when setting log level after initialization")
	}

	mu.Lock()
	if logLevel != LoggingLevelWarning {
		t.Errorf("expected logLevel to remain Warning after init, got %d", logLevel)
	}
	mu.Unlock()

	resetEnvironmentState()
}

func TestGetVersionStringWhenNotInitialized(t *testing.T) {
	resetEnvironmentState()

	version := GetVersionString()
	if version != "0.0.0-dev" {
		t.Errorf("expected version to be '0.0.0-dev' when not initialized, got %q", version)
	}

	resetEnvironmentState()
}

func TestInitializeEnvironmentWithoutLibraryPath(t *testing.T) {
	resetEnvironmentState()

	err := InitializeEnvironment()
	if err == nil {
		t.Error("expected error when library path not set")
	}

	if err.Error() != "library path not set, call SetSharedLibraryPath first" {
		t.Errorf("unexpected error message: %v", err)
	}

	resetEnvironmentState()
}

func TestReferenceCountingLogic(t *testing.T) {
	resetEnvironmentState()

	// Simulate initialized state
	mu.Lock()
	refCount = 1
	mu.Unlock()

	// First init increments
	err := InitializeEnvironment()
	if err != nil {
		t.Errorf("unexpected error on second init: %v", err)
	}

	mu.Lock()
	if refCount != 2 {
		t.Errorf("expected refCount to be 2, got %d", refCount)
	}
	mu.Unlock()

	// Third init increments again
	err = InitializeEnvironment()
	if err != nil {
		t.Errorf("unexpected error on third init: %v", err)
	}

	mu.Lock()
	if refCount != 3 {
		t.Errorf("expected refCount to be 3, got %d", refCount)
	}
	mu.Unlock()

	resetEnvironmentState()
}

func TestDestroyEnvironmentWhenNotInitialized(t *testing.T) {
	resetEnvironmentState()

	err := DestroyEnvironment()
	if err != nil {
		t.Errorf("unexpected error when destroying non-initialized environment: %v", err)
	}

	resetEnvironmentState()
}

func TestDestroyEnvironmentDecrements(t *testing.T) {
	resetEnvironmentState()

	// Simulate initialized state with refCount=3
	mu.Lock()
	refCount = 3
	mu.Unlock()

	// First destroy
	err := DestroyEnvironment()
	if err != nil {
		t.Errorf("unexpected error on destroy: %v", err)
	}

	mu.Lock()
	if refCount != 2 {
		t.Errorf("expected refCount to be 2, got %d", refCount)
	}
	mu.Unlock()

	// Second destroy
	err = DestroyEnvironment()
	if err != nil {
		t.Errorf("unexpected error on destroy: %v", err)
	}

	mu.Lock()
	if refCount != 1 {
		t.Errorf("expected refCount to be 1, got %d", refCount)
	}
	mu.Unlock()

	resetEnvironmentState()
}

func TestConcurrentInitialization(t *testing.T) {
	resetEnvironmentState()

	// Set a dummy library path
	if err := SetSharedLibraryPath("/nonexistent/path.so"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	var wg sync.WaitGroup
	concurrency := 10

	// Simulate initialized state first
	mu.Lock()
	refCount = 1
	mu.Unlock()

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = InitializeEnvironment()
		}()
	}

	wg.Wait()

	mu.Lock()
	expectedCount := 1 + concurrency
	if refCount != expectedCount {
		t.Errorf("expected refCount to be %d after concurrent inits, got %d", expectedCount, refCount)
	}
	mu.Unlock()

	resetEnvironmentState()
}

func TestConcurrentDestroy(t *testing.T) {
	resetEnvironmentState()

	concurrency := 10

	// Set initial refCount
	mu.Lock()
	refCount = concurrency
	mu.Unlock()

	var wg sync.WaitGroup

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = DestroyEnvironment()
		}()
	}

	wg.Wait()

	mu.Lock()
	if refCount != 0 {
		t.Errorf("expected refCount to be 0 after concurrent destroys, got %d", refCount)
	}
	mu.Unlock()

	resetEnvironmentState()
}

// TestInitializeWithActualLibrary tests with a real ONNX Runtime library if available
func TestInitializeWithActualLibrary(t *testing.T) {
	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		t.Skip("Skipping integration test: ONNXRUNTIME_LIB_PATH not set")
	}

	resetEnvironmentState()

	if err := SetSharedLibraryPath(libPath); err != nil {
		t.Fatalf("failed to set library path: %v", err)
	}

	err := InitializeEnvironment()
	if err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}

	if !IsInitialized() {
		t.Error("expected environment to be initialized")
	}

	// Test version string
	version := GetVersionString()
	if version == "0.0.0-dev" || version == "" {
		t.Errorf("expected valid version string, got %q", version)
	}
	t.Logf("ONNX Runtime version: %s", version)

	// Test double initialization (should increment ref count)
	err = InitializeEnvironment()
	if err != nil {
		t.Errorf("failed second initialization: %v", err)
	}

	// First destroy (should decrement)
	err = DestroyEnvironment()
	if err != nil {
		t.Errorf("failed first destroy: %v", err)
	}

	// Should still be initialized
	if !IsInitialized() {
		t.Error("expected environment to still be initialized after first destroy")
	}

	// Second destroy (should actually destroy)
	err = DestroyEnvironment()
	if err != nil {
		t.Errorf("failed second destroy: %v", err)
	}

	// Now should be uninitialized
	if IsInitialized() {
		t.Error("expected environment to be uninitialized after final destroy")
	}

	resetEnvironmentState()
}

func TestGetErrorMessageWithNullStatus(t *testing.T) {
	result := getErrorMessage(0)
	if result != "" {
		t.Errorf("expected empty string for null status, got %q", result)
	}
}

func TestGetErrorMessageWhenNotInitialized(t *testing.T) {
	resetEnvironmentState()

	// When getErrorMessageFunc is nil, should return empty string
	result := getErrorMessage(1234)
	if result != "" {
		t.Errorf("expected empty string when not initialized, got %q", result)
	}

	resetEnvironmentState()
}

func TestReleaseStatusWithNullStatus(t *testing.T) {
	// Should not panic
	releaseStatus(0)
}

func TestReleaseStatusWhenNotInitialized(t *testing.T) {
	resetEnvironmentState()

	// When releaseStatusFunc is nil, should not panic
	releaseStatus(1234)

	resetEnvironmentState()
}

func TestErrorMessageIntegrationWithFailedInit(t *testing.T) {
	// Test that error messages are properly extracted during failed initialization
	resetEnvironmentState()

	if err := SetSharedLibraryPath("/nonexistent/path/libonnxruntime.so"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	err := InitializeEnvironment()
	if err == nil {
		t.Fatal("expected error when loading non-existent library")
	}

	// Verify error message contains helpful information
	errMsg := err.Error()
	if !strings.Contains(errMsg, "failed to load ONNX Runtime library") {
		t.Errorf("expected error message to mention library loading failure, got: %v", errMsg)
	}

	// Verify the error provides context about what went wrong
	if errMsg == "" {
		t.Error("expected non-empty error message")
	}

	resetEnvironmentState()
}

func TestErrorMessageFormattingQuality(t *testing.T) {
	// Test that error messages follow good practices
	resetEnvironmentState()

	testCases := []struct {
		name         string
		setup        func() error
		shouldError  bool
		errorPattern string
	}{
		{
			name: "missing library path",
			setup: func() error {
				return InitializeEnvironment()
			},
			shouldError:  true,
			errorPattern: "library path not set",
		},
		{
			name: "cannot change path after init",
			setup: func() error {
				mu.Lock()
				refCount = 1
				mu.Unlock()
				return SetSharedLibraryPath("/new/path.so")
			},
			shouldError:  true,
			errorPattern: "cannot change library path after environment is initialized",
		},
		{
			name: "cannot change log level after init",
			setup: func() error {
				mu.Lock()
				refCount = 1
				mu.Unlock()
				return SetLogLevel(LoggingLevelError)
			},
			shouldError:  true,
			errorPattern: "cannot change log level after environment is initialized",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resetEnvironmentState()

			err := tc.setup()

			if tc.shouldError {
				if err == nil {
					t.Errorf("expected error but got nil")
					return
				}

				errMsg := err.Error()

				// Check that error message matches expected pattern
				if !strings.Contains(errMsg, tc.errorPattern) {
					t.Errorf("expected error message to contain %q, got: %v", tc.errorPattern, errMsg)
				}

				// Check error message quality
				if len(errMsg) < 10 {
					t.Errorf("error message too short (< 10 chars): %q", errMsg)
				}

				// Error messages should start with lowercase (Go convention for wrapped errors)
				// or be a complete sentence
				if errMsg[0] >= 'A' && errMsg[0] <= 'Z' {
					// Capital letter is OK if it's a proper noun or acronym
					if !strings.HasPrefix(errMsg, "ONNX") && !strings.HasPrefix(errMsg, "ORT") {
						// This is fine, just noting that it starts with capital
					}
				}
			} else {
				if err != nil {
					t.Errorf("expected no error but got: %v", err)
				}
			}

			resetEnvironmentState()
		})
	}
}

// Error path tests with real failure conditions

func TestInitializeWithNonExistentLibrary(t *testing.T) {
	resetEnvironmentState()

	if err := SetSharedLibraryPath("/nonexistent/path/libonnxruntime.so"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	err := InitializeEnvironment()
	if err == nil {
		t.Error("expected error when loading non-existent library")
	}
	if err != nil && !strings.Contains(err.Error(), "failed to load ONNX Runtime library") {
		t.Errorf("expected load error, got: %v", err)
	}

	resetEnvironmentState()
}

func TestInitializeWithInvalidLibrary(t *testing.T) {
	resetEnvironmentState()

	// Use the test binary itself as an invalid library
	// It exists as a file but doesn't have the ONNX Runtime symbols
	if err := SetSharedLibraryPath("/bin/sh"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	err := InitializeEnvironment()
	if err == nil {
		t.Error("expected error when loading invalid library")
		_ = DestroyEnvironment() // Clean up if it somehow succeeded
	}

	resetEnvironmentState()
}

func TestMultipleInitializeAfterDestroy(t *testing.T) {
	resetEnvironmentState()

	// Set library path
	if err := SetSharedLibraryPath("/nonexistent/path.so"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	// Simulate a successful initialization
	mu.Lock()
	refCount = 1
	mu.Unlock()

	// Destroy
	err := DestroyEnvironment()
	if err != nil {
		t.Errorf("unexpected error on destroy: %v", err)
	}

	// Should be able to set library path again after destroy
	if err := SetSharedLibraryPath("/different/path.so"); err != nil {
		t.Errorf("expected to be able to change library path after destroy, got error: %v", err)
	}

	mu.Lock()
	if libPath != "/different/path.so" {
		t.Errorf("expected libPath to be updated after destroy, got %q", libPath)
	}
	mu.Unlock()

	resetEnvironmentState()
}

// Benchmarks

func BenchmarkInitializeEnvironment(b *testing.B) {
	// Benchmark the reference counting path (already initialized)
	// This is the fast path that most applications will hit
	resetEnvironmentState()

	// Simulate already initialized state
	mu.Lock()
	refCount = 1
	mu.Unlock()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = InitializeEnvironment()
	}
	b.StopTimer()

	resetEnvironmentState()
}

func BenchmarkDestroyEnvironment(b *testing.B) {
	// Benchmark the reference counting path (decrement without actual cleanup)
	// This is the fast path when refCount > 1
	resetEnvironmentState()

	// Set high refCount so we never reach zero
	mu.Lock()
	refCount = b.N + 1
	mu.Unlock()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DestroyEnvironment()
	}
	b.StopTimer()

	resetEnvironmentState()
}

func BenchmarkInitializeDestroyPair(b *testing.B) {
	// Benchmark a complete init/destroy pair
	// This measures the full lifecycle with reference counting
	resetEnvironmentState()

	// Start with refCount=1 to avoid actual library operations
	mu.Lock()
	refCount = 1
	mu.Unlock()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = InitializeEnvironment() // Increments to 2
		_ = DestroyEnvironment()    // Decrements back to 1
	}
	b.StopTimer()

	resetEnvironmentState()
}

func BenchmarkSetSharedLibraryPath(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		resetEnvironmentState()
		b.StartTimer()

		_ = SetSharedLibraryPath("/path/to/library.so")
	}
}

func BenchmarkIsInitialized(b *testing.B) {
	resetEnvironmentState()

	// Test both initialized and uninitialized states
	b.Run("uninitialized", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = IsInitialized()
		}
	})

	b.Run("initialized", func(b *testing.B) {
		mu.Lock()
		refCount = 1
		mu.Unlock()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = IsInitialized()
		}
	})

	resetEnvironmentState()
}

func BenchmarkGetVersionString(b *testing.B) {
	resetEnvironmentState()

	// Test uninitialized path (fast path - no C call)
	b.Run("uninitialized", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = GetVersionString()
		}
	})

	// Note: We can't easily benchmark the initialized path without a real library
	// That would require integration testing with actual ONNX Runtime

	resetEnvironmentState()
}
