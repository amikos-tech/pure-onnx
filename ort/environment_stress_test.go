package ort

import (
	"sync"
	"testing"
)

// The stress tests below seed a baseline reference count of one under mu before
// spawning workers, mirroring TestConcurrentInitialization. This baseline
// guarantees refCount never drops below one while workers run, so every
// InitializeEnvironment call takes the fast increment-only path and never
// reaches the real (failing, against a nonexistent path) library loader. They
// are gated behind short mode so the repository's default test entry points can
// skip them.

// TestStressConcurrentInitDestroy hammers strictly paired InitializeEnvironment/
// DestroyEnvironment cycles from many goroutines to guard the refcount/mutex
// lifecycle against corruption, deadlocks, and panics under load.
func TestStressConcurrentInitDestroy(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")
	}

	resetEnvironmentState()
	defer resetEnvironmentState()

	if err := SetSharedLibraryPath("/nonexistent/path.so"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	mu.Lock()
	refCount = 1
	mu.Unlock()

	const goroutines = 100
	const iterations = 1000

	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				_ = InitializeEnvironment()
				_ = DestroyEnvironment()
			}
		}()
	}
	wg.Wait()

	mu.Lock()
	if refCount != 1 {
		t.Errorf("expected refCount to be 1 (untouched baseline) after workers, got %d", refCount)
	}
	mu.Unlock()

	_ = DestroyEnvironment()

	mu.Lock()
	if refCount != 0 {
		t.Errorf("expected refCount to be 0 after retiring baseline, got %d", refCount)
	}
	mu.Unlock()
}

// TestStressRapidInitDestroy uses a different concurrency shape (more goroutines,
// fewer iterations each) to stress mu contention differently while exercising
// the same strictly paired, seeded-baseline refcount lifecycle.
func TestStressRapidInitDestroy(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")
	}

	resetEnvironmentState()
	defer resetEnvironmentState()

	if err := SetSharedLibraryPath("/nonexistent/path.so"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	mu.Lock()
	refCount = 1
	mu.Unlock()

	const goroutines = 200
	const iterations = 500

	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				_ = InitializeEnvironment()
				_ = DestroyEnvironment()
			}
		}()
	}
	wg.Wait()

	mu.Lock()
	if refCount != 1 {
		t.Errorf("expected refCount to be 1 (untouched baseline) after workers, got %d", refCount)
	}
	mu.Unlock()

	_ = DestroyEnvironment()

	mu.Lock()
	if refCount != 0 {
		t.Errorf("expected refCount to be 0 after retiring baseline, got %d", refCount)
	}
	mu.Unlock()
}

// TestStressMixedOperationsUnderLoad interleaves read-only and rejected calls
// (IsInitialized, GetVersionString, SetLogLevel) with the strictly paired
// Init/Destroy sequence. Because each iteration's Init/Destroy pair is adjacent
// and unconditional, the accounting stays as tight as the paired-only tests
// while adding coverage of the concurrent read paths.
func TestStressMixedOperationsUnderLoad(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")
	}

	resetEnvironmentState()
	defer resetEnvironmentState()

	if err := SetSharedLibraryPath("/nonexistent/path.so"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	mu.Lock()
	refCount = 1
	mu.Unlock()

	const goroutines = 50
	const iterations = 500

	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				_ = InitializeEnvironment()
				_ = IsInitialized()
				_ = GetVersionString()
				if j%10 == 0 {
					// refCount > 0 throughout, so this always returns an error
					// and has no refCount side effect -- ignore it.
					_ = SetLogLevel(LoggingLevelWarning)
				}
				_ = DestroyEnvironment()
			}
		}()
	}
	wg.Wait()

	mu.Lock()
	if refCount != 1 {
		t.Errorf("expected refCount to be 1 (untouched baseline) after workers, got %d", refCount)
	}
	mu.Unlock()

	_ = DestroyEnvironment()

	mu.Lock()
	if refCount != 0 {
		t.Errorf("expected refCount to be 0 after retiring baseline, got %d", refCount)
	}
	mu.Unlock()
}
