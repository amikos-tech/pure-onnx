package ort

import (
	"os"
	"sync"
	"testing"
)

// Most stress tests below seed a baseline reference count of one under mu before
// spawning workers, mirroring TestConcurrentInitialization. This baseline
// guarantees refCount never drops below one while workers run, so every
// InitializeEnvironment call takes the fast increment-only path and never
// reaches the real (failing, against a nonexistent path) library loader. They
// stress the refcount/mutex accounting only. TestStressRealInitTeardownTransition
// is the exception: it allows refCount to cross 0<->1 against a real library so
// the genuine load/teardown path is exercised. They are gated behind short mode
// so the repository's default test entry points can skip them.

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

// TestStressRealInitTeardownTransition drives the real 0<->1 environment
// transition -- dlopen/dlclose, OrtGetApiBase/CreateEnv/ReleaseEnv, and purego
// symbol registration -- concurrently, so the genuinely racy load/teardown path
// (not just the seeded fast-path refcount accounting the other stress tests
// cover) is exercised under the race detector. It requires a real ONNX Runtime
// library; without ONNXRUNTIME_LIB_PATH it skips.
//
// Init and Destroy each hold ortCallMu exclusively, so real load/teardown is
// serialized -- but strict per-goroutine Init->Destroy pairing under contention
// still crosses refCount 0<->1 repeatedly, reloading and releasing the library
// and re-mutating the ortLib/ortEnv/ortAPI globals many times.
func TestStressRealInitTeardownTransition(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")
	}

	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		t.Skip("skipping real init/teardown stress: ONNXRUNTIME_LIB_PATH not set")
	}

	resetEnvironmentState()
	defer resetEnvironmentState()

	if err := SetSharedLibraryPath(libPath); err != nil {
		t.Fatalf("failed to set library path: %v", err)
	}

	const goroutines = 8
	const iterations = 20

	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				if err := InitializeEnvironment(); err != nil {
					t.Errorf("InitializeEnvironment failed: %v", err)
					return
				}
				if err := DestroyEnvironment(); err != nil {
					t.Errorf("DestroyEnvironment failed: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()

	mu.Lock()
	if refCount != 0 {
		t.Errorf("expected refCount to be 0 after strictly paired real init/destroy, got %d", refCount)
	}
	mu.Unlock()

	if IsInitialized() {
		t.Error("expected environment to be uninitialized after all real init/destroy cycles")
	}
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
