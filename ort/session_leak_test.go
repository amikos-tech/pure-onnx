package ort

import (
	"fmt"
	"runtime"
	"testing"
)

func TestAdvancedSessionRunWithAllMiniLML6V2MemoryStability(t *testing.T) {
	cleanup := setupTestEnvironment(t)
	defer cleanup()

	modelPath := resolveAllMiniLMModelPath(t)
	sequenceLength := allMiniLMSequenceLength(t)
	iterations := envIntOrDefault(t, "ONNXRUNTIME_TEST_LEAK_ITERATIONS", 80, 1)
	// Keep this aligned with CI default to reduce local-vs-CI divergence; value leaves
	// headroom for allocator variance across ORT versions and runner profiles.
	maxGrowthMB := envIntOrDefault(t, "ONNXRUNTIME_TEST_LEAK_MAX_GROWTH_MB", 96, 1)

	t.Log("memory stability check measures Go heap growth only; use native tooling (ASan/Valgrind) for ORT allocator leaks")

	// Double-GC improves stability of heap snapshots for this coarse regression check.
	runtime.GC()
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	for i := 0; i < iterations; i++ {
		output := runAllMiniLMInference(t, modelPath, sequenceLength)
		if i == 0 || i == iterations-1 {
			requireFiniteFloat32Slice(t, fmt.Sprintf("all-MiniLM output (iteration %d/%d)", i+1, iterations), output)
		}
		if (i+1)%10 == 0 {
			runtime.GC()
		}
	}

	runtime.GC()
	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	heapGrowth := int64(after.HeapAlloc) - int64(before.HeapAlloc)
	if heapGrowth < 0 {
		heapGrowth = 0
	}
	maxAllowedGrowth := int64(maxGrowthMB) * 1024 * 1024
	if heapGrowth > maxAllowedGrowth {
		t.Fatalf("heap growth exceeded threshold after %d runs: growth=%s limit=%s (set ONNXRUNTIME_TEST_LEAK_MAX_GROWTH_MB to tune)",
			iterations, formatMB(heapGrowth), formatMB(maxAllowedGrowth))
	}
}
