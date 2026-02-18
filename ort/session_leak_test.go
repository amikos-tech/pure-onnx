package ort

import (
	"runtime"
	"testing"
)

func TestAdvancedSessionRunWithAllMiniLML6V2MemoryStability(t *testing.T) {
	cleanup := setupTestEnvironment(t)
	defer cleanup()

	modelPath := resolveAllMiniLMModelPath(t)
	sequenceLength := allMiniLMSequenceLength(t)
	iterations := envIntOrDefault(t, "ONNXRUNTIME_TEST_LEAK_ITERATIONS", 80, 1)
	maxGrowthMB := envIntOrDefault(t, "ONNXRUNTIME_TEST_LEAK_MAX_GROWTH_MB", 64, 1)

	runtime.GC()
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	for i := 0; i < iterations; i++ {
		runAllMiniLMInferenceOnce(t, modelPath, sequenceLength)
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
