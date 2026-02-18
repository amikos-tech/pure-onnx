package ort

import (
	"fmt"
	"os"
	"strconv"
	"testing"
)

func requireDestroy(tb testing.TB, name string, destroy func() error) {
	tb.Helper()
	if err := destroy(); err != nil {
		tb.Fatalf("failed to destroy %s: %v", name, err)
	}
}

func envIntOrDefault(tb testing.TB, key string, defaultValue int, minValue int) int {
	tb.Helper()

	raw := os.Getenv(key)
	if raw == "" {
		return defaultValue
	}

	value, err := strconv.Atoi(raw)
	if err != nil {
		tb.Fatalf("invalid %s %q: %v", key, raw, err)
	}
	if value < minValue {
		tb.Fatalf("%s must be >= %d, got %d", key, minValue, value)
	}

	return value
}

func runAllMiniLMInferenceOnce(tb testing.TB, modelPath string, sequenceLength int) {
	tb.Helper()

	inputShape := Shape{1, int64(sequenceLength)}
	outputShape := Shape{1, int64(sequenceLength), allMiniLMOutputEmbeddingDim}
	inputIDs, attentionMask, tokenTypeIDs := makeAllMiniLMInputs(sequenceLength)

	inputIDsTensor, err := NewTensor[int64](inputShape, inputIDs)
	if err != nil {
		tb.Fatalf("failed to create input_ids tensor: %v", err)
	}
	attentionMaskTensor, err := NewTensor[int64](inputShape, attentionMask)
	if err != nil {
		requireDestroy(tb, "input_ids tensor", inputIDsTensor.Destroy)
		tb.Fatalf("failed to create attention_mask tensor: %v", err)
	}
	tokenTypeIDsTensor, err := NewTensor[int64](inputShape, tokenTypeIDs)
	if err != nil {
		requireDestroy(tb, "attention_mask tensor", attentionMaskTensor.Destroy)
		requireDestroy(tb, "input_ids tensor", inputIDsTensor.Destroy)
		tb.Fatalf("failed to create token_type_ids tensor: %v", err)
	}
	outputTensor, err := NewEmptyTensor[float32](outputShape)
	if err != nil {
		requireDestroy(tb, "token_type_ids tensor", tokenTypeIDsTensor.Destroy)
		requireDestroy(tb, "attention_mask tensor", attentionMaskTensor.Destroy)
		requireDestroy(tb, "input_ids tensor", inputIDsTensor.Destroy)
		tb.Fatalf("failed to create output tensor: %v", err)
	}

	session, err := NewAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		[]Value{inputIDsTensor, attentionMaskTensor, tokenTypeIDsTensor},
		[]Value{outputTensor},
		nil,
	)
	if err != nil {
		requireDestroy(tb, "output tensor", outputTensor.Destroy)
		requireDestroy(tb, "token_type_ids tensor", tokenTypeIDsTensor.Destroy)
		requireDestroy(tb, "attention_mask tensor", attentionMaskTensor.Destroy)
		requireDestroy(tb, "input_ids tensor", inputIDsTensor.Destroy)
		tb.Fatalf("failed to create all-MiniLM session: %v", err)
	}

	if err := session.Run(); err != nil {
		requireDestroy(tb, "session", session.Destroy)
		requireDestroy(tb, "output tensor", outputTensor.Destroy)
		requireDestroy(tb, "token_type_ids tensor", tokenTypeIDsTensor.Destroy)
		requireDestroy(tb, "attention_mask tensor", attentionMaskTensor.Destroy)
		requireDestroy(tb, "input_ids tensor", inputIDsTensor.Destroy)
		tb.Fatalf("all-MiniLM inference failed: %v", err)
	}

	expectedOutputSize := sequenceLength * int(allMiniLMOutputEmbeddingDim)
	if got := len(outputTensor.GetData()); got != expectedOutputSize {
		requireDestroy(tb, "session", session.Destroy)
		requireDestroy(tb, "output tensor", outputTensor.Destroy)
		requireDestroy(tb, "token_type_ids tensor", tokenTypeIDsTensor.Destroy)
		requireDestroy(tb, "attention_mask tensor", attentionMaskTensor.Destroy)
		requireDestroy(tb, "input_ids tensor", inputIDsTensor.Destroy)
		tb.Fatalf("unexpected output length: got %d, want %d", got, expectedOutputSize)
	}

	destroyErrors := []error{
		session.Destroy(),
		outputTensor.Destroy(),
		tokenTypeIDsTensor.Destroy(),
		attentionMaskTensor.Destroy(),
		inputIDsTensor.Destroy(),
	}
	for i, err := range destroyErrors {
		if err != nil {
			tb.Fatalf("cleanup failed at step %d: %v", i, err)
		}
	}
}

func formatMB(bytes int64) string {
	return fmt.Sprintf("%.2f MiB", float64(bytes)/1024.0/1024.0)
}
