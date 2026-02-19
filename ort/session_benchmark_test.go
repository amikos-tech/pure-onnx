package ort

import "testing"

func BenchmarkAdvancedSessionRunWarmWithAllMiniLML6V2(b *testing.B) {
	cleanup := setupTestEnvironment(b)
	defer cleanup()

	modelPath := resolveAllMiniLMModelPath(b)
	sequenceLength := allMiniLMSequenceLength(b)

	inputShape := Shape{1, int64(sequenceLength)}
	outputShape := Shape{1, int64(sequenceLength), allMiniLMOutputEmbeddingDim}
	inputIDs, attentionMask, tokenTypeIDs := makeAllMiniLMInputs(b, sequenceLength)

	inputIDsTensor, err := NewTensor[int64](inputShape, inputIDs)
	if err != nil {
		b.Fatalf("failed to create input_ids tensor: %v", err)
	}
	defer requireDestroy(b, "input_ids tensor", inputIDsTensor.Destroy)

	attentionMaskTensor, err := NewTensor[int64](inputShape, attentionMask)
	if err != nil {
		b.Fatalf("failed to create attention_mask tensor: %v", err)
	}
	defer requireDestroy(b, "attention_mask tensor", attentionMaskTensor.Destroy)

	tokenTypeIDsTensor, err := NewTensor[int64](inputShape, tokenTypeIDs)
	if err != nil {
		b.Fatalf("failed to create token_type_ids tensor: %v", err)
	}
	defer requireDestroy(b, "token_type_ids tensor", tokenTypeIDsTensor.Destroy)

	outputTensor, err := NewEmptyTensor[float32](outputShape)
	if err != nil {
		b.Fatalf("failed to create output tensor: %v", err)
	}
	defer requireDestroy(b, "output tensor", outputTensor.Destroy)

	session, err := NewAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		[]Value{inputIDsTensor, attentionMaskTensor, tokenTypeIDsTensor},
		[]Value{outputTensor},
		nil,
	)
	if err != nil {
		b.Fatalf("failed to create all-MiniLM session: %v", err)
	}
	defer requireDestroy(b, "session", session.Destroy)

	expectedOutputSize := sequenceLength * int(allMiniLMOutputEmbeddingDim)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := session.Run(); err != nil {
			b.Fatalf("all-MiniLM inference failed: %v", err)
		}
		if got := len(outputTensor.GetData()); got != expectedOutputSize {
			b.Fatalf("unexpected output length: got %d, want %d", got, expectedOutputSize)
		}
	}
}

func BenchmarkAdvancedSessionCreateRunDestroyWithAllMiniLML6V2(b *testing.B) {
	cleanup := setupTestEnvironment(b)
	defer cleanup()

	modelPath := resolveAllMiniLMModelPath(b)
	sequenceLength := allMiniLMSequenceLength(b)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runAllMiniLMInferenceOnce(b, modelPath, sequenceLength)
	}
}
