package minilm

import (
	"math"
	"strings"
	"testing"
)

func TestMeanPoolAndNormalizeSingleMaskedToken(t *testing.T) {
	embeddings, err := meanPoolAndNormalize(
		[]float32{1, 2, 3, 4},
		[]int64{1, 0},
		1,
		2,
		2,
	)
	if err != nil {
		t.Fatalf("meanPoolAndNormalize failed: %v", err)
	}
	if len(embeddings) != 1 {
		t.Fatalf("expected 1 embedding row, got %d", len(embeddings))
	}
	if len(embeddings[0]) != 2 {
		t.Fatalf("expected embedding width 2, got %d", len(embeddings[0]))
	}

	expected := []float32{0.4472136, 0.8944272}
	for i := range expected {
		if !float32Near(embeddings[0][i], expected[i], 1e-6) {
			t.Fatalf("unexpected embedding[%d]: got %.7f, want %.7f", i, embeddings[0][i], expected[i])
		}
	}
}

func TestMeanPoolAndNormalizeZeroMask(t *testing.T) {
	embeddings, err := meanPoolAndNormalize(
		[]float32{10, 20, 30, 40},
		[]int64{0, 0},
		1,
		2,
		2,
	)
	if err != nil {
		t.Fatalf("meanPoolAndNormalize failed: %v", err)
	}
	for i, value := range embeddings[0] {
		if value != 0 {
			t.Fatalf("expected zero embedding value at %d, got %f", i, value)
		}
	}
}

func TestMeanPoolAndNormalizeValidation(t *testing.T) {
	_, err := meanPoolAndNormalize([]float32{1, 2}, []int64{1, 1}, 1, 2, 2)
	if err == nil || !strings.Contains(err.Error(), "last_hidden_state length mismatch") {
		t.Fatalf("expected last_hidden_state length mismatch error, got: %v", err)
	}

	_, err = meanPoolAndNormalize([]float32{1, 2, 3, 4}, []int64{1}, 1, 2, 2)
	if err == nil || !strings.Contains(err.Error(), "attention mask length mismatch") {
		t.Fatalf("expected attention mask length mismatch error, got: %v", err)
	}
}

func TestDeriveAttentionMask(t *testing.T) {
	dst := make([]int64, 4)
	deriveAttentionMask(dst, []int64{101, 2023, 0, 0})

	expected := []int64{1, 1, 0, 0}
	for i := range expected {
		if dst[i] != expected[i] {
			t.Fatalf("unexpected attention mask at %d: got %d, want %d", i, dst[i], expected[i])
		}
	}
}

func TestFillUint32AsInt64TruncatesToDestinationLength(t *testing.T) {
	dst := make([]int64, 3)
	fillUint32AsInt64(dst, []uint32{1, 2, 3, 4, 5})

	expected := []int64{1, 2, 3}
	for i := range expected {
		if dst[i] != expected[i] {
			t.Fatalf("unexpected dst[%d]: got %d, want %d", i, dst[i], expected[i])
		}
	}
}

func TestEmbedQueryValidation(t *testing.T) {
	var embedder *Embedder
	_, err := embedder.EmbedQuery("test")
	if err == nil || !strings.Contains(err.Error(), "embedder is nil") {
		t.Fatalf("expected nil embedder error, got: %v", err)
	}
}

func float32Near(got float32, want float32, tolerance float64) bool {
	return math.Abs(float64(got-want)) <= tolerance
}
