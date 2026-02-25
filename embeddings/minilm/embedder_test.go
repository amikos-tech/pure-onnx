package minilm

import (
	"math"
	"strings"
	"testing"

	"github.com/amikos-tech/pure-onnx/embeddings/internal/ortutil"
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

func TestWithMaxCachedBatchSessionsValidation(t *testing.T) {
	cfg := defaultConfig()
	if err := WithMaxCachedBatchSessions(0)(&cfg); err == nil {
		t.Fatalf("expected validation error for zero max cached sessions")
	}
	if err := WithMaxCachedBatchSessions(2)(&cfg); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
	if cfg.maxCachedBatchCount != 2 {
		t.Fatalf("unexpected maxCachedBatchCount: got %d, want 2", cfg.maxCachedBatchCount)
	}
}

func TestWithEmbeddingDimensionValidation(t *testing.T) {
	cfg := defaultConfig()
	if err := WithEmbeddingDimension(0)(&cfg); err == nil {
		t.Fatalf("expected validation error for zero embedding dimension")
	}
	if err := WithEmbeddingDimension(768)(&cfg); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
	if cfg.embeddingDimension != 768 {
		t.Fatalf("unexpected embeddingDimension: got %d, want 768", cfg.embeddingDimension)
	}
}

func TestPoolingOptions(t *testing.T) {
	cfg := defaultConfig()

	if err := WithNoPooling()(&cfg); err != nil {
		t.Fatalf("WithNoPooling failed: %v", err)
	}
	if cfg.poolingStrategy != PoolingStrategyNone {
		t.Fatalf("unexpected pooling strategy: got %q, want %q", cfg.poolingStrategy, PoolingStrategyNone)
	}

	if err := WithCLSPooling()(&cfg); err != nil {
		t.Fatalf("WithCLSPooling failed: %v", err)
	}
	if cfg.poolingStrategy != PoolingStrategyCLS {
		t.Fatalf("unexpected pooling strategy: got %q, want %q", cfg.poolingStrategy, PoolingStrategyCLS)
	}

	if err := WithMeanPooling()(&cfg); err != nil {
		t.Fatalf("WithMeanPooling failed: %v", err)
	}
	if cfg.poolingStrategy != PoolingStrategyMean {
		t.Fatalf("unexpected pooling strategy: got %q, want %q", cfg.poolingStrategy, PoolingStrategyMean)
	}
}

func TestNormalizationOptions(t *testing.T) {
	cfg := defaultConfig()

	if err := WithoutL2Normalization()(&cfg); err != nil {
		t.Fatalf("WithoutL2Normalization failed: %v", err)
	}
	if cfg.l2Normalize {
		t.Fatalf("expected l2Normalize=false after WithoutL2Normalization")
	}

	if err := WithL2Normalization()(&cfg); err != nil {
		t.Fatalf("WithL2Normalization failed: %v", err)
	}
	if !cfg.l2Normalize {
		t.Fatalf("expected l2Normalize=true after WithL2Normalization")
	}
}

func TestWithoutTokenTypeIDsInput(t *testing.T) {
	cfg := defaultConfig()
	if err := WithoutTokenTypeIDsInput()(&cfg); err != nil {
		t.Fatalf("WithoutTokenTypeIDsInput failed: %v", err)
	}
	if cfg.useTokenTypeIDs {
		t.Fatalf("expected useTokenTypeIDs=false")
	}
	if cfg.tokenTypeIDsName != "" {
		t.Fatalf("expected tokenTypeIDsName to be empty, got %q", cfg.tokenTypeIDsName)
	}
}

func TestWithInputOutputNamesAllowsEmptyTokenTypeIDs(t *testing.T) {
	cfg := defaultConfig()
	if err := WithInputOutputNames("ids", "mask", "", "output")(&cfg); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
	if cfg.useTokenTypeIDs {
		t.Fatalf("expected useTokenTypeIDs=false when tokenTypeIDsName is empty")
	}
}

func TestPostProcessDenseOutputCLSPooling(t *testing.T) {
	embeddings, err := postProcessDenseOutput(
		[]float32{1, 2, 3, 4, 5, 6},
		[]int64{1, 1, 1},
		1,
		3,
		2,
		PoolingStrategyCLS,
		false,
	)
	if err != nil {
		t.Fatalf("postProcessDenseOutput failed: %v", err)
	}
	want := []float32{1, 2}
	assertVectorNearLocal(t, "CLS pooling", embeddings[0], want, 1e-6)
}

func TestPostProcessDenseOutputNoPooling(t *testing.T) {
	embeddings, err := postProcessDenseOutput(
		[]float32{1, 2, 3, 4},
		[]int64{1, 1},
		1,
		2,
		2,
		PoolingStrategyNone,
		false,
	)
	if err != nil {
		t.Fatalf("postProcessDenseOutput failed: %v", err)
	}
	want := []float32{1, 2, 3, 4}
	assertVectorNearLocal(t, "No pooling", embeddings[0], want, 1e-6)
}

func TestPostProcessDenseOutputCLSPoolingBatchTwo(t *testing.T) {
	embeddings, err := postProcessDenseOutput(
		[]float32{
			1, 2, 3, 4, 5, 6, // row 0 tokens
			7, 8, 9, 10, 11, 12, // row 1 tokens
		},
		[]int64{1, 1, 1, 1, 1, 1},
		2,
		3,
		2,
		PoolingStrategyCLS,
		false,
	)
	if err != nil {
		t.Fatalf("postProcessDenseOutput failed: %v", err)
	}
	if len(embeddings) != 2 {
		t.Fatalf("expected 2 embedding rows, got %d", len(embeddings))
	}
	assertVectorNearLocal(t, "CLS pooling row 0", embeddings[0], []float32{1, 2}, 1e-6)
	assertVectorNearLocal(t, "CLS pooling row 1", embeddings[1], []float32{7, 8}, 1e-6)
}

func TestPostProcessDenseOutputNoPoolingBatchTwo(t *testing.T) {
	embeddings, err := postProcessDenseOutput(
		[]float32{
			1, 2, 3, 4, // row 0 tokens
			5, 6, 7, 8, // row 1 tokens
		},
		[]int64{1, 1, 1, 1},
		2,
		2,
		2,
		PoolingStrategyNone,
		false,
	)
	if err != nil {
		t.Fatalf("postProcessDenseOutput failed: %v", err)
	}
	if len(embeddings) != 2 {
		t.Fatalf("expected 2 embedding rows, got %d", len(embeddings))
	}
	assertVectorNearLocal(t, "No pooling row 0", embeddings[0], []float32{1, 2, 3, 4}, 1e-6)
	assertVectorNearLocal(t, "No pooling row 1", embeddings[1], []float32{5, 6, 7, 8}, 1e-6)
}

func TestPostProcessDenseOutputCLSPoolingWithL2(t *testing.T) {
	embeddings, err := postProcessDenseOutput(
		[]float32{
			3, 4, 10, 20, // row 0 tokens
			5, 12, 1, 2, // row 1 tokens
		},
		[]int64{1, 1, 1, 1},
		2,
		2,
		2,
		PoolingStrategyCLS,
		true,
	)
	if err != nil {
		t.Fatalf("postProcessDenseOutput failed: %v", err)
	}
	assertVectorNearLocal(t, "CLS + L2 row 0", embeddings[0], []float32{0.6, 0.8}, 1e-6)
	assertVectorNearLocal(t, "CLS + L2 row 1", embeddings[1], []float32{5.0 / 13.0, 12.0 / 13.0}, 1e-6)
}

func TestPostProcessDenseOutputNoPoolingWithL2(t *testing.T) {
	embeddings, err := postProcessDenseOutput(
		[]float32{3, 4, 0, 0},
		[]int64{1, 1},
		1,
		2,
		2,
		PoolingStrategyNone,
		true,
	)
	if err != nil {
		t.Fatalf("postProcessDenseOutput failed: %v", err)
	}
	assertVectorNearLocal(t, "No pooling + L2", embeddings[0], []float32{0.6, 0.8, 0, 0}, 1e-6)
}

func TestPostProcessDenseOutputCLSPoolingWithL2ZeroVector(t *testing.T) {
	embeddings, err := postProcessDenseOutput(
		[]float32{0, 0, 5, 6},
		[]int64{1, 1},
		1,
		2,
		2,
		PoolingStrategyCLS,
		true,
	)
	if err != nil {
		t.Fatalf("postProcessDenseOutput failed: %v", err)
	}
	assertVectorNearLocal(t, "CLS + L2 zero vector", embeddings[0], []float32{0, 0}, 1e-6)
}

func TestPostProcessDenseOutputNoPoolingWithL2ZeroVector(t *testing.T) {
	embeddings, err := postProcessDenseOutput(
		[]float32{0, 0, 0, 0},
		[]int64{1, 1},
		1,
		2,
		2,
		PoolingStrategyNone,
		true,
	)
	if err != nil {
		t.Fatalf("postProcessDenseOutput failed: %v", err)
	}
	assertVectorNearLocal(t, "No pooling + L2 zero vector", embeddings[0], []float32{0, 0, 0, 0}, 1e-6)
}

func TestPostProcessDenseOutputInvalidPooling(t *testing.T) {
	_, err := postProcessDenseOutput(
		[]float32{1, 2, 3, 4},
		[]int64{1, 1},
		1,
		2,
		2,
		PoolingStrategy("invalid"),
		false,
	)
	if err == nil || !strings.Contains(err.Error(), "unsupported pooling strategy") {
		t.Fatalf("expected unsupported pooling strategy error, got: %v", err)
	}
}

type nilSafeDestroyer struct{}

func (d *nilSafeDestroyer) Destroy() error {
	return nil
}

func TestDestroyAllIgnoresTypedNil(t *testing.T) {
	var typedNil *nilSafeDestroyer
	if err := ortutil.DestroyAll(typedNil); err != nil {
		t.Fatalf("destroyAll should ignore typed nil destroyers, got: %v", err)
	}
}

func assertVectorNearLocal(tb testing.TB, label string, got []float32, want []float32, tolerance float64) {
	tb.Helper()
	if len(got) != len(want) {
		tb.Fatalf("%s length mismatch: got %d want %d", label, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tolerance {
			tb.Fatalf("%s mismatch at %d: got %.8f want %.8f", label, i, got[i], want[i])
		}
	}
}

func float32Near(got float32, want float32, tolerance float64) bool {
	return math.Abs(float64(got-want)) <= tolerance
}
