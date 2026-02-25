package splade

import "testing"

var spladeBenchmarkDocs = []string{
	"this is a test",
	"hello world",
	"neural search sparse retrieval",
	"dense and sparse hybrid ranking",
}

func BenchmarkSPLADEEmbedDocumentsWarmTopK128(b *testing.B) {
	cleanup := setupORTEnvironment(b)
	defer cleanup()

	modelPath, tokenizerPath := resolvePinnedSpladeAssets(b)
	embedder := mustNewBenchmarkEmbedder(b, modelPath, tokenizerPath, false)
	defer mustCloseBenchmarkEmbedder(b, embedder)

	warm, err := embedder.EmbedDocuments(spladeBenchmarkDocs)
	if err != nil {
		b.Fatalf("warm EmbedDocuments failed: %v", err)
	}
	if len(warm) != len(spladeBenchmarkDocs) {
		b.Fatalf("warm row count mismatch: got %d want %d", len(warm), len(spladeBenchmarkDocs))
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vectors, err := embedder.EmbedDocuments(spladeBenchmarkDocs)
		if err != nil {
			b.Fatalf("EmbedDocuments failed: %v", err)
		}
		if len(vectors) != len(spladeBenchmarkDocs) {
			b.Fatalf("row count mismatch: got %d want %d", len(vectors), len(spladeBenchmarkDocs))
		}
	}
}

func BenchmarkSPLADEEmbedDocumentsWarmTopK128WithLabels(b *testing.B) {
	cleanup := setupORTEnvironment(b)
	defer cleanup()

	modelPath, tokenizerPath := resolvePinnedSpladeAssets(b)
	embedder := mustNewBenchmarkEmbedder(b, modelPath, tokenizerPath, true)
	defer mustCloseBenchmarkEmbedder(b, embedder)

	warm, err := embedder.EmbedDocuments(spladeBenchmarkDocs)
	if err != nil {
		b.Fatalf("warm EmbedDocuments failed: %v", err)
	}
	if len(warm) != len(spladeBenchmarkDocs) {
		b.Fatalf("warm row count mismatch: got %d want %d", len(warm), len(spladeBenchmarkDocs))
	}
	for row := range warm {
		if len(warm[row].Labels) != len(warm[row].Indices) {
			b.Fatalf("warm row %d label/index mismatch: labels=%d indices=%d", row, len(warm[row].Labels), len(warm[row].Indices))
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vectors, err := embedder.EmbedDocuments(spladeBenchmarkDocs)
		if err != nil {
			b.Fatalf("EmbedDocuments failed: %v", err)
		}
		if len(vectors) != len(spladeBenchmarkDocs) {
			b.Fatalf("row count mismatch: got %d want %d", len(vectors), len(spladeBenchmarkDocs))
		}
		for row := range vectors {
			if len(vectors[row].Labels) != len(vectors[row].Indices) {
				b.Fatalf("row %d label/index mismatch: labels=%d indices=%d", row, len(vectors[row].Labels), len(vectors[row].Indices))
			}
		}
	}
}

func BenchmarkSPLADECreateRunDestroyTopK128(b *testing.B) {
	cleanup := setupORTEnvironment(b)
	defer cleanup()

	modelPath, tokenizerPath := resolvePinnedSpladeAssets(b)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		embedder := mustNewBenchmarkEmbedder(b, modelPath, tokenizerPath, false)
		vectors, err := embedder.EmbedDocuments(spladeBenchmarkDocs)
		if err != nil {
			b.Fatalf("EmbedDocuments failed: %v", err)
		}
		if len(vectors) != len(spladeBenchmarkDocs) {
			b.Fatalf("row count mismatch: got %d want %d", len(vectors), len(spladeBenchmarkDocs))
		}
		mustCloseBenchmarkEmbedder(b, embedder)
	}
}

func mustNewBenchmarkEmbedder(tb testing.TB, modelPath string, tokenizerPath string, withLabels bool) *Embedder {
	tb.Helper()

	opts := []Option{
		WithInputOutputNames(
			spladeDefaultInputIDsName,
			spladeDefaultAttentionMaskName,
			spladeDefaultTokenTypeIDsName,
			spladeDefaultOutputName,
		),
		WithTokenLogitsOutput(),
		WithTopK(128),
		WithPruneThreshold(0),
		WithLog1pReLU(),
	}
	if withLabels {
		opts = append(opts, WithReturnLabels())
	}

	embedder, err := NewEmbedder(modelPath, tokenizerPath, opts...)
	if err != nil {
		tb.Fatalf("failed to create SPLADE embedder: %v", err)
	}
	return embedder
}

func mustCloseBenchmarkEmbedder(tb testing.TB, embedder *Embedder) {
	tb.Helper()
	if err := embedder.Close(); err != nil {
		tb.Fatalf("failed to close SPLADE embedder: %v", err)
	}
}
