package splade

import (
	"math"
	"strings"
	"testing"

	tokenizers "github.com/amikos-tech/pure-tokenizers"
)

func TestSparseFromOutputTokenLogitsTopK(t *testing.T) {
	output := []float32{
		0, 1, 2, -1,
		0.5, 3, 1, 4,
	}
	attentionMask := []int64{1, 1}

	embeddings, err := sparseFromOutput(
		output,
		attentionMask,
		1,
		2,
		4,
		OutputLayoutTokenLogits,
		1.0,
		2,
		true,
	)
	if err != nil {
		t.Fatalf("sparseFromOutput failed: %v", err)
	}
	if len(embeddings) != 1 {
		t.Fatalf("expected one embedding row, got %d", len(embeddings))
	}

	got := embeddings[0]
	wantIndices := []int{1, 3}
	wantValues := []float32{float32(math.Log1p(3)), float32(math.Log1p(4))}
	if len(got.Indices) != len(wantIndices) {
		t.Fatalf("unexpected sparse index count: got %d, want %d", len(got.Indices), len(wantIndices))
	}
	for i := range wantIndices {
		if got.Indices[i] != wantIndices[i] {
			t.Fatalf("unexpected index at %d: got %d, want %d", i, got.Indices[i], wantIndices[i])
		}
		if !float32Near(got.Values[i], wantValues[i], 1e-6) {
			t.Fatalf("unexpected value at %d: got %.7f, want %.7f", i, got.Values[i], wantValues[i])
		}
	}
}

func TestSparseFromOutputTokenLogitsRespectsAttentionMask(t *testing.T) {
	embeddings, err := sparseFromOutput(
		[]float32{
			0, 2, -1, // token 0
			50, 60, 70, // token 1 (masked out)
		},
		[]int64{1, 0},
		1,
		2,
		3,
		OutputLayoutTokenLogits,
		0,
		0,
		false,
	)
	if err != nil {
		t.Fatalf("sparseFromOutput failed: %v", err)
	}
	if len(embeddings) != 1 {
		t.Fatalf("expected one embedding row, got %d", len(embeddings))
	}
	assertIntSliceEqual(t, embeddings[0].Indices, []int{1})
	if len(embeddings[0].Values) != 1 || !float32Near(embeddings[0].Values[0], 2, 1e-6) {
		t.Fatalf("unexpected masked sparse values: %v", embeddings[0].Values)
	}
}

func TestSparseFromOutputTokenLogitsMultiBatch(t *testing.T) {
	embeddings, err := sparseFromOutput(
		[]float32{
			1, 0, 2, // row0 token0
			0, 3, 1, // row0 token1
			4, 5, 0, // row1 token0
			100, 100, 100, // row1 token1 (masked out)
		},
		[]int64{1, 1, 1, 0},
		2,
		2,
		3,
		OutputLayoutTokenLogits,
		0,
		0,
		false,
	)
	if err != nil {
		t.Fatalf("sparseFromOutput failed: %v", err)
	}
	if len(embeddings) != 2 {
		t.Fatalf("expected two embedding rows, got %d", len(embeddings))
	}
	assertIntSliceEqual(t, embeddings[0].Indices, []int{0, 1, 2})
	assertIntSliceEqual(t, embeddings[1].Indices, []int{0, 1})
	if !float32Near(embeddings[0].Values[0], 1, 1e-6) || !float32Near(embeddings[0].Values[1], 3, 1e-6) || !float32Near(embeddings[0].Values[2], 2, 1e-6) {
		t.Fatalf("unexpected row0 sparse values: %v", embeddings[0].Values)
	}
	if !float32Near(embeddings[1].Values[0], 4, 1e-6) || !float32Near(embeddings[1].Values[1], 5, 1e-6) {
		t.Fatalf("unexpected row1 sparse values: %v", embeddings[1].Values)
	}
}

func TestSparseFromOutputDocumentLogitsWithoutTransform(t *testing.T) {
	embeddings, err := sparseFromOutput(
		[]float32{0.1, 0.8, 0.5, -0.1},
		[]int64{1, 1},
		1,
		2,
		4,
		OutputLayoutDocumentLogits,
		0.4,
		0,
		false,
	)
	if err != nil {
		t.Fatalf("sparseFromOutput failed: %v", err)
	}
	if len(embeddings) != 1 {
		t.Fatalf("expected one embedding row, got %d", len(embeddings))
	}

	got := embeddings[0]
	wantIndices := []int{1, 2}
	wantValues := []float32{0.8, 0.5}
	if len(got.Indices) != len(wantIndices) {
		t.Fatalf("unexpected sparse index count: got %d, want %d", len(got.Indices), len(wantIndices))
	}
	for i := range wantIndices {
		if got.Indices[i] != wantIndices[i] {
			t.Fatalf("unexpected index at %d: got %d, want %d", i, got.Indices[i], wantIndices[i])
		}
		if !float32Near(got.Values[i], wantValues[i], 1e-6) {
			t.Fatalf("unexpected value at %d: got %.7f, want %.7f", i, got.Values[i], wantValues[i])
		}
	}
}

func TestSparseFromOutputDocumentLogitsWithTransform(t *testing.T) {
	embeddings, err := sparseFromOutput(
		[]float32{1, 0, -2, 3},
		[]int64{1, 1},
		1,
		2,
		4,
		OutputLayoutDocumentLogits,
		0,
		0,
		true,
	)
	if err != nil {
		t.Fatalf("sparseFromOutput failed: %v", err)
	}
	if len(embeddings) != 1 {
		t.Fatalf("expected one embedding row, got %d", len(embeddings))
	}

	got := embeddings[0]
	assertIntSliceEqual(t, got.Indices, []int{0, 3})
	if len(got.Values) != 2 {
		t.Fatalf("unexpected sparse value count: got %d, want 2", len(got.Values))
	}
	if !float32Near(got.Values[0], float32(math.Log1p(1)), 1e-6) || !float32Near(got.Values[1], float32(math.Log1p(3)), 1e-6) {
		t.Fatalf("unexpected transformed sparse values: %v", got.Values)
	}
}

func TestSparseFromOutputDocumentLogitsMultiBatch(t *testing.T) {
	embeddings, err := sparseFromOutput(
		[]float32{
			0.2, 0.9, -1, 0.5,
			1.1, 0.2, 0.7, 0,
		},
		[]int64{1, 1, 1, 1},
		2,
		2,
		4,
		OutputLayoutDocumentLogits,
		0.4,
		0,
		false,
	)
	if err != nil {
		t.Fatalf("sparseFromOutput failed: %v", err)
	}
	if len(embeddings) != 2 {
		t.Fatalf("expected two embedding rows, got %d", len(embeddings))
	}

	assertIntSliceEqual(t, embeddings[0].Indices, []int{1, 3})
	assertIntSliceEqual(t, embeddings[1].Indices, []int{0, 2})
	if !float32Near(embeddings[0].Values[0], 0.9, 1e-6) || !float32Near(embeddings[0].Values[1], 0.5, 1e-6) {
		t.Fatalf("unexpected row 0 sparse values: %v", embeddings[0].Values)
	}
	if !float32Near(embeddings[1].Values[0], 1.1, 1e-6) || !float32Near(embeddings[1].Values[1], 0.7, 1e-6) {
		t.Fatalf("unexpected row 1 sparse values: %v", embeddings[1].Values)
	}
}

func TestSparseFromOutputValidation(t *testing.T) {
	_, err := sparseFromOutput(
		[]float32{1, 2, 3, 4},
		[]int64{1},
		1,
		2,
		2,
		OutputLayoutTokenLogits,
		0,
		0,
		true,
	)
	if err == nil || !strings.Contains(err.Error(), "attention mask length mismatch") {
		t.Fatalf("expected attention mask length mismatch error, got: %v", err)
	}

	_, err = sparseFromOutput(
		[]float32{1, 2, 3},
		[]int64{1, 1},
		1,
		2,
		2,
		OutputLayoutDocumentLogits,
		0,
		0,
		true,
	)
	if err == nil || !strings.Contains(err.Error(), "document logits length mismatch") {
		t.Fatalf("expected document logits length mismatch error, got: %v", err)
	}

	_, err = sparseFromOutput(
		[]float32{1, 2, 3},
		[]int64{1, 1},
		1,
		2,
		2,
		OutputLayoutTokenLogits,
		0,
		0,
		true,
	)
	if err == nil || !strings.Contains(err.Error(), "token logits length mismatch") {
		t.Fatalf("expected token logits length mismatch error, got: %v", err)
	}

	_, err = sparseFromOutput(
		[]float32{1, 2, 3, 4},
		[]int64{1, 1},
		1,
		2,
		2,
		OutputLayout("unknown"),
		0,
		0,
		true,
	)
	if err == nil || !strings.Contains(err.Error(), "unsupported output layout") {
		t.Fatalf("expected unsupported output layout error, got: %v", err)
	}
}

func TestWithTopKValidation(t *testing.T) {
	cfg := defaultConfig()
	if err := WithTopK(-1)(&cfg); err == nil {
		t.Fatalf("expected validation error for negative topK")
	}
	if err := WithTopK(42)(&cfg); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
	if cfg.topK != 42 {
		t.Fatalf("unexpected topK: got %d, want 42", cfg.topK)
	}
}

func TestWithReturnLabelsOption(t *testing.T) {
	cfg := defaultConfig()
	if err := WithReturnLabels()(&cfg); err != nil {
		t.Fatalf("WithReturnLabels failed: %v", err)
	}
	if !cfg.returnLabels {
		t.Fatalf("expected returnLabels=true")
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

func TestWithPreProcessorValidation(t *testing.T) {
	cfg := defaultConfig()
	if err := WithPreProcessor(nil)(&cfg); err == nil {
		t.Fatalf("expected validation error for nil pre-processor")
	}

	trimSpace := strings.TrimSpace
	if err := WithPreProcessor(trimSpace)(&cfg); err != nil {
		t.Fatalf("unexpected WithPreProcessor error: %v", err)
	}
	if cfg.preProcessor == nil {
		t.Fatalf("expected preProcessor to be set")
	}
	if got := cfg.preProcessor("  x  "); got != "x" {
		t.Fatalf("unexpected preProcessor result: got %q, want %q", got, "x")
	}
}

func TestPreprocessDocuments(t *testing.T) {
	embedder := &Embedder{
		preProcessor: func(input string) string {
			return strings.ToUpper(strings.TrimSpace(input))
		},
	}
	documents := []string{"  hello", "world  "}

	processed, err := embedder.preprocessDocuments(documents)
	if err != nil {
		t.Fatalf("preprocessDocuments failed: %v", err)
	}
	if len(processed) != len(documents) {
		t.Fatalf("unexpected processed documents count: got %d, want %d", len(processed), len(documents))
	}
	if processed[0] != "HELLO" || processed[1] != "WORLD" {
		t.Fatalf("unexpected processed documents: got %v", processed)
	}
}

func TestPreprocessDocumentsWithPanickingPreProcessor(t *testing.T) {
	embedder := &Embedder{
		preProcessor: func(string) string {
			panic("boom")
		},
	}

	_, err := embedder.preprocessDocuments([]string{"hello"})
	if err == nil || !strings.Contains(err.Error(), "pre-processor panic on document 0: boom") {
		t.Fatalf("expected panic conversion error, got: %v", err)
	}
}

func TestSparseVectorValidate(t *testing.T) {
	tests := []struct {
		name    string
		vector  SparseVector
		wantErr string
	}{
		{
			name:    "valid without labels",
			vector:  SparseVector{Indices: []int{1, 2}, Values: []float32{0.1, 0.2}},
			wantErr: "",
		},
		{
			name:    "valid with labels",
			vector:  SparseVector{Indices: []int{1}, Values: []float32{0.1}, Labels: []string{"tok"}},
			wantErr: "",
		},
		{
			name:    "mismatched indices and values",
			vector:  SparseVector{Indices: []int{1, 2}, Values: []float32{0.1}},
			wantErr: "mismatched indices/values",
		},
		{
			name:    "mismatched labels and indices",
			vector:  SparseVector{Indices: []int{1, 2}, Values: []float32{0.1, 0.2}, Labels: []string{"tok"}},
			wantErr: "mismatched labels/indices",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.vector.Validate()
			if tc.wantErr == "" && err != nil {
				t.Fatalf("expected no validation error, got: %v", err)
			}
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected validation error containing %q, got: %v", tc.wantErr, err)
				}
			}
		})
	}
}

func TestEmbedQueryValidation(t *testing.T) {
	var embedder *Embedder
	_, err := embedder.EmbedQuery("test")
	if err == nil || !strings.Contains(err.Error(), "embedder is nil") {
		t.Fatalf("expected nil embedder error, got: %v", err)
	}
}

func TestSplitEncodingIntoWindows(t *testing.T) {
	encoding := &tokenizers.EncodeResult{
		IDs:           []uint32{101, 11, 12, 13, 14, 102},
		AttentionMask: []uint32{1, 1, 1, 1, 1, 1},
		TypeIDs:       []uint32{0, 0, 0, 0, 0, 0},
	}

	windows, err := splitEncodingIntoWindows(encoding, 4, 3, true)
	if err != nil {
		t.Fatalf("splitEncodingIntoWindows failed: %v", err)
	}
	if len(windows) != 2 {
		t.Fatalf("unexpected window count: got %d, want 2", len(windows))
	}

	assertInt64SliceEqual(t, windows[0].inputIDs, []int64{101, 11, 12, 13})
	assertInt64SliceEqual(t, windows[0].attentionMask, []int64{1, 1, 1, 1})
	assertInt64SliceEqual(t, windows[0].tokenTypeIDs, []int64{0, 0, 0, 0})

	assertInt64SliceEqual(t, windows[1].inputIDs, []int64{13, 14, 102, 0})
	assertInt64SliceEqual(t, windows[1].attentionMask, []int64{1, 1, 1, 0})
	assertInt64SliceEqual(t, windows[1].tokenTypeIDs, []int64{0, 0, 0, 0})
}

func TestSplitEncodingIntoWindowsValidation(t *testing.T) {
	encoding := &tokenizers.EncodeResult{
		IDs: []uint32{1, 2, 3},
	}
	if _, err := splitEncodingIntoWindows(encoding, 4, 0, false); err == nil {
		t.Fatalf("expected stride validation error")
	}
	if _, err := splitEncodingIntoWindows(encoding, 4, 5, false); err == nil {
		t.Fatalf("expected stride > sequence length validation error")
	}
}

func TestFillSessionFromWindowsValidation(t *testing.T) {
	oneTokenWindows := []tokenWindow{
		{inputIDs: []int64{1}, attentionMask: []int64{1}},
	}

	tests := []struct {
		name           string
		session        *embeddingSession
		windows        []tokenWindow
		sequenceLength int
		wantErr        string
	}{
		{
			name:           "nil session",
			session:        nil,
			windows:        oneTokenWindows,
			sequenceLength: 1,
			wantErr:        "embedding session is nil",
		},
		{
			name:           "non-positive sequence length",
			session:        &embeddingSession{inputIDs: make([]int64, 1), attentionMask: make([]int64, 1)},
			windows:        oneTokenWindows,
			sequenceLength: 0,
			wantErr:        "sequence length must be > 0",
		},
		{
			name:           "empty windows",
			session:        &embeddingSession{inputIDs: make([]int64, 1), attentionMask: make([]int64, 1)},
			windows:        nil,
			sequenceLength: 1,
			wantErr:        "window batch cannot be empty",
		},
		{
			name:           "session token buffer mismatch",
			session:        &embeddingSession{inputIDs: make([]int64, 2), attentionMask: make([]int64, 1)},
			windows:        oneTokenWindows,
			sequenceLength: 1,
			wantErr:        "session token buffer length mismatch",
		},
		{
			name:           "session token type ids buffer mismatch",
			session:        &embeddingSession{inputIDs: make([]int64, 1), attentionMask: make([]int64, 1), tokenTypeIDs: make([]int64, 2)},
			windows:        oneTokenWindows,
			sequenceLength: 1,
			wantErr:        "session token_type_ids buffer length mismatch",
		},
		{
			name:           "window sequence length mismatch",
			session:        &embeddingSession{inputIDs: make([]int64, 2), attentionMask: make([]int64, 2)},
			windows:        []tokenWindow{{inputIDs: []int64{1}, attentionMask: []int64{1, 1}}},
			sequenceLength: 2,
			wantErr:        "invalid sequence length",
		},
		{
			name:           "window token type ids mismatch",
			session:        &embeddingSession{inputIDs: make([]int64, 2), attentionMask: make([]int64, 2), tokenTypeIDs: make([]int64, 2)},
			windows:        []tokenWindow{{inputIDs: []int64{1, 2}, attentionMask: []int64{1, 1}, tokenTypeIDs: []int64{0}}},
			sequenceLength: 2,
			wantErr:        "invalid token_type_ids length",
		},
		{
			name:           "unexpected window token type ids",
			session:        &embeddingSession{inputIDs: make([]int64, 1), attentionMask: make([]int64, 1)},
			windows:        []tokenWindow{{inputIDs: []int64{1}, attentionMask: []int64{1}, tokenTypeIDs: []int64{0}}},
			sequenceLength: 1,
			wantErr:        "includes token_type_ids but session does not expect them",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := fillSessionFromWindows(tc.session, tc.windows, tc.sequenceLength)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
			}
		})
	}
}

func TestFillSessionFromWindowsCopiesBuffers(t *testing.T) {
	session := &embeddingSession{
		inputIDs:      make([]int64, 4),
		attentionMask: make([]int64, 4),
		tokenTypeIDs:  make([]int64, 4),
	}
	windows := []tokenWindow{
		{inputIDs: []int64{11, 12}, attentionMask: []int64{1, 1}, tokenTypeIDs: []int64{0, 0}},
		{inputIDs: []int64{21, 22}, attentionMask: []int64{1, 0}, tokenTypeIDs: []int64{0, 1}},
	}
	if err := fillSessionFromWindows(session, windows, 2); err != nil {
		t.Fatalf("fillSessionFromWindows failed: %v", err)
	}
	assertInt64SliceEqual(t, session.inputIDs, []int64{11, 12, 21, 22})
	assertInt64SliceEqual(t, session.attentionMask, []int64{1, 1, 1, 0})
	assertInt64SliceEqual(t, session.tokenTypeIDs, []int64{0, 0, 0, 1})
}

func TestMergeWindowEmbeddings(t *testing.T) {
	merged, err := mergeWindowEmbeddings(
		[]SparseVector{
			{Indices: []int{1, 5}, Values: []float32{0.2, 0.7}},
			{Indices: []int{1, 3}, Values: []float32{0.9, 0.6}},
			{Indices: []int{4}, Values: []float32{0.8}},
		},
		0.5,
		2,
	)
	if err != nil {
		t.Fatalf("mergeWindowEmbeddings failed: %v", err)
	}

	assertIntSliceEqual(t, merged.Indices, []int{1, 4})
	if len(merged.Values) != 2 {
		t.Fatalf("unexpected merged value count: got %d, want 2", len(merged.Values))
	}
	if !float32Near(merged.Values[0], 0.9, 1e-6) {
		t.Fatalf("unexpected merged value at 0: got %.8f want 0.9", merged.Values[0])
	}
	if !float32Near(merged.Values[1], 0.8, 1e-6) {
		t.Fatalf("unexpected merged value at 1: got %.8f want 0.8", merged.Values[1])
	}
}

func TestMergeWindowEmbeddingsRejectsMismatchedVectors(t *testing.T) {
	_, err := mergeWindowEmbeddings(
		[]SparseVector{
			{Indices: []int{1, 2}, Values: []float32{0.5}},
		},
		0,
		0,
	)
	if err == nil || !strings.Contains(err.Error(), "mismatched indices/values lengths") {
		t.Fatalf("expected mismatched indices/values error, got: %v", err)
	}
}

func assertInt64SliceEqual(t *testing.T, got []int64, want []int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("mismatch at %d: got %d want %d", i, got[i], want[i])
		}
	}
}

func assertIntSliceEqual(t *testing.T, got []int, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("mismatch at %d: got %d want %d", i, got[i], want[i])
		}
	}
}

func float32Near(got float32, want float32, tolerance float64) bool {
	return math.Abs(float64(got-want)) <= tolerance
}
