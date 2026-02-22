package splade

import (
	"math"
	"testing"
)

const (
	spladeGoldenTopK           = 16
	spladeGoldenValueTolerance = float32(1e-3)
)

type spladeGoldenRow struct {
	text    string
	indices []int
	values  []float32
	labels  []string
}

var spladeGoldenRows = []spladeGoldenRow{
	{
		text:    "this is a test",
		indices: []int{1037, 2003, 2023, 2122, 2832, 3231, 3267, 3800, 4106, 4638, 5604, 5852, 6210, 6845, 7709, 11360},
		values:  []float32{1.3415627, 0.64121246, 1.8329592, 0.7313759, 0.2665748, 2.9719844, 1.3294482, 0.3659603, 0.44352913, 0.7388122, 1.7083762, 1.6688603, 0.7622402, 2.0686655, 0.71458447, 0.95829326},
		labels:  []string{"a", "is", "this", "these", "process", "test", "nature", "purpose", "analysis", "check", "testing", "tests", "definition", "lab", "procedure", "exam"},
	},
	{
		text:    "hello world",
		indices: []int{2040, 2047, 2088, 2299, 2748, 3011, 3376, 3795, 4774, 5304, 5798, 6160, 7592, 7632, 8484},
		values:  []float32{0.42195213, 0.43201068, 2.7665792, 0.33145443, 1.3951724, 0.021566132, 0.43770817, 0.0008107936, 0.51871943, 0.17124648, 0.14741021, 0.81428343, 2.8032634, 2.1904366, 1.053133},
		labels:  []string{"who", "new", "world", "song", "yes", "earth", "beautiful", "global", "planet", "universe", "birthday", "welcome", "hello", "hi", "worlds"},
	},
	{
		text:    "neural search sparse retrieval",
		indices: []int{2424, 3638, 3945, 4167, 5371, 5527, 6575, 7809, 9742, 9896, 12850, 15698, 15756, 20288, 24961, 26384},
		values:  []float32{0.8252983, 0.37466505, 1.975485, 1.4173999, 0.39054248, 0.8664049, 1.1029733, 0.28875235, 1.4478122, 0.49653327, 1.6323423, 1.2257673, 2.3365622, 2.4321601, 0.4298108, 2.2844098},
		labels:  []string{"find", "memory", "search", "brain", "file", "storage", "searching", "database", "dense", "algorithm", "retrieve", "neurons", "neural", "sparse", "sparsely", "retrieval"},
	},
}

func TestSPLADEGoldenRegressionTopK16WithLabels(t *testing.T) {
	cleanup := setupORTEnvironment(t)
	defer cleanup()

	modelPath, tokenizerPath := resolvePinnedSpladeAssets(t)

	embedder, err := NewEmbedder(
		modelPath,
		tokenizerPath,
		WithInputOutputNames(
			spladeDefaultInputIDsName,
			spladeDefaultAttentionMaskName,
			spladeDefaultTokenTypeIDsName,
			spladeDefaultOutputName,
		),
		WithTokenLogitsOutput(),
		WithTopK(spladeGoldenTopK),
		WithPruneThreshold(0),
		WithLog1pReLU(),
		WithReturnLabels(),
	)
	if err != nil {
		t.Fatalf("failed to create SPLADE embedder: %v", err)
	}
	defer func() {
		if err := embedder.Close(); err != nil {
			t.Errorf("failed to close SPLADE embedder: %v", err)
		}
	}()

	texts := make([]string, len(spladeGoldenRows))
	for i := range spladeGoldenRows {
		texts[i] = spladeGoldenRows[i].text
	}

	got, err := embedder.EmbedDocuments(texts)
	if err != nil {
		t.Fatalf("EmbedDocuments failed: %v", err)
	}
	if len(got) != len(spladeGoldenRows) {
		t.Fatalf("unexpected row count: got %d, want %d", len(got), len(spladeGoldenRows))
	}

	for i := range spladeGoldenRows {
		assertSparseVectorGoldenRow(t, i, got[i], spladeGoldenRows[i])
	}
}

func TestSPLADERepeatabilityTopK16(t *testing.T) {
	cleanup := setupORTEnvironment(t)
	defer cleanup()

	modelPath, tokenizerPath := resolvePinnedSpladeAssets(t)

	embedder, err := NewEmbedder(
		modelPath,
		tokenizerPath,
		WithTopK(spladeGoldenTopK),
		WithPruneThreshold(0),
		WithLog1pReLU(),
		WithReturnLabels(),
	)
	if err != nil {
		t.Fatalf("failed to create SPLADE embedder: %v", err)
	}
	defer func() {
		if err := embedder.Close(); err != nil {
			t.Errorf("failed to close SPLADE embedder: %v", err)
		}
	}()

	texts := []string{"this is a test", "hello world"}
	baseline, err := embedder.EmbedDocuments(texts)
	if err != nil {
		t.Fatalf("baseline EmbedDocuments failed: %v", err)
	}

	const runs = 5
	for run := 0; run < runs; run++ {
		got, err := embedder.EmbedDocuments(texts)
		if err != nil {
			t.Fatalf("repeat run %d failed: %v", run, err)
		}
		if len(got) != len(baseline) {
			t.Fatalf("repeat run %d row count mismatch: got %d want %d", run, len(got), len(baseline))
		}
		for row := range baseline {
			if len(got[row].Indices) != len(baseline[row].Indices) {
				t.Fatalf("repeat run %d row %d index length mismatch: got %d want %d", run, row, len(got[row].Indices), len(baseline[row].Indices))
			}
			if len(got[row].Values) != len(baseline[row].Values) {
				t.Fatalf("repeat run %d row %d value length mismatch: got %d want %d", run, row, len(got[row].Values), len(baseline[row].Values))
			}
			if len(got[row].Labels) != len(baseline[row].Labels) {
				t.Fatalf("repeat run %d row %d label length mismatch: got %d want %d", run, row, len(got[row].Labels), len(baseline[row].Labels))
			}
			for i := range baseline[row].Indices {
				if got[row].Indices[i] != baseline[row].Indices[i] {
					t.Fatalf("repeat run %d row %d index[%d] mismatch: got %d want %d", run, row, i, got[row].Indices[i], baseline[row].Indices[i])
				}
				if math.Abs(float64(got[row].Values[i]-baseline[row].Values[i])) > float64(1e-6) {
					t.Fatalf("repeat run %d row %d value[%d] mismatch: got %.8f want %.8f", run, row, i, got[row].Values[i], baseline[row].Values[i])
				}
				if got[row].Labels[i] != baseline[row].Labels[i] {
					t.Fatalf("repeat run %d row %d label[%d] mismatch: got %q want %q", run, row, i, got[row].Labels[i], baseline[row].Labels[i])
				}
			}
		}
	}
}

func assertSparseVectorGoldenRow(t *testing.T, row int, got SparseVector, want spladeGoldenRow) {
	t.Helper()

	if len(got.Indices) != len(want.indices) {
		t.Fatalf("row %d index length mismatch: got %d want %d", row, len(got.Indices), len(want.indices))
	}
	if len(got.Values) != len(want.values) {
		t.Fatalf("row %d value length mismatch: got %d want %d", row, len(got.Values), len(want.values))
	}
	if len(got.Labels) != len(want.labels) {
		t.Fatalf("row %d label length mismatch: got %d want %d", row, len(got.Labels), len(want.labels))
	}

	for i := range want.indices {
		if got.Indices[i] != want.indices[i] {
			t.Fatalf("row %d index[%d] mismatch: got %d want %d", row, i, got.Indices[i], want.indices[i])
		}
		if math.Abs(float64(got.Values[i]-want.values[i])) > float64(spladeGoldenValueTolerance) {
			t.Fatalf("row %d value[%d] mismatch: got %.8f want %.8f tolerance %.8f", row, i, got.Values[i], want.values[i], spladeGoldenValueTolerance)
		}
		if got.Labels[i] != want.labels[i] {
			t.Fatalf("row %d label[%d] mismatch: got %q want %q", row, i, got.Labels[i], want.labels[i])
		}
	}
}
