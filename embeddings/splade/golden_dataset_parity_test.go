package splade

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

const goldenRowsRequestTimeout = 60 * time.Second

type goldenDatasetRow struct {
	ID      string    `json:"id"`
	Text    string    `json:"text"`
	Indices []int     `json:"indices"`
	Values  []float32 `json:"values"`
	Labels  []string  `json:"labels"`
}

func TestSPLADEGoldenDatasetParity(t *testing.T) {
	jsonlURL := strings.TrimSpace(os.Getenv("ONNXRUNTIME_TEST_SPLADE_GOLDEN_JSONL_URL"))
	if jsonlURL == "" {
		// Compatibility alias for older setups.
		jsonlURL = strings.TrimSpace(os.Getenv("ONNXRUNTIME_TEST_SPLADE_PRIVATE_GOLDEN_JSONL_URL"))
	}
	if jsonlURL == "" {
		repo := strings.TrimSpace(os.Getenv("HF_DATASET_REPO"))
		if repo == "" {
			t.Skip("set ONNXRUNTIME_TEST_SPLADE_GOLDEN_JSONL_URL or HF_DATASET_REPO to run SPLADE golden dataset parity test")
		}
		jsonlURL = fmt.Sprintf("https://huggingface.co/datasets/%s/resolve/main/splade_endpoint_golden/v1/splade_pp_en_v1_endpoint_topk24_labels_v1.jsonl", repo)
	}

	rows, err := downloadGoldenRows(jsonlURL, strings.TrimSpace(os.Getenv("HF_TOKEN")))
	if err != nil {
		t.Skipf("unable to load golden dataset from %s: %v", jsonlURL, err)
	}
	if len(rows) == 0 {
		t.Fatalf("golden dataset is empty")
	}

	cleanup := setupORTEnvironment(t)
	defer cleanup()

	modelPath, tokenizerPath := resolvePinnedSpladeAssets(t)
	hasLabels := false
	for i := range rows {
		if len(rows[i].Labels) > 0 {
			hasLabels = true
			break
		}
	}

	opts := []Option{
		WithInputOutputNames(
			spladeDefaultInputIDsName,
			spladeDefaultAttentionMaskName,
			spladeDefaultTokenTypeIDsName,
			spladeDefaultOutputName,
		),
		WithTokenLogitsOutput(),
		WithTopK(0), // Match endpoint dataset generation (no top-k cap).
		WithPruneThreshold(0),
		WithLog1pReLU(),
	}
	if hasLabels {
		opts = append(opts, WithReturnLabels())
	}

	embedder, err := NewEmbedder(modelPath, tokenizerPath, opts...)
	if err != nil {
		t.Fatalf("failed to create SPLADE embedder: %v", err)
	}
	defer func() {
		if err := embedder.Close(); err != nil {
			t.Errorf("failed to close SPLADE embedder: %v", err)
		}
	}()

	texts := make([]string, len(rows))
	for i := range rows {
		texts[i] = rows[i].Text
	}

	got, err := embedder.EmbedDocuments(texts)
	if err != nil {
		t.Fatalf("EmbedDocuments failed: %v", err)
	}
	if len(got) != len(rows) {
		t.Fatalf("row count mismatch: got %d want %d", len(got), len(rows))
	}

	tolerance, err := parseFloat32EnvFromKeys(
		[]string{
			"ONNXRUNTIME_TEST_SPLADE_GOLDEN_TOLERANCE",
			// Compatibility alias for older setups.
			"ONNXRUNTIME_TEST_SPLADE_PRIVATE_GOLDEN_TOLERANCE",
		},
		1e-4,
	)
	if err != nil {
		t.Fatalf("invalid SPLADE golden tolerance env var: %v", err)
	}
	for row := range rows {
		if len(got[row].Indices) != len(rows[row].Indices) {
			t.Fatalf("row %d index length mismatch: got %d want %d", row, len(got[row].Indices), len(rows[row].Indices))
		}
		if len(got[row].Values) != len(rows[row].Values) {
			t.Fatalf("row %d value length mismatch: got %d want %d", row, len(got[row].Values), len(rows[row].Values))
		}
		for i := range rows[row].Indices {
			if got[row].Indices[i] != rows[row].Indices[i] {
				t.Fatalf("row %d index[%d] mismatch: got %d want %d", row, i, got[row].Indices[i], rows[row].Indices[i])
			}
			diff := got[row].Values[i] - rows[row].Values[i]
			if diff < 0 {
				diff = -diff
			}
			if diff > tolerance {
				t.Fatalf("row %d value[%d] mismatch: got %.8f want %.8f tolerance %.8f", row, i, got[row].Values[i], rows[row].Values[i], tolerance)
			}
		}
		if hasLabels {
			if len(got[row].Labels) != len(rows[row].Labels) {
				t.Fatalf("row %d labels length mismatch: got %d want %d", row, len(got[row].Labels), len(rows[row].Labels))
			}
			for i := range rows[row].Labels {
				if got[row].Labels[i] != rows[row].Labels[i] {
					t.Fatalf("row %d label[%d] mismatch: got %q want %q", row, i, got[row].Labels[i], rows[row].Labels[i])
				}
			}
		}
	}
}

func downloadGoldenRows(url string, token string) ([]goldenDatasetRow, error) {
	request, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}

	client := &http.Client{Timeout: goldenRowsRequestTimeout}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = response.Body.Close()
	}()

	if response.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(response.Body, 512))
		return nil, fmt.Errorf("unexpected HTTP status %d: %s", response.StatusCode, string(snippet))
	}

	scanner := bufio.NewScanner(response.Body)
	rows := make([]goldenDatasetRow, 0, 64)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var row goldenDatasetRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("invalid jsonl row: %w", err)
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return rows, nil
}

func parseFloat32EnvFromKeys(keys []string, fallback float32) (float32, error) {
	for _, key := range keys {
		raw := strings.TrimSpace(os.Getenv(key))
		if raw == "" {
			continue
		}
		parsed, err := strconv.ParseFloat(raw, 32)
		if err != nil {
			return fallback, fmt.Errorf("%s=%q is not a valid float32: %w", key, raw, err)
		}
		return float32(parsed), nil
	}
	return fallback, nil
}
