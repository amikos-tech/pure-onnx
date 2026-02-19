package ort

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

const (
	allMiniLMModelURL           = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
	allMiniLMOutputEmbeddingDim = int64(384)
)

var allMiniLMTemplateTokenIDs = []int64{101, 2023, 2003, 1037, 3231, 102}

func requireDestroy(tb testing.TB, name string, destroy func() error) {
	tb.Helper()
	if err := destroy(); err != nil {
		tb.Errorf("failed to destroy %s: %v", name, err)
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
	defer requireDestroy(tb, "input_ids tensor", inputIDsTensor.Destroy)

	attentionMaskTensor, err := NewTensor[int64](inputShape, attentionMask)
	if err != nil {
		tb.Fatalf("failed to create attention_mask tensor: %v", err)
	}
	defer requireDestroy(tb, "attention_mask tensor", attentionMaskTensor.Destroy)

	tokenTypeIDsTensor, err := NewTensor[int64](inputShape, tokenTypeIDs)
	if err != nil {
		tb.Fatalf("failed to create token_type_ids tensor: %v", err)
	}
	defer requireDestroy(tb, "token_type_ids tensor", tokenTypeIDsTensor.Destroy)

	outputTensor, err := NewEmptyTensor[float32](outputShape)
	if err != nil {
		tb.Fatalf("failed to create output tensor: %v", err)
	}
	defer requireDestroy(tb, "output tensor", outputTensor.Destroy)

	session, err := NewAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		[]Value{inputIDsTensor, attentionMaskTensor, tokenTypeIDsTensor},
		[]Value{outputTensor},
		nil,
	)
	if err != nil {
		tb.Fatalf("failed to create all-MiniLM session: %v", err)
	}
	defer requireDestroy(tb, "session", session.Destroy)

	if err := session.Run(); err != nil {
		tb.Fatalf("all-MiniLM inference failed: %v", err)
	}

	expectedOutputSize := sequenceLength * int(allMiniLMOutputEmbeddingDim)
	if got := len(outputTensor.GetData()); got != expectedOutputSize {
		tb.Fatalf("unexpected output length: got %d, want %d", got, expectedOutputSize)
	}
}

func allMiniLMSequenceLength(tb testing.TB) int {
	tb.Helper()

	raw := os.Getenv("ONNXRUNTIME_TEST_ALL_MINILM_SEQUENCE_LENGTH")
	if raw == "" {
		return 8
	}

	sequenceLength, err := strconv.Atoi(raw)
	if err != nil {
		tb.Fatalf("invalid ONNXRUNTIME_TEST_ALL_MINILM_SEQUENCE_LENGTH %q: %v", raw, err)
	}
	minTokens := len(allMiniLMTemplateTokenIDs)
	if sequenceLength < minTokens {
		tb.Fatalf("ONNXRUNTIME_TEST_ALL_MINILM_SEQUENCE_LENGTH must be >= %d, got %d", minTokens, sequenceLength)
	}

	return sequenceLength
}

func resolveAllMiniLMModelPath(tb testing.TB) string {
	tb.Helper()

	if modelPath := os.Getenv("ONNXRUNTIME_TEST_ALL_MINILM_MODEL_PATH"); modelPath != "" {
		if _, err := os.Stat(modelPath); err != nil {
			tb.Fatalf("ONNXRUNTIME_TEST_ALL_MINILM_MODEL_PATH %q is not usable: %v", modelPath, err)
		}
		return modelPath
	}

	cacheRoot := os.Getenv("ONNXRUNTIME_TEST_MODEL_CACHE_DIR")
	if cacheRoot == "" {
		userCacheDir, err := os.UserCacheDir()
		if err != nil {
			tb.Skipf("cannot determine user cache directory: %v; set ONNXRUNTIME_TEST_ALL_MINILM_MODEL_PATH", err)
		}
		cacheRoot = filepath.Join(userCacheDir, "onnx-purego", "models")
	}

	modelPath := filepath.Join(cacheRoot, "all-MiniLM-L6-v2.onnx")
	if info, err := os.Stat(modelPath); err == nil && info.Size() > 0 {
		return modelPath
	}

	if err := os.MkdirAll(filepath.Dir(modelPath), 0o755); err != nil {
		tb.Fatalf("failed to create model cache directory: %v", err)
	}

	url := os.Getenv("ONNXRUNTIME_TEST_ALL_MINILM_MODEL_URL")
	if url == "" {
		url = allMiniLMModelURL
	}

	// Concurrent test processes may race this download. The downloader handles that
	// by writing to unique temp files and treating an existing destination as success.
	tb.Logf("downloading all-MiniLM model from %s", url)
	if err := downloadModelFile(modelPath, url); err != nil {
		tb.Skipf("unable to download all-MiniLM model: %v", err)
	}

	return modelPath
}

func downloadModelFile(destinationPath string, modelURL string) error {
	const maxAttempts = 3

	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if attempt > 1 {
			time.Sleep(time.Duration(attempt) * time.Second)
		}

		err := downloadModelFileOnce(destinationPath, modelURL)
		if err == nil {
			return nil
		}
		lastErr = err

		if attempt < maxAttempts {
			_, _ = fmt.Fprintf(os.Stderr, "all-MiniLM download attempt %d/%d failed: %v\n", attempt, maxAttempts, err)
		}
	}

	return fmt.Errorf("failed to download model from %s after %d attempts: %w", modelURL, maxAttempts, lastErr)
}

func downloadModelFileOnce(destinationPath string, modelURL string) (err error) {
	client := &http.Client{Timeout: 3 * time.Minute}
	response, err := client.Get(modelURL)
	if err != nil {
		return err
	}
	defer func() {
		closeErr := response.Body.Close()
		if err == nil && closeErr != nil {
			err = closeErr
		}
	}()

	if response.StatusCode != http.StatusOK {
		requestURL := modelURL
		if response.Request != nil && response.Request.URL != nil {
			requestURL = response.Request.URL.String()
		}
		return fmt.Errorf("unexpected HTTP status %d from %s", response.StatusCode, requestURL)
	}

	file, err := os.CreateTemp(filepath.Dir(destinationPath), "all-minilm-*.tmp")
	if err != nil {
		return err
	}
	tempPath := file.Name()
	defer func() {
		_ = os.Remove(tempPath)
	}()

	if _, err := io.Copy(file, response.Body); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}

	if err := os.Rename(tempPath, destinationPath); err != nil {
		if info, statErr := os.Stat(destinationPath); statErr == nil && info.Size() > 0 {
			return nil
		}
		return err
	}

	return nil
}

func makeAllMiniLMInputs(sequenceLength int) ([]int64, []int64, []int64) {
	// [CLS] this is a test [SEP] plus zero-padding as needed.
	inputIDs := make([]int64, sequenceLength)
	attentionMask := make([]int64, sequenceLength)
	tokenTypeIDs := make([]int64, sequenceLength)

	nonPaddingCount := len(allMiniLMTemplateTokenIDs)
	if sequenceLength < nonPaddingCount {
		nonPaddingCount = sequenceLength
	}

	copy(inputIDs, allMiniLMTemplateTokenIDs[:nonPaddingCount])
	for i := 0; i < nonPaddingCount; i++ {
		attentionMask[i] = 1
	}

	return inputIDs, attentionMask, tokenTypeIDs
}

func formatMB(bytes int64) string {
	return fmt.Sprintf("%.2f MiB", float64(bytes)/1024.0/1024.0)
}
