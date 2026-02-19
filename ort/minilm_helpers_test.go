package ort

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

const (
	allMiniLMModelURL           = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
	allMiniLMModelSHA256        = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
	allMiniLMOutputEmbeddingDim = int64(384)
	allMiniLMTemplateTokenCount = 6
)

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

func runAllMiniLMInference(tb testing.TB, modelPath string, sequenceLength int) []float32 {
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
	output := outputTensor.GetData()
	if got := len(output); got != expectedOutputSize {
		tb.Fatalf("unexpected output length: got %d, want %d", got, expectedOutputSize)
	}

	return output
}

func runAllMiniLMInferenceOnce(tb testing.TB, modelPath string, sequenceLength int) {
	tb.Helper()
	_ = runAllMiniLMInference(tb, modelPath, sequenceLength)
}

func requireFiniteFloat32Slice(tb testing.TB, label string, values []float32) {
	tb.Helper()
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			tb.Fatalf("%s contains non-finite value at index %d: %v", label, i, value)
		}
	}
}

func allMiniLMSequenceLength(tb testing.TB) int {
	tb.Helper()

	return envIntOrDefault(tb, "ONNXRUNTIME_TEST_ALL_MINILM_SEQUENCE_LENGTH", 8, allMiniLMTemplateTokenCount)
}

func resolveAllMiniLMModelPath(tb testing.TB) string {
	tb.Helper()

	if modelPath := os.Getenv("ONNXRUNTIME_TEST_ALL_MINILM_MODEL_PATH"); modelPath != "" {
		if _, err := os.Stat(modelPath); err != nil {
			tb.Fatalf("ONNXRUNTIME_TEST_ALL_MINILM_MODEL_PATH %q is not usable: %v", modelPath, err)
		}
		if expectedSHA := strings.TrimSpace(os.Getenv("ONNXRUNTIME_TEST_ALL_MINILM_MODEL_SHA256")); expectedSHA != "" {
			if err := verifyFileSHA256(modelPath, expectedSHA); err != nil {
				tb.Fatalf("ONNXRUNTIME_TEST_ALL_MINILM_MODEL_PATH failed checksum validation: %v", err)
			}
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

	if err := os.MkdirAll(filepath.Dir(modelPath), 0o755); err != nil {
		tb.Fatalf("failed to create model cache directory: %v", err)
	}

	url := os.Getenv("ONNXRUNTIME_TEST_ALL_MINILM_MODEL_URL")
	if url == "" {
		url = allMiniLMModelURL
	}
	expectedSHA := allMiniLMExpectedSHA256(url)

	if info, err := os.Stat(modelPath); err == nil && info.Size() > 0 {
		if expectedSHA == "" {
			return modelPath
		}
		if err := verifyFileSHA256(modelPath, expectedSHA); err == nil {
			return modelPath
		}

		tb.Logf("cached all-MiniLM checksum mismatch, re-downloading model")
		if removeErr := os.Remove(modelPath); removeErr != nil {
			tb.Fatalf("failed to remove stale cached model: %v", removeErr)
		}
	}

	// Concurrent test processes may race this download. The downloader handles that
	// by writing to unique temp files and treating an existing destination as success.
	tb.Logf("downloading all-MiniLM model from %s", url)
	if err := downloadModelFile(modelPath, url); err != nil {
		tb.Skipf("unable to download all-MiniLM model: %v", err)
	}
	if expectedSHA != "" {
		if err := verifyFileSHA256(modelPath, expectedSHA); err != nil {
			_ = os.Remove(modelPath)
			tb.Fatalf("downloaded all-MiniLM model failed checksum validation: %v", err)
		}
	}

	return modelPath
}

func allMiniLMExpectedSHA256(modelURL string) string {
	if expectedSHA := strings.TrimSpace(os.Getenv("ONNXRUNTIME_TEST_ALL_MINILM_MODEL_SHA256")); expectedSHA != "" {
		return strings.ToLower(expectedSHA)
	}
	if modelURL == allMiniLMModelURL {
		return allMiniLMModelSHA256
	}
	return ""
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

func verifyFileSHA256(path string, expected string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() {
		_ = file.Close()
	}()

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return err
	}

	actual := hex.EncodeToString(hash.Sum(nil))
	expected = strings.ToLower(strings.TrimSpace(expected))
	if actual != expected {
		return fmt.Errorf("sha256 mismatch for %s: got %s want %s", path, actual, expected)
	}

	return nil
}

func makeAllMiniLMInputs(sequenceLength int) ([]int64, []int64, []int64) {
	// [CLS] this is a test [SEP] plus zero-padding as needed.
	templateTokenIDs := [allMiniLMTemplateTokenCount]int64{101, 2023, 2003, 1037, 3231, 102}

	inputIDs := make([]int64, sequenceLength)
	attentionMask := make([]int64, sequenceLength)
	tokenTypeIDs := make([]int64, sequenceLength)

	nonPaddingCount := len(templateTokenIDs)
	if sequenceLength < nonPaddingCount {
		nonPaddingCount = sequenceLength
	}

	copy(inputIDs, templateTokenIDs[:nonPaddingCount])
	for i := 0; i < nonPaddingCount; i++ {
		attentionMask[i] = 1
	}

	return inputIDs, attentionMask, tokenTypeIDs
}

// Used by heap-growth assertions in session_leak_test.go.
func formatMB(bytes int64) string {
	return fmt.Sprintf("%.2f MiB", float64(bytes)/1024.0/1024.0)
}
