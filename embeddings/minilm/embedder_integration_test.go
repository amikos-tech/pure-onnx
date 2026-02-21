package minilm

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/amikos-tech/pure-onnx/ort"
)

const (
	allMiniLMModelURL    = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
	allMiniLMModelSHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"

	allMiniLMTokenizerURL    = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
	allMiniLMTokenizerSHA256 = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
)

var expectedThisIsATestEmbeddingPrefix = []float32{
	0.03061239,
	0.01383142,
	-0.02084381,
	0.01632802,
	-0.01023151,
	-0.04798428,
	-0.01731338,
	0.03728743,
}

func resolveMiniLMAssets(t *testing.T) (modelPath string, tokenizerPath string) {
	t.Helper()

	modelPath = resolveAssetPath(
		t,
		"ONNXRUNTIME_TEST_ALL_MINILM_MODEL_PATH",
		"ONNXRUNTIME_TEST_ALL_MINILM_MODEL_URL",
		"ONNXRUNTIME_TEST_ALL_MINILM_MODEL_SHA256",
		allMiniLMModelURL,
		allMiniLMModelSHA256,
		"models",
		"all-MiniLM-L6-v2.onnx",
	)

	tokenizerPath = resolveAssetPath(
		t,
		"ONNXRUNTIME_TEST_ALL_MINILM_TOKENIZER_PATH",
		"ONNXRUNTIME_TEST_ALL_MINILM_TOKENIZER_URL",
		"ONNXRUNTIME_TEST_ALL_MINILM_TOKENIZER_SHA256",
		allMiniLMTokenizerURL,
		allMiniLMTokenizerSHA256,
		"tokenizers",
		"all-MiniLM-L6-v2-tokenizer.json",
	)

	return modelPath, tokenizerPath
}

func TestEmbedDocumentsWithAllMiniLML6V2(t *testing.T) {
	cleanup := setupORTTestEnvironment(t)
	defer cleanup()

	modelPath, tokenizerPath := resolveMiniLMAssets(t)

	embedder, err := NewEmbedder(modelPath, tokenizerPath)
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	defer func() {
		if err := embedder.Close(); err != nil {
			t.Errorf("failed to close embedder: %v", err)
		}
	}()

	documents := []string{"this is a test", "hello world"}
	embeddings, err := embedder.EmbedDocuments(documents)
	if err != nil {
		t.Fatalf("EmbedDocuments failed: %v", err)
	}
	if len(embeddings) != len(documents) {
		t.Fatalf("unexpected embedding row count: got %d, want %d", len(embeddings), len(documents))
	}
	if len(embedder.sessionsByBatch) != 1 {
		t.Fatalf("expected exactly one cached session after first batch run, got %d", len(embedder.sessionsByBatch))
	}
	batchTwoSession := embedder.sessionsByBatch[len(documents)]
	if batchTwoSession == nil {
		t.Fatalf("missing cached session for batch size %d", len(documents))
	}

	for i, embedding := range embeddings {
		if len(embedding) != int(OutputEmbeddingDimension) {
			t.Fatalf("unexpected embedding width at row %d: got %d, want %d", i, len(embedding), OutputEmbeddingDimension)
		}
		assertFiniteVector(t, fmt.Sprintf("embedding row %d", i), embedding)
		assertApproxUnitNorm(t, fmt.Sprintf("embedding row %d", i), embedding, 1e-4)
	}

	queryEmbedding, err := embedder.EmbedQuery(documents[0])
	if err != nil {
		t.Fatalf("EmbedQuery failed: %v", err)
	}
	if len(queryEmbedding) != int(OutputEmbeddingDimension) {
		t.Fatalf("unexpected query embedding width: got %d, want %d", len(queryEmbedding), OutputEmbeddingDimension)
	}
	if len(embedder.sessionsByBatch) != 2 {
		t.Fatalf("expected two cached sessions after single-query run, got %d", len(embedder.sessionsByBatch))
	}
	batchOneSession := embedder.sessionsByBatch[1]
	if batchOneSession == nil {
		t.Fatalf("missing cached session for batch size 1")
	}
	assertPrefixNear(
		t,
		"EmbedQuery golden prefix (this is a test)",
		queryEmbedding,
		expectedThisIsATestEmbeddingPrefix,
		1e-4,
	)

	singleDocEmbeddings, err := embedder.EmbedDocuments([]string{documents[0]})
	if err != nil {
		t.Fatalf("single-doc EmbedDocuments failed: %v", err)
	}
	if len(singleDocEmbeddings) != 1 {
		t.Fatalf("unexpected single-doc row count: got %d, want 1", len(singleDocEmbeddings))
	}
	if len(embedder.sessionsByBatch) != 2 {
		t.Fatalf("expected session cache size to remain 2 after repeated single-doc call, got %d", len(embedder.sessionsByBatch))
	}
	if embedder.sessionsByBatch[1] != batchOneSession {
		t.Fatalf("expected batch size 1 session to be reused")
	}
	if embedder.sessionsByBatch[len(documents)] != batchTwoSession {
		t.Fatalf("expected batch size %d session to remain cached", len(documents))
	}

	assertVectorNear(t, "EmbedQuery parity", queryEmbedding, singleDocEmbeddings[0], 1e-6)
}

func TestEmbedderSessionCacheRespectsLRUBound(t *testing.T) {
	cleanup := setupORTTestEnvironment(t)
	defer cleanup()

	modelPath, tokenizerPath := resolveMiniLMAssets(t)

	embedder, err := NewEmbedder(modelPath, tokenizerPath, WithMaxCachedBatchSessions(2))
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	defer func() {
		if err := embedder.Close(); err != nil {
			t.Errorf("failed to close embedder: %v", err)
		}
	}()

	if _, err := embedder.EmbedDocuments([]string{"doc-1"}); err != nil {
		t.Fatalf("first batch run failed: %v", err)
	}
	if _, err := embedder.EmbedDocuments([]string{"doc-1", "doc-2"}); err != nil {
		t.Fatalf("second batch run failed: %v", err)
	}
	if len(embedder.sessionsByBatch) != 2 {
		t.Fatalf("expected two cached sessions after warm-up, got %d", len(embedder.sessionsByBatch))
	}
	sessionBatchOne := embedder.sessionsByBatch[1]
	if sessionBatchOne == nil {
		t.Fatalf("missing cached session for batch size 1")
	}

	// Touch batch size 1 so batch size 2 becomes least recently used.
	if _, err := embedder.EmbedDocuments([]string{"doc-3"}); err != nil {
		t.Fatalf("third batch run failed: %v", err)
	}

	// Insert batch size 3 and verify LRU eviction removed batch size 2.
	if _, err := embedder.EmbedDocuments([]string{"doc-4", "doc-5", "doc-6"}); err != nil {
		t.Fatalf("fourth batch run failed: %v", err)
	}
	if len(embedder.sessionsByBatch) != 2 {
		t.Fatalf("expected cache size to remain 2 after eviction, got %d", len(embedder.sessionsByBatch))
	}
	if embedder.sessionsByBatch[1] != sessionBatchOne {
		t.Fatalf("expected batch size 1 session to remain cached as recently used")
	}
	if _, ok := embedder.sessionsByBatch[2]; ok {
		t.Fatalf("expected batch size 2 session to be evicted")
	}
	if embedder.sessionsByBatch[3] == nil {
		t.Fatalf("expected batch size 3 session to be cached")
	}
}

func setupORTTestEnvironment(tb testing.TB) func() {
	tb.Helper()

	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		tb.Skip("ONNXRUNTIME_LIB_PATH not set, skipping integration test")
	}

	if err := ort.SetSharedLibraryPath(libPath); err != nil {
		tb.Fatalf("failed to set ONNX Runtime library path: %v", err)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		tb.Fatalf("failed to initialize ONNX Runtime: %v", err)
	}

	return func() {
		if err := ort.DestroyEnvironment(); err != nil {
			tb.Errorf("failed to destroy ONNX Runtime environment: %v", err)
		}
	}
}

func resolveAssetPath(tb testing.TB, envPathKey string, envURLKey string, envSHAKey string, defaultURL string, defaultSHA string, cacheSubdir string, cacheFilename string) string {
	tb.Helper()

	if path := os.Getenv(envPathKey); path != "" {
		if _, err := os.Stat(path); err != nil {
			tb.Fatalf("%s %q is not usable: %v", envPathKey, path, err)
		}
		if expectedSHA := strings.TrimSpace(os.Getenv(envSHAKey)); expectedSHA != "" {
			if err := verifyFileSHA256(path, expectedSHA); err != nil {
				tb.Fatalf("%s failed checksum validation: %v", envPathKey, err)
			}
		}
		return path
	}

	cacheRoot := os.Getenv("ONNXRUNTIME_TEST_MODEL_CACHE_DIR")
	if cacheRoot == "" {
		userCacheDir, err := os.UserCacheDir()
		if err != nil {
			tb.Skipf("cannot determine user cache directory: %v; set %s", err, envPathKey)
		}
		cacheRoot = filepath.Join(userCacheDir, "onnx-purego")
	}

	assetPath := filepath.Join(cacheRoot, cacheSubdir, cacheFilename)
	if err := os.MkdirAll(filepath.Dir(assetPath), 0o755); err != nil {
		tb.Fatalf("failed to create cache directory: %v", err)
	}

	url := os.Getenv(envURLKey)
	if url == "" {
		url = defaultURL
	}
	expectedSHA := strings.TrimSpace(os.Getenv(envSHAKey))
	if expectedSHA == "" && url == defaultURL {
		expectedSHA = defaultSHA
	}
	expectedSHA = strings.ToLower(expectedSHA)

	if info, err := os.Stat(assetPath); err == nil && info.Size() > 0 {
		if expectedSHA == "" {
			return assetPath
		}
		if err := verifyFileSHA256(assetPath, expectedSHA); err == nil {
			return assetPath
		}
		if removeErr := os.Remove(assetPath); removeErr != nil {
			tb.Fatalf("failed to remove stale cached asset: %v", removeErr)
		}
	}

	tb.Logf("downloading test asset from %s", url)
	if err := downloadFile(assetPath, url); err != nil {
		tb.Skipf("unable to download test asset from %s: %v", url, err)
	}
	if expectedSHA != "" {
		if err := verifyFileSHA256(assetPath, expectedSHA); err != nil {
			_ = os.Remove(assetPath)
			tb.Fatalf("downloaded asset failed checksum validation: %v", err)
		}
	}

	return assetPath
}

func downloadFile(destinationPath string, assetURL string) error {
	const maxAttempts = 3

	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if attempt > 1 {
			backoff := time.Second * time.Duration(1<<(attempt-1))
			if backoff > 8*time.Second {
				backoff = 8 * time.Second
			}
			time.Sleep(backoff)
		}

		err := downloadFileOnce(destinationPath, assetURL)
		if err == nil {
			return nil
		}
		lastErr = err
	}

	return fmt.Errorf("failed to download %s after %d attempts: %w", assetURL, maxAttempts, lastErr)
}

func downloadFileOnce(destinationPath string, assetURL string) (err error) {
	client := &http.Client{Timeout: 3 * time.Minute}
	response, err := client.Get(assetURL)
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
		requestURL := assetURL
		if response.Request != nil && response.Request.URL != nil {
			requestURL = response.Request.URL.String()
		}
		return fmt.Errorf("unexpected HTTP status %d from %s", response.StatusCode, requestURL)
	}

	file, err := os.CreateTemp(filepath.Dir(destinationPath), "minilm-*.tmp")
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

func assertFiniteVector(tb testing.TB, label string, values []float32) {
	tb.Helper()
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			tb.Fatalf("%s contains non-finite value at %d: %v", label, i, value)
		}
	}
}

func assertApproxUnitNorm(tb testing.TB, label string, values []float32, tolerance float64) {
	tb.Helper()
	normSquared := 0.0
	for _, value := range values {
		normSquared += float64(value * value)
	}
	norm := math.Sqrt(normSquared)
	if math.Abs(norm-1.0) > tolerance {
		tb.Fatalf("%s norm not close to 1: got %.8f", label, norm)
	}
}

func assertPrefixNear(tb testing.TB, label string, got []float32, prefix []float32, tolerance float64) {
	tb.Helper()
	if len(got) < len(prefix) {
		tb.Fatalf("%s length mismatch: got %d want at least %d", label, len(got), len(prefix))
	}
	for i := range prefix {
		if math.Abs(float64(got[i]-prefix[i])) > tolerance {
			tb.Fatalf("%s mismatch at %d: got %.8f want %.8f", label, i, got[i], prefix[i])
		}
	}
}

func assertVectorNear(tb testing.TB, label string, got []float32, want []float32, tolerance float64) {
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
