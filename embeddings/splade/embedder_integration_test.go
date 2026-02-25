package splade

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/amikos-tech/pure-onnx/ort"
)

const (
	spladeModelCommit = "762be6a7206e2f299182705972a65e5c46e62be2"

	spladeModelURL    = "https://huggingface.co/prithivida/Splade_PP_en_v1/resolve/" + spladeModelCommit + "/onnx/model.onnx"
	spladeModelSHA256 = "0934583a27a031a66b2e847cbc260fbbef29689e969f500436460ef5146a43f2"

	spladeTokenizerURL    = "https://huggingface.co/prithivida/Splade_PP_en_v1/resolve/" + spladeModelCommit + "/tokenizer.json"
	spladeTokenizerSHA256 = "2fc687b11de0bc1b3d8348f92e3b49ef1089a621506c7661fbf3248fcd54947e"

	spladeDefaultInputIDsName      = "input_ids"
	spladeDefaultAttentionMaskName = "input_mask"
	spladeDefaultTokenTypeIDsName  = "segment_ids"
	spladeDefaultOutputName        = "output"
)

func TestEmbedDocumentsWithSPLADEModel(t *testing.T) {
	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		t.Skip("ONNXRUNTIME_LIB_PATH not set, skipping integration test")
	}

	modelPath, tokenizerPath := resolveSpladeAssets(t)

	if err := ort.SetSharedLibraryPath(libPath); err != nil {
		t.Fatalf("failed to set ONNX Runtime library path: %v", err)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize ONNX Runtime: %v", err)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			t.Errorf("failed to destroy ONNX Runtime environment: %v", err)
		}
	}()

	inputIDsName := envOr("ONNXRUNTIME_TEST_SPLADE_INPUT_IDS_NAME", spladeDefaultInputIDsName)
	attentionMaskName := envOr("ONNXRUNTIME_TEST_SPLADE_ATTENTION_MASK_NAME", spladeDefaultAttentionMaskName)
	tokenTypeIDsName := envOr("ONNXRUNTIME_TEST_SPLADE_TOKEN_TYPE_IDS_NAME", spladeDefaultTokenTypeIDsName)
	if strings.TrimSpace(os.Getenv("ONNXRUNTIME_TEST_SPLADE_DISABLE_TOKEN_TYPE_IDS")) == "1" {
		tokenTypeIDsName = ""
	}
	outputName := envOr("ONNXRUNTIME_TEST_SPLADE_OUTPUT_NAME", spladeDefaultOutputName)

	opts := []Option{
		WithInputOutputNames(inputIDsName, attentionMaskName, tokenTypeIDsName, outputName),
		WithTopK(128),
		WithPruneThreshold(0),
		WithLog1pReLU(),
		WithReturnLabels(),
	}

	layout := strings.ToLower(strings.TrimSpace(os.Getenv("ONNXRUNTIME_TEST_SPLADE_OUTPUT_LAYOUT")))
	switch layout {
	case "", string(OutputLayoutTokenLogits):
		opts = append(opts, WithTokenLogitsOutput())
	case string(OutputLayoutDocumentLogits):
		opts = append(opts, WithDocumentLogitsOutput())
	default:
		t.Fatalf("unsupported ONNXRUNTIME_TEST_SPLADE_OUTPUT_LAYOUT: %q", layout)
	}

	if rawVocabSize := strings.TrimSpace(os.Getenv("ONNXRUNTIME_TEST_SPLADE_VOCAB_SIZE")); rawVocabSize != "" {
		vocabSize, err := strconv.Atoi(rawVocabSize)
		if err != nil {
			t.Fatalf("invalid ONNXRUNTIME_TEST_SPLADE_VOCAB_SIZE %q: %v", rawVocabSize, err)
		}
		opts = append(opts, WithVocabularySize(vocabSize))
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

	documents := []string{"this is a test", "hello world"}
	embeddings, err := embedder.EmbedDocuments(documents)
	if err != nil {
		t.Fatalf("EmbedDocuments failed: %v", err)
	}
	if len(embeddings) != len(documents) {
		t.Fatalf("unexpected embedding row count: got %d, want %d", len(embeddings), len(documents))
	}

	for rowIndex, embedding := range embeddings {
		if len(embedding.Indices) != len(embedding.Values) {
			t.Fatalf("row %d has mismatched sparse arrays: indices=%d values=%d", rowIndex, len(embedding.Indices), len(embedding.Values))
		}
		if len(embedding.Labels) != len(embedding.Indices) {
			t.Fatalf("row %d has mismatched sparse labels: labels=%d indices=%d", rowIndex, len(embedding.Labels), len(embedding.Indices))
		}
		if len(embedding.Indices) == 0 {
			continue
		}
		for i := range embedding.Indices {
			if i > 0 && embedding.Indices[i] <= embedding.Indices[i-1] {
				t.Fatalf("row %d sparse indices are not strictly increasing at position %d", rowIndex, i)
			}
			if embedding.Values[i] <= 0 {
				t.Fatalf("row %d contains non-positive sparse value at %d: %f", rowIndex, i, embedding.Values[i])
			}
		}
	}
}

func resolveSpladeAssets(t *testing.T) (modelPath string, tokenizerPath string) {
	t.Helper()

	modelPath = resolveAssetPath(
		t,
		"ONNXRUNTIME_TEST_SPLADE_MODEL_PATH",
		"ONNXRUNTIME_TEST_SPLADE_MODEL_URL",
		"ONNXRUNTIME_TEST_SPLADE_MODEL_SHA256",
		spladeModelURL,
		spladeModelSHA256,
		"models",
		"Splade_PP_en_v1.onnx",
	)

	tokenizerPath = resolveAssetPath(
		t,
		"ONNXRUNTIME_TEST_SPLADE_TOKENIZER_PATH",
		"ONNXRUNTIME_TEST_SPLADE_TOKENIZER_URL",
		"ONNXRUNTIME_TEST_SPLADE_TOKENIZER_SHA256",
		spladeTokenizerURL,
		spladeTokenizerSHA256,
		"tokenizers",
		"Splade_PP_en_v1-tokenizer.json",
	)

	return modelPath, tokenizerPath
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
	client := &http.Client{Timeout: 10 * time.Minute}
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

	file, err := os.CreateTemp(filepath.Dir(destinationPath), "splade-*.tmp")
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

func envOr(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}
