package splade

import (
	"os"
	"testing"

	"github.com/amikos-tech/pure-onnx/ort"
)

func setupORTEnvironment(tb testing.TB) func() {
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

func resolvePinnedSpladeAssets(tb testing.TB) (modelPath string, tokenizerPath string) {
	tb.Helper()

	modelPath = resolveAssetPath(
		tb,
		"ONNXRUNTIME_TEST_SPLADE_GOLDEN_MODEL_PATH",
		"ONNXRUNTIME_TEST_SPLADE_GOLDEN_MODEL_URL",
		"ONNXRUNTIME_TEST_SPLADE_GOLDEN_MODEL_SHA256",
		spladeModelURL,
		spladeModelSHA256,
		"models",
		"Splade_PP_en_v1.onnx",
	)

	tokenizerPath = resolveAssetPath(
		tb,
		"ONNXRUNTIME_TEST_SPLADE_GOLDEN_TOKENIZER_PATH",
		"ONNXRUNTIME_TEST_SPLADE_GOLDEN_TOKENIZER_URL",
		"ONNXRUNTIME_TEST_SPLADE_GOLDEN_TOKENIZER_SHA256",
		spladeTokenizerURL,
		spladeTokenizerSHA256,
		"tokenizers",
		"Splade_PP_en_v1-tokenizer.json",
	)

	return modelPath, tokenizerPath
}
