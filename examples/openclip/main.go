package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/amikos-tech/pure-onnx/embeddings/openclip"
	"github.com/amikos-tech/pure-onnx/ort"
)

const (
	defaultAssetsDir      = "./assets"
	defaultExampleLimit   = 30
	defaultRankingTopK    = 3
	manifestFileName      = "manifest.jsonl"
	assetsDirEnvVar       = "OPENCLIP_EXAMPLE_ASSETS_DIR"
	exampleLimitEnvVar    = "OPENCLIP_EXAMPLE_LIMIT"
	onnxRuntimeLibPathEnv = "ONNXRUNTIME_LIB_PATH"
)

type manifestRow struct {
	ID      string `json:"id"`
	File    string `json:"file"`
	Dataset string `json:"dataset"`
	Split   string `json:"split"`
	Index   int    `json:"index"`
	Prompt  string `json:"prompt"`
}

type rankingScore struct {
	PromptIndex int
	Score       float32
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (retErr error) {
	assetsDir := strings.TrimSpace(os.Getenv(assetsDirEnvVar))
	if assetsDir == "" {
		assetsDir = defaultAssetsDir
	}

	limit, err := parsePositiveIntEnv(exampleLimitEnvVar, defaultExampleLimit)
	if err != nil {
		return err
	}

	rows, images, texts, err := loadExamplesFromManifest(assetsDir, limit)
	if err != nil {
		return err
	}
	if len(rows) == 0 {
		return fmt.Errorf("no fixtures loaded from %s", filepath.Join(assetsDir, manifestFileName))
	}

	if err := initializeOrtEnvironment(); err != nil {
		return fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}
	defer func() {
		if destroyErr := ort.DestroyEnvironment(); destroyErr != nil {
			retErr = errors.Join(retErr, fmt.Errorf("failed to destroy ONNX Runtime environment: %w", destroyErr))
		}
	}()

	modelAssets, err := openclip.EnsureDefaultAssets()
	if err != nil {
		return fmt.Errorf("failed to resolve default OpenCLIP model assets: %w", err)
	}

	embedder, err := openclip.NewEmbedder(
		modelAssets.TextModelPath,
		modelAssets.VisionModelPath,
		modelAssets.TokenizerPath,
		modelAssets.PreprocessorConfigPath,
	)
	if err != nil {
		return fmt.Errorf("failed to create OpenCLIP embedder: %w", err)
	}
	defer func() {
		if closeErr := embedder.Close(); closeErr != nil {
			retErr = errors.Join(retErr, fmt.Errorf("failed to close OpenCLIP embedder: %w", closeErr))
		}
	}()

	textEmbeddings, err := embedder.EmbedTexts(texts)
	if err != nil {
		return fmt.Errorf("EmbedTexts failed: %w", err)
	}

	imageEmbeddings, err := embedder.EmbedImages(images)
	if err != nil {
		return fmt.Errorf("EmbedImages failed: %w", err)
	}

	logits, err := openclip.CLIPSimilarityLogits(imageEmbeddings, textEmbeddings, openclip.DefaultCLIPLogitScale)
	if err != nil {
		return fmt.Errorf("CLIPSimilarityLogits failed: %w", err)
	}
	if len(logits) == 0 {
		return fmt.Errorf("CLIPSimilarityLogits returned an empty matrix for %d rows", len(rows))
	}
	if err := validateLogitsMatrix(logits, len(rows)); err != nil {
		return fmt.Errorf("CLIPSimilarityLogits returned an unexpected shape: %w", err)
	}

	fmt.Printf("Loaded %d fixture cases from %s\n", len(rows), assetsDir)
	fmt.Printf("Model assets:\n")
	fmt.Printf("  text model: %s\n", modelAssets.TextModelPath)
	fmt.Printf("  vision model: %s\n", modelAssets.VisionModelPath)
	fmt.Printf("  tokenizer: %s\n", modelAssets.TokenizerPath)
	fmt.Printf("  preprocessor: %s\n", modelAssets.PreprocessorConfigPath)
	fmt.Println()

	printSimilarityRanking(rows, logits, defaultRankingTopK)
	return retErr
}

func parsePositiveIntEnv(key string, fallback int) (int, error) {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("%s must be an integer, got %q", key, raw)
	}
	if value <= 0 {
		return 0, fmt.Errorf("%s must be > 0, got %d", key, value)
	}
	return value, nil
}

func loadExamplesFromManifest(assetsDir string, limit int) (rows []manifestRow, images []image.Image, texts []string, retErr error) {
	manifestPath := filepath.Join(assetsDir, manifestFileName)
	file, err := os.Open(manifestPath)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to open manifest %q: %w", manifestPath, err)
	}
	defer func() {
		if closeErr := file.Close(); retErr == nil && closeErr != nil {
			retErr = fmt.Errorf("failed to close manifest %q: %w", manifestPath, closeErr)
		}
	}()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	rows = make([]manifestRow, 0, limit)
	images = make([]image.Image, 0, limit)
	texts = make([]string, 0, limit)

	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		if len(rows) >= limit {
			break
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var row manifestRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, nil, nil, fmt.Errorf("invalid JSON in %s at line %d: %w", manifestPath, lineNumber, err)
		}
		if err := validateManifestRow(row); err != nil {
			return nil, nil, nil, fmt.Errorf("invalid row in %s at line %d: %w", manifestPath, lineNumber, err)
		}

		imagePath := filepath.Join(assetsDir, row.File)
		img, err := decodeImageFile(imagePath)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to load image %q: %w", imagePath, err)
		}

		rows = append(rows, row)
		images = append(images, img)
		texts = append(texts, row.Prompt)
	}
	if err := scanner.Err(); err != nil {
		return nil, nil, nil, fmt.Errorf("failed while reading %s: %w", manifestPath, err)
	}

	return rows, images, texts, nil
}

func validateManifestRow(row manifestRow) error {
	if strings.TrimSpace(row.ID) == "" {
		return fmt.Errorf("id is empty")
	}
	if strings.TrimSpace(row.File) == "" {
		return fmt.Errorf("file is empty")
	}
	if filepath.IsAbs(row.File) {
		return fmt.Errorf("file must be relative, got absolute path %q", row.File)
	}
	if filepath.Base(row.File) != row.File {
		return fmt.Errorf("file must not include directories, got %q", row.File)
	}
	if strings.Contains(row.File, "..") {
		return fmt.Errorf("file contains path traversal: %q", row.File)
	}
	if strings.TrimSpace(row.Dataset) == "" {
		return fmt.Errorf("dataset is empty")
	}
	if strings.TrimSpace(row.Split) == "" {
		return fmt.Errorf("split is empty")
	}
	if row.Index < 0 {
		return fmt.Errorf("index must be >= 0, got %d", row.Index)
	}
	if strings.TrimSpace(row.Prompt) == "" {
		return fmt.Errorf("prompt is empty")
	}
	return nil
}

func decodeImageFile(path string) (retImg image.Image, retErr error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %q: %w", path, err)
	}
	defer func() {
		if closeErr := file.Close(); retErr == nil && closeErr != nil {
			retErr = fmt.Errorf("close %q: %w", path, closeErr)
		}
	}()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("decode %q: %w", path, err)
	}
	return img, nil
}

func printSimilarityRanking(rows []manifestRow, logits [][]float32, topK int) {
	for imageIndex := range rows {
		caseRow := rows[imageIndex]
		scores := make([]rankingScore, 0, len(rows))
		for promptIndex := range rows {
			scores = append(scores, rankingScore{
				PromptIndex: promptIndex,
				Score:       logits[imageIndex][promptIndex],
			})
		}

		sort.Slice(scores, func(i, j int) bool {
			if scores[i].Score == scores[j].Score {
				return scores[i].PromptIndex < scores[j].PromptIndex
			}
			return scores[i].Score > scores[j].Score
		})

		k := topK
		if k > len(scores) {
			k = len(scores)
		}

		fmt.Printf("[%02d] image=%s dataset=%s split=%s index=%d file=%s\n",
			imageIndex+1,
			caseRow.ID,
			caseRow.Dataset,
			caseRow.Split,
			caseRow.Index,
			caseRow.File,
		)
		for rank := 0; rank < k; rank++ {
			match := scores[rank]
			promptRow := rows[match.PromptIndex]
			fmt.Printf("  %d. score=%8.4f prompt=%q (prompt_id=%s)\n", rank+1, match.Score, promptRow.Prompt, promptRow.ID)
		}
		fmt.Println()
	}
}

func validateLogitsMatrix(logits [][]float32, expectedRows int) error {
	if len(logits) != expectedRows {
		return fmt.Errorf("row count mismatch: got %d, want %d", len(logits), expectedRows)
	}
	for i := range logits {
		if len(logits[i]) != expectedRows {
			return fmt.Errorf("row %d column count mismatch: got %d, want %d", i, len(logits[i]), expectedRows)
		}
	}
	return nil
}

func initializeOrtEnvironment() error {
	libPath := strings.TrimSpace(os.Getenv(onnxRuntimeLibPathEnv))
	if libPath != "" {
		if err := ort.SetSharedLibraryPath(libPath); err != nil {
			return fmt.Errorf("failed to set explicit ONNX Runtime library path %q: %w", libPath, err)
		}
		if err := ort.InitializeEnvironment(); err != nil {
			return fmt.Errorf("failed to initialize ONNX Runtime with explicit path %q: %w", libPath, err)
		}
		return nil
	}

	bootstrappedPath, err := ort.EnsureOnnxRuntimeSharedLibrary()
	if err != nil {
		return fmt.Errorf("failed to bootstrap ONNX Runtime shared library: %w", err)
	}
	log.Printf("%s not set; using bootstrapped library at %s", onnxRuntimeLibPathEnv, bootstrappedPath)

	if err := ort.SetSharedLibraryPath(bootstrappedPath); err != nil {
		return fmt.Errorf("failed to set bootstrapped ONNX Runtime library path %q: %w", bootstrappedPath, err)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("failed to initialize ONNX Runtime with bootstrapped path %q: %w", bootstrappedPath, err)
	}
	return nil
}
