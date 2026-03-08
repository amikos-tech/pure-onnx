package main

import (
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParsePositiveIntEnv(t *testing.T) {
	const key = "OPENCLIP_EXAMPLE_LIMIT_TEST"

	t.Run("uses fallback when unset", func(t *testing.T) {
		t.Setenv(key, "")
		got, err := parsePositiveIntEnv(key, 30)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != 30 {
			t.Fatalf("fallback mismatch: got %d, want 30", got)
		}
	})

	t.Run("parses explicit positive value", func(t *testing.T) {
		t.Setenv(key, "7")
		got, err := parsePositiveIntEnv(key, 30)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != 7 {
			t.Fatalf("parsed mismatch: got %d, want 7", got)
		}
	})

	t.Run("rejects non integer", func(t *testing.T) {
		t.Setenv(key, "abc")
		_, err := parsePositiveIntEnv(key, 30)
		if err == nil {
			t.Fatal("expected error for non integer")
		}
		if !strings.Contains(err.Error(), key) {
			t.Fatalf("expected key in error, got: %v", err)
		}
	})

	t.Run("rejects zero", func(t *testing.T) {
		t.Setenv(key, "0")
		_, err := parsePositiveIntEnv(key, 30)
		if err == nil {
			t.Fatal("expected error for zero")
		}
	})

	t.Run("rejects negative", func(t *testing.T) {
		t.Setenv(key, "-3")
		_, err := parsePositiveIntEnv(key, 30)
		if err == nil {
			t.Fatal("expected error for negative value")
		}
	})
}

func TestValidateManifestRow(t *testing.T) {
	valid := manifestRow{
		ID:      "mnist-001",
		File:    "mnist-001.png",
		Dataset: "ylecun/mnist",
		Split:   "test",
		Index:   0,
		Prompt:  "a grayscale handwritten digit",
	}

	if err := validateManifestRow(valid); err != nil {
		t.Fatalf("valid row rejected: %v", err)
	}

	tests := []struct {
		name string
		row  manifestRow
	}{
		{name: "empty id", row: manifestRow{File: "x.png", Dataset: "d", Split: "s", Prompt: "p"}},
		{name: "empty file", row: manifestRow{ID: "x", Dataset: "d", Split: "s", Prompt: "p"}},
		{name: "empty dataset", row: manifestRow{ID: "x", File: "x.png", Split: "s", Prompt: "p"}},
		{name: "empty split", row: manifestRow{ID: "x", File: "x.png", Dataset: "d", Prompt: "p"}},
		{name: "absolute path", row: manifestRow{ID: "x", File: "/tmp/x.png", Dataset: "d", Split: "s", Prompt: "p"}},
		{name: "nested path", row: manifestRow{ID: "x", File: "nested/x.png", Dataset: "d", Split: "s", Prompt: "p"}},
		{name: "traversal", row: manifestRow{ID: "x", File: "../x.png", Dataset: "d", Split: "s", Prompt: "p"}},
		{name: "negative index", row: manifestRow{ID: "x", File: "x.png", Dataset: "d", Split: "s", Index: -1, Prompt: "p"}},
		{name: "empty prompt", row: manifestRow{ID: "x", File: "x.png", Dataset: "d", Split: "s", Index: 0}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := validateManifestRow(tc.row); err == nil {
				t.Fatalf("expected error for case %q", tc.name)
			}
		})
	}
}

func TestValidateLogitsMatrix(t *testing.T) {
	if err := validateLogitsMatrix([][]float32{{1, 2}, {3, 4}}, 2); err != nil {
		t.Fatalf("valid matrix rejected: %v", err)
	}

	if err := validateLogitsMatrix([][]float32{{1, 2}}, 2); err == nil {
		t.Fatal("expected row mismatch error")
	}

	if err := validateLogitsMatrix([][]float32{{1}, {2}}, 2); err == nil {
		t.Fatal("expected column mismatch error")
	}
}

func TestLoadExamplesFromManifest(t *testing.T) {
	dir := t.TempDir()
	imagePath := filepath.Join(dir, "sample.png")
	if err := writeTestPNG(imagePath); err != nil {
		t.Fatalf("failed to write fixture image: %v", err)
	}

	manifestRows := []manifestRow{
		{ID: "r1", File: "sample.png", Dataset: "d", Split: "s", Index: 0, Prompt: "p1"},
		{ID: "r2", File: "sample.png", Dataset: "d", Split: "s", Index: 1, Prompt: "p2"},
	}
	if err := writeManifest(filepath.Join(dir, manifestFileName), manifestRows); err != nil {
		t.Fatalf("failed to write manifest: %v", err)
	}

	rows, images, texts, err := loadExamplesFromManifest(dir, 1)
	if err != nil {
		t.Fatalf("unexpected load error: %v", err)
	}
	if len(rows) != 1 || len(images) != 1 || len(texts) != 1 {
		t.Fatalf("unexpected loaded sizes: rows=%d images=%d texts=%d", len(rows), len(images), len(texts))
	}
	if rows[0].ID != "r1" || texts[0] != "p1" {
		t.Fatalf("unexpected first row content: %+v text=%q", rows[0], texts[0])
	}

	t.Run("invalid json row", func(t *testing.T) {
		badDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(badDir, manifestFileName), []byte("{bad json}\n"), 0o600); err != nil {
			t.Fatalf("failed to write bad manifest: %v", err)
		}
		_, _, _, err := loadExamplesFromManifest(badDir, 1)
		if err == nil {
			t.Fatal("expected error for invalid manifest JSON")
		}
	})

	t.Run("missing image file", func(t *testing.T) {
		badDir := t.TempDir()
		badRows := []manifestRow{{ID: "missing", File: "missing.png", Dataset: "d", Split: "s", Index: 0, Prompt: "p"}}
		if err := writeManifest(filepath.Join(badDir, manifestFileName), badRows); err != nil {
			t.Fatalf("failed to write manifest: %v", err)
		}
		_, _, _, err := loadExamplesFromManifest(badDir, 1)
		if err == nil {
			t.Fatal("expected error for missing image")
		}
		if !strings.Contains(err.Error(), "missing.png") {
			t.Fatalf("expected missing filename in error, got: %v", err)
		}
	})
}

func writeManifest(path string, rows []manifestRow) (retErr error) {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := file.Close(); retErr == nil && closeErr != nil {
			retErr = closeErr
		}
	}()

	encoder := json.NewEncoder(file)
	for i := range rows {
		if err := encoder.Encode(rows[i]); err != nil {
			return err
		}
	}
	return nil
}

func writeTestPNG(path string) (retErr error) {
	img := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			img.Set(x, y, color.NRGBA{R: 255, G: 0, B: 0, A: 255})
		}
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := file.Close(); retErr == nil && closeErr != nil {
			retErr = closeErr
		}
	}()

	return png.Encode(file, img)
}
