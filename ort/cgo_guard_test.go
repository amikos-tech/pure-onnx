package ort

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestNoCgoImportInOrtPackage enforces the project's no-CGO contract for ort/.
func TestNoCgoImportInOrtPackage(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to determine current file path")
	}
	ortDir := filepath.Dir(thisFile)

	entries, err := os.ReadDir(ortDir)
	if err != nil {
		t.Fatalf("failed to read ort package directory: %v", err)
	}

	fset := token.NewFileSet()
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		if !strings.HasSuffix(name, ".go") {
			continue
		}

		fullPath := filepath.Join(ortDir, name)
		file, err := parser.ParseFile(fset, fullPath, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatalf("failed to parse %s: %v", name, err)
		}

		for _, imp := range file.Imports {
			if imp.Path != nil && imp.Path.Value == "\"C\"" {
				t.Fatalf("CGO import detected in %s: import \"C\" is forbidden", name)
			}
		}
	}
}
