package ort

import (
	"fmt"
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
	ortDir, err := resolveOrtPackageDir()
	if err != nil {
		t.Fatal(err)
	}

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

func resolveOrtPackageDir() (string, error) {
	candidates := make([]string, 0, 4)

	if wd, err := os.Getwd(); err == nil && wd != "" {
		candidates = append(candidates, wd, filepath.Join(wd, "ort"))
	}

	if _, thisFile, _, ok := runtime.Caller(0); ok {
		callerDir := filepath.Dir(thisFile)
		candidates = append(candidates, callerDir)
	}

	for _, dir := range candidates {
		if dir == "" {
			continue
		}
		if isOrtPackageDir(dir) {
			return dir, nil
		}
	}

	return "", fmt.Errorf("failed to locate ort package directory; checked: %v", candidates)
}

func isOrtPackageDir(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
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
		file, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, parser.PackageClauseOnly)
		if err != nil {
			continue
		}
		return file.Name != nil && file.Name.Name == "ort"
	}

	return false
}
