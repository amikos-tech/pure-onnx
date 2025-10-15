// Package main generates the complete OrtApi struct from the C header
//
// NOTE: This generator uses simple regex-based parsing which works for the current
// ONNX Runtime C API but may be fragile with future header changes. In a future PR,
// we should consider using a proper C parser like tree-sitter-c for more robust
// parsing and potentially auto-generating purego function bindings.
//
// See: https://github.com/tree-sitter/tree-sitter-c
package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <path-to-onnxruntime_c_api.h>\n", os.Args[0])
		os.Exit(1)
	}

	headerPath := os.Args[1]
	file, err := os.Open(headerPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open header file: %v\n", err)
		os.Exit(1)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	// Find the start of OrtApi struct
	inStruct := false
	lineNum := 0
	structLineNum := 0
	var functions []FunctionPointer

	// Regex patterns
	ortApiPattern := regexp.MustCompile(`^struct OrtApi \{`)
	ortApi2StatusPattern := regexp.MustCompile(`ORT_API2_STATUS\((\w+),`)
	functionPtrPattern := regexp.MustCompile(`^\s+(OrtStatus|OrtErrorCode|const char|void)\s*\(\s*ORT_API_CALL\s*\*\s*(\w+)\)`)
	// Also match patterns like "OrtStatus*(ORT_API_CALL* CreateStatus)"
	functionPtrPattern2 := regexp.MustCompile(`^\s+(OrtStatus|OrtErrorCode|const char)\s*\*\s*\(\s*ORT_API_CALL\s*\*\s*(\w+)\)`)
	ortClassReleasePattern := regexp.MustCompile(`ORT_CLASS_RELEASE\((\w+)\)`)
	endStructPattern := regexp.MustCompile(`^\s*\};`)

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()

		if !inStruct {
			if ortApiPattern.MatchString(line) {
				inStruct = true
				structLineNum = lineNum
				fmt.Printf("// Found OrtApi struct at line %d\n", lineNum)
			}
			continue
		}

		// Check if we've reached the end of the struct
		if endStructPattern.MatchString(line) {
			break
		}

		// Skip comments and empty lines
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "/*") || strings.HasPrefix(trimmed, "*") || strings.HasPrefix(trimmed, "///") {
			continue
		}

		var funcName string

		// Check for ORT_API2_STATUS macro
		if matches := ortApi2StatusPattern.FindStringSubmatch(line); len(matches) > 1 {
			funcName = matches[1]
		} else if matches := functionPtrPattern.FindStringSubmatch(line); len(matches) > 2 {
			funcName = matches[2]
		} else if matches := functionPtrPattern2.FindStringSubmatch(line); len(matches) > 2 {
			funcName = matches[2]
		} else if matches := ortClassReleasePattern.FindStringSubmatch(line); len(matches) > 1 {
			funcName = "Release" + matches[1]
		}

		if funcName != "" {
			functions = append(functions, FunctionPointer{
				Name:    funcName,
				LineNum: lineNum,
			})
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
		os.Exit(1)
	}

	// Validate function count
	if len(functions) < 290 || len(functions) > 320 {
		fmt.Fprintf(os.Stderr, "Warning: Parsed %d functions, expected ~305 (valid range: 290-320). Header may have changed.\n", len(functions))
	}

	// Check for duplicate function names
	seen := make(map[string]bool)
	for _, fn := range functions {
		if seen[fn.Name] {
			fmt.Fprintf(os.Stderr, "Error: Duplicate function name: %s\n", fn.Name)
			os.Exit(1)
		}
		seen[fn.Name] = true
	}

	// Validate key function positions to catch parser bugs
	keyFunctions := map[string]int{
		"CreateEnv":                      4,
		"CreateTensorWithDataAsOrtValue": 50,
		"CreateMemoryInfo":               69,
		"ReleaseEnv":                     93,
	}

	for name, expectedPos := range keyFunctions {
		found := false
		for i, fn := range functions {
			if fn.Name == name {
				actualPos := i + 1 // 1-indexed
				if actualPos != expectedPos {
					fmt.Fprintf(os.Stderr, "Error: Key function '%s' found at position %d, expected %d. Parser may be broken.\n", name, actualPos, expectedPos)
					os.Exit(1)
				}
				found = true
				break
			}
		}
		if !found {
			fmt.Fprintf(os.Stderr, "Error: Key function '%s' not found. Parser may be broken.\n", name)
			os.Exit(1)
		}
	}

	fmt.Printf("// Parsed %d function pointers\n\n", len(functions))

	// Generate the Go struct
	generateGoStruct(functions, headerPath, structLineNum)
}

type FunctionPointer struct {
	Name    string
	LineNum int
}

func generateGoStruct(functions []FunctionPointer, headerPath string, structLineNum int) {
	fmt.Println("package ort")
	fmt.Println()
	fmt.Printf("// Auto-generated from: %s\n", headerPath)
	fmt.Printf("// Generated on: %s\n", time.Now().Format(time.RFC3339))
	fmt.Println("// Generator: tools/gen_ortapi.go")
	fmt.Printf("// Found OrtApi struct at line %d\n", structLineNum)
	fmt.Printf("// Parsed %d function pointers\n", len(functions))
	fmt.Println("//")
	fmt.Println("// OrtApi represents the ONNX Runtime C API function pointers")
	fmt.Println("// DO NOT EDIT MANUALLY - regenerate using tools/gen_ortapi.go")
	fmt.Println("type OrtApi struct {")

	for i, fn := range functions {
		fmt.Printf("\t%-50s uintptr // Function %d\n", fn.Name, i+1)
	}

	fmt.Println("}")
}
