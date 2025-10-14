// Package main generates the complete OrtApi struct from the C header
package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strings"
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

	fmt.Printf("// Parsed %d function pointers\n\n", len(functions))

	// Generate the Go struct
	generateGoStruct(functions)
}

type FunctionPointer struct {
	Name    string
	LineNum int
}

func generateGoStruct(functions []FunctionPointer) {
	fmt.Println("package ort")
	fmt.Println()
	fmt.Println("// OrtApi represents the ONNX Runtime C API function pointers")
	fmt.Println("// This struct is automatically generated from onnxruntime_c_api.h")
	fmt.Println("// DO NOT EDIT MANUALLY - regenerate using tools/gen_ortapi.go")
	fmt.Println("type OrtApi struct {")

	for i, fn := range functions {
		fmt.Printf("\t%-50s uintptr // Function %d\n", fn.Name, i+1)
	}

	fmt.Println("}")
}
