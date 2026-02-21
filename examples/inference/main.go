package main

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/amikos-tech/pure-onnx/ort"
)

func main() {
	modelPath := os.Getenv("ONNX_MODEL_PATH")
	if modelPath == "" {
		log.Fatal("set ONNX_MODEL_PATH to your .onnx file")
	}

	inputName := envOr("ONNX_INPUT_NAME", "input")
	outputName := envOr("ONNX_OUTPUT_NAME", "output")

	inputShape, err := parseShapeEnv("ONNX_INPUT_SHAPE")
	if err != nil {
		log.Fatal(err)
	}
	outputShape, err := parseShapeEnv("ONNX_OUTPUT_SHAPE")
	if err != nil {
		log.Fatal(err)
	}

	inputElementCount, err := ort.ShapeElementCount(inputShape)
	if err != nil {
		log.Fatalf("invalid input shape: %v", err)
	}

	inputData, err := parseInputData(os.Getenv("ONNX_INPUT_DATA"), inputElementCount)
	if err != nil {
		log.Fatalf("invalid ONNX_INPUT_DATA: %v", err)
	}

	if err := initializeOrtEnvironment(); err != nil {
		log.Fatalf("failed to initialize ONNX Runtime: %v", err)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			log.Printf("failed to destroy environment: %v", err)
		}
	}()

	inputTensor, err := ort.NewTensor[float32](inputShape, inputData)
	if err != nil {
		log.Fatalf("failed to create input tensor: %v", err)
	}
	defer func() {
		if err := inputTensor.Destroy(); err != nil {
			log.Printf("failed to destroy input tensor: %v", err)
		}
	}()

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Fatalf("failed to create output tensor: %v", err)
	}
	defer func() {
		if err := outputTensor.Destroy(); err != nil {
			log.Printf("failed to destroy output tensor: %v", err)
		}
	}()

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{inputName},
		[]string{outputName},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		log.Fatalf("failed to create session: %v", err)
	}
	defer func() {
		if err := session.Destroy(); err != nil {
			log.Printf("failed to destroy session: %v", err)
		}
	}()

	if err := session.Run(); err != nil {
		log.Fatalf("inference failed: %v", err)
	}

	output := outputTensor.GetData()
	fmt.Printf("inference completed: output shape=%v output elements=%d\n", outputShape, len(output))
	printPreview(output, 16)
}

func envOr(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

func parseShapeEnv(key string) (ort.Shape, error) {
	raw := os.Getenv(key)
	if raw == "" {
		return nil, fmt.Errorf("set %s (example: \"1,384\")", key)
	}

	shape, err := ort.ParseShape(raw)
	if err != nil {
		return nil, fmt.Errorf("%s is invalid: %w", key, err)
	}
	return shape, nil
}

func parseInputData(raw string, expected int) ([]float32, error) {
	if raw == "" {
		data := make([]float32, expected)
		for i := range data {
			data[i] = float32(i + 1)
		}
		return data, nil
	}

	parts := strings.Split(raw, ",")
	if len(parts) != expected {
		return nil, fmt.Errorf("expected %d elements, got %d", expected, len(parts))
	}

	data := make([]float32, expected)
	for i, part := range parts {
		part = strings.TrimSpace(part)
		value, err := strconv.ParseFloat(part, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid float at index %d (%q): %w", i, part, err)
		}
		data[i] = float32(value)
	}

	return data, nil
}

func printPreview(values []float32, max int) {
	if len(values) == 0 {
		fmt.Println("output preview: []")
		return
	}

	end := len(values)
	if end > max {
		end = max
	}

	fmt.Printf("output preview (%d/%d): %v\n", end, len(values), values[:end])
}

func initializeOrtEnvironment() error {
	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath != "" {
		if err := ort.SetSharedLibraryPath(libPath); err != nil {
			return fmt.Errorf("failed to set explicit ONNX Runtime library path: %w", err)
		}
		return ort.InitializeEnvironment()
	}

	bootstrappedPath, err := ort.EnsureOnnxRuntimeSharedLibrary()
	if err != nil {
		return fmt.Errorf("failed to bootstrap ONNX Runtime shared library: %w", err)
	}

	log.Printf("ONNXRUNTIME_LIB_PATH not set; using bootstrapped library at %s", bootstrappedPath)
	if err := ort.SetSharedLibraryPath(bootstrappedPath); err != nil {
		return fmt.Errorf("failed to set bootstrapped ONNX Runtime library path: %w", err)
	}

	return ort.InitializeEnvironment()
}
