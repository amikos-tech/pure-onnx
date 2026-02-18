# Inference Example

This example runs a real model inference using `ort.NewAdvancedSession` without CGO.

## Required environment variables

- `ONNXRUNTIME_LIB_PATH`: path to ONNX Runtime shared library (`.so`, `.dylib`, or `.dll`)
- `ONNX_MODEL_PATH`: path to your `.onnx` model file
- `ONNX_INPUT_SHAPE`: comma-separated input tensor shape (example: `1,384`)
- `ONNX_OUTPUT_SHAPE`: comma-separated output tensor shape (example: `1,384`)

## Optional environment variables

- `ONNX_INPUT_NAME` (default: `input`)
- `ONNX_OUTPUT_NAME` (default: `output`)
- `ONNX_INPUT_DATA`: comma-separated `float32` values. If omitted, the example generates `1..N`.

## Run

```bash
export ONNXRUNTIME_LIB_PATH="/path/to/libonnxruntime.so"
export ONNX_MODEL_PATH="/path/to/model.onnx"
export ONNX_INPUT_NAME="input"
export ONNX_OUTPUT_NAME="output"
export ONNX_INPUT_SHAPE="1,384"
export ONNX_OUTPUT_SHAPE="1,384"

go run ./examples/inference
```

The output prints a short preview of the first values from the output tensor.
