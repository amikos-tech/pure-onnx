# Inference Example

This example runs a real model inference using `ort.NewAdvancedSession` without CGO.

## Required environment variables

- `ONNX_MODEL_PATH`: path to your `.onnx` model file
- `ONNX_INPUT_SHAPE`: comma-separated input tensor shape (example: `1,384`)
- `ONNX_OUTPUT_SHAPE`: comma-separated output tensor shape (example: `1,384`)

## Optional environment variables

- `ONNXRUNTIME_LIB_PATH`: explicit ONNX Runtime shared library path (`.so`, `.dylib`, or `.dll`)
- `ONNXRUNTIME_VERSION`: bootstrap download version (default: `1.23.1`)
- `ONNXRUNTIME_CACHE_DIR`: bootstrap cache location
- `ONNXRUNTIME_DISABLE_DOWNLOAD=1`: disable bootstrap download and require existing cache/path
- `ONNXRUNTIME_SKIP_VERSION_CHECK=1`: skip runtime version warning during `InitializeEnvironment`
- `ONNX_INPUT_NAME` (default: `input`)
- `ONNX_OUTPUT_NAME` (default: `output`)
- `ONNX_INPUT_DATA`: comma-separated `float32` values. If omitted, the example generates `1..N`.

## Run

```bash
export ONNX_MODEL_PATH="/path/to/model.onnx"
export ONNX_INPUT_NAME="input"
export ONNX_OUTPUT_NAME="output"
export ONNX_INPUT_SHAPE="1,384"
export ONNX_OUTPUT_SHAPE="1,384"

go run ./examples/inference
```

If `ONNXRUNTIME_LIB_PATH` is omitted, the example automatically bootstraps ONNX Runtime into the local cache.

The output prints a short preview of the first values from the output tensor.
