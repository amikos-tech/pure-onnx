# OpenCLIP Runnable Example

This example runs OpenCLIP text+image embeddings end-to-end and prints a
text-image similarity ranking for bundled fixture images.

## What It Demonstrates

- loading OpenCLIP assets with `openclip.EnsureDefaultAssets()`
- embedding text prompts with `EmbedTexts`
- embedding local image fixtures with `EmbedImages`
- ranking prompt similarity per image with `openclip.CLIPSimilarityLogits`

## Prerequisites

- Go toolchain
- ONNX Runtime shared library (either explicit path or bootstrap download)
- Network access on first run to download missing runtime/model assets

## Run

```bash
cd examples/openclip
go run .
```

## Environment Variables

### Example-specific

- `OPENCLIP_EXAMPLE_ASSETS_DIR`
  - optional
  - default: `./assets`
  - points to a directory containing `manifest.jsonl` and PNG files
- `OPENCLIP_EXAMPLE_LIMIT`
  - optional
  - default: `30`
  - limits how many manifest rows are loaded (first `N` rows)

### ONNX Runtime / model bootstrap

- `ONNXRUNTIME_LIB_PATH`
  - optional explicit path to `.so` / `.dylib` / `.dll`
  - if omitted, the example uses `ort.EnsureOnnxRuntimeSharedLibrary()`
- `ONNXRUNTIME_VERSION`, `ONNXRUNTIME_CACHE_DIR`, `ONNXRUNTIME_DISABLE_DOWNLOAD`
  - optional runtime bootstrap controls
- `ONNXRUNTIME_OPENCLIP_CACHE_DIR`
  - optional OpenCLIP model cache directory
- `HF_TOKEN`
  - optional Hugging Face token for gated/private assets

## Example Output

The example prints:

- loaded fixture count
- resolved OpenCLIP asset paths
- top-3 prompt matches for each image with similarity logits

## Troubleshooting

- **Model/preprocessor mismatch**
  - If you mix files from different exports, embedding calls can fail or produce incorrect rankings.
  - Use `text_model.onnx`, `vision_model.onnx`, `tokenizer.json`, and `preprocessor_config.json` from the same bundle.
- **Wrong model I/O contract**
  - This example expects the pinned OpenCLIP contract (text: `input_ids`, `attention_mask` -> `text_embeds`; vision: `pixel_values` -> `image_embeds`).
- **Image preprocessing mistakes**
  - Do not manually normalize or resize image tensors before passing images; `embeddings/openclip` applies its configured preprocessing pipeline.
- **ORT library load failures**
  - Set `ONNXRUNTIME_LIB_PATH` explicitly if auto-bootstrap is unavailable in your environment.

## Fixture Sources

Bundled fixtures are resized derivatives from public datasets.
See [`ATTRIBUTION.md`](./ATTRIBUTION.md) for source links and license notes.
