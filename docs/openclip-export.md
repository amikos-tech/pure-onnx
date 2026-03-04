# OpenCLIP Export Runbook

This runbook documents how to export a pinned OpenCLIP checkpoint into ONNX
artifacts used by this repository.

The tooling lives at:
- `tools/openclip_export_onnx.py`

## Scope

The exporter produces a split ONNX contract:
- `text_model.onnx` (`input_ids`, `attention_mask` -> `text_embeds`)
- `vision_model.onnx` (`pixel_values` -> `image_embeds`)

It also copies model metadata from Hugging Face and writes a checksum manifest.

## Default Source Model (Pinned)

- model id: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
- revision: `1a25a446712ba5ee05982a381eed697ef9b435cf`

This corresponds to OpenCLIP:
- `model_name="ViT-B-32"`
- `checkpoint="laion2b_s34b_b79k"`

## Prerequisites

Use Python 3.10+ and install:

```bash
pip install -r ./tools/requirements-openclip.txt
# or:
pip install torch transformers huggingface_hub numpy onnxruntime==1.23.1
```

Optional for private/gated models:
- `HF_TOKEN` with read access

Optional for publishing:
- `HF_TOKEN` with write access to the target repo/user/org
- or local cached auth via `huggingface-cli login`

## Export Locally

```bash
python3 ./tools/openclip_export_onnx.py \
  --output-dir ./build/openclip-vit-b-32-laion2b-s34b-b79k-onnx \
  --clean-output-dir
```

## Publish to Hugging Face Hub

```bash
export HF_TOKEN=<hf_token_with_model_write>

python3 ./tools/openclip_export_onnx.py \
  --output-dir ./build/openclip-vit-b-32-laion2b-s34b-b79k-onnx \
  --clean-output-dir \
  --push-to-hub-repo <org-or-user>/<repo-name>
```

You can also omit `HF_TOKEN` if you have already authenticated locally:

```bash
huggingface-cli login
python3 ./tools/openclip_export_onnx.py \
  --output-dir ./build/openclip-vit-b-32-laion2b-s34b-b79k-onnx \
  --clean-output-dir \
  --push-to-hub-repo <org-or-user>/<repo-name>
```

Optional flags:
- `--push-to-hub-private` to create a private repo if it does not exist.
- `--push-to-hub-revision <branch>` (default: `main`).
- `--push-to-hub-commit-message <message>`.

## Expected Artifact Layout

After a successful run:
- `text_model.onnx`
- `vision_model.onnx`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `preprocessor_config.json`
- `manifest.json`

Optional copied files (when present upstream):
- `special_tokens_map.json`
- `open_clip_config.json`
- `vocab.json`
- `merges.txt`
- `README.md`

## Manifest Contract

`manifest.json` includes:
- `schema_version`
- `generated_at_utc`
- `generator`
- source model id + requested/resolved revision
- exporter settings (opset, torch version)
- text/vision model input/output contract and dimensions
- artifact checksum list (path, size, SHA256)
- publish metadata (`repo_id`, `revision`, `base_url`) when push is used

## Verify Published Artifacts

```bash
curl -sL https://huggingface.co/<org-or-user>/<repo-name>/resolve/main/manifest.json
```

Optional: compare local checksum to manifest:

```bash
shasum -a 256 ./build/openclip-vit-b-32-laion2b-s34b-b79k-onnx/text_model.onnx
```

## Troubleshooting

### NumPy / ONNX Runtime ABI mismatch

Symptom:
- `AttributeError: _ARRAY_API not found`
- message about module compiled with NumPy 1.x and running in NumPy 2.x

Fix:
1. Pin ONNX Runtime to `1.23.1` (project baseline):
   ```bash
   pip install --upgrade onnxruntime==1.23.1
   ```
2. Re-run export.

Workaround:
- add `--skip-verify` to skip ONNX Runtime verification phase.

## Notes

- ONNX export may print `TracerWarning` messages from PyTorch; these are expected
  for this export path.
- ONNX conversion is a format conversion only; model license remains upstream
  (`MIT` for the default source model).
