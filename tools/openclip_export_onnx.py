#!/usr/bin/env python3
"""Export OpenCLIP/CLIP models from Hugging Face into split ONNX artifacts.

The script exports:
  - text_model.onnx (input_ids + attention_mask -> text_embeds)
  - vision_model.onnx (pixel_values -> image_embeds)

It also copies metadata files required by downstream consumers and writes a
manifest.json with checksums and export metadata.

Implementation note:
  This exporter uses Hugging Face transformers CLIP classes
  (CLIPTextModelWithProjection / CLIPVisionModelWithProjection). The
  open_clip Python package is not required.

Example:
  python3 tools/openclip_export_onnx.py \
    --output-dir /tmp/openclip_onnx

  python3 tools/openclip_export_onnx.py \
    --output-dir /tmp/openclip_onnx \
    --push-to-hub-repo tazarov/openclip-vit-b-32-laion2b-onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
from huggingface_hub import HfApi, model_info, snapshot_download
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

DEFAULT_MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
DEFAULT_MODEL_REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"
DEFAULT_OPSET = 17
DEFAULT_MANIFEST_NAME = "manifest.json"

REQUIRED_METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
)

OPTIONAL_METADATA_FILES = (
    "special_tokens_map.json",
    "open_clip_config.json",
    "vocab.json",
    "merges.txt",
    "README.md",
)


@dataclass
class ExportConfig:
    model_id: str
    requested_revision: str
    resolved_revision: str
    output_dir: Path
    opset: int
    manifest_name: str
    verify: bool
    hf_token: str | None
    push_to_hub_repo: str | None
    push_to_hub_revision: str
    push_to_hub_private: bool
    push_to_hub_commit_message: str


class _TextExportModule(torch.nn.Module):
    def __init__(self, model: CLIPTextModelWithProjection):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.text_embeds


class _VisionExportModule(torch.nn.Module):
    def __init__(self, model: CLIPVisionModelWithProjection):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        return outputs.image_embeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a pinned CLIP/OpenCLIP Hugging Face model into split ONNX artifacts."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model repo id (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help=(
            "Model revision (commit hash recommended for reproducibility). "
            f"Default: {DEFAULT_MODEL_REVISION}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for ONNX artifacts and metadata files.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"ONNX opset version to use during export (default: {DEFAULT_OPSET}).",
    )
    parser.add_argument(
        "--manifest-name",
        default=DEFAULT_MANIFEST_NAME,
        help=f"Manifest filename to write in output dir (default: {DEFAULT_MANIFEST_NAME}).",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="Optional explicit Hugging Face token (overrides --hf-token-env lookup).",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable containing Hugging Face token (default: HF_TOKEN).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX Runtime verification pass after export.",
    )
    parser.add_argument(
        "--push-to-hub-repo",
        default="",
        help="Optional Hugging Face repo id to upload artifacts to (e.g. org/model-name).",
    )
    parser.add_argument(
        "--push-to-hub-revision",
        default="main",
        help="Target Hub branch/revision for upload (default: main).",
    )
    parser.add_argument(
        "--push-to-hub-private",
        action="store_true",
        help="Create the Hub repo as private if it does not exist.",
    )
    parser.add_argument(
        "--push-to-hub-commit-message",
        default="Add OpenCLIP ONNX export artifacts",
        help="Commit message for upload when --push-to-hub-repo is used.",
    )
    parser.add_argument(
        "--clean-output-dir",
        action="store_true",
        help="Delete existing files in --output-dir before export.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_hf_token(explicit_token: str, env_var_name: str) -> str | None:
    explicit = explicit_token.strip()
    if explicit:
        return explicit
    env_value = os.getenv(env_var_name, "").strip()
    if env_value:
        return env_value
    return None


def resolve_model_revision(model_id: str, requested_revision: str, hf_token: str | None) -> str:
    try:
        info = model_info(repo_id=model_id, revision=requested_revision, token=hf_token)
    except Exception as exc:
        raise RuntimeError(
            "failed to fetch model metadata from Hugging Face Hub for "
            f"{model_id!r} at revision {requested_revision!r}: {type(exc).__name__}: {exc}"
        ) from exc

    resolved = (info.sha or "").strip()
    if not resolved:
        raise RuntimeError(
            f"Unable to resolve revision SHA for model {model_id!r} and revision {requested_revision!r}."
        )
    return resolved


def ensure_output_dir(path: Path, clean: bool) -> None:
    try:
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"output path exists and is not a directory: {path}")

        if path.exists() and not clean:
            try:
                has_entries = any(path.iterdir())
            except OSError as exc:
                raise RuntimeError(f"failed to inspect existing output directory {path}: {exc}") from exc
            if has_entries:
                raise RuntimeError(
                    f"output directory already exists and is not empty: {path}. "
                    "Use --clean-output-dir to overwrite an existing directory."
                )
            return

        if clean and path.exists():
            cleanup_target = path.with_name(
                f"{path.name}.cleanup-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            )
            path.rename(cleanup_target)
            try:
                path.mkdir(parents=True, exist_ok=False)
            except OSError as exc:
                raise RuntimeError(
                    f"failed to recreate output directory {path} after moving existing contents to {cleanup_target}: {exc}. "
                    f"Previous contents are preserved under {cleanup_target}"
                ) from exc
            try:
                shutil.rmtree(cleanup_target)
            except OSError as exc:
                raise RuntimeError(
                    "failed to remove previous output directory snapshot "
                    f"{cleanup_target}: {exc}"
                ) from exc
            return

        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"failed to prepare output directory {path}: {exc}") from exc


def copy_metadata_files(cfg: ExportConfig) -> list[str]:
    allow_patterns = list(REQUIRED_METADATA_FILES) + list(OPTIONAL_METADATA_FILES)
    snapshot_path = Path(
        snapshot_download(
            repo_id=cfg.model_id,
            revision=cfg.resolved_revision,
            allow_patterns=allow_patterns,
            token=cfg.hf_token,
        )
    )

    copied: list[str] = []
    for filename in REQUIRED_METADATA_FILES:
        src = snapshot_path / filename
        if not src.exists():
            raise RuntimeError(
                f"Required metadata file {filename!r} was not found in model {cfg.model_id}@{cfg.resolved_revision}."
            )
        dst = cfg.output_dir / filename
        shutil.copy2(src, dst)
        copied.append(filename)

    for filename in OPTIONAL_METADATA_FILES:
        src = snapshot_path / filename
        if not src.exists():
            continue
        dst = cfg.output_dir / filename
        shutil.copy2(src, dst)
        copied.append(filename)

    return copied


def export_text_model(cfg: ExportConfig) -> dict[str, Any]:
    print("Loading CLIPTextModelWithProjection...", file=sys.stderr)
    model = CLIPTextModelWithProjection.from_pretrained(
        cfg.model_id,
        revision=cfg.resolved_revision,
        token=cfg.hf_token,
    )
    sequence_length = int(model.config.max_position_embeddings)
    projection_dim = int(model.config.projection_dim)

    wrapper = _TextExportModule(model)
    wrapper.eval()

    dummy_input_ids = torch.ones((1, sequence_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.long)
    output_path = cfg.output_dir / "text_model.onnx"

    print(f"Exporting {output_path.name} (sequence_length={sequence_length}, projection_dim={projection_dim})...", file=sys.stderr)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            str(output_path),
            export_params=True,
            opset_version=cfg.opset,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["text_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "text_embeds": {0: "batch_size"},
            },
        )

    return {
        "file": output_path.name,
        "inputs": ["input_ids", "attention_mask"],
        "outputs": ["text_embeds"],
        "sequence_length": sequence_length,
        "embedding_dimension": projection_dim,
    }


def export_vision_model(cfg: ExportConfig) -> dict[str, Any]:
    print("Loading CLIPVisionModelWithProjection...", file=sys.stderr)
    model = CLIPVisionModelWithProjection.from_pretrained(
        cfg.model_id,
        revision=cfg.resolved_revision,
        token=cfg.hf_token,
    )
    image_size = int(model.config.image_size)
    projection_dim = int(model.config.projection_dim)
    num_channels = int(getattr(model.config, "num_channels", 3))

    wrapper = _VisionExportModule(model)
    wrapper.eval()

    dummy_pixel_values = torch.zeros((1, num_channels, image_size, image_size), dtype=torch.float32)
    output_path = cfg.output_dir / "vision_model.onnx"

    print(
        f"Exporting {output_path.name} (image_size={image_size}, projection_dim={projection_dim})...",
        file=sys.stderr,
    )
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy_pixel_values,),
            str(output_path),
            export_params=True,
            opset_version=cfg.opset,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_embeds": {0: "batch_size"},
            },
        )

    return {
        "file": output_path.name,
        "inputs": ["pixel_values"],
        "outputs": ["image_embeds"],
        "image_size": image_size,
        "num_channels": num_channels,
        "embedding_dimension": projection_dim,
    }


def verify_exports(cfg: ExportConfig, text_meta: dict[str, Any], vision_meta: dict[str, Any]) -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - import guard for local tooling
        raise RuntimeError(
            "ONNX Runtime verification requested but numpy is not installed. "
            "Install numpy or pass --skip-verify."
        ) from exc

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - import guard for local tooling
        raise RuntimeError(
            "ONNX Runtime verification requested but dependencies are missing. "
            "Install onnxruntime+numpy or pass --skip-verify."
        ) from exc
    except Exception as exc:  # pragma: no cover - import guard for local tooling
        message = str(exc) or repr(exc)
        if "_ARRAY_API" in message or "compiled using NumPy 1.x" in message:
            raise RuntimeError(
                "ONNX Runtime failed to import due to a NumPy ABI mismatch. "
                "Install compatible versions (for this repo baseline: onnxruntime==1.23.1) "
                "or pass --skip-verify."
            ) from exc
        raise RuntimeError(f"ONNX Runtime failed to import: {message}") from exc

    text_path = cfg.output_dir / text_meta["file"]
    text_session = ort.InferenceSession(str(text_path), providers=["CPUExecutionProvider"])
    text_input_ids = np.zeros((2, int(text_meta["sequence_length"])), dtype=np.int64)
    text_attention = np.ones((2, int(text_meta["sequence_length"])), dtype=np.int64)
    text_outputs = text_session.run(
        None,
        {
            "input_ids": text_input_ids,
            "attention_mask": text_attention,
        },
    )
    if len(text_outputs) != 1:
        raise RuntimeError(f"Unexpected text model output count: got {len(text_outputs)}, want 1")
    text_shape = tuple(text_outputs[0].shape)
    expected_text_shape = (2, int(text_meta["embedding_dimension"]))
    if text_shape != expected_text_shape:
        raise RuntimeError(f"Unexpected text output shape: got {text_shape}, want {expected_text_shape}")

    vision_path = cfg.output_dir / vision_meta["file"]
    vision_session = ort.InferenceSession(str(vision_path), providers=["CPUExecutionProvider"])
    vision_input = np.zeros(
        (2, int(vision_meta["num_channels"]), int(vision_meta["image_size"]), int(vision_meta["image_size"])),
        dtype=np.float32,
    )
    vision_outputs = vision_session.run(None, {"pixel_values": vision_input})
    if len(vision_outputs) != 1:
        raise RuntimeError(f"Unexpected vision model output count: got {len(vision_outputs)}, want 1")
    vision_shape = tuple(vision_outputs[0].shape)
    expected_vision_shape = (2, int(vision_meta["embedding_dimension"]))
    if vision_shape != expected_vision_shape:
        raise RuntimeError(f"Unexpected vision output shape: got {vision_shape}, want {expected_vision_shape}")


def collect_artifact_checksums(output_dir: Path, manifest_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for child in sorted(output_dir.iterdir()):
        if not child.is_file():
            continue
        if child.name == manifest_name:
            continue
        records.append(
            {
                "path": child.name,
                "size_bytes": child.stat().st_size,
                "sha256": file_sha256(child),
            }
        )
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def push_to_hub(cfg: ExportConfig) -> None:
    repo_id = cfg.push_to_hub_repo
    if repo_id is None:
        raise RuntimeError("internal error: push_to_hub called without a target repo")
    print(f"Uploading artifacts to Hugging Face Hub repo {repo_id}...", file=sys.stderr)
    api = HfApi(token=cfg.hf_token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=cfg.push_to_hub_private,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(cfg.output_dir),
        path_in_repo="",
        revision=cfg.push_to_hub_revision,
        commit_message=cfg.push_to_hub_commit_message,
    )


def build_publish_info(cfg: ExportConfig) -> dict[str, Any] | None:
    if cfg.push_to_hub_repo is None:
        return None
    return {
        "repo_id": cfg.push_to_hub_repo,
        "revision": cfg.push_to_hub_revision,
        "base_url": f"https://huggingface.co/{cfg.push_to_hub_repo}/resolve/{cfg.push_to_hub_revision}",
    }


def build_manifest(
    cfg: ExportConfig,
    text_meta: dict[str, Any],
    vision_meta: dict[str, Any],
    copied_metadata_files: list[str],
    artifact_files: list[dict[str, Any]],
    push_info: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/openclip_export_onnx.py",
        "source": {
            "model_id": cfg.model_id,
            "requested_revision": cfg.requested_revision,
            "resolved_revision": cfg.resolved_revision,
        },
        "export": {
            "opset": cfg.opset,
            "torch_version": torch.__version__,
            "text_model": text_meta,
            "vision_model": vision_meta,
            "copied_metadata_files": copied_metadata_files,
        },
        "artifacts": artifact_files,
        "publish": push_info,
    }


def make_export_config(args: argparse.Namespace) -> ExportConfig:
    if args.opset < 14:
        raise ValueError(f"--opset must be >= 14, got {args.opset}")
    manifest_name = args.manifest_name.strip()
    if not manifest_name:
        raise ValueError("--manifest-name cannot be empty")
    if manifest_name in {".", ".."}:
        raise ValueError("--manifest-name must be a regular file name, not '.' or '..'")
    for path_type in (PurePosixPath, PureWindowsPath):
        parsed = path_type(manifest_name)
        if parsed.name != manifest_name or len(parsed.parts) != 1:
            raise ValueError("--manifest-name must be a file name, not a path")

    hf_token = resolve_hf_token(args.hf_token, args.hf_token_env)
    push_to_hub_repo = args.push_to_hub_repo.strip() or None
    if push_to_hub_repo is not None:
        try:
            HfApi(token=hf_token).whoami()
        except Exception as exc:
            raise ValueError(
                "--push-to-hub-repo requires Hugging Face authentication. "
                "Either pass a token via --hf-token/--hf-token-env or log in via "
                "`huggingface-cli login` so cached credentials can be used."
            ) from exc
    resolved_revision = resolve_model_revision(args.model_id, args.model_revision, hf_token)

    return ExportConfig(
        model_id=args.model_id,
        requested_revision=args.model_revision,
        resolved_revision=resolved_revision,
        output_dir=args.output_dir,
        opset=args.opset,
        manifest_name=manifest_name,
        verify=not args.skip_verify,
        hf_token=hf_token,
        push_to_hub_repo=push_to_hub_repo,
        push_to_hub_revision=args.push_to_hub_revision.strip() or "main",
        push_to_hub_private=args.push_to_hub_private,
        push_to_hub_commit_message=args.push_to_hub_commit_message,
    )


def main() -> int:
    args = parse_args()
    try:
        ensure_output_dir(args.output_dir, clean=args.clean_output_dir)
        cfg = make_export_config(args)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(
        "Using source model "
        f"{cfg.model_id}@{cfg.resolved_revision} (requested revision: {cfg.requested_revision})",
        file=sys.stderr,
    )
    push_info = build_publish_info(cfg)

    try:
        copied_metadata_files = copy_metadata_files(cfg)
        text_meta = export_text_model(cfg)
        vision_meta = export_vision_model(cfg)

        if cfg.verify:
            print("Verifying ONNX exports with ONNX Runtime...", file=sys.stderr)
            verify_exports(cfg, text_meta, vision_meta)

        artifact_files = collect_artifact_checksums(cfg.output_dir, cfg.manifest_name)
        manifest = build_manifest(
            cfg=cfg,
            text_meta=text_meta,
            vision_meta=vision_meta,
            copied_metadata_files=copied_metadata_files,
            artifact_files=artifact_files,
            push_info=push_info,
        )
        manifest_path = cfg.output_dir / cfg.manifest_name
        # Write manifest before upload so it is included in pushed artifacts.
        write_json(manifest_path, manifest)
    except Exception as exc:
        print(f"error: export failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    if cfg.push_to_hub_repo is not None:
        try:
            push_to_hub(cfg)
        except Exception as exc:
            print(f"error: upload failed after successful export ({type(exc).__name__}): {exc}", file=sys.stderr)
            print(f"local artifacts are available at: {cfg.output_dir}", file=sys.stderr)
            print(f"local manifest is available at: {cfg.output_dir / cfg.manifest_name}", file=sys.stderr)
            return 1

    print(f"Wrote artifacts to: {cfg.output_dir}", file=sys.stderr)
    print(f"Wrote manifest: {cfg.output_dir / cfg.manifest_name}", file=sys.stderr)
    if push_info is not None:
        print(f"Uploaded artifacts to: {push_info['base_url']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
