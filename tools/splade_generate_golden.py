#!/usr/bin/env python3
"""Generate SPLADE golden JSONL data locally using Hugging Face Transformers.

Example:
  python3 tools/splade_generate_golden.py \
    --texts-file /tmp/texts.txt \
    --output-jsonl /tmp/splade_endpoint_golden/v1/splade_pp_en_v1_local_topk24_v1.jsonl \
    --metadata-path /tmp/splade_endpoint_golden/v1/metadata.json \
    --model-name prithivida/Splade_PP_en_v1 \
    --top-k 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


@dataclass
class EncodeConfig:
    model_name: str
    sequence_length: int
    batch_size: int
    top_k: int
    prune_threshold: float
    with_labels: bool
    device: str
    hf_token: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SPLADE sparse vectors locally and write golden JSONL output."
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Input text (can be passed multiple times).",
    )
    parser.add_argument(
        "--texts-file",
        type=Path,
        help="Path to text file with one document per line.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Destination JSONL file for rows {id,text,indices,values,labels}.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        help="Destination metadata.json path (default: alongside --output-jsonl).",
    )
    parser.add_argument(
        "--model-name",
        default="prithivida/Splade_PP_en_v1",
        help="Hugging Face model repo id.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=256,
        help="Tokenizer truncation/padding length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for local inference.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=24,
        help="Max sparse dimensions per row (0 means unbounded).",
    )
    parser.add_argument(
        "--prune-threshold",
        type=float,
        default=0.0,
        help="Drop dimensions where value <= threshold.",
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="Include token labels in JSONL rows.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable containing Hugging Face token.",
    )
    return parser.parse_args()


def load_texts(inline_texts: list[str], texts_file: Path | None) -> list[str]:
    texts: list[str] = list(inline_texts)
    if texts_file is not None:
        with texts_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    texts.append(text)
    return texts


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.inference_mode()
def encode_batch_sparse(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    cfg: EncodeConfig,
) -> list[dict[str, object]]:
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=cfg.sequence_length,
    )
    tokenized = {key: value.to(cfg.device) for key, value in tokenized.items()}

    outputs = model(**tokenized)
    logits = outputs.logits  # [batch, seq_len, vocab]
    transformed = torch.log1p(torch.relu(logits))
    masked = transformed * tokenized["attention_mask"].unsqueeze(-1)
    dense = masked.max(dim=1).values  # [batch, vocab]

    vectors: list[dict[str, object]] = []
    for row in dense:
        vectors.append(
            dense_row_to_sparse(
                row=row,
                tokenizer=tokenizer,
                top_k=cfg.top_k,
                prune_threshold=cfg.prune_threshold,
                with_labels=cfg.with_labels,
            )
        )
    return vectors


def dense_row_to_sparse(
    row: torch.Tensor,
    tokenizer: AutoTokenizer,
    top_k: int,
    prune_threshold: float,
    with_labels: bool,
) -> dict[str, object]:
    pairs: list[tuple[int, float]] = []
    values = row.detach().cpu().tolist()
    for idx, value in enumerate(values):
        if value <= prune_threshold:
            continue
        pairs.append((idx, float(value)))

    if top_k > 0 and len(pairs) > top_k:
        pairs.sort(key=lambda item: (-item[1], item[0]))
        pairs = pairs[:top_k]

    pairs.sort(key=lambda item: item[0])
    indices = [idx for idx, _ in pairs]
    sparse_values = [value for _, value in pairs]

    labels: list[str] = []
    if with_labels and indices:
        labels = tokenizer.convert_ids_to_tokens(indices)

    return {
        "indices": indices,
        "values": sparse_values,
        "labels": labels,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    texts = load_texts(args.text, args.texts_file)
    if not texts:
        print("error: provide at least one --text or --texts-file with non-empty lines", file=sys.stderr)
        return 2
    if args.sequence_length <= 0:
        print("error: --sequence-length must be > 0", file=sys.stderr)
        return 2
    if args.batch_size <= 0:
        print("error: --batch-size must be > 0", file=sys.stderr)
        return 2
    if args.top_k < 0:
        print("error: --top-k must be >= 0", file=sys.stderr)
        return 2
    if args.prune_threshold < 0:
        print("error: --prune-threshold must be >= 0", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    hf_token = os.getenv(args.hf_token_env, "").strip() or None
    cfg = EncodeConfig(
        model_name=args.model_name,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        top_k=args.top_k,
        prune_threshold=args.prune_threshold,
        with_labels=args.with_labels,
        device=device,
        hf_token=hf_token,
    )

    print(f"Loading tokenizer/model: {cfg.model_name}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=cfg.hf_token)
    model = AutoModelForMaskedLM.from_pretrained(cfg.model_name, token=cfg.hf_token)
    model = model.to(cfg.device)
    model.eval()

    rows: list[dict[str, object]] = []
    for i, batch in enumerate(batched(texts, cfg.batch_size), start=1):
        print(f"Encoding batch {i} ({len(batch)} docs)", file=sys.stderr)
        sparse_vectors = encode_batch_sparse(batch, tokenizer, model, cfg)
        for text, sparse in zip(batch, sparse_vectors):
            row = {
                "id": f"s{len(rows) + 1}",
                "text": text,
                "indices": sparse["indices"],
                "values": sparse["values"],
                "labels": sparse["labels"],
            }
            rows.append(row)

    write_jsonl(args.output_jsonl, rows)
    digest = sha256_file(args.output_jsonl)

    metadata_path = args.metadata_path
    if metadata_path is None:
        metadata_path = args.output_jsonl.parent / "metadata.json"

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "local:tools/splade_generate_golden.py",
        "source_type": "local_transformers",
        "model_repo": cfg.model_name,
        "row_count": len(rows),
        "dataset_digest_sha256": digest,
        "settings": {
            "sequence_length": cfg.sequence_length,
            "batch_size": cfg.batch_size,
            "top_k": cfg.top_k,
            "prune_threshold": cfg.prune_threshold,
            "with_labels": cfg.with_labels,
            "device": cfg.device,
        },
        "request_payload": {
            "texts": "batch_of_strings",
        },
        "response_shape": "vectors[{indices,values,labels}]",
    }
    write_metadata(metadata_path, metadata)

    print(f"Wrote JSONL: {args.output_jsonl}", file=sys.stderr)
    print(f"Wrote metadata: {metadata_path}", file=sys.stderr)
    print(f"Digest (SHA-256): {digest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
