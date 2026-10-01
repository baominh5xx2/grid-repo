"""Prepare pinned official and leakage-clean ViANLI target splits.

By default this writes the complete official mapped splits to ``data/raw`` and
the deterministic primary clean protocol to ``data/processed/vianli_clean``.
Only target-train rows whose normalized exact pair appears in official dev/test
are removed.  Dev and test are preserved unchanged in both destinations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import unicodedata
from collections import Counter

from huggingface_hub import hf_hub_download

HF_REPO = "uitnlp/ViANLI"
REVISION = "0fec8d6ecb043a61c609f9b51f80401fdf1e84d3"
SPLIT_FILES = {
    "train": "vianli_train.jsonl",
    "dev": "vianli_dev.jsonl",
    "test": "vianli_test.jsonl",
}
EXPECTED_SOURCE_COUNTS = {"train": 8_012, "dev": 1_000, "test": 1_000}
EXPECTED_CLEAN_TRAIN_COUNT = 8_010
LABEL_MAP = {"entailment": "E", "contradiction": "C", "neutral": "N"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).lower()).strip()


def pair_key(row: dict) -> tuple[str, str]:
    return normalize(row["premise"]), normalize(row["hypothesis"])


def row_id(row: dict) -> str:
    value = row.get("uid", row.get("id"))
    if value is None:
        raise ValueError("ViANLI row is missing uid/id")
    return str(value)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_source_jsonl(raw: bytes, split: str) -> list[dict]:
    rows = []
    for index, line in enumerate(raw.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        source = json.loads(line)
        missing = {field for field in ("uid", "premise", "hypothesis", "label") if field not in source}
        if missing:
            raise ValueError(f"ViANLI {split} row {index} missing fields {sorted(missing)}")
        if source["label"] not in LABEL_MAP:
            raise ValueError(f"unknown ViANLI label {source['label']!r} at {split} row {index}")
        rows.append(source)
    return rows


def map_rows(rows: list[dict]) -> list[dict]:
    return [{
        "id": row_id(row),
        "premise": row["premise"],
        "hypothesis": row["hypothesis"],
        "label": LABEL_MAP[row["label"]] if row["label"] in LABEL_MAP else row["label"],
    } for row in rows]


def clean_train_rows(
    train_rows: list[dict], dev_rows: list[dict], test_rows: list[dict]
) -> tuple[list[dict], list[str]]:
    """Remove train rows overlapping either holdout, preserving source order."""
    holdout_keys = {pair_key(row) for row in [*dev_rows, *test_rows]}
    clean = []
    removed_ids = []
    for row in train_rows:
        if pair_key(row) in holdout_keys:
            removed_ids.append(row_id(row))
        else:
            clean.append(row)
    return clean, removed_ids


def encode_jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def split_metadata(source_raw: bytes, output: bytes, rows: list[dict], source_url: str) -> dict:
    return {
        "source_url": source_url,
        "source_sha256": sha256(source_raw),
        "output_sha256": sha256(output),
        "row_count": len(rows),
        "label_distribution": dict(sorted(Counter(row["label"] for row in rows).items())),
    }


def prepare_dataset(raw_dir: pathlib.Path, clean_dir: pathlib.Path) -> dict:
    raw_dir, clean_dir = pathlib.Path(raw_dir), pathlib.Path(clean_dir)
    source_rows, source_payloads = {}, {}
    for split in ("train", "dev", "test"):
        source_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=SPLIT_FILES[split],
            repo_type="dataset",
            revision=REVISION,
        )
        payload = pathlib.Path(source_path).read_bytes()
        rows = parse_source_jsonl(payload, split)
        expected = EXPECTED_SOURCE_COUNTS[split]
        if len(rows) != expected:
            raise ValueError(f"ViANLI {split} row count {len(rows)} != official {expected}")
        source_rows[split], source_payloads[split] = rows, payload

    clean_source_train, removed_ids = clean_train_rows(
        source_rows["train"], source_rows["dev"], source_rows["test"]
    )
    if len(clean_source_train) != EXPECTED_CLEAN_TRAIN_COUNT:
        raise ValueError(
            f"ViANLI cleaned train row count {len(clean_source_train)} != expected {EXPECTED_CLEAN_TRAIN_COUNT}"
        )

    official_rows = {split: map_rows(source_rows[split]) for split in ("train", "dev", "test")}
    clean_rows = {
        "train": map_rows(clean_source_train),
        "dev": official_rows["dev"],
        "test": official_rows["test"],
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    official_meta, clean_meta = {}, {}
    for split in ("train", "dev", "test"):
        source_url = (
            f"https://huggingface.co/datasets/{HF_REPO}/resolve/{REVISION}/{SPLIT_FILES[split]}"
        )
        official_payload = encode_jsonl(official_rows[split])
        clean_payload = encode_jsonl(clean_rows[split])
        (raw_dir / f"{split}.jsonl").write_bytes(official_payload)
        (clean_dir / f"{split}.jsonl").write_bytes(clean_payload)
        official_meta[split] = split_metadata(
            source_payloads[split], official_payload, official_rows[split], source_url
        )
        clean_meta[split] = split_metadata(
            source_payloads[split], clean_payload, clean_rows[split], source_url
        )
        print(
            f"[prepare_vianli] {split}: official={len(official_rows[split])}, "
            f"clean={len(clean_rows[split])}"
        )

    manifest = {
        "dataset": "ViANLI-primary-clean",
        "source": {"repository": HF_REPO, "revision": REVISION},
        "official_output_directory": str(raw_dir),
        "clean_output_directory": str(clean_dir),
        "official_splits": official_meta,
        "clean_splits": clean_meta,
        "cleaning": {
            "rule": "remove target train rows whose NFC/lowercase/whitespace-normalized exact pair occurs in official dev or test",
            "removed_ids": removed_ids,
            "removed_row_count": len(removed_ids),
            "expected_clean_train_count": EXPECTED_CLEAN_TRAIN_COUNT,
        },
    }
    (clean_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def convert_split(split: str, out_path: pathlib.Path) -> int:
    """Compatibility helper for writing one pinned official (uncleaned) split."""
    source_path = hf_hub_download(
        repo_id=HF_REPO, filename=SPLIT_FILES[split], repo_type="dataset", revision=REVISION
    )
    rows = parse_source_jsonl(pathlib.Path(source_path).read_bytes(), split)
    if len(rows) != EXPECTED_SOURCE_COUNTS[split]:
        raise ValueError(f"ViANLI {split} row count {len(rows)} != official {EXPECTED_SOURCE_COUNTS[split]}")
    payload = encode_jsonl(map_rows(rows))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    return len(rows)


def main(args) -> None:
    manifest = prepare_dataset(pathlib.Path(args.raw_dir), pathlib.Path(args.clean_dir))
    print(
        f"[prepare_vianli] done: official train={manifest['official_splits']['train']['row_count']}, "
        f"clean train={manifest['clean_splits']['train']['row_count']} at revision {REVISION}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", "--out-dir", dest="raw_dir", default="data/raw")
    parser.add_argument("--clean-dir", default="data/processed/vianli_clean")
    main(parser.parse_args())
