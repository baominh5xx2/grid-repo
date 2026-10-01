"""Prepare the immutable three-way ViNLI source dataset.

The upstream CSVs are pinned to a reviewed Git commit.  Numeric labels follow the
ViNLI release semantics (0=entailment, 1=contradiction, 2=neutral).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pathlib
from collections import Counter
from urllib.request import urlopen

SOURCE_REPOSITORY = "trantranuit/ViHLM_NLI_Project"
REVISION = "47bd78ac5d075bd00a3cb4cdd3ede4eec4acf8c2"
RAW_BASE = f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{REVISION}/data/vinli"
SPLIT_FILES = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}
EXPECTED_SPLIT_COUNTS = {"train": 18_282, "dev": 2_255, "test": 2_264}
LABEL_MAP = {0: "E", 1: "C", 2: "N"}
DOMAIN = "vinli"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_bytes(url: str) -> bytes:
    with urlopen(url, timeout=120) as response:
        return response.read()


def parse_csv(raw: bytes, split: str) -> list[dict]:
    """Parse one pinned CSV without silently coercing or dropping rows."""
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    required = {"sentence1", "sentence2", "label"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError(f"ViNLI {split} must contain columns {sorted(required)}; got {reader.fieldnames}")

    rows = []
    for index, source in enumerate(reader):
        try:
            numeric_label = int(source["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid ViNLI label {source.get('label')!r} at {split} row {index}") from exc
        if numeric_label not in LABEL_MAP:
            raise ValueError(f"unknown ViNLI label {numeric_label!r} at {split} row {index}")
        if source["sentence1"] is None or source["sentence2"] is None:
            raise ValueError(f"missing ViNLI text at {split} row {index}")
        rows.append({
            "id": f"vinli-{split}-{index:06d}",
            "premise": source["sentence1"],
            "hypothesis": source["sentence2"],
            "label": LABEL_MAP[numeric_label],
            "domain": DOMAIN,
        })
    return rows


def encode_jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def convert_split(split: str, out_path: pathlib.Path) -> tuple[int, dict]:
    url = f"{RAW_BASE}/{SPLIT_FILES[split]}"
    raw = download_bytes(url)
    rows = parse_csv(raw, split)
    expected = EXPECTED_SPLIT_COUNTS[split]
    if len(rows) != expected:
        raise ValueError(f"ViNLI {split} row count {len(rows)} != canonical {expected}")
    payload = encode_jsonl(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    distribution = dict(sorted(Counter(row["label"] for row in rows).items()))
    return len(rows), distribution


def prepare_dataset(out_dir: pathlib.Path) -> dict:
    out_dir = pathlib.Path(out_dir)
    split_manifest = {}
    for split in ("train", "dev", "test"):
        url = f"{RAW_BASE}/{SPLIT_FILES[split]}"
        raw = download_bytes(url)
        rows = parse_csv(raw, split)
        expected = EXPECTED_SPLIT_COUNTS[split]
        if len(rows) != expected:
            raise ValueError(f"ViNLI {split} row count {len(rows)} != canonical {expected}")
        payload = encode_jsonl(rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{split}.jsonl").write_bytes(payload)
        split_manifest[split] = {
            "source_url": url,
            "source_sha256": sha256(raw),
            "output_sha256": sha256(payload),
            "row_count": len(rows),
            "label_distribution": dict(sorted(Counter(row["label"] for row in rows).items())),
        }
        print(f"[prepare_vinli] {split}: {len(rows)} rows")

    manifest = {
        "dataset": "ViNLI-three-way",
        "source": {"repository": SOURCE_REPOSITORY, "revision": REVISION},
        "label_map": {str(key): value for key, value in LABEL_MAP.items()},
        "splits": split_manifest,
        "cleaning": {"rule": "none", "removed_ids": []},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(args) -> None:
    manifest = prepare_dataset(pathlib.Path(args.out_dir))
    total = sum(item["row_count"] for item in manifest["splits"].values())
    print(f"[prepare_vinli] done: {total} rows at revision {REVISION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/external/vinli")
    main(parser.parse_args())
