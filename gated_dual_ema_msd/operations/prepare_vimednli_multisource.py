"""Prepare ViMedNLI from the immutable official refined TSV release.

The official files contain two tab-separated fields per row: a combined
``sent1: ... sent2: ...`` field and the label.  Every row in a normalized pair
group carrying more than one label is removed from *training only*; official
dev and test rows are preserved byte-for-semantics in the converted output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pathlib
import re
import unicodedata
from collections import Counter, defaultdict
from urllib.request import urlopen

SOURCE_REPOSITORY = "justinphan3110/ViPubmed"
REVISION = "2cd94305ba48ae1ccf8782c1df9819ddad7f035f"
RAW_BASE = f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{REVISION}/data/vi_mednli"
SPLIT_FILES = {
    "train": "train_vi_refined.tsv",
    "dev": "dev_vi_refined.tsv",
    "test": "test_vi_refined.tsv",
}
EXPECTED_SOURCE_COUNTS = {"train": 11_232, "dev": 1_395, "test": 1_422}
EXPECTED_CLEAN_TRAIN_COUNT = 11_217
LABEL_MAP = {"entailment": "E", "contradiction": "C", "neutral": "N"}
DOMAIN = "vimednli"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).lower()).strip()


def pair_key(row: dict) -> tuple[str, str]:
    return normalize(row["premise"]), normalize(row["hypothesis"])


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_bytes(url: str) -> bytes:
    with urlopen(url, timeout=120) as response:
        return response.read()


def parse_official_tsv(raw: bytes, split: str) -> list[dict]:
    """Parse the headerless official two-column refined TSV exactly."""
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), delimiter="\t")
    rows = []
    for index, fields in enumerate(reader):
        if len(fields) != 2:
            raise ValueError(
                f"ViMedNLI {split} row {index} must have exactly two tab-separated fields; got {len(fields)}"
            )
        combined, raw_label = fields
        prefix, separator = "sent1: ", " sent2: "
        if not combined.startswith(prefix) or separator not in combined[len(prefix):]:
            raise ValueError(f"ViMedNLI {split} row {index} must use exact sent1: ... sent2: ... format")
        premise, hypothesis = combined[len(prefix):].split(separator, 1)
        if not premise or not hypothesis:
            raise ValueError(f"ViMedNLI {split} row {index} has an empty sent1 or sent2")
        label = LABEL_MAP.get(raw_label.strip().lower())
        if label is None:
            raise ValueError(f"unknown ViMedNLI label {raw_label!r} at {split} row {index}")
        rows.append({
            "id": f"vimednli-{split}-{index:06d}",
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "domain": DOMAIN,
        })
    return rows


def remove_conflicting_pair_groups(rows: list[dict]) -> tuple[list[dict], list[str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[pair_key(row)].append(row)
    conflicting_keys = {key for key, members in groups.items() if len({r["label"] for r in members}) > 1}
    clean = [row for row in rows if pair_key(row) not in conflicting_keys]
    removed_ids = [row["id"] for row in rows if pair_key(row) in conflicting_keys]
    details = []
    for key in sorted(conflicting_keys):
        members = groups[key]
        details.append({
            "normalized_premise": key[0],
            "normalized_hypothesis": key[1],
            "labels": sorted({row["label"] for row in members}),
            "removed_ids": [row["id"] for row in members],
        })
    return clean, removed_ids, details


def conflicting_pair_groups(rows: list[dict]) -> list[dict]:
    _, _, groups = remove_conflicting_pair_groups(rows)
    return groups


def encode_jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def prepare_dataset(out_dir: pathlib.Path) -> dict:
    out_dir = pathlib.Path(out_dir)
    source_rows = {}
    raw_payloads = {}
    for split in ("train", "dev", "test"):
        url = f"{RAW_BASE}/{SPLIT_FILES[split]}"
        raw = download_bytes(url)
        rows = parse_official_tsv(raw, split)
        expected = EXPECTED_SOURCE_COUNTS[split]
        if len(rows) != expected:
            raise ValueError(f"ViMedNLI {split} row count {len(rows)} != official {expected}")
        source_rows[split], raw_payloads[split] = rows, raw

    clean_train, removed_ids, train_conflicts = remove_conflicting_pair_groups(source_rows["train"])
    if len(clean_train) != EXPECTED_CLEAN_TRAIN_COUNT:
        raise ValueError(
            f"ViMedNLI cleaned train row count {len(clean_train)} != expected {EXPECTED_CLEAN_TRAIN_COUNT}"
        )
    output_rows = {"train": clean_train, "dev": source_rows["dev"], "test": source_rows["test"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {}
    for split in ("train", "dev", "test"):
        payload = encode_jsonl(output_rows[split])
        (out_dir / f"{split}.jsonl").write_bytes(payload)
        split_manifest[split] = {
            "source_url": f"{RAW_BASE}/{SPLIT_FILES[split]}",
            "source_sha256": sha256(raw_payloads[split]),
            "source_row_count": len(source_rows[split]),
            "output_sha256": sha256(payload),
            "row_count": len(output_rows[split]),
            "label_distribution": dict(sorted(Counter(row["label"] for row in output_rows[split]).items())),
        }
        print(f"[prepare_vimednli] {split}: {len(output_rows[split])} rows")

    manifest = {
        "dataset": "ViMedNLI-refined",
        "source": {"repository": SOURCE_REPOSITORY, "revision": REVISION},
        "splits": split_manifest,
        "cleaning": {
            "rule": "remove every train row in a normalized exact-pair group with conflicting labels; preserve dev/test",
            "removed_ids": removed_ids,
            "removed_row_count": len(removed_ids),
            "conflicting_train_groups": train_conflicts,
        },
        "preserved_eval_conflicting_groups": {
            "dev": conflicting_pair_groups(source_rows["dev"]),
            "test": conflicting_pair_groups(source_rows["test"]),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def convert_split(split: str, out_path: pathlib.Path) -> tuple[int, dict]:
    """Compatibility helper; applies the train cleaning rule when split=train."""
    raw = download_bytes(f"{RAW_BASE}/{SPLIT_FILES[split]}")
    rows = parse_official_tsv(raw, split)
    if len(rows) != EXPECTED_SOURCE_COUNTS[split]:
        raise ValueError(f"ViMedNLI {split} row count {len(rows)} != official {EXPECTED_SOURCE_COUNTS[split]}")
    if split == "train":
        rows, _, _ = remove_conflicting_pair_groups(rows)
        if len(rows) != EXPECTED_CLEAN_TRAIN_COUNT:
            raise ValueError(f"ViMedNLI cleaned train row count {len(rows)} != expected {EXPECTED_CLEAN_TRAIN_COUNT}")
    payload = encode_jsonl(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    distribution = dict(sorted(Counter(row["label"] for row in rows).items()))
    return len(rows), distribution


def main(args) -> None:
    manifest = prepare_dataset(pathlib.Path(args.out_dir))
    total = sum(item["row_count"] for item in manifest["splits"].values())
    print(f"[prepare_vimednli] done: {total} rows at revision {REVISION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/external/vimednli")
    main(parser.parse_args())
