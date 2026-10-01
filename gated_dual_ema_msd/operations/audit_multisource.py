"""Hard-gating leakage and integrity audit for multi-source transfer.

The audit compares target splits internally and each source train split against
the target.  Normalized exact-pair overlap, invalid labels, or conflicting
labels in any train split make ``all_checks_passed`` false and exit with code 1.
Premise-only overlaps and conflicts in preserved evaluation splits are reported
as diagnostics rather than silently changing official evaluation data.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import re
import unicodedata
from collections import defaultdict

VALID_LABELS = {"E", "C", "N"}
DATASETS = ("vianli", "vinli", "vimednli")
SPLITS = ("train", "dev", "test")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).lower()).strip()


def pair_tuple(row: dict, normalized: bool = True) -> tuple[str, str]:
    premise, hypothesis = row["premise"], row["hypothesis"]
    return (normalize(premise), normalize(hypothesis)) if normalized else (premise, hypothesis)


def pair_key(row: dict) -> str:
    premise, hypothesis = pair_tuple(row)
    return f"{premise} [SEP] {hypothesis}"


def premise_key(row: dict, normalized: bool = True) -> str:
    return normalize(row["premise"]) if normalized else row["premise"]


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {field for field in ("id", "premise", "hypothesis", "label") if field not in row}
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields {sorted(missing)}")
            rows.append(row)
    return rows


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlap_summary(left: list[dict], right: list[dict]) -> dict:
    """Count raw/normalized pair and premise overlap between two row collections."""
    right_raw_pairs = {pair_tuple(row, normalized=False) for row in right}
    right_normalized_pairs = {pair_tuple(row) for row in right}
    right_raw_premises = {premise_key(row, normalized=False) for row in right}
    right_normalized_premises = {premise_key(row) for row in right}
    raw_pair_matches, normalized_pair_matches = [], []
    raw_premise_matches, normalized_premise_matches = [], []
    seen = {"raw_pair": set(), "normalized_pair": set(), "raw_premise": set(), "normalized_premise": set()}
    for row in left:
        raw_pair, normalized_pair = pair_tuple(row, False), pair_tuple(row, True)
        raw_premise, normalized_premise = premise_key(row, False), premise_key(row, True)
        if raw_pair in right_raw_pairs and raw_pair not in seen["raw_pair"]:
            raw_pair_matches.append({"id": row["id"], "premise": raw_pair[0], "hypothesis": raw_pair[1]})
            seen["raw_pair"].add(raw_pair)
        if normalized_pair in right_normalized_pairs and normalized_pair not in seen["normalized_pair"]:
            normalized_pair_matches.append({"id": row["id"], "key": f"{normalized_pair[0]} [SEP] {normalized_pair[1]}"})
            seen["normalized_pair"].add(normalized_pair)
        if raw_premise in right_raw_premises and raw_premise not in seen["raw_premise"]:
            raw_premise_matches.append({"id": row["id"], "premise": raw_premise})
            seen["raw_premise"].add(raw_premise)
        if normalized_premise in right_normalized_premises and normalized_premise not in seen["normalized_premise"]:
            normalized_premise_matches.append({"id": row["id"], "premise": normalized_premise})
            seen["normalized_premise"].add(normalized_premise)
    return {
        "exact_pair_count": len(raw_pair_matches),
        "normalized_pair_count": len(normalized_pair_matches),
        "exact_premise_count": len(raw_premise_matches),
        "normalized_premise_count": len(normalized_premise_matches),
        "normalized_pair_examples": normalized_pair_matches[:20],
        "normalized_premise_examples": normalized_premise_matches[:20],
    }


def check_invalid_labels(name: str, rows: list[dict]) -> list[dict]:
    return [
        {"dataset": name, "id": row.get("id"), "label": row.get("label")}
        for row in rows
        if row.get("label") not in VALID_LABELS
    ]


def conflicting_pair_groups(name: str, rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[pair_tuple(row)].append(row)
    conflicts = []
    for key in sorted(groups):
        members = groups[key]
        labels = sorted({row["label"] for row in members})
        if len(labels) > 1:
            conflicts.append({
                "dataset": name,
                "key": f"{key[0]} [SEP] {key[1]}",
                "labels": labels,
                "ids": [row["id"] for row in members],
            })
    return conflicts


def target_internal_overlaps(target: dict[str, list[dict]]) -> dict:
    return {
        "train_vs_dev": overlap_summary(target["train"], target["dev"]),
        "train_vs_test": overlap_summary(target["train"], target["test"]),
        "dev_vs_test": overlap_summary(target["dev"], target["test"]),
    }


def source_target_overlaps(source_train: list[dict], target: dict[str, list[dict]]) -> dict:
    return {f"train_vs_{split}": overlap_summary(source_train, target[split]) for split in SPLITS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vianli-dir", default="data/processed/vianli_clean")
    parser.add_argument("--vinli-dir", default="data/external/vinli")
    parser.add_argument("--vimednli-dir", default="data/external/vimednli")
    parser.add_argument("--out", default="data/audit/multisource_audit.json")
    return parser


def main(args=None) -> None:
    if args is None or isinstance(args, (list, tuple)):
        args = build_parser().parse_args(args)
    directories = {
        "vianli": pathlib.Path(args.vianli_dir),
        "vinli": pathlib.Path(args.vinli_dir),
        "vimednli": pathlib.Path(args.vimednli_dir),
    }
    paths = {
        (dataset, split): directories[dataset] / f"{split}.jsonl"
        for dataset in DATASETS
        for split in SPLITS
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        for path in missing:
            print(f"[audit] MISSING required file: {path}")
        raise SystemExit(2)

    rows = {(dataset, split): load_jsonl(path) for (dataset, split), path in paths.items()}
    target = {split: rows[("vianli", split)] for split in SPLITS}
    target_overlaps = target_internal_overlaps(target)
    source_overlaps = {
        source: source_target_overlaps(rows[(source, "train")], target)
        for source in ("vinli", "vimednli")
    }

    target_pair_count = sum(item["normalized_pair_count"] for item in target_overlaps.values())
    checks = [{
        "name": "vianli_internal_normalized_pair_overlap",
        "passed": target_pair_count == 0,
        "overlap_count": target_pair_count,
        "comparisons": target_overlaps,
    }]
    for source in ("vinli", "vimednli"):
        holdout_count = sum(
            source_overlaps[source][f"train_vs_{split}"]["normalized_pair_count"]
            for split in ("dev", "test")
        )
        checks.append({
            "name": f"{source}_train_normalized_pair_overlap_vianli_devtest",
            "passed": holdout_count == 0,
            "overlap_count": holdout_count,
            "comparisons": {
                split: source_overlaps[source][f"train_vs_{split}"] for split in ("dev", "test")
            },
        })

    invalid = []
    for dataset in DATASETS:
        for split in SPLITS:
            invalid.extend(check_invalid_labels(f"{dataset}_{split}", rows[(dataset, split)]))
    checks.append({
        "name": "invalid_labels",
        "passed": not invalid,
        "invalid_count": len(invalid),
        "invalid_examples": invalid[:20],
    })

    train_conflicts = []
    for dataset in DATASETS:
        train_conflicts.extend(conflicting_pair_groups(f"{dataset}_train", rows[(dataset, "train")]))
    checks.append({
        "name": "train_duplicate_conflicting_labels",
        "passed": not train_conflicts,
        "conflicting_group_count": len(train_conflicts),
        "conflicting_groups": train_conflicts[:20],
    })

    eval_conflicts = []
    for dataset in DATASETS:
        for split in ("dev", "test"):
            eval_conflicts.extend(conflicting_pair_groups(f"{dataset}_{split}", rows[(dataset, split)]))

    all_checks_passed = all(item["passed"] for item in checks)
    leakage_passed = all(item["passed"] for item in checks[:3])
    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "counts": {
            f"{dataset}_{split}": len(rows[(dataset, split)])
            for dataset in DATASETS
            for split in SPLITS
        },
        "fingerprints": {
            f"{dataset}_{split}": {
                "path": str(paths[(dataset, split)]),
                "sha256": file_sha256(paths[(dataset, split)]),
            }
            for dataset in DATASETS
            for split in SPLITS
        },
        "overlaps": {
            "target_internal": target_overlaps,
            "sources_vs_target": source_overlaps,
        },
        "checks": checks,
        "diagnostics": {
            "eval_conflicting_pair_groups": {
                "group_count": len(eval_conflicts),
                "groups": eval_conflicts[:20],
                "blocking": False,
                "reason": "official dev/test splits are preserved and conflicts are disclosed",
            }
        },
        "leakage_passed": leakage_passed,
        "all_checks_passed": all_checks_passed,
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for item in checks:
        print(f"[audit] {item['name']}: {'PASS' if item['passed'] else 'FAIL'}")
    if not all_checks_passed:
        print(f"[audit] HARD FAIL: all_checks_passed=false; see {out_path}")
        raise SystemExit(1)
    print(f"[audit] ALL CHECKS PASSED -> {out_path}")


if __name__ == "__main__":
    main()
