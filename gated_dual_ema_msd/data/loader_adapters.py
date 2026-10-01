"""Self-contained, robust dataset loader adhering strictly to Rule 0b:

  - ViANLI: max_length = 512
  - ViMedNLI: max_length = 256
  - ViNLI: max_length = 512

Exact 100% match with src/data/dataset.py & continual_trainer.py (NO .strip() on premise/hypothesis).
"""
from __future__ import annotations

import csv
import io
import json
import os
import pathlib
import re
import unicodedata
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from gated_dual_ema_msd.config.contracts import (
    DATASET_MAX_LENGTHS,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    parse_label,
    require_max_length,
)

ROOT = pathlib.Path(__file__).resolve().parents[3]
LABEL_MAP = {
    0: 0, 1: 1, 2: 2,
    "E": 0, "C": 1, "N": 2,
    "entailment": 0, "contradiction": 1, "neutral": 2,
    "e": 0, "c": 1, "n": 2,
}



def _norm_str(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text)).lower()).strip()


def _read_jsonl(file_path: pathlib.Path) -> List[dict]:
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class RawDictDataset(Dataset):
    """Memory-efficient dataset wrapping raw dict records (exact 100% clone of continual_trainer.py)."""

    def __init__(self, rows: List[dict], tokenizer: Any, max_len: int = 512):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        lbl = parse_label(r["label"], str(r.get("id", f"sample_{idx}")))
        enc = self.tokenizer(
            str(r["premise"]),
            str(r["hypothesis"]),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(lbl, dtype=torch.long)
        return item


# Backward compatibility alias
NLIDataset = RawDictDataset


def collate_eval_rows(rows: Sequence[dict], tokenizer: Any, max_length: int) -> dict:
    """Dynamic padding='longest' collation for evaluation (exact 100% clone of dataset.py)."""
    encs = []
    for r in rows:
        e = tokenizer(
            str(r["premise"]),
            str(r["hypothesis"]),
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        encs.append(e)
    enc = tokenizer.pad(encs, padding="longest", return_tensors="pt")
    batch = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor(
            [parse_label(r["label"], str(r.get("id", f"sample_{i}"))) for i, r in enumerate(rows)],
            dtype=torch.long,
        ),
        "sample_id": [str(r.get("id", f"sample_{i}")) for i, r in enumerate(rows)],
        "gold_label": [r.get("label", "E") for r in rows],
    }

    if "token_type_ids" in enc:
        batch["token_type_ids"] = enc["token_type_ids"]
    return batch


def eval_batches(rows: Sequence[dict], batch_size: int):
    for i in range(0, len(rows), batch_size):
        yield rows[i:i + batch_size]


def _materialize_vianli(out_dir: pathlib.Path) -> Dict[str, List[dict]]:
    from huggingface_hub import hf_hub_download
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = "uitnlp/ViANLI"
    rev = "0fec8d6ecb043a61c609f9b51f80401fdf1e84d3"
    files = {"train": "vianli_train.jsonl", "dev": "vianli_dev.jsonl", "test": "vianli_test.jsonl"}
    lbl_map = {"entailment": "E", "contradiction": "C", "neutral": "N"}

    raw = {}
    for split, fn in files.items():
        src_path = hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset", revision=rev)
        rows = []
        with open(src_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    rows.append({
                        "id": str(item.get("uid", item.get("id"))),
                        "premise": item["premise"],
                        "hypothesis": item["hypothesis"],
                        "label": lbl_map.get(item["label"], item["label"]),
                    })
        raw[split] = rows

    holdout_pairs = {(_norm_str(r["premise"]), _norm_str(r["hypothesis"])) for r in raw["dev"] + raw["test"]}
    clean_train = [r for r in raw["train"] if (_norm_str(r["premise"]), _norm_str(r["hypothesis"])) not in holdout_pairs]
    raw["train"] = clean_train

    for split in ("train", "dev", "test"):
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in raw[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return raw


def _materialize_vimednli(out_dir: pathlib.Path) -> Dict[str, List[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/justinphan3110/ViPubmed/2cd94305ba48ae1ccf8782c1df9819ddad7f035f/data/vi_mednli"
    files = {"train": "train_vi_refined.tsv", "dev": "dev_vi_refined.tsv", "test": "test_vi_refined.tsv"}
    lbl_map = {"entailment": "E", "contradiction": "C", "neutral": "N"}

    raw = {}
    for split, fn in files.items():
        url = f"{base_url}/{fn}"
        with urllib.request.urlopen(url, timeout=120) as resp:
            content = resp.read().decode("utf-8-sig")
        reader = csv.reader(io.StringIO(content), delimiter="\t")
        rows = []
        for idx, fields in enumerate(reader):
            if len(fields) == 2:
                comb, raw_lbl = fields
                if comb.startswith("sent1: ") and " sent2: " in comb[7:]:
                    p, h = comb[7:].split(" sent2: ", 1)
                    lbl = lbl_map.get(raw_lbl.strip().lower())
                    if lbl:
                        rows.append({
                            "id": f"vimednli-{split}-{idx:06d}",
                            "premise": p.strip(),
                            "hypothesis": h.strip(),
                            "label": lbl,
                        })
        raw[split] = rows

    from collections import defaultdict
    groups = defaultdict(list)
    for r in raw["train"]:
        groups[(_norm_str(r["premise"]), _norm_str(r["hypothesis"]))].append(r)
    conflicting = {k for k, members in groups.items() if len({m["label"] for m in members}) > 1}
    raw["train"] = [r for r in raw["train"] if (_norm_str(r["premise"]), _norm_str(r["hypothesis"])) not in conflicting]

    for split in ("train", "dev", "test"):
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in raw[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return raw


def _materialize_vinli(out_dir: pathlib.Path) -> Dict[str, List[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/trantranuit/ViHLM_NLI_Project/47bd78ac5d075bd00a3cb4cdd3ede4eec4acf8c2/data/vinli"
    files = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}
    lbl_map = {0: "E", 1: "C", 2: "N"}

    raw = {}
    for split, fn in files.items():
        url = f"{base_url}/{fn}"
        with urllib.request.urlopen(url, timeout=120) as resp:
            content = resp.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))
        rows = []
        for idx, r in enumerate(reader):
            lbl_int = int(r["label"])
            rows.append({
                "id": f"vinli-{split}-{idx:06d}",
                "premise": r["sentence1"],
                "hypothesis": r["sentence2"],
                "label": lbl_map[lbl_int],
            })
        raw[split] = rows

    for split in ("train", "dev", "test"):
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in raw[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return raw


def load_nli_dataset(
    domain: str,
    root_dir: Optional[pathlib.Path] = None,
    model_name: Optional[str] = None,
    revision: Optional[str] = None,
) -> Tuple[Dict[str, List[dict]], AutoTokenizer]:
    root = root_dir or pathlib.Path.cwd()
    chosen_model = model_name or "uitnlp/CafeBERT"
    chosen_rev = revision if revision is not None else ("af76fcf2a04096b2b54b348a3e4eb48253c93c5d" if chosen_model == "uitnlp/CafeBERT" else None)

    tokenizer = AutoTokenizer.from_pretrained(
        chosen_model,
        revision=chosen_rev,
        trust_remote_code=True,
    )
    domain = domain.lower()
    raw_data: Dict[str, List[dict]] = {}

    if domain == "vianli":
        candidates = [
            root / "data/processed/vianli_clean",
            root / "vianli_clean",
            ROOT / "data/processed/vianli_clean",
            root / "gated_dual_ema_msd/data/processed/vianli_clean",
        ]
        target_dir = next((d for d in candidates if (d / "train.jsonl").exists()), root / "data/processed/vianli_clean")
        if (target_dir / "train.jsonl").exists():
            for split in ("train", "dev", "test"):
                raw_data[split] = _read_jsonl(target_dir / f"{split}.jsonl")
        else:
            print("[Data] Downloading & materializing clean ViANLI from HuggingFace...")
            raw_data = _materialize_vianli(target_dir)
        print(f"  [OK] ViANLI: Train={len(raw_data['train']):,}, Dev={len(raw_data['dev']):,}, Test={len(raw_data['test']):,}")

    elif domain == "vimednli":
        candidates = [
            root / "data/external/vimednli",
            root / "vimednli",
            ROOT / "data/external/vimednli",
            root / "gated_dual_ema_msd/data/external/vimednli",
        ]
        target_dir = next((d for d in candidates if (d / "train.jsonl").exists()), root / "data/external/vimednli")
        if (target_dir / "train.jsonl").exists():
            for split in ("train", "dev", "test"):
                raw_data[split] = _read_jsonl(target_dir / f"{split}.jsonl")
        else:
            print("[Data] Downloading & materializing clean ViMedNLI...")
            raw_data = _materialize_vimednli(target_dir)
        print(f"  [OK] ViMedNLI: Train={len(raw_data['train']):,}, Dev={len(raw_data['dev']):,}, Test={len(raw_data['test']):,}")

    elif domain == "vinli":
        candidates = [
            root / "data/external/vinli",
            root / "vinli",
            ROOT / "data/external/vinli",
            root / "gated_dual_ema_msd/data/external/vinli",
        ]
        target_dir = next((d for d in candidates if (d / "train.jsonl").exists()), root / "data/external/vinli")
        if (target_dir / "train.jsonl").exists():
            for split in ("train", "dev", "test"):
                raw_data[split] = _read_jsonl(target_dir / f"{split}.jsonl")
        else:
            print("[Data] Downloading & materializing ViNLI...")
            raw_data = _materialize_vinli(target_dir)
        print(f"  [OK] ViNLI: Train={len(raw_data['train']):,}, Dev={len(raw_data['dev']):,}, Test={len(raw_data['test']):,}")

    else:
        raise ValueError(f"Unknown dataset domain: '{domain}'. Choose from ['vianli', 'vimednli', 'vinli']")

    return raw_data, tokenizer


def make_dataloader(
    rows: List[dict],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        RawDictDataset(rows, tokenizer, max_len=max_length),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


build_dataloader = make_dataloader
