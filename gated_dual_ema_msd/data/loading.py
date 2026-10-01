"""Split loading and dataset materialization with search isolation."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

from gated_dual_ema_msd.config.contracts import (
    MODEL_NAME,
    MODEL_REVISION,
    parse_label,
    require_max_length,
)
from gated_dual_ema_msd.data.records import NLIRecord


def read_jsonl(file_path: pathlib.Path | str) -> List[dict]:
    """Read a JSONL file into a list of dictionaries."""
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                rows.append(json.loads(line_str))
    return rows


def read_splits(
    directory: pathlib.Path,
    dataset: str,
    *,
    include_test: bool = False,
) -> Dict[str, List[NLIRecord]]:
    """Read dataset splits ensuring test split is isolated unless explicitly requested."""
    names = ("train", "dev", "test") if include_test else ("train", "dev")
    result: Dict[str, List[NLIRecord]] = {}

    for name in names:
        split_path = directory / f"{name}.jsonl"
        rows = read_jsonl(split_path)
        formatted: List[NLIRecord] = []
        for r in rows:
            # Strictly validate label
            parse_label(r["label"], str(r.get("id", "")))
            formatted.append(
                {
                    "id": str(r.get("id", "")),
                    "premise": str(r.get("premise", "")),
                    "hypothesis": str(r.get("hypothesis", "")),
                    "label": str(r.get("label", "")),
                    "domain": dataset,
                }
            )
        result[name] = formatted

    return result


def load_nli_dataset(
    domain: str,
    root_dir: Optional[pathlib.Path] = None,
    model_name: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    revision: Optional[str] = None,
    *,
    include_test: bool = False,
) -> Tuple[Dict[str, List[dict]], AutoTokenizer]:
    """Load dataset splits with search split isolation and tokenization."""
    if (
        model_name is not None
        and tokenizer_name is not None
        and model_name != tokenizer_name
    ):
        raise ValueError(
            f"Conflicting model_name ({model_name!r}) and tokenizer_name ({tokenizer_name!r})"
        )

    # Validate domain and enforce Rule 0b max_length
    require_max_length(domain)

    chosen_model = model_name or tokenizer_name or MODEL_NAME
    chosen_rev = revision if revision is not None else (
        MODEL_REVISION if chosen_model == MODEL_NAME else None
    )

    tokenizer = AutoTokenizer.from_pretrained(
        chosen_model,
        revision=chosen_rev,
        trust_remote_code=True,
    )

    root = root_dir or pathlib.Path.cwd()
    domain_lower = domain.lower()

    if domain_lower == "vianli":
        candidates = [
            root / "data/processed/vianli_clean",
            root / "vianli_clean",
            root / "data/external/vianli",
        ]
    elif domain_lower == "vimednli":
        candidates = [
            root / "data/external/vimednli",
            root / "vimednli",
        ]
    elif domain_lower == "vinli":
        candidates = [
            root / "data/external/vinli",
            root / "vinli",
        ]
    else:
        raise ValueError(f"Unknown domain: {domain}")

    target_dir = next((d for d in candidates if (d / "train.jsonl").exists()), candidates[0])
    if not (target_dir / "train.jsonl").exists():
        raise FileNotFoundError(
            f"Dataset files for '{domain}' not found in {candidates}. "
            f"Please run data preparation scripts before loading."
        )

    splits = read_splits(target_dir, domain_lower, include_test=include_test)
    return splits, tokenizer
