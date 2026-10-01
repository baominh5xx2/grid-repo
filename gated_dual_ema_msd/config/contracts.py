"""Shared Non-Negotiable Contracts and Immutable Provenance.

Rule 0b:
- ViNLI = 512
- ViANLI = 512
- ViMedNLI = 256

Canonical label mapping:
0 = Entailment (E)
1 = Contradiction (C)
2 = Neutral (N)
"""
from __future__ import annotations

from typing import Any, Mapping

DATASET_MAX_LENGTHS: Mapping[str, int] = {
    "vinli": 512,
    "vianli": 512,
    "vimednli": 256,
}

LABELS = ["E", "C", "N"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

MODEL_NAME = "uitnlp/CafeBERT"
MODEL_REVISION = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d"

DATASET_REVISIONS: Mapping[str, str] = {
    "vianli": "0fec8d6ecb043a61c609f9b51f80401fdf1e84d3",
    "vinli": "47bd78ac5d075bd00a3cb4cdd3ede4eec4acf8c2",
    "vimednli": "2cd94305ba48ae1ccf8782c1df9819ddad7f035f",
}

STILTS_R1_CHECKPOINT = "trinhtrantran122/hier-nli-e-first-flat-cafebert-vinli"
STILTS_R1_REVISION = "3ae7df0b009d14ece648dbdf4cfd88d7ad37f570"


def require_max_length(dataset: str, requested: int | None = None) -> int:
    """Validate and return canonical max_length according to Rule 0b."""
    key = str(dataset).lower()
    if key not in DATASET_MAX_LENGTHS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from {list(DATASET_MAX_LENGTHS.keys())}"
        )
    expected = DATASET_MAX_LENGTHS[key]
    if requested is not None and requested != expected:
        raise ValueError(
            f"max_length violation: {dataset} expected {expected}, got {requested}"
        )
    return expected


def parse_label(value: Any, sample_id: str = "") -> int:
    """Parse and validate an NLI label strictly into canonical int {0, 1, 2}.

    Rejects booleans, floats, out-of-range ints, and unknown strings.
    """
    if isinstance(value, bool):
        raise ValueError(f"Illegal NLI label (boolean) for {sample_id}: {value!r}")
    if isinstance(value, int) and value in (0, 1, 2):
        return value
    if isinstance(value, float):
        raise ValueError(f"Illegal NLI label (float) for {sample_id}: {value!r}")

    aliases = {
        "entailment": 0,
        "contradiction": 1,
        "neutral": 2,
        "e": 0,
        "c": 1,
        "n": 2,
        "0": 0,
        "1": 1,
        "2": 2,
    }
    key = str(value).strip().lower()
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Illegal NLI label for {sample_id}: {value!r}")
