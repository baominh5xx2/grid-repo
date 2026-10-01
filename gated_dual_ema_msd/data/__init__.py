"""Package component data."""
from __future__ import annotations

from gated_dual_ema_msd.data.dataset import (
    LABELS,
    LABEL2ID,
    ID2LABEL,
    MultiSourceBatchMixer,
    collate_rows,
    load_jsonl,
)
from gated_dual_ema_msd.data.collator import NLIDataCollator
from gated_dual_ema_msd.data.records import NLIRecord
from gated_dual_ema_msd.data.loading import (
    read_jsonl,
    read_splits,
    load_nli_dataset,
)
from gated_dual_ema_msd.data.loader_adapters import (
    RawDictDataset,
    build_dataloader,
    make_dataloader,
)

__all__ = [
    "LABELS",
    "LABEL2ID",
    "ID2LABEL",
    "MultiSourceBatchMixer",
    "collate_rows",
    "load_jsonl",
    "NLIDataCollator",
    "NLIRecord",
    "read_jsonl",
    "read_splits",
    "RawDictDataset",
    "build_dataloader",
    "make_dataloader",
    "load_nli_dataset",
]
