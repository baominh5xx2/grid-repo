"""Object-Oriented DataModule & Dataset for NLI Benchmarks."""
from __future__ import annotations

import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset

from gated_dual_ema_msd.data.collator import NLIDataCollator, DOMAIN_MAX_LENGTHS


class NLIDataset(Dataset):
    """Lightweight PyTorch Dataset holding in-memory sample records."""
    def __init__(self, records: List[dict], domain: Optional[str] = None):
        self.records = records
        self.domain = domain
        if domain:
            for r in self.records:
                r.setdefault("domain", domain)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


class NLIDataModule:
    """Encapsulates dataset setup, splits, tokenization, and DataLoader creation."""
    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        batch_size: int = 16,
        eval_batch_size: int = 32,
        num_workers: int = 0,
        max_length: Optional[int] = None,
    ):
        self.dataset_name = dataset_name.lower()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        # Determine domain max length
        self.max_length = max_length or DOMAIN_MAX_LENGTHS.get(self.dataset_name, 512)

        self.collator = NLIDataCollator(
            tokenizer=self.tokenizer,
            default_max_length=self.max_length,
        )

        self.train_dataset: Optional[NLIDataset] = None
        self.val_dataset: Optional[NLIDataset] = None
        self.test_dataset: Optional[NLIDataset] = None

    def setup_datasets(
        self,
        train_records: List[dict],
        val_records: List[dict],
        test_records: Optional[List[dict]] = None,
    ) -> None:
        """Initialize Dataset instances from raw record dictionaries."""
        self.train_dataset = NLIDataset(train_records, domain=self.dataset_name)
        self.val_dataset = NLIDataset(val_records, domain=self.dataset_name)
        if test_records is not None:
            self.test_dataset = NLIDataset(test_records, domain=self.dataset_name)

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("train_dataset is not initialized. Call setup_datasets() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError("val_dataset is not initialized. Call setup_datasets() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("test_dataset is not initialized. Call setup_datasets() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
