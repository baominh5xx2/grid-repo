"""Data loading + exact-count multi-source batch mixing for Flat CafeBERT.

Every row (ViANLI, ViNLI, ViMedNLI) is a plain dict {id, premise, hypothesis, label,
domain?} loaded straight from jsonl — ViANLI rows have no "domain" key on disk (they're
always the target), so it defaults to "vianli" when tagging.

MultiSourceBatchMixer is the single mechanism used for BOTH Stage 1 (source_batch_mix)
and Stage 2 (batch_mix): every batch is built by taking EXACTLY `count` samples from
each named source, concatenated — never an average over many batches. This is what
guarantees acceptance criterion #1 ("unit test confirms source batch mixer gives exact
requested counts"); `datasets.interleave_datasets` only guarantees the ratio on average,
so it isn't used here.
"""
from __future__ import annotations
import json, pathlib, random
from typing import Dict, List, Optional, Sequence

import torch

from gated_dual_ema_msd.config.contracts import (
    DATASET_MAX_LENGTHS,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    parse_label,
    require_max_length,
)


def load_jsonl(path: str | pathlib.Path, default_domain: str) -> List[dict]:
    """Load a jsonl split and tag every row with its domain (source jsonl rows already
    carry "domain"; ViANLI rows don't, since it's always the target — defaulted here)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r.setdefault("domain", default_domain)
            rows.append(r)
    return rows


def collate_rows(rows: Sequence[dict], tokenizer, max_length: int,
                 max_lengths: Optional[Dict[str, int]] = None) -> dict:
    """Tokenize a list of raw rows into a single batch of tensors.

    max_lengths maps dataset/domain -> truncation length (e.g. {"vinli": 512,
    "vianli": 512, "vimednli": 256}); each row is truncated to its own source
    limit and the batch is padded to the longest actual sequence.
    """
    encs = []
    for r in rows:
        domain = r.get("domain")
        if domain:
            ml = require_max_length(domain)
        elif max_lengths:
            ml = max_lengths.get(domain, max_length)
        else:
            ml = max_length
        e = tokenizer(r["premise"], r["hypothesis"], truncation=True,
                      max_length=ml, return_tensors=None)
        encs.append(e)
    enc = tokenizer.pad(encs, padding="longest", return_tensors="pt")
    batch = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor([parse_label(r["label"], str(r.get("id", ""))) for r in rows], dtype=torch.long),
        "sample_id": [r["id"] for r in rows],
        "gold_label": [r["label"] for r in rows],
        "domain": [r["domain"] for r in rows],
    }
    if "token_type_ids" in enc:
        batch["token_type_ids"] = enc["token_type_ids"]
    return batch


class _SourceCycler:
    """Infinite stream of row indices for one source: shuffles once per internal pass,
    reshuffles (new RNG draw, deterministic given the seeded rng) and wraps on exhaustion
    so a source can be sampled far more times than it has rows (needed for small
    "replay" sources in Stage 2)."""

    def __init__(self, n: int, rng: random.Random):
        if n <= 0:
            raise ValueError("source has 0 rows, cannot cycle")
        self.n = n
        self.rng = rng
        self._queue: List[int] = []

    def _refill(self):
        idx = list(range(self.n))
        self.rng.shuffle(idx)
        self._queue.extend(idx)

    def next_indices(self, count: int) -> List[int]:
        out = []
        while len(out) < count:
            if not self._queue:
                self._refill()
            take = min(count - len(out), len(self._queue))
            out.extend(self._queue[:take])
            del self._queue[:take]
        return out


class MultiSourceBatchMixer:
    """Builds batches with an EXACT, fixed per-source composition.

    name_to_rows: {source_name: [row dict, ...]}
    batch_mix:    {source_name: count_in_every_batch} — every yielded batch has exactly
                  these counts from each named source, concatenated in `batch_mix`
                  iteration order (dict order, i.e. insertion order).
    primary_sources: subset of batch_mix.keys() whose full coverage defines "1 epoch".
                  Stage 1 single-source: that source. Stage 1 dual-source (50/50 mix):
                  BOTH sources (epoch = full pass of whichever needs more steps at its
                  per-batch rate, so the larger source doesn't get under-covered).
                  Stage 2: ONLY the target (e.g. vianli) — replay sources are excluded
                  from the epoch-length calc on purpose, they're meant to cycle/repeat
                  many times over a single target-driven epoch, not bound it.
    """

    def __init__(self, name_to_rows: Dict[str, List[dict]], batch_mix: Dict[str, int],
                 primary_sources: Sequence[str], seed: int):
        missing = set(batch_mix) - set(name_to_rows)
        if missing:
            raise ValueError(f"batch_mix references unknown source(s): {missing}")
        bad_primary = set(primary_sources) - set(batch_mix)
        if bad_primary:
            raise ValueError(f"primary_sources not in batch_mix: {bad_primary}")
        if not primary_sources:
            raise ValueError("primary_sources must be non-empty")

        self.name_to_rows = name_to_rows
        self.batch_mix = dict(batch_mix)  # preserve insertion order
        self.primary_sources = list(primary_sources)
        self.batch_size = sum(self.batch_mix.values())
        rng = random.Random(seed)
        self._cyclers = {name: _SourceCycler(len(name_to_rows[name]), random.Random(rng.randrange(2**31)))
                          for name in self.batch_mix}

    def epoch_steps(self) -> int:
        import math
        return max(math.ceil(len(self.name_to_rows[name]) / self.batch_mix[name]) for name in self.primary_sources)

    def next_batch_rows(self) -> List[dict]:
        """One batch worth of raw rows, exact per-source counts, source order = batch_mix order."""
        out: List[dict] = []
        for name, count in self.batch_mix.items():
            idxs = self._cyclers[name].next_indices(count)
            rows = self.name_to_rows[name]
            out.extend(rows[i] for i in idxs)
        return out

    def batches(self, n_steps: Optional[int] = None):
        """Yields raw row-lists (not yet tokenized) for n_steps batches (default: 1 epoch)."""
        if n_steps is None:
            n_steps = self.epoch_steps()
        for _ in range(n_steps):
            yield self.next_batch_rows()


def eval_batches(rows: Sequence[dict], batch_size: int):
    """Deterministic, in-order, non-overlapping batches over a full split — for dev/test
    evaluation (no mixing, no shuffling, every row seen exactly once)."""
    for i in range(0, len(rows), batch_size):
        yield rows[i:i + batch_size]
