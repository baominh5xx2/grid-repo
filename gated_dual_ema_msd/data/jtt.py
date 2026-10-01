"""Compute-matched Just Train Twice helpers for EXP-001-R2.

The official JTT MultiNLI recipe trains an identification model through epoch 2,
then adds four extra copies of each identification error for a five-epoch final
run.  Literal duplication can inflate compute by roughly 3x when the error set
is large.  This module preserves the corresponding weighted-ERM objective
(hard row weight 5, ordinary row weight 1) while drawing a fixed, baseline-size
number of samples per epoch with replacement.
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence


class ComputeMatchedJTTBatchMixer:
    """Deterministic fixed-compute weighted sampler for a frozen JTT error set."""

    def __init__(
        self,
        rows: Sequence[dict],
        hard_example_ids: Iterable[str],
        *,
        batch_size: int,
        hard_weight: float = 5.0,
        ordinary_weight: float = 1.0,
        seed: int = 42,
        draws_per_epoch: int | None = None,
    ) -> None:
        if not rows:
            raise ValueError("JTT rows must be non-empty")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("JTT batch_size must be a positive integer")
        if hard_weight <= ordinary_weight or ordinary_weight <= 0:
            raise ValueError("JTT requires hard_weight > ordinary_weight > 0")

        ids = [str(row.get("id", "")) for row in rows]
        if any(not sample_id for sample_id in ids):
            raise ValueError("every JTT row requires a non-empty id")
        if len(set(ids)) != len(ids):
            raise ValueError("JTT row ids must be unique")

        hard_ids = {str(sample_id) for sample_id in hard_example_ids}
        unknown = hard_ids - set(ids)
        if unknown:
            raise ValueError(f"JTT hard-example ids are absent from training rows: {sorted(unknown)[:3]}")
        if not hard_ids:
            raise ValueError("JTT identification produced an empty hard-example set")

        minimum_draws = math.ceil(len(rows) / batch_size) * batch_size
        if draws_per_epoch is None:
            draws_per_epoch = minimum_draws
        if draws_per_epoch != minimum_draws or draws_per_epoch % batch_size:
            raise ValueError(
                "JTT draws_per_epoch must equal the baseline padded epoch size "
                f"{minimum_draws}, got {draws_per_epoch}"
            )

        self.rows = list(rows)
        self.batch_size = batch_size
        self.draws_per_epoch = draws_per_epoch
        self.hard_example_ids = frozenset(hard_ids)
        self.hard_weight = float(hard_weight)
        self.ordinary_weight = float(ordinary_weight)
        self._rng = random.Random(seed)
        self._weights = [
            self.hard_weight if sample_id in self.hard_example_ids else self.ordinary_weight
            for sample_id in ids
        ]

    def epoch_steps(self) -> int:
        return self.draws_per_epoch // self.batch_size

    @property
    def expected_hard_draw_fraction(self) -> float:
        hard_mass = len(self.hard_example_ids) * self.hard_weight
        ordinary_mass = (len(self.rows) - len(self.hard_example_ids)) * self.ordinary_weight
        return hard_mass / (hard_mass + ordinary_mass)

    def next_batch_rows(self) -> list[dict]:
        indices = self._rng.choices(
            range(len(self.rows)), weights=self._weights, k=self.batch_size
        )
        return [self.rows[index] for index in indices]

    def batches(self, n_steps: int | None = None):
        if n_steps is None:
            n_steps = self.epoch_steps()
        for _ in range(n_steps):
            yield self.next_batch_rows()


def frozen_error_ids(
    prediction_rows: Sequence[Mapping[str, object]],
    training_rows: Sequence[Mapping[str, object]],
) -> frozenset[str]:
    """Validate full train inference and return IDs where argmax != gold."""
    expected = {str(row.get("id", "")): str(row.get("label", "")) for row in training_rows}
    if "" in expected or len(expected) != len(training_rows):
        raise ValueError("training rows require unique non-empty ids")
    if len(prediction_rows) != len(training_rows):
        raise ValueError(
            f"JTT identification must predict every training row: "
            f"{len(prediction_rows)} != {len(training_rows)}"
        )

    seen: set[str] = set()
    errors: set[str] = set()
    valid_labels = {"E", "C", "N"}
    for row in prediction_rows:
        sample_id = str(row.get("sample_id", ""))
        gold = str(row.get("gold_label", ""))
        pred = str(row.get("pred_label", ""))
        if sample_id not in expected:
            raise ValueError(f"unknown JTT identification sample_id: {sample_id}")
        if sample_id in seen:
            raise ValueError(f"duplicate JTT identification sample_id: {sample_id}")
        if gold != expected[sample_id]:
            raise ValueError(
                f"JTT identification gold-label mismatch for {sample_id}: "
                f"{gold} != {expected[sample_id]}"
            )
        if pred not in valid_labels:
            raise ValueError(f"invalid JTT predicted label for {sample_id}: {pred}")
        seen.add(sample_id)
        if pred != gold:
            errors.add(sample_id)

    missing = set(expected) - seen
    if missing:
        raise ValueError(f"JTT identification predictions missing rows: {sorted(missing)[:3]}")
    if not errors:
        raise ValueError("JTT identification produced no errors; weighted phase would be a no-op")
    return frozenset(errors)
