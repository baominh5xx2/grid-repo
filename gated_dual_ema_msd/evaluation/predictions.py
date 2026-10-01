"""Predictions schema validation and sanity assertions."""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

from gated_dual_ema_msd.config.contracts import LABEL2ID


def validate_predictions(frame: pd.DataFrame, expected_ids: Sequence[str]) -> None:
    """Validate prediction dataframe schema, order, uniqueness, finite logits, and argmax match."""
    req_cols = {"sample_id", "gold_label", "pred_label", "logit_E", "logit_C", "logit_N"}
    missing = req_cols - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required prediction column(s): {missing}")

    if frame["sample_id"].tolist() != list(expected_ids):
        raise ValueError("prediction ID order/count mismatch")

    if not frame["sample_id"].is_unique:
        raise ValueError("duplicate prediction IDs")

    for col in ("gold_label", "pred_label"):
        if not set(frame[col]).issubset({"E", "C", "N"}):
            raise ValueError(f"illegal label: {col}")

    logits = frame[["logit_E", "logit_C", "logit_N"]].to_numpy(dtype=float)
    if not np.isfinite(logits).all():
        raise ValueError("non-finite logits")

    predicted = np.array([LABEL2ID[x] for x in frame["pred_label"]])
    if not np.array_equal(logits.argmax(1), predicted):
        raise ValueError("pred != argmax(logits)")
