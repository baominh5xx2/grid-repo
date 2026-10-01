"""Evaluation CLI Script for Gated-Dual CafeBERT models."""
from __future__ import annotations

import argparse
import json
import pathlib
import torch
import pandas as pd

from typing import Sequence

from gated_dual_ema_msd.config.test_policy import allow_final_test
from gated_dual_ema_msd.data import load_nli_dataset
from gated_dual_ema_msd.experiment_registry import (
    DATASET_MAX_LENGTHS,
    EXPERIMENTS,
    get_experiment,
    model_kwargs_for_experiment,
)
from gated_dual_ema_msd.models.factory import create_nli_model
from gated_dual_ema_msd.trainer import evaluate_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained NLI model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--dataset", type=str, default="vianli", choices=["vianli", "vimednli", "vinli"])
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--experiment_id", type=str.upper, default="M3_FULL", choices=list(EXPERIMENTS))
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--frozen_final", action="store_true", help="Required for test evaluation")
    parser.add_argument("--no_test", action="store_true", help="Skip evaluation on the test split")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.split == "test" and not allow_final_test(frozen_final=args.frozen_final, no_test=args.no_test):
        raise ValueError("Test is locked. Pass --frozen_final only for an approved final evaluation.")
    max_length = DATASET_MAX_LENGTHS[args.dataset]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data, tokenizer = load_nli_dataset(args.dataset)

    experiment = get_experiment(args.experiment_id)
    model = create_nli_model(
        experiment.model_type,
        **model_kwargs_for_experiment(
            experiment, sep_token_id=getattr(tokenizer, "sep_token_id", None) or 2
        ),
    ).to(device)
    try:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"\nEvaluating {args.experiment_id} on {args.dataset.upper()} ({args.split} split, max_length={max_length})...")
    metrics, df = evaluate_model(model, data[args.split], tokenizer, device, max_length=max_length)

    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    if args.output_csv:
        out_p = pathlib.Path(args.output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"\n✓ Predictions exported to: {out_p}")


if __name__ == "__main__":
    main()
