"""Evaluation CLI Script for Gated-Dual CafeBERT models."""
from __future__ import annotations

import argparse
import json
import pathlib
import time
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
from gated_dual_ema_msd.config.contracts import DATASET_REVISIONS, MODEL_REVISION
from gated_dual_ema_msd.training.r2_runtime import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained NLI model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--dataset", type=str, default="vianli", choices=["vianli", "vimednli", "vinli"])
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--experiment_id", type=str.upper, default="M3_FULL", choices=list(EXPERIMENTS))
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None, help="Evaluation ledger; required implicitly alongside final CSV")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--resume_unstarted", action="store_true", help="Retry matching setup ledger only if no test inference started")
    parser.add_argument("--frozen_final", action="store_true", help="Required for test evaluation")
    parser.add_argument("--no_test", action="store_true", help="Skip evaluation on the test split")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.split == "test" and not allow_final_test(frozen_final=args.frozen_final, no_test=args.no_test):
        raise ValueError("Test is locked. Pass --frozen_final only for an approved final evaluation.")
    if args.split == "test" and not (args.output_csv or args.output_json):
        raise ValueError("Final test requires an output ledger (--output_csv or --output_json)")
    ledger_path = pathlib.Path(args.output_json) if args.output_json else (pathlib.Path(args.output_csv).with_suffix(".json") if args.output_csv else None)
    if args.split == "test" and ledger_path.exists() and not args.resume_unstarted:
        raise ValueError(f"Final evaluation ledger already exists: {ledger_path}")
    max_length = DATASET_MAX_LENGTHS[args.dataset]

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("Final evaluation requires a CUDA runtime")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ledger = dict(state="initializing", dataset=args.dataset, split=args.split, experiment_id=args.experiment_id,
                  checkpoint_sha256=sha256_file(pathlib.Path(args.checkpoint)), model_revision=MODEL_REVISION,
                  data_revision=DATASET_REVISIONS[args.dataset], max_length=max_length,
                  eval_precision="fp32", test_evaluations=0, prior_test_exposure=True if args.dataset == "vinli" else None)
    if ledger_path:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        resuming = args.split == "test" and ledger_path.exists()
        if resuming:
            previous = json.loads(ledger_path.read_text(encoding="utf-8"))
            if previous != ledger:
                raise ValueError("Final evaluation already started or setup provenance differs; refusing rerun")
        # Exclusive creation reserves the output before expensive model loading.
        if not resuming:
            mode = "x" if args.split == "test" else "w"
            with ledger_path.open(mode, encoding="utf-8") as stream:
                json.dump(ledger, stream, indent=2)
    data, tokenizer = load_nli_dataset(args.dataset, include_test=args.split == "test")

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
    started = time.perf_counter()
    if ledger_path:
        ledger.update(state="evaluating", test_evaluations=1 if args.split == "test" else 0)
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    metrics, df = evaluate_model(model, data[args.split], tokenizer, device, max_length=max_length)

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    if args.output_csv:
        out_p = pathlib.Path(args.output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"\nPredictions exported to: {out_p}")
        ledger["prediction_sha256"] = sha256_file(out_p)
    ledger.update(state="complete", metrics=metrics, row_count=len(df), elapsed_seconds=time.perf_counter()-started)
    if ledger_path:
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
