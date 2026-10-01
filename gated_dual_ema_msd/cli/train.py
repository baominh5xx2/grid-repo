"""Single-run CLI for the canonical 11-configuration experiment registry."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Sequence

import torch

from gated_dual_ema_msd.config.test_policy import allow_final_test
from gated_dual_ema_msd.data import load_nli_dataset
from gated_dual_ema_msd.experiment_registry import (
    DATASET_MAX_LENGTHS,
    EMA_DECAY,
    EMA_START_STEP,
    EXPERIMENTS,
    get_experiment,
    model_kwargs_for_experiment,
)
from gated_dual_ema_msd.models import create_nli_model
from gated_dual_ema_msd.trainer import DirectTrainer, msd_metadata, parameter_counts
from gated_dual_ema_msd.training.runtime import prepare_model, set_seed
from gated_dual_ema_msd.training.precision import bf16_enabled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one canonical baseline/ablation configuration"
    )
    parser.add_argument(
        "--experiment_id",
        default="M3_FULL",
        type=str.upper,
        choices=list(EXPERIMENTS),
    )
    parser.add_argument(
        "--dataset", default="vianli", choices=list(DATASET_MAX_LENGTHS)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--physical_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--ema_decay", type=float, default=EMA_DECAY)
    parser.add_argument("--ema_start_step", type=int, default=EMA_START_STEP)
    parser.add_argument("--output_dir", default="outputs/ablation_matrix")
    parser.add_argument(
        "--frozen_final",
        action="store_true",
        default=False,
        help="Evaluate the dev-selected checkpoint on test exactly once",
    )
    parser.add_argument(
        "--no_test",
        action="store_true",
        help="Skip evaluation on the test split",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="gated-relation-cafebert")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--hf_repo_id", default=None)
    parser.add_argument("--hf_private", action="store_true")
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Hugging Face backbone identifier (e.g. uitnlp/CafeBERT, vinai/phobert-base-v2)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        type=str,
        help="Hugging Face model commit SHA / revision",
    )
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="Fail instead of silently falling back to CPU when this is a GPU matrix job.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    experiment = get_experiment(args.experiment_id)
    if args.seed not in experiment.seeds:
        raise ValueError(
            f"Seed {args.seed} is outside the registered policy for "
            f"{experiment.experiment_id}: {list(experiment.seeds)}"
        )

    max_length = DATASET_MAX_LENGTHS[args.dataset]
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("This matrix job requires CUDA, but no CUDA device is available")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_bf16 = bf16_enabled(device)
    allow_test = allow_final_test(frozen_final=args.frozen_final, no_test=args.no_test)
    data, tokenizer = load_nli_dataset(
        args.dataset,
        model_name=args.model_name,
        revision=args.revision,
        include_test=allow_test,
    )
    model_kwargs = model_kwargs_for_experiment(
        experiment,
        sep_token_id=getattr(tokenizer, "sep_token_id", None) or 2,
        model_name=args.model_name,
        revision=args.revision,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
    )
    model = prepare_model(
        args.seed,
        lambda: create_nli_model(experiment.model_type, **model_kwargs),
    )
    run_name = f"{args.dataset}-{experiment.experiment_id.lower()}-seed{args.seed}"
    run_output = (
        pathlib.Path(args.output_dir)
        / args.dataset
        / experiment.experiment_id
        / f"seed{args.seed}"
    )

    if args.preflight:
        report = {
            "run_name": run_name,
            "dataset": args.dataset,
            "max_length": max_length,
            "train_precision": "bf16" if use_bf16 else "fp32",
            "eval_precision": "fp32",
            "experiment": experiment.to_dict(),
            "parameters": parameter_counts(model),
            "msd": msd_metadata(model),
            "split_sizes": {key: len(value) for key, value in data.items()},
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    trainer = DirectTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=run_output,
        dataset=args.dataset,
        experiment_id=experiment.experiment_id,
        model_configuration=experiment.to_dict(),
        max_length=max_length,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_epochs=args.epochs,
        eval_steps=args.eval_steps,
        patience=args.patience,
        physical_batch_size=args.physical_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        seed=args.seed,
        use_ema=experiment.use_ema,
        ema_decay=args.ema_decay,
        ema_start_step=args.ema_start_step,
        evaluate_test=allow_test,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        use_hf=bool(args.hf_repo_id),
        hf_repo_id=args.hf_repo_id,
        hf_private=args.hf_private,
    )
    trainer.train(
        data["train"],
        data["dev"],
        data.get("test") if allow_test else None,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
