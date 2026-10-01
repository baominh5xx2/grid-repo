"""Phased launcher for the canonical baseline/ablation experiment matrix.

The two notebook cohorts map directly to CLI cohorts:
  - seed42: 11 configurations for one selected dataset
  - robustness: M0--M3 x seeds {2024, 3407} for one selected dataset
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import torch

from gated_dual_ema_msd.data import load_nli_dataset
from gated_dual_ema_msd.experiment_registry import (
    DATASET_MAX_LENGTHS,
    EMA_DECAY,
    EMA_START_STEP,
    EXPERIMENTS,
    MAIN_EXPERIMENTS,
    SIGNAL_EXPERIMENTS,
    get_experiment,
    iter_matrix,
    model_kwargs_for_experiment,
)
from gated_dual_ema_msd.models import create_nli_model
from gated_dual_ema_msd.trainer import DirectTrainer
from gated_dual_ema_msd.training.runtime import prepare_model
from gated_dual_ema_msd.training.precision import bf16_enabled


PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[1]
PROJECT_PARENT = PACKAGE_DIR.parent


@dataclass(frozen=True)
class MatrixJob:
    dataset: str
    experiment_id: str
    seed: int

    @property
    def run_name(self) -> str:
        return f"{self.dataset}-{self.experiment_id.lower()}-seed{self.seed}"

    def output_dir(self, root: pathlib.Path) -> pathlib.Path:
        return root / self.dataset / self.experiment_id / f"seed{self.seed}"


def build_jobs(cohort: str, datasets: Sequence[str]) -> List[MatrixJob]:
    cohort = cohort.lower()
    if cohort == "signal":
        tuples = [
            (dataset, experiment_id, 42)
            for dataset in datasets
            for experiment_id in SIGNAL_EXPERIMENTS
        ]
    elif cohort == "seed42":
        tuples = [
            (dataset, experiment_id, 42)
            for dataset in datasets
            for experiment_id in EXPERIMENTS
        ]
    elif cohort == "robustness":
        tuples = [
            (dataset, experiment_id, seed)
            for dataset in datasets
            for experiment_id in MAIN_EXPERIMENTS
            for seed in (2024, 3407)
        ]
    elif cohort in ("all", "multiseed"):
        tuples = iter_matrix(datasets=datasets, experiment_ids=EXPERIMENTS)
    else:
        raise ValueError(f"Unsupported cohort={cohort!r}")
    return [MatrixJob(*item) for item in tuples]


def _read_result(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(result: Dict[str, Any], split: str, name: str) -> Optional[float]:
    values = result.get(split)
    if not values:
        return None
    value = values.get(name)
    return float(value) if value is not None else None


def seed42_signal_report(output_root: pathlib.Path, dataset: str) -> Dict[str, Any]:
    results: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for experiment_id in SIGNAL_EXPERIMENTS:
        path = output_root / dataset / experiment_id / "seed42" / "result.json"
        if path.exists():
            results[experiment_id] = _read_result(path)
        else:
            missing.append(str(path))
    if missing:
        return {"approved": False, "reason": "missing seed-42 signal results", "missing": missing}
    b0 = float(results["B0_CLS"]["best_dev_macro_f1"])
    m0 = float(results["M0_RELATION_GATE"]["best_dev_macro_f1"])
    m3 = float(results["M3_FULL"]["best_dev_macro_f1"])
    return {
        "approved": bool(m0 > b0 and m3 >= m0),
        "dataset": dataset,
        "B0_CLS": b0,
        "M0_RELATION_GATE": m0,
        "M3_FULL": m3,
        "criteria": "M0 > B0 and M3 >= M0 on dev Macro-F1",
    }


def validate_robustness_gate(
    output_root: pathlib.Path,
    datasets: Sequence[str],
    signal_approved: bool,
) -> None:
    reports = [seed42_signal_report(output_root, dataset) for dataset in datasets]
    if all(report.get("approved") for report in reports):
        return
    if signal_approved:
        # Explicit operator approval supports the separate-notebook workflow
        # where seed-42 artifacts may live in another immutable store.
        return
    raise RuntimeError(
        "Robustness cohort is gated until seed-42 signal is established. "
        "Provide the seed-42 result directories or pass --signal_approved "
        "after verifying them externally. Reports:\n"
        + json.dumps(reports, indent=2)
    )


def collect_results(output_root: pathlib.Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in output_root.glob("*/*/seed*/result.json"):
        result = _read_result(path)
        rows.append(
            {
                "dataset": result.get("dataset"),
                "experiment_id": result.get("experiment_id"),
                "seed": result.get("seed"),
                "selected_weight_source": result.get("selected_weight_source"),
                "dev_macro_f1": _metric(result, "final_dev", "macro_f1"),
                "dev_accuracy": _metric(result, "final_dev", "accuracy"),
                "test_macro_f1": _metric(result, "test", "macro_f1"),
                "test_accuracy": _metric(result, "test", "accuracy"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "result_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def write_summaries(output_root: pathlib.Path) -> None:
    runs = collect_results(output_root)
    if runs.empty:
        return
    runs = runs.sort_values(["dataset", "experiment_id", "seed"])
    runs.to_csv(output_root / "all_runs.csv", index=False)
    summary = (
        runs.groupby(["dataset", "experiment_id"], dropna=False)
        .agg(
            seeds=("seed", "count"),
            dev_macro_f1_mean=("dev_macro_f1", "mean"),
            dev_macro_f1_std=("dev_macro_f1", "std"),
            dev_accuracy_mean=("dev_accuracy", "mean"),
            test_macro_f1_mean=("test_macro_f1", "mean"),
            test_macro_f1_std=("test_macro_f1", "std"),
            test_accuracy_mean=("test_accuracy", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_root / "paper_summary.csv", index=False)


def write_vianli_error_analysis(output_root: pathlib.Path) -> None:
    frames: Dict[str, pd.DataFrame] = {}
    for experiment_id in ("B0_CLS", "M0_RELATION_GATE", "M3_FULL"):
        path = output_root / "vianli" / experiment_id / "seed42" / "test_predictions.csv"
        if not path.exists():
            return
        frame = pd.read_csv(path)
        keep = [
            "sample_id",
            "premise",
            "hypothesis",
            "gold_label",
            "pred_label",
            "prob_E",
            "prob_C",
            "prob_N",
            "gate_mean",
        ]
        frame = frame[[column for column in keep if column in frame.columns]].copy()
        rename = {
            column: f"{experiment_id}_{column}"
            for column in frame.columns
            if column not in ("sample_id", "premise", "hypothesis", "gold_label")
        }
        frames[experiment_id] = frame.rename(columns=rename)

    merged = frames["B0_CLS"]
    for experiment_id in ("M0_RELATION_GATE", "M3_FULL"):
        merged = merged.merge(
            frames[experiment_id],
            on=["sample_id", "premise", "hypothesis", "gold_label"],
            how="inner",
            validate="one_to_one",
        )
    analysis_dir = output_root / "vianli" / "error_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(analysis_dir / "B0_M0_M3_predictions.csv", index=False)
    b0_correct = merged["B0_CLS_pred_label"] == merged["gold_label"]
    m3_correct = merged["M3_FULL_pred_label"] == merged["gold_label"]
    merged[~b0_correct & m3_correct].to_csv(
        analysis_dir / "B0_wrong_M3_correct.csv", index=False
    )
    merged[b0_correct & ~m3_correct].to_csv(
        analysis_dir / "B0_correct_M3_wrong.csv", index=False
    )
    merged[~b0_correct & ~m3_correct].to_csv(
        analysis_dir / "both_wrong.csv", index=False
    )


def run_jobs_in_process(
    jobs: Sequence[MatrixJob],
    *,
    output_root: pathlib.Path,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    label_smoothing: float,
    dropout: float,
    epochs: int,
    eval_steps: int,
    patience: int,
    physical_batch_size: int,
    grad_accum: int,
    frozen_final: bool,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    hf_repo_prefix: Optional[str],
    hf_private: bool,
    rerun: bool,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bf16_enabled(device)
    cached_data: Dict[str, Any] = {}
    cached_tokenizers: Dict[str, Any] = {}

    for index, job in enumerate(jobs, start=1):
        run_output = job.output_dir(output_root)
        result_path = run_output / "result.json"
        if result_path.exists() and not rerun:
            print(f"[{index}/{len(jobs)}] SKIP completed {job.run_name}")
            continue
        if job.dataset not in cached_data:
            cached_data[job.dataset], cached_tokenizers[job.dataset] = load_nli_dataset(
                job.dataset
            )
        data = cached_data[job.dataset]
        tokenizer = cached_tokenizers[job.dataset]
        experiment = get_experiment(job.experiment_id)
        model = prepare_model(
            job.seed,
            lambda: create_nli_model(
                experiment.model_type,
                **model_kwargs_for_experiment(
                    experiment,
                    sep_token_id=getattr(tokenizer, "sep_token_id", None) or 2,
                    label_smoothing=label_smoothing,
                    dropout=dropout,
                ),
            ),
        )
        hf_repo_id = None
        if hf_repo_prefix:
            hf_repo_id = (
                f"{hf_repo_prefix}-{job.dataset}-"
                f"{job.experiment_id.lower().replace('_', '-')}-seed{job.seed}"
            )
        print(f"[{index}/{len(jobs)}] RUN {job.run_name}", flush=True)
        trainer = DirectTrainer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            output_dir=run_output,
            dataset=job.dataset,
            experiment_id=job.experiment_id,
            model_configuration=experiment.to_dict(),
            max_length=DATASET_MAX_LENGTHS[job.dataset],
            lr=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            max_epochs=epochs,
            eval_steps=eval_steps,
            patience=patience,
            physical_batch_size=physical_batch_size,
            gradient_accumulation_steps=grad_accum,
            seed=job.seed,
            use_ema=experiment.use_ema,
            ema_decay=EMA_DECAY,
            ema_start_step=EMA_START_STEP,
            evaluate_test=frozen_final,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            use_hf=bool(hf_repo_id),
            hf_repo_id=hf_repo_id,
            hf_private=hf_private,
        )
        trainer.train(
            data["train"],
            data["dev"],
            data["test"] if frozen_final else None,
            run_name=job.run_name,
        )
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_summaries(output_root)
    write_summaries(output_root)
    if frozen_final:
        write_vianli_error_analysis(output_root)


def _estimated_cost(job: MatrixJob) -> float:
    """A deterministic longest-job-first heuristic for five GPU slots.

    Every model is trained independently, so cross-dataset mixing is safe.
    Ordering larger datasets first avoids leaving one long ViNLI run as the
    lone tail after the other four slots go idle.
    """
    dataset_weight = {"vinli": 1.0, "vimednli": 0.63, "vianli": 0.45}
    relation_weight = 1.08 if job.experiment_id != "B0_CLS" else 1.0
    msd_weight = 1.04 if job.experiment_id in {"M1_RELATION_MSD", "M3_FULL"} else 1.0
    return dataset_weight[job.dataset] * relation_weight * msd_weight


def _job_command(
    job: MatrixJob,
    *,
    output_root: pathlib.Path,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    label_smoothing: float,
    dropout: float,
    epochs: int,
    eval_steps: int,
    patience: int,
    physical_batch_size: int,
    grad_accum: int,
    frozen_final: bool,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    hf_repo_prefix: Optional[str],
    hf_private: bool,
    rerun: bool,
) -> List[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "gated_dual_ema_msd.train",
        "--experiment_id",
        job.experiment_id,
        "--dataset",
        job.dataset,
        "--seed",
        str(job.seed),
        "--output_dir",
        str(output_root),
        "--lr",
        str(lr),
        "--weight_decay",
        str(weight_decay),
        "--warmup_ratio",
        str(warmup_ratio),
        "--label_smoothing",
        str(label_smoothing),
        "--dropout",
        str(dropout),
        "--epochs",
        str(epochs),
        "--eval_steps",
        str(eval_steps),
        "--patience",
        str(patience),
        "--physical_batch_size",
        str(physical_batch_size),
        "--grad_accum",
        str(grad_accum),
        "--wandb_project",
        wandb_project,
        "--require_cuda",
    ]
    if frozen_final:
        command.append("--frozen_final")
    else:
        command.append("--no_test")
    if not use_wandb:
        command.append("--no_wandb")
    elif wandb_entity:
        command.extend(["--wandb_entity", wandb_entity])
    if hf_repo_prefix:
        repo_id = (
            f"{hf_repo_prefix}-{job.dataset}-"
            f"{job.experiment_id.lower().replace('_', '-')}-seed{job.seed}"
        )
        command.extend(["--hf_repo_id", repo_id])
        if hf_private:
            command.append("--hf_private")
    return command


def run_jobs_parallel(
    jobs: Sequence[MatrixJob],
    *,
    output_root: pathlib.Path,
    parallel_jobs: int,
    poll_seconds: float,
    **kwargs: Any,
) -> None:
    """Run independent train processes in up to ``parallel_jobs`` GPU slots.

    A process owns exactly one model, optimizer, W&B run, and output directory.
    This is intentionally process-based rather than threaded: CUDA allocator
    state and random seeds cannot leak between experimental cells.
    """
    if parallel_jobs < 1:
        raise ValueError("parallel_jobs must be >= 1")
    if parallel_jobs == 1:
        run_jobs_in_process(jobs, output_root=output_root, **kwargs)
        return

    pending = sorted(jobs, key=_estimated_cost, reverse=True)
    running: Dict[int, tuple[subprocess.Popen[bytes], MatrixJob, Any]] = {}
    failures: List[str] = []
    base_env = os.environ.copy()
    # Five Python trainers otherwise oversubscribe the Colab CPU with tokenizers
    # and BLAS workers, starving the one GPU we want to keep busy.
    base_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    base_env.setdefault("OMP_NUM_THREADS", "2")
    base_env.setdefault("MKL_NUM_THREADS", "2")

    def launch(job: MatrixJob) -> None:
        run_dir = job.output_dir(output_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_handle = (run_dir / "worker.log").open("wb")
        command = _job_command(job, output_root=output_root, **kwargs)
        process = subprocess.Popen(
            command,
            cwd=PROJECT_PARENT,
            env=base_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        running[process.pid] = (process, job, log_handle)
        print(
            f"[parallel {len(running)}/{parallel_jobs}] START {job.run_name} "
            f"pid={process.pid}",
            flush=True,
        )

    while pending or running:
        while pending and len(running) < parallel_jobs:
            job = pending.pop(0)
            result_path = job.output_dir(output_root) / "result.json"
            if result_path.exists() and not kwargs["rerun"]:
                print(f"[parallel] SKIP completed {job.run_name}", flush=True)
                continue
            launch(job)

        if not running:
            continue
        time.sleep(poll_seconds)
        for pid, (process, job, log_handle) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del running[pid]
            if return_code != 0:
                failures.append(
                    f"{job.run_name} exited {return_code}; inspect "
                    f"{job.output_dir(output_root) / 'worker.log'}"
                )
                continue
            result_path = job.output_dir(output_root) / "result.json"
            if not result_path.exists():
                failures.append(
                    f"{job.run_name} exited 0 without result.json; inspect "
                    f"{job.output_dir(output_root) / 'worker.log'}"
                )
                continue
            print(f"[parallel] DONE {job.run_name}", flush=True)
            write_summaries(output_root)

        if failures:
            # Do not keep charging GPU time once an isolated run failed. The
            # surviving workers finish their current atomic run; nothing new is
            # admitted, preserving their output and logs for diagnosis.
            pending.clear()

    write_summaries(output_root)
    if kwargs["frozen_final"]:
        write_vianli_error_analysis(output_root)
    if failures:
        raise RuntimeError("Parallel matrix failed:\n" + "\n".join(failures))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical experiment matrix")
    parser.add_argument(
        "--cohort", choices=["signal", "seed42", "robustness", "all", "multiseed"], default="signal"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASET_MAX_LENGTHS), default=["vianli"]
    )
    parser.add_argument("--output_dir", default="outputs/ablation_matrix")
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument("--signal_approved", action="store_true")
    parser.add_argument("--frozen_final", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--physical_batch_size",
        type=int,
        default=4,
        help="Per-process batch size for each isolated GPU trainer.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Keeps effective batch size at 16 when physical_batch_size=4.",
    )
    parser.add_argument(
        "--parallel_jobs",
        type=int,
        default=5,
        help="Maximum isolated trainer processes sharing CUDA:0 (1 disables parallelism).",
    )
    parser.add_argument("--poll_seconds", type=float, default=5.0)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="gated-relation-cafebert")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--hf_repo_prefix", default=None)
    parser.add_argument("--hf_private", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    jobs = build_jobs(args.cohort, args.datasets)
    print(
        json.dumps(
            [
                {"dataset": job.dataset, "experiment_id": job.experiment_id, "seed": job.seed}
                for job in jobs
            ],
            indent=2,
        )
    )
    print(f"Total jobs: {len(jobs)}")
    if args.plan_only:
        return

    output_root = pathlib.Path(args.output_dir)
    common_kwargs = dict(
        output_root=output_root,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        epochs=args.epochs,
        eval_steps=args.eval_steps,
        patience=args.patience,
        physical_batch_size=args.physical_batch_size,
        grad_accum=args.grad_accum,
        frozen_final=args.frozen_final,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        hf_repo_prefix=args.hf_repo_prefix,
        hf_private=args.hf_private,
        rerun=args.rerun,
    )

    def execute(ready_jobs: Sequence[MatrixJob]) -> None:
        run_jobs_parallel(
            ready_jobs,
            parallel_jobs=args.parallel_jobs,
            poll_seconds=args.poll_seconds,
            **common_kwargs,
        )

    if args.cohort == "robustness":
        validate_robustness_gate(output_root, args.datasets, args.signal_approved)
        execute(jobs)
    elif args.cohort == "all":
        # First spend two five-slot waves only on the evidence that decides
        # whether the rest of the matrix is scientifically worth the GPU time.
        signal_jobs = [
            job
            for job in jobs
            if job.seed == 42 and job.experiment_id in SIGNAL_EXPERIMENTS
        ]
        execute(signal_jobs)
        validate_robustness_gate(output_root, args.datasets, args.signal_approved)
        seed42_jobs = [
            job
            for job in jobs
            if job.seed == 42 and job.experiment_id not in SIGNAL_EXPERIMENTS
        ]
        robustness_jobs = [job for job in jobs if job.seed != 42]
        execute(seed42_jobs)
        execute(robustness_jobs)
    else:
        execute(jobs)


if __name__ == "__main__":
    main()
