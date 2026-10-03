"""Explicit test-aware exploration; peaks are selected on the test set itself."""
from __future__ import annotations

import json
import math
import pathlib

import pandas as pd

from gated_dual_ema_msd.cli.matrix import MatrixJob
from gated_dual_ema_msd.evaluation.predictions import validate_predictions
from gated_dual_ema_msd.operations.architecture_search import SCREENING_METHODS, _linked_manifests
from gated_dual_ema_msd.operations.notebook_batch import manifest_digest, validate_run
from gated_dual_ema_msd.training.direct import compute_metrics
from gated_dual_ema_msd.tracking.hf import sha256_file


def verified_peak_result(root: pathlib.Path, method: str, seed: int) -> dict:
    root = pathlib.Path(root)
    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("test_locked") is not False or manifest.get("selection_policy") != "exploratory_test_macro_f1"
            or manifest.get("config", {}).get("test_peak_exploratory") is not True
            or manifest.get("config", {}).get("frozen_final") is not False):
        raise ValueError("Peak selection requires an explicit test-aware exploratory manifest")
    job = MatrixJob("vinli", method, seed)
    folder = job.output_dir(root)
    result = json.loads((folder / "verified_run.json").read_text(encoding="utf-8"))
    validate_run(result, job, test_peak_exploratory=True)
    revision = result.get("hf_revision", "")
    if (not result.get("artifact_readback_verified") or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
            or result.get("batch_manifest_sha256") != manifest_digest(manifest)
            or result.get("git_sha") != manifest["git_sha"]
            or not result.get("hf_exploratory_peak_checkpoint_path")):
        raise ValueError("Unverified peak checkpoint or mismatched study provenance")
    digest = result.get("hf_exploratory_peak_checkpoint_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Missing verified peak checkpoint digest")
    metrics = result.get("peak_test_metrics", {})
    for key in ("macro_f1", "accuracy", "f1_E", "f1_C", "f1_N"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Missing/invalid peak test metric: {key}")
    if (result.get("test_curve_sha256") != sha256_file(folder / "test_curve.csv")
            or result.get("peak_prediction_sha256") != sha256_file(folder / "test_predictions_peak.csv")):
        raise ValueError("Test curve/peak prediction hash mismatch")
    curve = pd.read_csv(folder / "test_curve.csv")
    values = curve.test_macro_f1
    if (len(curve) != result["test_evaluations"] - 1 or curve.empty
            or not all(math.isfinite(v) and 0 <= v <= 1 for v in values)
            or not curve.optimizer_step.is_unique or not curve.optimizer_step.is_monotonic_increasing
            or set(curve.weight_source) != {"ema"}):
        raise ValueError("Invalid test peak curve")
    peak = curve.loc[values.idxmax()]
    if (int(peak.optimizer_step) != result["peak_test_step"]
            or not math.isclose(float(peak.test_macro_f1), metrics["macro_f1"], abs_tol=1e-10)
            or not math.isclose(result["peak_test_macro_f1"], metrics["macro_f1"], abs_tol=1e-10)):
        raise ValueError("Recorded test peak differs from its curve")
    metadata_path = folder / "test_peak.json"
    if result.get("peak_metadata_sha256") != sha256_file(metadata_path):
        raise ValueError("Peak metadata hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = dict(optimizer_step=result["peak_test_step"], epoch=result.get("peak_test_epoch"), weight_source="ema",
                    metrics=metrics, dev_metrics_at_peak=result.get("peak_test_dev_metrics"),
                    selection_policy="exploratory_test_macro_f1")
    if (any(metadata.get(k) != v for k, v in expected.items())
            or result.get("peak_test_weight_source") != "ema" or int(peak.epoch) != expected["epoch"]):
        raise ValueError("Peak metadata differs from saved step/epoch/EMA evidence")
    frame = pd.read_csv(folder / "test_predictions_peak.csv", dtype={"sample_id": str})
    validate_predictions(frame, frame.sample_id.tolist())
    mapping = {"E": 0, "C": 1, "N": 2}
    recomputed = compute_metrics(frame.pred_label.map(mapping).to_numpy(), frame.gold_label.map(mapping).to_numpy())
    if any(not math.isclose(recomputed[k], metrics[k], abs_tol=1e-10) for k in ("macro_f1", "accuracy", "f1_E", "f1_C", "f1_N")):
        raise ValueError("Peak test metrics differ from saved predictions")
    return result


def test_peak_table(output_root: pathlib.Path) -> pd.DataFrame:
    rows = []
    for method in SCREENING_METHODS:
        if not MatrixJob("vinli", method, 42).output_dir(pathlib.Path(output_root)).joinpath("verified_run.json").exists():
            continue
        result = verified_peak_result(output_root, method, 42)
        metrics = result["peak_test_metrics"]
        rows.append(dict(experiment_id=method, seed=42, peak_test_macro_f1=metrics["macro_f1"],
                         peak_test_accuracy=metrics["accuracy"], peak_test_step=result["peak_test_step"],
                         peak_test_f1_E=metrics["f1_E"], peak_test_f1_C=metrics["f1_C"], peak_test_f1_N=metrics["f1_N"],
                         reference_dev_macro_f1=result["final_dev"]["macro_f1"],
                         dev_macro_f1_at_test_peak=result.get("peak_test_dev_metrics", {}).get("macro_f1"),
                         head_parameters=result["hparams"].get("additional_head_parameters"),
                         train_seconds_per_optimizer_step=result.get("train_seconds_per_optimizer_step"),
                         peak_gpu_memory_bytes=result.get("peak_gpu_memory_bytes"),
                         hf_repo_id=result["hf_repo_id"], hf_revision=result["hf_revision"],
                         hf_peak_checkpoint=result["hf_exploratory_peak_checkpoint_path"],
                         selection_policy="exploratory_test_macro_f1"))
    table = pd.DataFrame(rows)
    if not table.empty:
        table["_tie_order"] = table.experiment_id.map({m: i for i, m in enumerate(SCREENING_METHODS)})
        table = table.sort_values(["peak_test_macro_f1", "_tie_order"], ascending=[False, True]).drop(columns="_tie_order").reset_index(drop=True)
    return table


def select_peak_candidate(output_root: pathlib.Path) -> str:
    table = test_peak_table(output_root)
    if table.empty or set(table.experiment_id) != set(SCREENING_METHODS):
        raise ValueError("Complete all four verified seed-42 test-peak runs first")
    # Exact ties prefer the existing control, then deterministic registry order.
    best = table.peak_test_macro_f1.max()
    tied = set(table.loc[table.peak_test_macro_f1 == best, "experiment_id"])
    return next(method for method in SCREENING_METHODS if method in tied)


def peak_confirmation_jobs(output_root: pathlib.Path, candidate: str) -> list[MatrixJob]:
    if candidate != select_peak_candidate(output_root):
        raise ValueError("Candidate differs from the verified seed-42 test-peak winner")
    methods = list(dict.fromkeys(("M3_FULL", candidate)))
    return [MatrixJob("vinli", method, seed) for seed in (2024, 3407) for method in methods]


def peak_capacity_control_jobs(output_root: pathlib.Path) -> list[MatrixJob]:
    control = {"ARCH_ALIGN256": "ARCH_REL304", "ARCH_CONDPOOL128": "ARCH_REL230", "ARCH_REL256": "ARCH_REL512"}.get(select_peak_candidate(output_root))
    return [MatrixJob("vinli", control, 42)] if control else []


def peak_robustness_summary(screening_root: pathlib.Path, confirmation_root: pathlib.Path, candidate: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    peak_confirmation_jobs(screening_root, candidate)
    _linked_manifests(screening_root, confirmation_root, candidate,
                      protocol="vinli_architecture_test_peak_robustness")
    rows = []
    for seed in (42, 2024, 3407):
        root = screening_root if seed == 42 else confirmation_root
        for method in dict.fromkeys(("M3_FULL", candidate)):
            result = verified_peak_result(root, method, seed)
            rows.append(dict(experiment_id=method, seed=seed, peak_test_macro_f1=result["peak_test_macro_f1"],
                             peak_test_step=result["peak_test_step"], hf_repo_id=result["hf_repo_id"], hf_revision=result["hf_revision"],
                             selection_policy="exploratory_test_macro_f1"))
    runs = pd.DataFrame(rows)
    summary = runs.groupby("experiment_id").agg(seeds=("seed", "nunique"),
        peak_test_macro_f1_mean=("peak_test_macro_f1", "mean"), peak_test_macro_f1_std=("peak_test_macro_f1", "std")).reset_index()
    summary["selection_policy"] = "exploratory_test_macro_f1"
    return runs, summary
