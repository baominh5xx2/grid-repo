"""Dev-only decisions for the single-model ViNLI head study."""
from __future__ import annotations

import json
import math
import pathlib
import statistics

import pandas as pd

from gated_dual_ema_msd.cli.matrix import MatrixJob
from gated_dual_ema_msd.operations.notebook_batch import manifest_digest, validate_run

SCREENING_METHODS = ("M3_FULL", "ARCH_ALIGN256", "ARCH_REL256", "ARCH_CONDPOOL128")


def environment_contract(metadata: dict) -> dict:
    """Stable runtime fields affecting paired training/inference; no host details."""
    return {**{k: metadata.get(k) for k in ("python", "cuda_version", "gpu")},
            "packages": {k: v for k, v in metadata.get("packages", {}).items() if k != "wandb"}}


def _linked_manifests(screening_root: pathlib.Path, confirmation_root: pathlib.Path, candidate: str,
                      protocol: str = "vinli_architecture_confirmation_train_dev") -> dict:
    try:
        pilot = json.loads((pathlib.Path(screening_root) / "batch_manifest.json").read_text(encoding="utf-8"))
        paired = json.loads((pathlib.Path(confirmation_root) / "batch_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Both frozen study manifests are required for confirmation provenance") from error
    common = ("git_sha", "model_name", "model_revision", "precision", "evaluation_precision", "test_locked",
              "label_order", "datasets", "data_fingerprints", "environment_contract", "selection_policy")
    if (paired.get("screening_manifest_sha256") != manifest_digest(pilot)
            or any(key not in pilot or paired.get(key) != pilot[key] for key in common)):
        raise ValueError("Confirmation provenance differs from the frozen screening manifest")
    # Output grouping may change; every training/artifact policy value is fixed.
    recipes = [{k: v for k, v in m.get("config", {}).items() if k != "hf_prefix"} for m in (pilot, paired)]
    if not recipes[0] or recipes[0] != recipes[1]:
        raise ValueError("Confirmation recipe differs from screening")
    methods = list(dict.fromkeys(("M3_FULL", candidate)))
    expected_jobs = [dict(dataset="vinli", experiment_id=m, seed=s)
                     for s in (2024, 3407) for m in methods]
    if (paired.get("protocol") != protocol
            or paired.get("selected_candidate") != candidate or paired.get("seeds") != [2024, 3407]
            or paired.get("methods") != methods or paired.get("jobs") != expected_jobs):
        raise ValueError("Confirmation manifest does not describe the selected paired study")
    return pilot


def verified_result(root: pathlib.Path, method: str, seed: int) -> dict:
    root = pathlib.Path(root)
    job = MatrixJob("vinli", method, seed)
    path = job.output_dir(root) / "verified_run.json"
    if not path.exists():
        raise ValueError(f"Missing verified result: {job.run_name}")
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_run(result, job)
    if (not result.get("artifact_readback_verified") or len(result.get("hf_revision", "")) != 40
            or any(c not in "0123456789abcdef" for c in result["hf_revision"])
            or result.get("selection_policy") != "dev_macro_f1"
            or result.get("selected_weight_source") != "ema"):
        raise ValueError(f"Unverified or incorrectly selected result: {job.run_name}")
    manifest = root / "batch_manifest.json"
    if manifest.exists() and result.get("batch_manifest_sha256") != manifest_digest(json.loads(manifest.read_text())):
        raise ValueError("Result differs from its frozen batch manifest")
    for key in ("macro_f1", "f1_E", "f1_C", "f1_N"):
        value = result.get("final_dev", {}).get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Missing/invalid dev metric: {job.run_name}/{key}")
    return result


def screening_table(output_root: pathlib.Path) -> pd.DataFrame:
    """Return completed pilot results; require all four before advancing."""
    records = {}
    for method in SCREENING_METHODS:
        path = MatrixJob("vinli", method, 42).output_dir(pathlib.Path(output_root)) / "verified_run.json"
        if path.exists():
            records[method] = verified_result(output_root, method, 42)
    control = records.get("M3_FULL")
    rows = []
    for method, result in records.items():
        metrics = result["final_dev"]
        delta = metrics["macro_f1"] - control["final_dev"]["macro_f1"] if control else None
        seconds = result.get("train_seconds_per_optimizer_step")
        base_seconds = control.get("train_seconds_per_optimizer_step") if control else None
        valid_timing = all(isinstance(t, (float, int)) and math.isfinite(t) and t > 0 for t in (seconds, base_seconds))
        ratio = seconds / base_seconds if valid_timing else None
        reasons = []
        if method == "M3_FULL":
            reasons.append("control")
        elif delta is None:
            reasons.append("missing control")
        else:
            if delta < 0.003 - 1e-12: reasons.append("dev gain below 0.30 pp")
            if metrics["f1_E"] < control["final_dev"]["f1_E"] - .003 - 1e-12:
                reasons.append("E decreases more than 0.30 pp")
            if (metrics["f1_C"] + metrics["f1_N"] <= control["final_dev"]["f1_C"] + control["final_dev"]["f1_N"]):
                reasons.append("C/N mean does not improve")
            if not valid_timing: reasons.append("missing training throughput")
            elif ratio > 1.5 and delta < .005 - 1e-12:
                reasons.append("cost >1.5x without 0.50 pp gain")
        rows.append(dict(experiment_id=method, seed=42, dev_macro_f1=metrics["macro_f1"],
                         delta_pp=delta * 100 if delta is not None else None,
                         dev_f1_E=metrics["f1_E"], dev_f1_C=metrics["f1_C"], dev_f1_N=metrics["f1_N"],
                         head_parameters=result["hparams"].get("additional_head_parameters"),
                         train_seconds_per_optimizer_step=seconds, relative_training_time=ratio,
                         peak_gpu_memory_bytes=result.get("peak_gpu_memory_bytes"),
                         dev_evaluation_seconds=result.get("evaluation_seconds"),
                         passes_screen=not reasons, reason="; ".join(reasons) or "eligible"))
    return pd.DataFrame(rows)


def select_candidate(output_root: pathlib.Path) -> str:
    table = screening_table(output_root)
    if table.empty or set(table.experiment_id) != set(SCREENING_METHODS):
        raise ValueError("Complete all four verified seed-42 screening jobs first")
    eligible = table[table.passes_screen]
    if eligible.empty:
        raise ValueError("No architecture passes the dev/class/throughput screening gate")
    return str(eligible.sort_values(["dev_macro_f1", "experiment_id"], ascending=[False, True]).iloc[0].experiment_id)


def confirmation_jobs(output_root: pathlib.Path, candidate: str) -> list[MatrixJob]:
    if candidate != select_candidate(output_root):
        raise ValueError("Confirmation candidate must be the preselected dev winner")
    return [MatrixJob("vinli", method, seed) for seed in (2024, 3407) for method in ("M3_FULL", candidate)]


def capacity_control_jobs(output_root: pathlib.Path) -> list[MatrixJob]:
    """One conditional capacity comparison; it never changes pilot selection."""
    controls = {"ARCH_ALIGN256": "ARCH_REL304", "ARCH_CONDPOOL128": "ARCH_REL230", "ARCH_REL256": "ARCH_REL512"}
    return [MatrixJob("vinli", controls[select_candidate(output_root)], 42)]


def confirmation_decision(screening_root: pathlib.Path, confirmation_root: pathlib.Path, candidate: str) -> dict:
    confirmation_jobs(screening_root, candidate)
    _linked_manifests(screening_root, confirmation_root, candidate)
    pairs = []
    for seed in (42, 2024, 3407):
        root = screening_root if seed == 42 else confirmation_root
        try:
            base = verified_result(root, "M3_FULL", seed)
            selected = verified_result(root, candidate, seed)
        except ValueError as error:
            return dict(complete=False, passes_confirmation=False, reason=str(error), candidate=candidate)
        pairs.append(dict(seed=seed, control_dev=base["final_dev"]["macro_f1"],
                          candidate_dev=selected["final_dev"]["macro_f1"],
                          delta=selected["final_dev"]["macro_f1"] - base["final_dev"]["macro_f1"]))
    mean = statistics.mean(p["delta"] for p in pairs)
    wins = sum(p["delta"] > 0 for p in pairs)
    return dict(complete=True, candidate=candidate, pairs=pairs, mean_delta_pp=mean * 100,
                improved_seeds=wins, passes_confirmation=mean >= .002 - 1e-12 and wins >= 2,
                criterion="mean dev gain >=0.20 pp; positive gain in at least 2/3 paired seeds")
