"""Resumable notebook batch: isolated BF16 runs and verified HF artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd
from transformers import AutoConfig, AutoTokenizer

from gated_dual_ema_msd.cli.matrix import MatrixJob, write_summaries
from gated_dual_ema_msd.config.contracts import DATASET_MAX_LENGTHS, DATASET_REVISIONS, MODEL_NAME, MODEL_REVISION, parse_label
from gated_dual_ema_msd.data.loading import read_jsonl
from gated_dual_ema_msd.evaluation.predictions import validate_predictions
from gated_dual_ema_msd.tracking.hf import push_run_artifacts
from gated_dual_ema_msd.training.r2_runtime import sha256_file

DATA_DIRS = {"vinli": "data/external/vinli", "vianli": "data/processed/vianli_clean", "vimednli": "data/external/vimednli"}
COUNTS = {"vinli": (18282, 2255), "vianli": (8010, 1000), "vimednli": (11217, 1395)}
TEST_COUNTS = {"vinli": 2264, "vianli": 1000, "vimednli": 1422}


def write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def bind_manifest(output_root: pathlib.Path, manifest: dict) -> str:
    path = pathlib.Path(output_root) / "batch_manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing batch manifest differs. Restore the original settings/source SHA or choose a new RUN_GROUP.")
    write_json(path, manifest)
    return manifest_digest(manifest)


def prepare_data(repo: pathlib.Path, datasets: list[str], *, frozen_final: bool = False) -> dict:
    """Bind pinned data; include test provenance only for the approved final path."""
    from gated_dual_ema_msd.operations import prepare_vianli, prepare_vinli_multisource, prepare_vimednli_multisource
    for dataset in datasets:
        directory = repo / DATA_DIRS[dataset]
        if not (directory / "manifest.json").exists():
            if dataset == "vianli":
                prepare_vianli.prepare_dataset(repo / "data/raw", directory)
            elif dataset == "vinli":
                prepare_vinli_multisource.prepare_dataset(directory)
            else:
                prepare_vimednli_multisource.prepare_dataset(directory)
    fingerprints = {}
    for dataset in datasets:
        directory = repo / DATA_DIRS[dataset]
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["source"]["revision"] != DATASET_REVISIONS[dataset]:
            raise ValueError(f"Pinned data revision mismatch: {dataset}")
        metadata = manifest.get("clean_splits", manifest.get("splits"))
        splits = {}
        expected_splits = dict(zip(("train", "dev"), COUNTS[dataset]))
        if frozen_final:
            expected_splits["test"] = TEST_COUNTS[dataset]
        for split, expected_count in expected_splits.items():
            path = directory / f"{split}.jsonl"
            records = read_jsonl(path)
            ids = [str(row["id"]) for row in records]
            if len(records) != expected_count or len(set(ids)) != len(ids):
                raise ValueError(f"Unexpected count or duplicate IDs: {dataset}/{split}")
            for row in records:
                parse_label(row["label"], str(row["id"]))
            digest = sha256_file(path)
            if digest != metadata[split]["output_sha256"]:
                raise ValueError(f"Prepared data hash mismatch: {dataset}/{split}")
            splits[split] = {"row_count": len(records), "sha256": digest}
        fingerprints[dataset] = {"revision": DATASET_REVISIONS[dataset], "max_length": DATASET_MAX_LENGTHS[dataset], "splits": splits}
    return fingerprints


def train_command(job: MatrixJob, config: dict, output_root: pathlib.Path) -> list[str]:
    if config.get("test_peak_exploratory", False) and config.get("frozen_final", False):
        raise ValueError("Exploratory test scans and frozen-final are separate protocols")
    command = [sys.executable, "-u", "-m", "gated_dual_ema_msd.cli.train", "--experiment_id", job.experiment_id,
               "--dataset", job.dataset, "--seed", str(job.seed), "--output_dir", str(output_root), "--require_cuda",
               "--test_peak_exploratory" if config.get("test_peak_exploratory", False) else
               ("--frozen_final" if config.get("frozen_final", False) else "--no_test"),
               "--no_wandb"]
    for name, flag in {"epochs": "--epochs", "eval_steps": "--eval_steps", "patience": "--patience", "lr": "--lr",
                       "weight_decay": "--weight_decay", "warmup_ratio": "--warmup_ratio", "label_smoothing": "--label_smoothing",
                       "dropout": "--dropout", "physical_batch_size": "--physical_batch_size", "grad_accum": "--grad_accum"}.items():
        command.extend([flag, str(config[name])])
    for name in ("ema_decay", "ema_start_step"):
        if name in config:
            command.extend(["--" + name, str(config[name])])
    return command


def run_logged(command: list[str], cwd: pathlib.Path, log_path: pathlib.Path) -> None:
    """Keep a live log and stop the child when the notebook cell is interrupted."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise RuntimeError(f"Training failed; inspect {log_path}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def validate_run(result: dict, job: MatrixJob, *, frozen_final: bool = False, test_peak_exploratory: bool = False) -> None:
    if (result.get("dataset"), result.get("experiment_id"), result.get("seed")) != (job.dataset, job.experiment_id, job.seed):
        raise ValueError("Result does not belong to this job")
    hparams = result["hparams"]
    if hparams.get("train_precision") != "bf16" or hparams.get("fp32_eval") is not True:
        raise ValueError("Result precision is not BF16 train / FP32 eval")
    if hparams["max_length"] != DATASET_MAX_LENGTHS[job.dataset]:
        raise ValueError("Result max_length violates dataset contract")
    if test_peak_exploratory:
        if frozen_final or result.get("test_peak_exploratory") is not True or result.get("test_selection_policy") != "exploratory_test_macro_f1":
            raise ValueError("Test-peak runs must be explicitly marked exploratory")
        if (result.get("selection_policy") != "dev_macro_f1" or result.get("test_evaluations", 0) < 2
                or not result.get("peak_test_checkpoint") or not isinstance(result.get("peak_test_step"), int)
                or result["peak_test_step"] < 1):
            raise ValueError("Exploratory peak checkpoint/step/evaluations are missing")
        peak = result.get("peak_test_macro_f1")
        if not isinstance(peak, (int, float)) or not math.isfinite(peak) or not 0 <= peak <= 1:
            raise ValueError("Invalid exploratory peak metric")
    elif frozen_final:
        if (result.get("selection_policy") != "dev_macro_f1" or result.get("test_evaluations") != 1
                or result.get("test_peak_exploratory", False)):
            raise ValueError("Frozen-final requires dev selection and exactly one test evaluation")
        metrics = result.get("test")
        if not isinstance(metrics, dict) or any(not isinstance(metrics.get(key), (float, int))
                or not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1 for key in ("macro_f1", "accuracy")):
            raise ValueError("Frozen-final test metrics are missing or invalid")
        if any(result.get(key) is not None for key in ("peak_test_step", "peak_test_macro_f1", "peak_test_checkpoint")):
            raise ValueError("Test-based checkpoint selection is forbidden")
    elif result.get("test") is not None or result.get("test_evaluations") != 0:
        raise ValueError("This multi-seed batch must keep test locked")


def publish_run(run_dir: pathlib.Path, repo: pathlib.Path, job: MatrixJob, config: dict, batch_manifest: dict) -> dict:
    """Upload the selected weights and verify every file at an immutable revision."""
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    frozen_final = bool(config.get("frozen_final", False))
    exploratory = bool(config.get("test_peak_exploratory", False))
    validate_run(result, job, frozen_final=frozen_final, test_peak_exploratory=exploratory)
    prediction_paths = {}
    split_outputs = [("dev", "dev_predictions.csv")]
    if frozen_final or exploratory:
        split_outputs.append(("test", "test_predictions.csv"))
    if exploratory:
        split_outputs.append(("test", "test_predictions_peak.csv"))
    for split, filename in split_outputs:
        prediction_path = run_dir / filename
        frame = pd.read_csv(prediction_path, dtype={"sample_id": str})
        split_path = repo / DATA_DIRS[job.dataset] / f"{split}.jsonl"
        expected_split = batch_manifest["data_fingerprints"][job.dataset]["splits"][split]
        if sha256_file(split_path) != expected_split["sha256"]:
            raise ValueError(f"Prepared data changed since protocol freeze: {job.dataset}/{split}")
        rows = read_jsonl(split_path)
        validate_predictions(frame, [str(row["id"]) for row in rows])
        if frame["gold_label"].tolist() != [row["label"] for row in rows]:
            raise ValueError(f"Prediction gold labels differ from the pinned {split} split")
        key = f"{job.dataset}_{split}" + ("_exploratory_peak" if filename == "test_predictions_peak.csv" else "")
        prediction_paths[key] = prediction_path
    if exploratory:
        curve = pd.read_csv(run_dir / "test_curve.csv")
        if len(curve) != result["test_evaluations"] - 1 or curve.empty:
            raise ValueError("Exploratory test curve/evaluation count mismatch")
        best = curve.loc[curve["test_macro_f1"].idxmax()]
        if int(best["optimizer_step"]) != result["peak_test_step"] or not math.isclose(float(best["test_macro_f1"]), result["peak_test_macro_f1"]):
            raise ValueError("Exploratory peak differs from recorded test curve")
    repo_id = f"{config['hf_namespace']}/{config['hf_prefix']}-{job.dataset}-{job.experiment_id.lower().replace('_', '-')}-seed{job.seed}"
    metadata = {**result, "git_sha": batch_manifest["git_sha"], "model_revision": MODEL_REVISION,
                "data_fingerprints": batch_manifest["data_fingerprints"][job.dataset], "target_test_accessed": frozen_final or exploratory,
                "protocol": batch_manifest.get("protocol"), "paper_differences": batch_manifest.get("paper_differences"),
                "wandb_enabled": False, "artifact_backend": "huggingface",
                "publication_helper_patch_git_sha": globals().get("BATCH_HELPER_PATCH_SHA"),
                "batch_manifest_sha256": manifest_digest(batch_manifest)}
    from gated_dual_ema_msd.training.r2_runtime import environment_metadata
    metadata["environment"] = environment_metadata()
    with tempfile.TemporaryDirectory(prefix="nli-publish-", dir=run_dir) as temporary:
        temporary = pathlib.Path(temporary)
        checkpoint = temporary / "checkpoint"
        checkpoint.mkdir()
        shutil.copy2(run_dir / "best_model.pt", checkpoint / "pytorch_model.bin")
        if exploratory:
            shutil.copy2(run_dir / "best_test_model.pt", checkpoint / "exploratory_best_test_model.pt")
            shutil.copy2(run_dir / "test_curve.csv", checkpoint / "exploratory_test_curve.csv")
        AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).save_pretrained(checkpoint)
        AutoConfig.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).save_pretrained(checkpoint)
        write_json(checkpoint / "model_identity.json", {"experiment": result["hparams"]["model_configuration"],
                   "selected_weight_source": result["selected_weight_source"], "model_name": MODEL_NAME,
                   "model_revision": MODEL_REVISION, "label_order": ["E", "C", "N"],
                   "test_peak_exploratory": exploratory,
                   "exploratory_peak_weights": "exploratory_best_test_model.pt" if exploratory else None})
        hf_config = {"hf_hub": {"enabled": True, "repo_id": repo_id, "private": config["hf_private"],
                                "readback_cache_dir": str(temporary / "readback")}}
        repo_id, revision = push_run_artifacts(hf_config, checkpoint, prediction_paths, metadata)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("HF did not return an immutable commit SHA")
    result.update(hf_repo_id=repo_id, hf_revision=revision, artifact_readback_verified=True,
                  wandb_enabled=False, artifact_backend="huggingface",
                  publication_helper_patch_git_sha=globals().get("BATCH_HELPER_PATCH_SHA"),
                  batch_manifest_sha256=manifest_digest(batch_manifest), git_sha=batch_manifest["git_sha"],
                  hf_checkpoint_path="stage2_checkpoint/pytorch_model.bin",
                  hf_exploratory_peak_checkpoint_path="stage2_checkpoint/exploratory_best_test_model.pt" if exploratory else None)
    write_json(run_dir / "result.json", result)
    return result


def run_batch(jobs: list[MatrixJob], repo: pathlib.Path, work_root: pathlib.Path, output_root: pathlib.Path,
              config: dict, batch_manifest: dict) -> None:
    signature = bind_manifest(output_root, batch_manifest)
    work_root.mkdir(parents=True, exist_ok=True)
    bind_manifest(work_root, batch_manifest)
    for index, job in enumerate(jobs, 1):
        stored_dir = job.output_dir(output_root)
        marker = stored_dir / "verified_run.json"
        if marker.exists():
            previous = json.loads(marker.read_text(encoding="utf-8"))
            if (previous.get("batch_manifest_sha256") != signature
                    or not previous.get("artifact_readback_verified")
                    or not re.fullmatch(r"[0-9a-f]{40}", previous.get("hf_revision", ""))):
                raise ValueError(f"Invalid resume marker: {marker}")
            validate_run(previous, job, frozen_final=bool(config.get("frozen_final", False)),
                         test_peak_exploratory=bool(config.get("test_peak_exploratory", False)))
            print(f"[{index}/{len(jobs)}] SKIP verified {job.run_name}", flush=True)
            continue
        run_dir = job.output_dir(work_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(jobs)}] RUN {job.run_name}", flush=True)
        write_json(output_root / "progress.json", {"state": "training", "job": job.run_name, "index": index, "total": len(jobs)})
        if not (run_dir / "result.json").exists():
            run_logged(train_command(job, config, work_root), repo, stored_dir / "worker.log")
        write_json(output_root / "progress.json", {"state": "artifact_upload", "job": job.run_name, "index": index, "total": len(jobs)})
        result = publish_run(run_dir, repo, job, config, batch_manifest)
        stored_dir.mkdir(parents=True, exist_ok=True)
        for path in run_dir.iterdir():
            if path.is_file() and path.suffix in (".csv", ".json", ".log"):
                shutil.copy2(path, stored_dir / path.name)
        write_json(marker, result)
        write_summaries(output_root)
        # Only our local generated checkpoint copies are removed after complete
        # HF read-back. HF remains the source of truth; Drive stores the ledger.
        if not config.get("keep_local_checkpoints", False):
            for name in ("best_model.pt", "best_current_model.pt", "best_test_model.pt", "pytorch_model.bin"):
                (run_dir / name).unlink(missing_ok=True)
    write_summaries(output_root)
    write_json(output_root / "progress.json", {"state": "complete", "total": len(jobs)})
