"""Dedicated runner for EXP-001: corrected ViNLI checkpoint -> ViANLI only.

This runner intentionally has no source replay and never opens ViANLI test.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import random
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from gated_dual_ema_msd.data.dataset import LABEL2ID, MultiSourceBatchMixer, collate_rows, load_jsonl
from gated_dual_ema_msd.evaluation.inference import run_inference
from gated_dual_ema_msd.evaluation.metrics import compute_metrics
from gated_dual_ema_msd.compatibility.root_models.flat_cafebert import FlatCafeBERT
from gated_dual_ema_msd.utils import hf_hub_helper, wandb_helper
from gated_dual_ema_msd.training.precision import bf16_enabled

ROOT = pathlib.Path(__file__).resolve().parents[2]

try:
    from gated_dual_ema_msd.operations.migrate_vinli_checkpoint import (  # noqa: E402
        migrate_checkpoint,
        migrate_local_checkpoint,
    )
except ImportError:
    sys.path.insert(0, str(ROOT / "scripts"))
    from migrate_vinli_checkpoint import (  # noqa: E402
        migrate_checkpoint,
        migrate_local_checkpoint,
    )



def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())


def validate_config(cfg: dict) -> None:
    if cfg["project"]["experiment_id"] != "EXP-001" or cfg["project"]["seed"] != 42:
        raise ValueError("runner accepts EXP-001 seed=42 only")
    if cfg["project"].get("target_test_locked") is not True:
        raise ValueError("ViANLI test must be locked")
    if cfg["training"].get("stage2_batch_mix") != {"vianli": 16}:
        raise ValueError("EXP-001 Stage 2 must be ViANLI-only batch=16 (no replay)")
    if cfg["evaluation"].get("target_test_enabled") is not False:
        raise ValueError("EXP-001 search config must disable target test")
    if cfg["model"].get("fallback_models"):
        raise ValueError("scientific model identity cannot use fallback models")
    for section, key in (("model", "revision"), ("initialization", "source_revision")):
        revision = str(cfg[section][key])
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
            raise ValueError(f"{section}.{key} must be an immutable 40-char SHA")
    if cfg["hf_hub"].get("private") is not True or cfg["wandb"].get("tracking_only") is not True:
        raise ValueError("EXP-001 requires private HF artifacts and tracking-only W&B")


def fresh_output_root(cfg: dict, debug: bool) -> pathlib.Path:
    root = ROOT / cfg["outputs"]["root"]
    if debug:
        root = root.with_name(root.name + "-debug")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output directory is not fresh: {root}; remove it explicitly before rerun")
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_audit(cfg: dict) -> dict:
    data = cfg["data"]
    cmd = [
        sys.executable, str(ROOT / "scripts" / "audit_multisource.py"),
        "--vianli-dir", str(ROOT / data["vianli_dir"]),
        "--vinli-dir", str(ROOT / data["vinli_dir"]),
        "--vimednli-dir", str(ROOT / data["vimednli_dir"]),
        "--out", str(ROOT / data["audit_report"]),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    report = json.loads((ROOT / data["audit_report"]).read_text(encoding="utf-8"))
    if report.get("all_checks_passed") is not True:
        raise RuntimeError("audit did not produce all_checks_passed=true")
    return report


def load_rows(cfg: dict, debug: bool):
    data = cfg["data"]
    target_dir = ROOT / data["vianli_dir"]
    source_dir = ROOT / data["vinli_dir"]
    rows = {
        "vianli_train": load_jsonl(target_dir / "train.jsonl", "vianli"),
        "vianli_dev": load_jsonl(target_dir / "dev.jsonl", "vianli"),
        "vinli_dev": load_jsonl(source_dir / "dev.jsonl", "vinli"),
        "vinli_test": load_jsonl(source_dir / "test.jsonl", "vinli"),
    }
    if not debug:
        expected = data["expected_counts"]
        for name, expected_count in expected.items():
            if len(rows[name]) != expected_count:
                raise RuntimeError(f"{name} count {len(rows[name])} != expected {expected_count}")
    else:
        rows["vianli_train"] = rows["vianli_train"][:64]
        rows["vianli_dev"] = rows["vianli_dev"][:32]
        rows["vinli_dev"] = rows["vinli_dev"][:32]
        rows["vinli_test"] = rows["vinli_test"][:32]
    return rows


def prepare_initialization(cfg: dict, out_dir: pathlib.Path) -> dict:
    init = cfg["initialization"]
    if init.get("local_checkpoint"):
        tokenizer_dir = pathlib.Path(init["local_tokenizer_dir"])
        tokenizer_files = [p for p in tokenizer_dir.iterdir() if p.is_file()]
        return migrate_local_checkpoint(
            pathlib.Path(init["local_checkpoint"]), out_dir, tokenizer_files=tokenizer_files,
            source_metadata={"source_repo_id": "local-debug-fixture", "source_revision": "debug"},
        )
    return migrate_checkpoint(
        out_dir, repo_id=init["source_repo_id"], revision=init["source_revision"],
        token=os.getenv("HF_TOKEN"),
    )


def verify_dataset_manifests(cfg: dict) -> dict:
    """Fail-closed binding of runtime data to pinned immutable releases."""
    manifests = {}
    for key, manifest_path in [
        ("vianli_clean", ROOT / "data/processed/vianli_clean/manifest.json"),
        ("vinli", ROOT / "data/external/vinli/manifest.json"),
        ("vimednli", ROOT / "data/external/vimednli/manifest.json"),
    ]:
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing provenance manifest: {manifest_path}")
        manifests[key] = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Pin checks
    if manifests["vianli_clean"]["source"]["revision"] != "0fec8d6ecb043a61c609f9b51f80401fdf1e84d3":
        raise RuntimeError("ViANLI revision mismatch")
    if manifests["vinli"]["source"]["revision"] != "47bd78ac5d075bd00a3cb4cdd3ede4eec4acf8c2":
        raise RuntimeError("ViNLI revision mismatch")
    if manifests["vimednli"]["source"]["revision"] != "2cd94305ba48ae1ccf8782c1df9819ddad7f035f":
        raise RuntimeError("ViMedNLI revision mismatch")
    # Verify on-disk hashes match manifest output_sha256
    for dataset, mapping in [
        ("vianli_clean", {"train": ROOT / "data/processed/vianli_clean/train.jsonl", "dev": ROOT / "data/processed/vianli_clean/dev.jsonl", "test": ROOT / "data/processed/vianli_clean/test.jsonl"}),
        ("vinli", {k: ROOT / f"data/external/vinli/{k}.jsonl" for k in ("train", "dev", "test")}),
        ("vimednli", {k: ROOT / f"data/external/vimednli/{k}.jsonl" for k in ("train", "dev", "test")}),
    ]:
        manifest = manifests[dataset]
        splits = manifest.get("clean_splits") or manifest.get("splits")
        # vianli_clean uses clean_splits, others use splits
        if dataset == "vianli_clean":
            splits = manifest["clean_splits"]
        else:
            splits = manifest["splits"]
        for split, path in mapping.items():
            if not path.exists():
                raise FileNotFoundError(f"missing dataset file: {path}")
            on_disk_hash = sha256_file(path)
            expected = splits[split]["output_sha256"]
            if on_disk_hash != expected:
                raise RuntimeError(f"dataset hash mismatch {dataset}/{split}: {on_disk_hash} != {expected}")
    return manifests


def save_predictions(df: pd.DataFrame, path: pathlib.Path, expected_rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    required = {"sample_id", "gold_label", "logit_E", "logit_C", "logit_N", "pred_label"}
    if not required.issubset(reloaded.columns):
        raise RuntimeError(f"prediction verification failed for {path}: missing columns")
    if len(reloaded) != len(expected_rows):
        raise RuntimeError(f"prediction verification failed for {path}: row count {len(reloaded)} != expected {len(expected_rows)}")
    expected_ids = [r["id"] for r in expected_rows]
    if list(reloaded["sample_id"]) != expected_ids:
        raise RuntimeError(f"prediction ID sequence mismatch for {path}")
    if len(set(reloaded["sample_id"])) != len(reloaded):
        raise RuntimeError(f"duplicate sample_id in {path}")
    if not set(reloaded["gold_label"]).issubset({"E", "C", "N"}) or not set(reloaded["pred_label"]).issubset({"E", "C", "N"}):
        raise RuntimeError(f"illegal label in {path}")
    import math as _math
    for col in ("logit_E", "logit_C", "logit_N"):
        if not all(isinstance(v, (int, float)) and _math.isfinite(float(v)) for v in reloaded[col]):
            raise RuntimeError(f"non-finite logit in {path}:{col}")
    # argmax consistency
    import numpy as _np
    logits = reloaded[["logit_E", "logit_C", "logit_N"]].to_numpy()
    argmax = _np.argmax(logits, axis=1)
    pred_ids = [LABEL2ID[x] for x in reloaded["pred_label"]]
    if not _np.array_equal(argmax, _np.array(pred_ids)):
        raise RuntimeError(f"pred_label != argmax(logits) for {path}")
    y_true = [LABEL2ID[x] for x in reloaded["gold_label"]]
    metrics = compute_metrics(pred_ids, y_true)
    return {"rows": len(reloaded), "sha256": sha256_file(path), "metrics": metrics}


def environment_metadata() -> dict:
    packages = {}
    for name in ("torch", "transformers", "numpy", "pandas", "scikit-learn", "wandb", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def evaluate_and_persist(model, rows: dict, tokenizer, cfg: dict, device, pred_dir: pathlib.Path,
                         prefix: str, splits: list[str]) -> tuple[dict, dict]:
    metrics_by_split, verification = {}, {}
    for split in splits:
        metrics, _, df = run_inference(
            model, rows[split], tokenizer, cfg["model"]["max_length"], device,
            cfg["training"]["batch_size"] * 2, compute_loss=False,
        )
        path = pred_dir / f"{prefix}_{split}_predictions.csv"
        check = save_predictions(df, path, rows[split])
        for key, value in metrics.items():
            if abs(float(value) - float(check["metrics"][key])) > 1e-12:
                raise RuntimeError(f"offline metric mismatch for {split}:{key}")
        metrics_by_split[split] = metrics
        verification[split] = {**check, "path": str(path)}
        print(f"[{prefix}] {split}: macro_f1={metrics['macro_f1']:.4f} rows={len(df)}")
    return metrics_by_split, verification


def verify_corrected_source_metrics(cfg: dict, metrics: dict, tolerance: float | None = None) -> None:
    """Verify migrated-source semantics within a declared cross-device envelope."""
    expected = cfg["initialization"].get("expected_corrected_metrics", {})
    if tolerance is None:
        tolerance = float(cfg["initialization"].get("expected_metric_tolerance", 1e-12))
    if not 0.0 <= tolerance <= 1e-2:
        raise RuntimeError(f"unsafe expected_metric_tolerance: {tolerance}")
    if set(expected) != {"vinli_dev", "vinli_test"}:
        raise RuntimeError("missing expected corrected ViNLI dev/test metrics")
    for split, expected_metrics in expected.items():
        for key, expected_value in expected_metrics.items():
            actual = metrics[split][key]
            if abs(float(actual) - float(expected_value)) > tolerance:
                raise RuntimeError(
                    f"corrected source metric mismatch {split}.{key}: {actual} != {expected_value}"
                )


def train_target(model, target_train: list[dict], target_dev: list[dict], tokenizer, cfg: dict,
                 device, checkpoint_dir: pathlib.Path, run) -> tuple[float, int]:
    tr = cfg["training"]
    mixer = MultiSourceBatchMixer(
        {"vianli": target_train}, {"vianli": tr["batch_size"]}, ["vianli"], cfg["project"]["seed"]
    )
    steps_per_epoch = mixer.epoch_steps()
    grad_accum = max(1, tr.get("grad_accum_steps", 1))
    total_optimizer_steps = math.ceil(steps_per_epoch / grad_accum) * tr["max_epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_optimizer_steps * tr["warmup_ratio"]), total_optimizer_steps
    )
    bf16 = bf16_enabled(device, requested=tr.get("bf16_train", tr.get("bf16", True)))
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    best_metric, no_improve, global_step = -1.0, 0, 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(step: int) -> bool:
        nonlocal best_metric, no_improve
        metrics, dev_loss, _ = run_inference(
            model, target_dev, tokenizer, cfg["model"]["max_length"], device,
            tr["batch_size"] * 2, compute_loss=True,
        )
        wandb_helper.log_step(run, {
            "stage2/vianli_dev_macro_f1": metrics["macro_f1"],
            "stage2/vianli_dev_loss": dev_loss,
        }, step=step)
        if metrics["macro_f1"] > best_metric:
            best_metric = metrics["macro_f1"]
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")
            tokenizer.save_pretrained(checkpoint_dir)
            (checkpoint_dir / "model_identity.json").write_text(json.dumps({
                "model_name": cfg["model"]["name"], "model_revision": cfg["model"]["revision"],
                "label_order": ["E", "C", "N"],
            }, indent=2), encoding="utf-8")
            print(f"[stage2] new best={best_metric:.4f} step={step}")
        else:
            no_improve += 1
        model.train()
        return no_improve >= tr["early_stopping_patience_evals"]

    stop = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, tr["max_epochs"] + 1):
        model.train()
        for batch_idx, raw_rows in enumerate(tqdm(mixer.batches(), total=steps_per_epoch, desc=f"stage2 epoch {epoch}"), 1):
            batch = collate_rows(raw_rows, tokenizer, cfg["model"]["max_length"])
            tensors = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids", "labels")}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                output = model(
                    input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"],
                    token_type_ids=tensors.get("token_type_ids"), labels=tensors["labels"],
                )
                loss = output["loss"] / grad_accum
            scaler.scale(loss).backward()
            if batch_idx % grad_accum == 0 or batch_idx == steps_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= before:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                eval_due = global_step % tr["eval_steps"] == 0
                if global_step % tr["logging_steps"] == 0:
                    wandb_helper.log_step(run, {
                        "stage2/train_loss": float(loss.item() * grad_accum),
                        "stage2/lr": scheduler.get_last_lr()[0], "stage2/epoch": epoch,
                    }, step=global_step, commit=not eval_due)
                if eval_due:
                    if evaluate(global_step):
                        stop = True
                        break
        if stop:
            print(f"[stage2] early stop at step={global_step}")
            break
    # Strict schedule: terminal evaluation only if last interval was missed
    if not stop and global_step % tr["eval_steps"] != 0:
        if evaluate(global_step):
            print(f"[stage2] terminal evaluation at step={global_step}")
    if best_metric < 0 or not (checkpoint_dir / "pytorch_model.bin").exists():
        raise RuntimeError("training completed without a selected checkpoint")
    return best_metric, global_step


def main(args) -> dict:
    cfg_path = pathlib.Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    validate_config(cfg)
    set_seed(cfg["project"]["seed"])
    if not args.debug and git_dirty():
        raise RuntimeError("scientific run requires a clean git worktree")
    if not args.debug:
        wandb_helper.require_wandb(cfg)
        hf_hub_helper.require_hf(cfg)
        # Preflight HF write capability before any GPU work
        hf_hub_helper.ensure_repo(cfg["hf_hub"]["repo_id"], private=True)
        print(f"[EXP-001] HF preflight OK: {cfg['hf_hub']['repo_id']}")

    out_root = fresh_output_root(cfg, args.debug)
    audit_report = run_audit(cfg)
    dataset_manifests = verify_dataset_manifests(cfg)
    rows = load_rows(cfg, args.debug)

    config_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    hparams = {
        "experiment_id": "EXP-001", "seed": 42, "config_sha256": config_hash,
        "git_commit": git_commit(), "model_name": cfg["model"]["name"],
        "model_revision": cfg["model"]["revision"], "source_repo_id": cfg["initialization"]["source_repo_id"],
        "source_revision": cfg["initialization"]["source_revision"], **cfg["training"],
        "counts": {k: len(v) for k, v in rows.items()},
    }
    # Initialize W&B before checkpoint migration and source verification so the
    # Colab run is observable from the first expensive operation, including a
    # gate failure.
    run = None if args.debug else wandb_helper.init_run(
        cfg, cfg["wandb"]["run_name"], hparams, tags=["EXP-001", "seed42", "no-replay"]
    )
    if run:
        wandb_helper.log_audit_status(run, audit_report)
        run.summary["lifecycle/status"] = "initialization"

    init_dir = out_root / "migrated_source_checkpoint"
    migration = prepare_initialization(cfg, init_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"], revision=cfg["model"]["revision"], trust_remote_code=True,
    )
    model = FlatCafeBERT(
        cfg["model"]["name"], fallback_models=[], dropout=cfg["model"]["dropout"],
        num_labels=3, label_smoothing=cfg["model"]["label_smoothing"], revision=cfg["model"]["revision"],
    )
    model.load_state_dict(torch.load(init_dir / "pytorch_model.bin", map_location="cpu", weights_only=True), strict=True)
    model.to(device)

    pred_dir = out_root / "predictions"
    pre_metrics, pre_verify = evaluate_and_persist(
        model, rows, tokenizer, cfg, device, pred_dir, "preadapt",
        ["vinli_dev", "vinli_test", "vianli_dev"],
    )
    if run:
        preadapt_payload = {
            f"preadapt/{split}_{metric}": value
            for split, split_metrics in pre_metrics.items()
            for metric, value in split_metrics.items()
        }
        wandb_helper.log_step(run, preadapt_payload, step=0)
        run.summary["lifecycle/status"] = "source_verification"
    if not args.debug:
        try:
            verify_corrected_source_metrics(cfg, pre_metrics)
        except Exception:
            if run:
                run.summary["lifecycle/status"] = "source_verification_failed"
                wandb_helper.finish(run)
            raise
        if run:
            run.summary["lifecycle/status"] = "training"
        print("[EXP-001] corrected ViNLI source metrics verified before adaptation")

    checkpoint_dir = out_root / "stage2_checkpoint"
    best_dev, optimizer_steps = train_target(
        model, rows["vianli_train"], rows["vianli_dev"], tokenizer, cfg, device, checkpoint_dir, run
    )
    model.load_state_dict(torch.load(checkpoint_dir / "pytorch_model.bin", map_location=device, weights_only=True))
    final_metrics, final_verify = evaluate_and_persist(
        model, rows, tokenizer, cfg, device, pred_dir, "final", ["vianli_dev"]
    )
    if abs(final_metrics["vianli_dev"]["macro_f1"] - best_dev) > 1e-12:
        raise RuntimeError("reloaded best checkpoint metric does not match selection metric")

    metadata = {
        "experiment_id": "EXP-001", "experiment_mode": cfg["project"]["experiment_mode"],
        "seed": 42, "debug": args.debug, "mock": False, "target_test_accessed": False,
        "frozen_final_evaluation": False, "git_commit": git_commit(), "config_sha256": config_hash,
        "model": {"name": cfg["model"]["name"], "revision": cfg["model"]["revision"]},
        "migration": migration, "audit": audit_report, "dataset_manifests": dataset_manifests, "pre_adaptation_metrics": pre_metrics,
        "final_metrics": final_metrics, "selection_metric": "vianli_dev_macro_f1",
        "best_vianli_dev_macro_f1": best_dev, "optimizer_steps": optimizer_steps,
        "prediction_verification": {"pre_adaptation": pre_verify, "final": final_verify},
        "environment": environment_metadata(), "wandb_run_url": run.url if run else None,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    metadata_path = out_root / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.debug:
        print("[EXP-001] debug integration complete; remote upload intentionally skipped")
    else:
        pred_paths = {
            "vinli_dev": pathlib.Path(pre_verify["vinli_dev"]["path"]),
            "vinli_test": pathlib.Path(pre_verify["vinli_test"]["path"]),
            "vianli_zero_shot_dev": pathlib.Path(pre_verify["vianli_dev"]["path"]),
            "vianli_dev": pathlib.Path(final_verify["vianli_dev"]["path"]),
        }
        repo_id, revision = hf_hub_helper.push_run_artifacts(cfg, checkpoint_dir, pred_paths, metadata)
        metadata["hf_repo_id"] = repo_id
        metadata["hf_revision"] = revision
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        wandb_helper.log_hf_link(run, repo_id, revision)
        for metric, value in final_metrics["vianli_dev"].items():
            run.summary[f"final/vianli_dev_{metric}"] = value
        for split in ("vinli_dev", "vinli_test", "vianli_dev"):
            frame = pd.read_csv(pre_verify[split]["path"])
            wandb_helper.log_confusion_matrix(
                run, frame["gold_label"].map(LABEL2ID), frame["pred_label"].map(LABEL2ID), ["E", "C", "N"],
                f"preadapt/{split}_confusion_matrix",
            )
        final_frame = pd.read_csv(final_verify["vianli_dev"]["path"])
        wandb_helper.log_confusion_matrix(
            run, final_frame["gold_label"].map(LABEL2ID), final_frame["pred_label"].map(LABEL2ID), ["E", "C", "N"],
            "final/vianli_dev_confusion_matrix",
        )
        run.summary["best_vianli_dev_macro_f1"] = best_dev
        wandb_helper.finish(run)
    print(f"[EXP-001] complete best_dev_macro_f1={best_dev:.4f} -> {out_root}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp001_vinli_stilts_no_replay.yaml")
    parser.add_argument("--debug", action="store_true", help="tiny real integration; never uploads or claims results")
    main(parser.parse_args())
