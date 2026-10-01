"""Data loading, manifest verification, and provenance checking for R2 protocol."""
from __future__ import annotations

import json
import pathlib
from typing import Any

from gated_dual_ema_msd.data import jtt, laurer_vi_nli
from gated_dual_ema_msd.data.dataset import load_jsonl
from gated_dual_ema_msd.config.r2_validation import (
    ROOT,
    VINLI_REVISION,
    VIANLI_REVISION,
    VIMEDNLI_REVISION,
    LAURER_MTVI_DOMAIN,
    LAURER_MTVI_REVISION,
    LAURER_MTVI_SPLITS,
    AUX_RUNS,
    BRIDGE_RUN_DATASETS,
    M25_RUN_NAME,
    M41_MNLI_BRIDGE_RUN_NAME,
    M42_ANLI_BRIDGE_RUN_NAME,
)
from gated_dual_ema_msd.training.r2_runtime import sha256_file

def verify_and_load_rows(cfg: dict, debug: bool) -> tuple[dict[str, Any], dict]:
    """Bind and load only configured scientific splits; target test stays unread when locked."""
    data = cfg["data"]
    manifest_paths = {name: ROOT / path for name, path in data["manifests"].items()}
    missing_manifests = [p for p in manifest_paths.values() if not p.exists()]
    if missing_manifests:
        print("[data] Dataset manifests not found. Automatically running data preparation pipeline...")
        from run import prepare as run_prepare
        run_prepare()

    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in manifest_paths.items()
    }
    if manifests["vinli"]["source"]["revision"] != VINLI_REVISION:
        raise RuntimeError("ViNLI manifest revision mismatch")
    if manifests["vianli"]["source"]["revision"] != VIANLI_REVISION:
        raise RuntimeError("ViANLI manifest revision mismatch")
    stage2_target = cfg["training"]["stage2"].get("dataset", "vianli")
    if stage2_target not in ("vianli", "vimednli"):
        raise ValueError(f"unsupported stage2 target dataset: {stage2_target}")
    if stage2_target == "vimednli" \
            and manifests["vimednli"]["source"]["revision"] != VIMEDNLI_REVISION:
        raise RuntimeError("ViMedNLI manifest revision mismatch")

    target_dir_key = "vimednli_dir" if stage2_target == "vimednli" else "vianli_dir"
    paths = {
        "vinli_train": ROOT / data["vinli_dir"] / "train.jsonl",
        "vinli_dev": ROOT / data["vinli_dir"] / "dev.jsonl",
        f"{stage2_target}_train": ROOT / data[target_dir_key] / "train.jsonl",
        f"{stage2_target}_dev": ROOT / data[target_dir_key] / "dev.jsonl",
        f"{stage2_target}_test": ROOT / data[target_dir_key] / "test.jsonl",
    }
    for split_key, split_path in paths.items():
        if not split_path.exists():
            print(f"[data] Split file {split_path} not found. Running data preparation...")
            from run import prepare as run_prepare
            run_prepare()
            break
    if cfg["evaluation"].get("target_test_enabled") is not True:
        # Do not even read target-test text or labels during a locked search run.
        paths.pop(f"{stage2_target}_test")
    target_split_meta = (
        manifests["vimednli"]["splits"]
        if stage2_target == "vimednli"
        else (manifests["vianli"].get("clean_splits") or manifests["vianli"].get("splits") or manifests["vianli"].get("counts"))
    )


    split_metadata = {
        "vinli_train": manifests["vinli"]["splits"]["train"],
        "vinli_dev": manifests["vinli"]["splits"]["dev"],
        f"{stage2_target}_train": target_split_meta["train"],
        f"{stage2_target}_dev": target_split_meta["dev"],
        f"{stage2_target}_test": target_split_meta["test"],
    }
    domains = {
        "vinli_train": "vinli", "vinli_dev": "vinli",
        f"{stage2_target}_train": stage2_target,
        f"{stage2_target}_dev": stage2_target,
        f"{stage2_target}_test": stage2_target,
    }
    rows: dict[str, Any] = {}
    for split, path in paths.items():
        expected = data["expected_counts"][split]
        meta = split_metadata[split]
        if meta["row_count"] != expected or sha256_file(path) != meta["output_sha256"]:
            raise RuntimeError(f"dataset integrity mismatch for {split}")
        rows[split] = load_jsonl(path, domains[split])
        if len(rows[split]) != expected:
            raise RuntimeError(f"{split} count {len(rows[split])} != expected {expected}")

    # Optional ViMedNLI (medical NLI, max_length 256) when Stage 1 is multi-source.
    stage1_cfg = cfg["training"].get("stage1") or {}
    stage1_mix = stage1_cfg.get("batch_mix", {"vinli": 4})
    if "vimednli" in stage1_mix:
        vm_dir = ROOT / data["vimednli_dir"]
        vm_path = vm_dir / "train.jsonl"
        vm_manifest = json.loads((ROOT / data["manifests"]["vimednli"]).read_text(encoding="utf-8"))
        if vm_manifest["source"]["revision"] != VIMEDNLI_REVISION:
            raise RuntimeError("ViMedNLI manifest revision mismatch")
        vm_meta = vm_manifest["splits"]["train"]
        expected_vm = data["expected_counts"]["vimednli_train"]
        if vm_meta["row_count"] != expected_vm or sha256_file(vm_path) != vm_meta["output_sha256"]:
            raise RuntimeError("ViMedNLI train integrity mismatch")
        rows["vimednli_train"] = load_jsonl(vm_path, "vimednli")
        if len(rows["vimednli_train"]) != expected_vm:
            raise RuntimeError(f"vimednli_train count {len(rows['vimednli_train'])} != expected {expected_vm}")

    bridge = cfg["training"].get("stage1b")
    if bridge is not None:
        bname = bridge["dataset"]
        bmanifest = json.loads(
            (ROOT / data["manifests"][bname]).read_text(encoding="utf-8")
        )
        if bmanifest["source"]["revision"] != bridge["dataset_revision"]:
            raise RuntimeError(f"{bname} bridge manifest revision mismatch")
        bpath = ROOT / pathlib.Path(bridge["dataset_dir"]) / "train.jsonl"
        bmeta = bmanifest["splits"]["train"]
        expected_bridge = data["expected_counts"][bridge["train_split"]]
        if bmeta["row_count"] != expected_bridge or sha256_file(bpath) != bmeta["output_sha256"]:
            raise RuntimeError(f"{bname} bridge train integrity mismatch")
        rows[bname] = load_jsonl(bpath, bridge["domain"])
        if len(rows[bname]) != expected_bridge:
            raise RuntimeError(
                f"{bname} bridge count {len(rows[bname])} != expected {expected_bridge}"
            )

    augmentation = stage1_cfg.get("mt_vi_augmentation")
    if augmentation is None:
        augmentation = (cfg["training"].get("stage2") or {}).get("mt_vi_augmentation")
    if augmentation is not None:
        binding = data.get("auxiliary_nli")
        if augmentation.get("enabled") is not True or binding is None:
            raise RuntimeError("auxiliary loading requested without its pinned binding")
        auxiliary_rows, verified_manifest = laurer_vi_nli.verify_materialized_dataset(
            ROOT / binding["manifest"],
            binding,
        )
        if manifests.get("laurer_vi_nli") != verified_manifest:
            raise RuntimeError("loaded auxiliary manifest changed during verification")
        rows["laurer_mtvi_splits"] = auxiliary_rows

    if debug:
        gate = cfg.get("debug", {})
        max_train = gate.get("max_train_rows_per_stage", 32)
        max_dev = gate.get("max_dev_rows_per_stage", 16)
        for split in ("vinli_train", f"{stage2_target}_train"):
            rows[split] = rows[split][:max_train]
        if bridge is not None:
            rows[bridge["dataset"]] = rows[bridge["dataset"]][:max_train]
        for split in ("vinli_dev", f"{stage2_target}_dev"):
            rows[split] = rows[split][:max_dev]
    return rows, manifests


def save_checkpoint(model, tokenizer, checkpoint_dir: pathlib.Path, cfg: dict,
                    stage_name: str, metric: float, optimizer_step: int) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(checkpoint_dir)
    model.backbone.config.save_pretrained(checkpoint_dir)
    identity = {
        "stage": stage_name,
        "base_model": cfg["model"]["name"],
        "base_revision": cfg["model"]["revision"],
        "label_order": ["E", "C", "N"],
        "selection_metric": "macro_f1",
        "best_dev_macro_f1": metric,
        "optimizer_step": optimizer_step,
        "model_config": dict(cfg["model"]),
    }
    (checkpoint_dir / "checkpoint_metadata.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8"
    )
