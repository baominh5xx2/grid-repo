"""EXP-001-R2 main protocol orchestrator."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import time
from collections import Counter
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from gated_dual_ema_msd.data import jtt, laurer_vi_nli
from gated_dual_ema_msd.evaluation.inference import run_inference
from gated_dual_ema_msd.compatibility.root_models.flat_cafebert import FlatCafeBERT
from gated_dual_ema_msd.training.lora import apply_lora_to_model, assert_lora_eval_noop
from gated_dual_ema_msd.training.train_exp001 import save_predictions
from gated_dual_ema_msd.utils import hf_hub_helper, wandb_helper
from gated_dual_ema_msd.config.r2_validation import (
    ROOT,
    DEFAULT_CONFIG,
    MODEL_REVISION,
    M25_RUN_NAME,
    M32_JTT_RUN_NAME,
    M45_SIGNED_DELTA_RUN_NAME,
    M46_HIER_E_FIRST_RUN_NAME,
    M47_MIRROR_SYMMETRY_RUN_NAME,
    M48_VIMEDNLI_CE_RUN_NAME,
    M49_VIMEDNLI_MIRROR_RUN_NAME,
    LAURER_MTVI_DOMAIN,
    validate_config,
    validate_runtime_args,
)
from gated_dual_ema_msd.data.provenance import verify_and_load_rows
from gated_dual_ema_msd.training.checkpoints import load_best
from gated_dual_ema_msd.training.r2_runtime import (
    fresh_output_root,
    environment_metadata,
    git_commit,
    git_dirty,
    set_seed,
    sha256_file,
)
from gated_dual_ema_msd.training.r2_stage import (
    StageResult,
    train_stage,
    inference_max_length,
    persist_selected_dev,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-process EXP-001-R2 runner")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--offline", action="store_true", help="run offline locally on full dataset without remote W&B/HF uploads")
    parser.add_argument("--debug", action="store_true", help="small local gate; disables every remote write")
    parser.add_argument("--tiny-model", help="debug-only local/HF tiny model identity (requires --debug)")
    return parser


def main(args) -> dict:
    cfg_path = pathlib.Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    validate_config(cfg)
    tiny_model = getattr(args, "tiny_model", None)
    wandb_enabled = (
        not args.debug
        and not getattr(args, "offline", False)
        and cfg.get("wandb", {}).get("enabled", True)
        and wandb_helper.wandb_enabled(cfg)
        and os.environ.get("WANDB_DISABLED") != "true"
        and os.environ.get("WANDB_MODE") != "offline"
    )
    hf_enabled = (
        not args.debug
        and not getattr(args, "offline", False)
        and cfg.get("hf_hub", {}).get("enabled", True)
        and hf_hub_helper.hf_enabled(cfg)
    )
    is_offline = not wandb_enabled and not hf_enabled
    if not args.debug and not is_offline and git_dirty():
        print("[git] Notice: worktree dirty or running uncommitted workspace archive")
    set_seed(cfg["project"].get("seed", 42))

    if hf_enabled:
        try:
            hf_hub_helper.ensure_repo(cfg["hf_hub"]["repo_id"], private=cfg["hf_hub"]["private"])
        except Exception as e:
            print(f"[hf_hub] warning: ensure_repo failed ({e})", flush=True)

    out_root = fresh_output_root(cfg, args.debug)
    rows, manifests = verify_and_load_rows(cfg, args.debug)
    config_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    row_counts = {
        name: (
            {split: len(split_rows) for split, split_rows in value.items()}
            if name == "laurer_mtvi_splits"
            else len(value)
        )
        for name, value in rows.items()
    }
    hparams = {
        "experiment_id": cfg["project"]["experiment_id"], "seed": cfg["project"].get("seed", 42),
        "config_sha256": config_hash, "git_commit": git_commit(),
        "model": cfg["model"], "training": cfg["training"],
        "counts": row_counts,
        "auxiliary_nli": cfg["data"].get("auxiliary_nli"),
        "debug": args.debug, "tiny_model": tiny_model,
        "offline": is_offline,
    }

    run = None
    run_finished = False
    try:
        # The only W&B initialization in this process; both stages share this run.
        if wandb_enabled:
            run = wandb_helper.init_run(
                cfg, cfg["wandb"]["run_name"], hparams,
                tags=[
                    "EXP-001-R2", "full-stage1-stage2", "maxlen512", f"seed{cfg['project'].get('seed', 42)}", "no-replay",
                    *(["m25", "mtvi", "macro-bi"] if cfg["wandb"]["run_name"] == M25_RUN_NAME else []),
                    *(
                        ["m32", "jtt", "compute-matched", "hard-example-weighting"]
                        if cfg["wandb"]["run_name"] == M32_JTT_RUN_NAME
                        else []
                    ),
                    *(
                        ["m45", "architecture", "signed-relation-delta"]
                        if cfg["wandb"]["run_name"] == M45_SIGNED_DELTA_RUN_NAME
                        else []
                    ),
                    *(
                        ["m46", "architecture", "hier-e-first"]
                        if cfg["wandb"]["run_name"] == M46_HIER_E_FIRST_RUN_NAME
                        else []
                    ),
                    *(
                        ["m47", "loss", "mirror-symmetry", "dev-only-test-locked"]
                        if cfg["wandb"]["run_name"] == M47_MIRROR_SYMMETRY_RUN_NAME
                        else []
                    ),
                ],
            )
            run.summary["lifecycle/status"] = "stage2_training" if cfg["training"].get("skip_stage1") else "stage1_training"
            if cfg["wandb"]["run_name"] == M25_RUN_NAME:
                run.summary["stage1/balance/configured_native_fraction"] = 0.5
                run.summary["stage1/balance/configured_auxiliary_fraction"] = 0.5

        model_name = tiny_model or cfg["model"]["name"]
        revision = None if tiny_model else cfg["model"]["revision"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        model = FlatCafeBERT(
            model_name, fallback_models=[], dropout=cfg["model"]["dropout"], num_labels=3,
            gradient_checkpointing=cfg["model"].get("gradient_checkpointing", False),
            label_smoothing=cfg["model"]["label_smoothing"], revision=revision,
            class_weights=cfg["model"].get("class_weights"),
            pool_mode=cfg["model"].get("pool_mode", "cls"),
            head_hidden=cfg["model"].get("head_hidden"),
            sep_token_id=getattr(tokenizer, "sep_token_id", None) or 2,
            alignment_dim=cfg["model"].get("alignment_dim", 256),
            relation_hidden=cfg["model"].get("relation_hidden"),
            gate_bias=cfg["model"].get("gate_bias", -2.0),
            relation_delta_mode=cfg["model"].get("relation_delta_mode", "absolute"),
            hierarchical_e_first=cfg["model"].get("hierarchical_e_first", False),
            use_layer_mix=cfg["model"].get("use_layer_mix", False),
            focal_gamma=cfg["model"].get("focal_gamma", 0.0),
            class_margins=cfg["model"].get("class_margins"),
            use_multi_sample_dropout=cfg["model"].get("use_multi_sample_dropout", False),
            msd_dropouts=cfg["model"].get("msd_dropouts"),
            relation_features_mode=cfg["model"].get("relation_features_mode", "standard"),
            use_supcon=cfg["model"].get("use_supcon", False),
            supcon_weight=cfg["model"].get("supcon_weight", 0.1),
            supcon_temperature=cfg["model"].get("supcon_temperature", 0.07),
            head_architecture=cfg["model"].get("head_architecture", "standard"),
            classifier_mlp=cfg["model"].get("classifier_mlp", False),
            classifier_type=cfg["model"].get("classifier_type", "linear"),
            classifier_hidden=cfg["model"].get("classifier_hidden", 256),
            classifier_dropout=cfg["model"].get("classifier_dropout"),
            classifier_act=cfg["model"].get("classifier_act", "gelu"),
            segment_pooling=cfg["model"].get("segment_pooling", "attentive"),
            use_gate=cfg["model"].get("use_gate", True),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        stage1_dir = out_root / "stage1_best"
        if hf_enabled:
            hf_repo = cfg["hf_hub"]["repo_id"]
            hf_private = cfg["hf_hub"]["private"]
            try:
                hf_hub_helper.ensure_repo(hf_repo, private=hf_private)
                print(f"[hf_hub] ☁️ Hugging Face repository verified: {hf_repo}", flush=True)
            except Exception as e:
                print(f"[hf_hub] warning: ensure_repo failed ({e})", flush=True)

        # Lưu checkpoint cao nhất cục bộ trong lúc train, CHỈ upload lên HF 1 lần duy nhất khi run hoàn tất
        stage1_best_upload = stage1b_best_upload = stage2_best_upload = None

        skip_stage1 = (
            cfg["training"].get("skip_stage1", False)
            or cfg["training"].get("stage1") is None
            or cfg["project"].get("experiment_mode") in ("target_only", "no_stilts")
        )
        predictions_dir = out_root / "predictions"
        if skip_stage1:
            stage1 = None
            stage1_dev = None
            stage1_step_offset = 0
            print("[EXP-001-R2] Stage 1 skipped: target-only / no-STILTs ablation active ✓", flush=True)
        else:
            stage1_mix = cfg["training"]["stage1"]["batch_mix"]
            if cfg["training"]["stage1"].get("mt_vi_augmentation"):
                stage1_sets = {
                    "vinli": rows["vinli_train"],
                    LAURER_MTVI_DOMAIN: rows["laurer_mtvi_splits"],
                }
                print(
                    "[EXP-001-R2] M25 stage1 MACRO-BI mapping: "
                    "vinli:2 + laurer_mtvi:2; uniform five-split cycle seed=42",
                    flush=True,
                )
            else:
                stage1_sets = {src: rows.get(f"{src}_train") for src in stage1_mix}
            if any(v is None for v in stage1_sets.values()):
                raise RuntimeError(f"stage1 sources missing rows: {stage1_mix}")
            if "vimednli" in stage1_mix:
                print("[EXP-001-R2] stage1 multi-source: vinli:2 + vimednli:2", flush=True)
            stage1 = train_stage(
                model, stage1_sets, rows["vinli_dev"], tokenizer, cfg,
                "stage1", device, stage1_dir, run, debug=args.debug,
                on_best_improve=stage1_best_upload,
            )
            load_best(model, stage1_dir, device)
            stage1_dev = persist_selected_dev(
                model, rows["vinli_dev"], tokenizer, cfg, device,
                predictions_dir / "stage1_best_vinli_dev_predictions.csv", stage1.best_metric,
            )
            stage1_step_offset = stage1.optimizer_steps

        # Optional English NLI knowledge bridge (M41 MNLI / M42 ANLI) between
        # Stage 1 and Stage 2. Selection stays on ViNLI dev; ViANLI test is
        # never touched here. Bridge best checkpoint is pushed to HF and later
        # published as stage1b_best in the immutable artifact payload.
        bridge_result = None
        bridge_dir = out_root / "stage1b_best"
        stage1b_dev = None
        bridge_step_offset = 0
        bridge_cfg = cfg["training"].get("stage1b")
        if bridge_cfg is not None:
            bname = bridge_cfg["dataset"]
            if run:
                run.summary["lifecycle/status"] = "bridge_training"
            print(
                f"[EXP-001-R2] English NLI bridge active: {bname} "
                f"rows={len(rows[bname])} selection=vinli_dev lr={bridge_cfg['learning_rate']}",
                flush=True,
            )
            bridge_result = train_stage(
                model,
                {bname: rows[bname]},
                rows["vinli_dev"],
                tokenizer,
                cfg,
                "stage1b",
                device,
                bridge_dir,
                run,
                wandb_step_offset=stage1.optimizer_steps,
                debug=args.debug,
                on_best_improve=stage1b_best_upload,
            )
            load_best(model, bridge_dir, device)
            stage1b_dev = persist_selected_dev(
                model, rows["vinli_dev"], tokenizer, cfg, device,
                predictions_dir / "stage1b_best_vinli_dev_predictions.csv",
                bridge_result.best_metric,
            )
            bridge_step_offset = bridge_result.optimizer_steps

        # Stage 2 begins from the reloaded best Stage-1 weights. M32 first
        # trains the exact two-epoch JTT identification model, freezes every
        # train error, reloads Stage 1, and then runs compute-matched weighted
        # ERM for five epochs. Other runs retain the historical single phase.
        stage2_target = cfg["training"]["stage2"].get("dataset", "vianli")
        stage2_sets = {stage2_target: rows[f"{stage2_target}_train"]}
        if cfg["training"]["stage2"].get("mt_vi_augmentation"):
            stage2_sets[LAURER_MTVI_DOMAIN] = rows["laurer_mtvi_splits"]
        if cfg["training"]["stage2"].get("replay"):
            stage2_sets["vinli"] = rows["vinli_train"]
            print(f"[EXP-001-R2] stage2 replay: {cfg['training']['stage2']['batch_mix']}", flush=True)

        identification = None
        identification_verification = None
        identification_dir = out_root / "stage2_jtt_identification_final"
        identification_path = (
            predictions_dir / "jtt_identification_vianli_train_predictions.csv"
        )
        hard_example_ids: frozenset[str] | None = None
        hard_example_rows_by_gold: Counter[str] = Counter()
        stage2_step_offset = (stage1.optimizer_steps if stage1 is not None else 0) + bridge_step_offset
        if cfg["wandb"]["run_name"] == M32_JTT_RUN_NAME:
            if run:
                run.summary["lifecycle/status"] = "stage2_jtt_identification"
            identification = train_stage(
                model,
                {"vianli": rows["vianli_train"]},
                rows["vianli_dev"],
                tokenizer,
                cfg,
                "stage2_identification",
                device,
                identification_dir,
                run,
                wandb_step_offset=stage1.optimizer_steps,
                debug=args.debug,
            )
            load_best(model, identification_dir, device)
            _, _, identification_frame = run_inference(
                model,
                rows["vianli_train"],
                tokenizer,
                cfg["model"]["max_length"],
                device,
                cfg["training"]["physical_batch_size"] * 2,
                compute_loss=False,
            )
            identification_verification = save_predictions(
                identification_frame,
                identification_path,
                rows["vianli_train"],
            )
            hard_example_ids = jtt.frozen_error_ids(
                identification_frame.to_dict(orient="records"),
                rows["vianli_train"],
            )
            hard_example_rows_by_gold.update(
                row["label"]
                for row in rows["vianli_train"]
                if row["id"] in hard_example_ids
            )
            if not args.debug and len(hard_example_ids) >= len(rows["vianli_train"]):
                raise RuntimeError("M32 JTT identification marked every training row hard")
            stage2_step_offset += identification.optimizer_steps
            if run:
                run.summary["stage2/jtt/identification_error_rows"] = len(hard_example_ids)
                run.summary["stage2/jtt/identification_error_fraction"] = (
                    len(hard_example_ids) / len(rows["vianli_train"])
                )
                for label in ("E", "C", "N"):
                    run.summary[
                        f"stage2/jtt/identification_error_rows_gold_{label}"
                    ] = hard_example_rows_by_gold[label]
                run.summary["stage2/jtt/hard_example_weight"] = 5.0
                run.summary["stage2/jtt/ordinary_example_weight"] = 1.0
                run.summary["stage2/jtt/compute_matched"] = True
            # The final JTT model must start from the exact same selected Stage-1
            # checkpoint, never from the identification model.
            load_best(model, stage1_dir, device)

        stage2_lora = cfg["training"]["stage2"].get("lora")
        if stage2_lora is not None:
            reference_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            replaced = apply_lora_to_model(
                model,
                r=int(stage2_lora["r"]),
                alpha=float(stage2_lora["alpha"]),
                lora_dropout=float(stage2_lora["lora_dropout"]),
            )
            if not args.debug and replaced != int(stage2_lora["expected_target_modules"]):
                raise RuntimeError(
                    f"LoRA replaced {replaced} modules != configured "
                    f"{stage2_lora['expected_target_modules']}"
                )
            if args.debug and replaced < 2:
                raise RuntimeError(
                    f"LoRA debug gate replaced too few modules: {replaced}"
                )
            assert_lora_eval_noop(model, reference_state)
            # Adapters are created with the wrapped weight's device, but call .to()
            # again so any later wrap/init cannot leave parameters off-device.
            model.to(device)
            if run:
                run.summary["stage2/lora/replaced_modules"] = replaced
                run.summary["stage2/lora/r"] = int(stage2_lora["r"])
                run.summary["stage2/lora/alpha"] = float(stage2_lora["alpha"])
            print(
                f"[EXP-001-R2] LoRA active (stage2 only): r={stage2_lora['r']} "
                f"alpha={stage2_lora['alpha']} modules={replaced}",
                flush=True,
            )

        if run:
            run.summary["lifecycle/status"] = "stage2_training"
        stage2_dir = out_root / "stage2_best"
        stage2 = train_stage(
            model, stage2_sets, rows[f"{stage2_target}_dev"], tokenizer, cfg,
            "stage2", device, stage2_dir, run,
            wandb_step_offset=stage2_step_offset, debug=args.debug,
            on_best_improve=stage2_best_upload,
            hard_example_ids=hard_example_ids,
        )
        load_best(model, stage2_dir, device)
        stage2_dev = persist_selected_dev(
            model, rows[f"{stage2_target}_dev"], tokenizer, cfg, device,
            predictions_dir / f"stage2_best_{stage2_target}_dev_predictions.csv",
            stage2.best_metric,
        )
        # Target-test inference is only enabled for pre-approved end-to-end
        # runs. ViANLI target test stays locked; M48/M49 target ViMedNLI test
        # has never been exposed and is used as the paper number.
        target_test = None
        target_test_key = f"{stage2_target}_test"
        if cfg.get("evaluation", {}).get("target_test_enabled", True) and target_test_key in rows:
            test_metrics, _, test_frame = run_inference(
                model, rows[target_test_key], tokenizer,
                inference_max_length(cfg, rows[target_test_key]), device,
                cfg["training"]["physical_batch_size"] * 2, compute_loss=False,
            )
            target_test = {
                **save_predictions(
                    test_frame,
                    predictions_dir / f"stage2_best_{stage2_target}_test_predictions.csv",
                    rows[f"{stage2_target}_test"],
                ),
                "metrics": test_metrics,
                "path": str(
                    predictions_dir / f"stage2_best_{stage2_target}_test_predictions.csv"
                ),
            }
            print("\n" + "=" * 80, flush=True)
            print(f"🎯 FINAL PAPER RESULTS: [{stage2_target.upper()} TEST SET EVALUATION]", flush=True)
            print("=" * 80, flush=True)
            print(f"  • Test Macro-F1:  {test_metrics['macro_f1']:.4f} ({test_metrics['macro_f1']*100:.2f}%)", flush=True)
            print(f"  • Test Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)", flush=True)
            print(f"  • Per-Class F1:   Entailment={test_metrics['f1_E']:.4f} | Contradiction={test_metrics['f1_C']:.4f} | Neutral={test_metrics['f1_N']:.4f}", flush=True)
            if stage1 is not None:
                print(f"  • Stage 1 Best (ViNLI Dev):  {stage1.best_metric:.4f}", flush=True)
            print(f"  • Stage 2 Best ({stage2_target.upper()} Dev): {stage2.best_metric:.4f}", flush=True)
            print(f"  • Predictions: {predictions_dir / f'stage2_best_{stage2_target}_test_predictions.csv'}", flush=True)
            print("=" * 80 + "\n", flush=True)
            if run:
                test_summary = {
                    f"stage2/{stage2_target}_test_macro_f1": test_metrics["macro_f1"],
                    f"stage2/{stage2_target}_test_accuracy": test_metrics["accuracy"],
                    f"stage2/{stage2_target}_test_f1_E": test_metrics["f1_E"],
                    f"stage2/{stage2_target}_test_f1_C": test_metrics["f1_C"],
                    f"stage2/{stage2_target}_test_f1_N": test_metrics["f1_N"],
                }
                for key, value in test_summary.items():
                    run.summary[key] = value
                # The final Stage-2 dev evaluation may already own the last optimizer
                # step. Commit real test metrics at the next monotonic W&B step so
                # the history payload cannot be discarded as stale.
                wandb_helper.log_step(
                    run, test_summary,
                    step=stage2_step_offset + stage2.optimizer_steps + 1,
                )

        metadata = {
            "experiment_id": "EXP-001-R2", "run_name": cfg["wandb"]["run_name"],
            "protocol": (
                "one_process_fresh_vinli_then_vimednli_mirror_symmetry"
                if cfg["wandb"]["run_name"] == M49_VIMEDNLI_MIRROR_RUN_NAME
                else (
                    "one_process_fresh_vinli_then_vimednli_ce"
                    if cfg["wandb"]["run_name"] == M48_VIMEDNLI_CE_RUN_NAME
                    else (
                        "one_process_fresh_vinli_then_vianli_mirror_symmetry_stage2_dev_only"
                        if cfg["wandb"]["run_name"] == M47_MIRROR_SYMMETRY_RUN_NAME
                        else (
                            "one_process_m25_50_50_vinli_mtvi_then_clean_vianli_no_replay"
                            if cfg["wandb"]["run_name"] == M25_RUN_NAME
                            else (
                                "one_process_fresh_vinli_then_compute_matched_jtt_vianli"
                                if cfg["wandb"]["run_name"] == M32_JTT_RUN_NAME
                                else (
                                    "one_process_fresh_vinli_then_en_bridge_then_clean_vianli_no_replay"
                                    if bridge_result is not None
                                    else "one_process_fresh_vinli_then_clean_vianli_no_replay"
                                )
                            )
                        )
                    )
                )
            ),
            "seed": 42, "debug": args.debug, "tiny_model": tiny_model,
            "target_test_accessed": cfg["evaluation"].get("target_test_enabled") is True,
            "target_dataset": stage2_target,
            "source_test_accessed": False,
            "frozen_final_evaluation": cfg["evaluation"].get("method_frozen") is True,
            "git_commit": git_commit(),
            "config_sha256": config_hash,
            "model": dict(cfg["model"]),
            "dataset_revisions": {
                "vinli": manifests["vinli"]["source"]["revision"],
                "vianli": manifests["vianli"]["source"]["revision"],
                **(
                    {"vimednli": manifests["vimednli"]["source"]["revision"]}
                    if "vimednli" in manifests
                    else {}
                ),
                **(
                    {"laurer_vi_nli": manifests["laurer_vi_nli"]["source"]["revision"]}
                    if "laurer_vi_nli" in manifests
                    else {}
                ),
                **(
                    {
                        bridge_cfg["dataset"]: manifests[bridge_cfg["dataset"]]["source"]["revision"]
                    }
                    if bridge_cfg is not None
                    else {}
                ),
            },
            "auxiliary_nli": cfg["data"].get("auxiliary_nli"),
            "stage1": ({**stage1.__dict__, "selection_split": "vinli_dev", "dev_prediction_verification": stage1_dev} if stage1 is not None else None),
            "stage1b": (
                {
                    **bridge_result.__dict__,
                    "dataset": bridge_cfg["dataset"],
                    "selection_split": "vinli_dev",
                    "dev_prediction_verification": stage1b_dev,
                }
                if bridge_result is not None
                else None
            ),
            "stage2": {
                **stage2.__dict__,
                "selection_split": f"{stage2_target}_dev",
                "replay": False,
                "dev_prediction_verification": stage2_dev,
                "test_prediction_verification": target_test,
                **(
                    {
                        "jtt": {
                            "identification": identification.__dict__,
                            "identification_prediction_verification": (
                                identification_verification
                            ),
                            "identification_error_rows": len(hard_example_ids or ()),
                            "identification_error_fraction": (
                                len(hard_example_ids or ()) / len(rows["vianli_train"])
                            ),
                            "identification_error_rows_by_gold": {
                                label: hard_example_rows_by_gold[label]
                                for label in ("E", "C", "N")
                            },
                            "hard_example_ids_sha256": hashlib.sha256(
                                "\n".join(sorted(hard_example_ids or ())).encode("utf-8")
                            ).hexdigest(),
                            "hard_example_weight": 5.0,
                            "ordinary_example_weight": 1.0,
                            "compute_matched": True,
                        }
                    }
                    if identification is not None
                    else {}
                ),
            },
            "checkpoint_directories": {
                **({"stage1_best": str(stage1_dir)} if stage1 is not None else {}),
                **({"stage1b_best": str(bridge_dir)} if bridge_result is not None else {}),
                "stage2_best": str(stage2_dir),
                **(
                    {"stage2_jtt_identification_final": str(identification_dir)}
                    if identification is not None
                    else {}
                ),
            },
            "wandb_run_url": run.url if run else None,
            "environment": environment_metadata(),
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        metadata_path = out_root / "run_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        if args.debug:
            print("[EXP-001-R2] debug/tiny gate complete; W&B and HF writes skipped")
        else:
            pred_paths = {
                **({"vinli_dev": pathlib.Path(stage1_dev["path"])} if stage1_dev is not None else {}),
                f"{stage2_target}_dev": pathlib.Path(stage2_dev["path"]),
                **(
                    {f"{stage2_target}_test": pathlib.Path(target_test["path"])}
                    if target_test is not None
                    else {}
                ),
                **(
                    {"stage1b_vinli_dev": pathlib.Path(stage1b_dev["path"])}
                    if stage1b_dev is not None
                    else {}
                ),
                **(
                    {
                        "jtt_identification_vianli_train": pathlib.Path(
                            identification_path
                        )
                    }
                    if identification is not None
                    else {}
                ),
            }
            if hf_enabled:
                if skip_stage1:
                    repo_id, revision = hf_hub_helper.push_run_artifacts(
                        cfg, stage2_dir, pred_paths, metadata
                    )
                else:
                    checkpoint_dirs = {
                        "stage1_best": stage1_dir,
                        "stage2_best": stage2_dir,
                        **({"stage1b_best": bridge_dir} if bridge_result is not None else {}),
                    }
                    repo_id, revision = hf_hub_helper.push_dual_stage_artifacts(
                        cfg, checkpoint_dirs, pred_paths, metadata
                    )
                metadata["hf_repo_id"] = repo_id
                metadata["hf_revision"] = revision
                metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"\n[hf_hub] 🚀 All models & predictions successfully published to HF Hub: https://huggingface.co/{repo_id} (tag: {revision})\n", flush=True)
                if run:
                    wandb_helper.log_hf_link(run, repo_id, revision)
            else:
                metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            if run:
                if stage1 is not None:
                    run.summary["stage1/best_vinli_dev_macro_f1"] = stage1.best_metric
                if bridge_result is not None:
                    run.summary["stage1b/best_vinli_dev_macro_f1"] = bridge_result.best_metric
                run.summary[f"stage2/best_{stage2_target}_dev_macro_f1"] = stage2.best_metric
                run.summary["lifecycle/status"] = "complete"
        return metadata
    except Exception:
        if run:
            run.summary["lifecycle/status"] = "failed"
        raise
    finally:
        if run is not None and not run_finished:
            wandb_helper.finish(run)
            run_finished = True
