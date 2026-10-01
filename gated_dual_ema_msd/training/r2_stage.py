"""Stage execution, multi-source mixing, and checkpoint persistence for R2."""
from __future__ import annotations

import datetime as dt
import math
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from gated_dual_ema_msd.data import jtt, laurer_vi_nli
from gated_dual_ema_msd.data.dataset import MultiSourceBatchMixer, collate_rows
from gated_dual_ema_msd.evaluation.inference import run_inference
from gated_dual_ema_msd.training.consistency_losses import (
    R3F_DEFAULT_SIGMA,
    freelb_delta_step,
    freelb_params_from_cfg,
    init_freelb_delta,
    mirror_contradiction_symmetry_loss,
    r3f_loss,
    rdrop_loss,
    resolve_consistency_mode,
    sam_params_from_cfg,
)
from gated_dual_ema_msd.training.recadam import RecAdam
from gated_dual_ema_msd.training.sam import SAM, gradients_are_finite
from gated_dual_ema_msd.training.train_exp001 import save_predictions
from gated_dual_ema_msd.utils import hf_hub_helper, wandb_helper
from gated_dual_ema_msd.config.r2_validation import (
    ROOT,
    M25_RUN_NAME,
    M32_JTT_RUN_NAME,
    M47_MIRROR_SYMMETRY_RUN_NAME,
    VIMEDNLI_TARGET_RUNS,
    LAURER_MTVI_DOMAIN,
)
from gated_dual_ema_msd.training.checkpoints import save_checkpoint

@dataclass(frozen=True)
class StageResult:
    best_metric: float
    optimizer_steps: int
    best_eval_step: int
    balance_history: tuple[dict[str, Any], ...] = ()


def build_stage_mixer(
    train_sets: dict[str, Any],
    stage: dict,
    seed: int,
    hard_example_ids: frozenset[str] | None = None,
):
    """Build the registered generic mixer or the M25-M31 exact-ratio mixer."""
    jtt_cfg = stage.get("jtt")
    if jtt_cfg is not None:
        if set(train_sets) != {"vianli"}:
            raise RuntimeError("JTT final stage must receive exactly ViANLI rows")
        if hard_example_ids is None:
            raise RuntimeError("JTT final stage requires the frozen identification error set")
        rows = train_sets["vianli"]
        batch_size = stage["batch_mix"]["vianli"]
        baseline_draws = math.ceil(len(rows) / batch_size) * batch_size
        configured_draws = int(jtt_cfg["draws_per_epoch"])
        draws_per_epoch = configured_draws if len(rows) == 8010 else baseline_draws
        return jtt.ComputeMatchedJTTBatchMixer(
            rows,
            hard_example_ids,
            batch_size=batch_size,
            hard_weight=float(jtt_cfg["hard_example_weight"]),
            ordinary_weight=float(jtt_cfg["ordinary_example_weight"]),
            seed=int(jtt_cfg["seed"]),
            draws_per_epoch=draws_per_epoch,
        )
    augmentation = stage.get("mt_vi_augmentation")
    if augmentation is not None:
        native_key = augmentation["native_source"]
        non_native = set(train_sets) - ({native_key} | {LAURER_MTVI_DOMAIN})
        if non_native or set(train_sets) != {native_key, LAURER_MTVI_DOMAIN}:
            raise RuntimeError("auxiliary stage must receive exactly native and Laurer MT rows")
        auxiliary_rows = train_sets[LAURER_MTVI_DOMAIN]
        if not isinstance(auxiliary_rows, dict):
            raise RuntimeError("auxiliary rows must remain split-indexed without concatenation")
        selected_splits = augmentation["auxiliary_splits"]
        selected_rows = {split: auxiliary_rows[split] for split in selected_splits}
        return laurer_vi_nli.MtViMacroBiBatchMixer(
            train_sets[native_key],
            selected_rows,
            native_per_batch=stage["batch_mix"][native_key],
            auxiliary_per_batch=stage["batch_mix"][LAURER_MTVI_DOMAIN],
            seed=seed,
            text_mode=augmentation["text_mode"],
            aux_splits=augmentation["auxiliary_splits"],
            code_switch_fraction=augmentation["code_switch_fraction"],
            epoch_anchor=augmentation.get("epoch_anchor", "aux"),
        )
    return MultiSourceBatchMixer(
        train_sets,
        stage["batch_mix"],
        list(train_sets),
        seed,
    )


def configured_batch_fraction(stage: dict, source: str) -> float:
    """Return one source's configured physical-batch fraction, fail closed."""
    batch_mix = stage.get("batch_mix") or {}
    if source not in batch_mix:
        raise ValueError(f"source {source!r} is absent from batch_mix")
    if any(not isinstance(value, int) or value <= 0 for value in batch_mix.values()):
        raise ValueError("batch_mix counts must be positive integers")
    return batch_mix[source] / sum(batch_mix.values())


def inference_max_length(cfg: dict, rows: list[dict]) -> int:
    """Per-domain max length for evaluation/inference."""
    if not rows:
        raise ValueError("inference rows must be non-empty to resolve domain max length")
    domain = rows[0].get("domain")
    if domain == "vimednli":
        ml = cfg.get("data", {}).get("max_lengths", {}).get("vimednli")
        if ml is not None and int(ml) != 256:
            raise RuntimeError("ViMedNLI evaluation requires max_length 256")
        return 256
    ml = cfg.get("data", {}).get("max_lengths") or {}
    if domain in ml:
        return int(ml[domain])
    return int(cfg.get("model", {}).get("max_length", 512))



def train_stage(model, train_sets: dict[str, Any], dev_rows: list[dict], tokenizer,
                cfg: dict, stage_name: str, device, checkpoint_dir: pathlib.Path,
                run, wandb_step_offset: int = 0, debug: bool = False,
                on_best_improve=None,
                hard_example_ids: frozenset[str] | None = None) -> StageResult:
    """Train one stage with a fresh optimizer; W&B steps remain global/monotonic.

    train_sets maps dataset name -> rows (e.g. {"vianli": [...]} or, with replay,
    {"vianli": [...], "vinli": [...]}); batch_mix from the stage config defines the
    exact per-step composition. Every time a better dev checkpoint is found it is
    saved locally, then on_best_improve(checkpoint_dir, metric, optimizer_step) is
    invoked (used to overwrite the experiment's best artifact on HF). Early
    stopping patience=3 stops the stage; the best checkpoint is always kept.
    """
    training, stage = cfg["training"], cfg["training"][stage_name]
    mixout_cfg = stage.get("mixout")
    recadam_cfg = stage.get("recadam")
    if mixout_cfg is not None:
        mixout_counts = apply_mixout_to_model(model, float(mixout_cfg["p"]))
        print(
            f"[{stage_name}] Mixout active: p={mixout_cfg['p']} "
            f"reference=stage entry weights {mixout_counts}",
            flush=True,
        )
    physical_batch = training["physical_batch_size"]
    mixer = build_stage_mixer(
        train_sets, stage, cfg["project"]["seed"], hard_example_ids=hard_example_ids
    )
    steps_per_epoch = mixer.epoch_steps()
    grad_accum = training["gradient_accumulation_steps"]
    max_epochs = cfg["debug"]["max_epochs_per_stage"] if debug else stage["max_epochs"]
    eval_steps = cfg["debug"]["eval_optimizer_steps"] if debug else stage["eval_optimizer_steps"]
    total_optimizer_steps = math.ceil(steps_per_epoch / grad_accum) * max_epochs
    stage_weight_decay = float(stage.get("weight_decay", training["weight_decay"]))
    if recadam_cfg is not None:
        entry_state = {
            name: value.detach().clone().cpu()
            for name, value in model.state_dict().items()
        }
        base_optimizer = RecAdam(
            model.parameters(),
            reference_state=entry_state,
            named_params=list(model.named_parameters()),
            lr=stage["learning_rate"],
            weight_decay=stage_weight_decay,
            gamma=float(recadam_cfg["gamma"]),
            anneal_k=float(recadam_cfg["anneal_k"]),
            anneal_t0=float(recadam_cfg["anneal_t0"]),
        )
        print(f"[{stage_name}] RecAdam active: gamma={recadam_cfg['gamma']} "
              f"k={recadam_cfg['anneal_k']} t0={recadam_cfg['anneal_t0']} "
              f"(reference=stage entry weights)", flush=True)
    else:
        head_lr = stage.get("head_learning_rate") or stage.get("head_lr")
        if head_lr is not None and hasattr(model, "backbone"):
            backbone_params = list(model.backbone.parameters())
            backbone_param_ids = set(id(p) for p in backbone_params)
            head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
            param_groups = [
                {"params": backbone_params, "lr": stage["learning_rate"]},
                {"params": head_params, "lr": head_lr},
            ]
            base_optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=stage_weight_decay,
            )
        else:
            base_optimizer = torch.optim.AdamW(
                model.parameters(), lr=stage["learning_rate"],
                weight_decay=stage_weight_decay,
            )
    sam_params = sam_params_from_cfg(cfg["model"])
    optimizer = (
        SAM(model.parameters(), base_optimizer=base_optimizer, rho=sam_params["rho"])
        if sam_params is not None
        else base_optimizer
    )
    # SAM calls ``base_optimizer.step()`` from second_step; attach the scheduler
    # to that optimizer so PyTorch observes the correct step order and state.
    scheduler_optimizer = base_optimizer if sam_params is not None else optimizer
    scheduler = get_linear_schedule_with_warmup(
        scheduler_optimizer,
        int(total_optimizer_steps * training["warmup_ratio"]),
        total_optimizer_steps,
    )
    use_fp16 = device.type == "cuda" and training["fp16_train"]
    scaler = torch.amp.GradScaler(device.type, enabled=use_fp16)
    sam_second_scaler = (
        torch.amp.GradScaler(device.type, enabled=use_fp16)
        if sam_params is not None else None
    )
    best_metric, best_eval_step, no_improve, optimizer_step = -1.0, -1, 0, 0
    stopped = False
    balance_history: list[dict[str, Any]] = []
    current_epoch = 0
    current_epoch_complete = False
    epoch_source_counts: Counter[str] = Counter()
    epoch_aux_counts: Counter[str] = Counter()
    epoch_jtt_counts: Counter[str] = Counter()
    mtvi_mixer = (
        mixer if isinstance(mixer, laurer_vi_nli.MtViMacroBiBatchMixer) else None
    )
    jtt_mixer = mixer if isinstance(mixer, jtt.ComputeMatchedJTTBatchMixer) else None
    is_m25_balance = mtvi_mixer is not None

    def balance_payload(epoch: int, complete: bool) -> dict[str, Any]:
        if jtt_mixer is not None:
            total_rows = epoch_jtt_counts["total"]
            hard_rows = epoch_jtt_counts["hard"]
            return {
                f"{stage_name}/balance/epoch": epoch,
                f"{stage_name}/balance/epoch_complete": int(complete),
                f"{stage_name}/balance/jtt_hard_weight": jtt_mixer.hard_weight,
                f"{stage_name}/balance/jtt_ordinary_weight": jtt_mixer.ordinary_weight,
                f"{stage_name}/balance/jtt_hard_set_rows": len(
                    jtt_mixer.hard_example_ids
                ),
                f"{stage_name}/balance/jtt_expected_hard_draw_fraction": (
                    jtt_mixer.expected_hard_draw_fraction
                ),
                f"{stage_name}/balance/jtt_observed_hard_rows": hard_rows,
                f"{stage_name}/balance/jtt_observed_total_rows": total_rows,
                f"{stage_name}/balance/jtt_observed_hard_draw_fraction": (
                    hard_rows / total_rows if total_rows else 0.0
                ),
            }
        if not is_m25_balance:
            return {}
        native_key = stage["mt_vi_augmentation"]["native_source"]
        native_rows = epoch_source_counts[native_key]
        auxiliary_rows = epoch_source_counts[LAURER_MTVI_DOMAIN]
        total_rows = native_rows + auxiliary_rows
        native_per_batch = stage["batch_mix"][native_key]
        auxiliary_per_batch = stage["batch_mix"][LAURER_MTVI_DOMAIN]
        if native_per_batch <= 0 or auxiliary_per_batch <= 0:
            raise RuntimeError(f"{stage_name} batch mix must include both sources")
        expected_native_frac = configured_batch_fraction(stage, native_key)
        observed_frac = native_rows / total_rows if total_rows else 0.0
        if abs(observed_frac - expected_native_frac) > 1e-9:
            raise RuntimeError(
                f"{stage_name} observed Stage ratio {observed_frac:.4f} "
                f"!= configured {expected_native_frac:.4f}"
            )
        if complete and mtvi_mixer is not None and mtvi_mixer.epoch_anchor == "aux":
            expected_split_counts = mtvi_mixer.full_epoch_aux_counts()
            observed_split_counts = {
                split: epoch_aux_counts[split] for split in LAURER_MTVI_SPLITS
                if epoch_aux_counts[split] or split in expected_split_counts
            }
            if observed_split_counts != expected_split_counts:
                raise RuntimeError(
                    f"{stage_name} complete epoch did not consume the pinned auxiliary "
                    f"composition: {observed_split_counts} != {expected_split_counts}"
                )
        payload = {
            f"{stage_name}/balance/epoch": epoch,
            f"{stage_name}/balance/epoch_complete": int(complete),
            f"{stage_name}/balance/configured_native_fraction": expected_native_frac,
            f"{stage_name}/balance/configured_auxiliary_fraction": 1.0 - expected_native_frac,
            f"{stage_name}/balance/observed_native_rows": native_rows,
            f"{stage_name}/balance/observed_auxiliary_rows": auxiliary_rows,
            f"{stage_name}/balance/observed_native_fraction": (
                native_rows / total_rows if total_rows else 0.0
            ),
            f"{stage_name}/balance/observed_auxiliary_fraction": (
                auxiliary_rows / total_rows if total_rows else 0.0
            ),
        }
        payload.update(
            {
                f"{stage_name}/balance/observed_{split}_rows": epoch_aux_counts[split]
                for split in LAURER_MTVI_SPLITS
            }
        )
        return payload

    def record_balance(epoch: int, complete: bool) -> dict[str, Any]:
        payload = balance_payload(epoch, complete)
        if not payload:
            return payload
        record = {
            key.removeprefix(f"{stage_name}/balance/"): value
            for key, value in payload.items()
        }
        balance_history.append(record)
        if jtt_mixer is not None:
            print(
                f"[{stage_name}] JTT balance epoch={epoch} complete={complete} "
                f"hard={record['jtt_observed_hard_rows']}/"
                f"{record['jtt_observed_total_rows']} "
                f"expected_fraction={record['jtt_expected_hard_draw_fraction']:.4f}",
                flush=True,
            )
            if run:
                prefix = f"{stage_name}/balance/epoch_{epoch}"
                for key, value in record.items():
                    run.summary[f"{prefix}/{key}"] = value
            return payload
        print(
            f"[{stage_name}] M25 balance epoch={epoch} complete={complete} "
            f"native={record['observed_native_rows']} "
            f"mt={record['observed_auxiliary_rows']} "
            f"aux_splits={dict(epoch_aux_counts)}",
            flush=True,
        )
        if run:
            prefix = f"{stage_name}/balance/epoch_{epoch}"
            run.summary[f"{prefix}/complete"] = complete
            run.summary[f"{prefix}/configured_native_fraction"] = record[
                "configured_native_fraction"
            ]
            run.summary[f"{prefix}/configured_auxiliary_fraction"] = record[
                "configured_auxiliary_fraction"
            ]
            run.summary[f"{prefix}/observed_native_rows"] = record["observed_native_rows"]
            run.summary[f"{prefix}/observed_auxiliary_rows"] = record[
                "observed_auxiliary_rows"
            ]
            for split in LAURER_MTVI_SPLITS:
                run.summary[f"{prefix}/observed_{split}_rows"] = epoch_aux_counts[split]
        return payload
    consistency_mode, consistency_params = resolve_consistency_mode(cfg["model"])
    if consistency_mode == "mirror_symmetry" and stage_name != consistency_params["stage"]:
        # M47 follows the paper's constrained-continuation protocol: Stage 1 is
        # ordinary supervised ViNLI fine-tuning, Stage 2 alone gets mirrored pairs.
        consistency_mode, consistency_params = None, {}
    if consistency_mode is not None:
        print(f"[{stage_name}] consistency loss active: {consistency_mode} "
              f"{consistency_params} (every optimizer step)", flush=True)
    freelb_params = freelb_params_from_cfg(cfg["model"])
    if freelb_params is not None:
        print(f"[{stage_name}] FreeLB active: {freelb_params} (every step)", flush=True)
    if sam_params is not None:
        print(f"[{stage_name}] SAM active: rho={sam_params['rho']} "
              "(two-pass, AMP with independent GradScalers)", flush=True)
    child_tune_p = cfg["model"].get("childtune_p")
    if child_tune_p is not None:
        print(f"[{stage_name}] Child-Tuning-F active: p_F={child_tune_p} "
              f"(gradient mask resampled every optimizer step, rescaled by 1/p)", flush=True)

    def evaluate(epoch: int | None = None) -> bool:
        nonlocal best_metric, best_eval_step, no_improve
        metrics, dev_loss, _ = run_inference(
            model, dev_rows, tokenizer, inference_max_length(cfg, dev_rows), device,
            physical_batch * 2, compute_loss=True,
        )
        tracking_step = wandb_step_offset + optimizer_step
        eval_payload = {
            f"{stage_name}/{stage['selection_split']}_macro_f1": metrics["macro_f1"],
            f"{stage_name}/{stage['selection_split']}_accuracy": metrics["accuracy"],
            f"{stage_name}/{stage['selection_split']}_f1_E": metrics["f1_E"],
            f"{stage_name}/{stage['selection_split']}_f1_C": metrics["f1_C"],
            f"{stage_name}/{stage['selection_split']}_f1_N": metrics["f1_N"],
            f"{stage_name}/{stage['selection_split']}_loss": dev_loss,
        }
        eval_payload.update(balance_payload(current_epoch, current_epoch_complete))
        wandb_helper.log_step(run, eval_payload, step=tracking_step)
        improved = metrics["macro_f1"] > best_metric
        if improved:
            best_metric = metrics["macro_f1"]
            best_eval_step = optimizer_step
            no_improve = 0
            save_checkpoint(model, tokenizer, checkpoint_dir, cfg, stage_name, best_metric, optimizer_step)
            if on_best_improve is not None:
                on_best_improve(checkpoint_dir, best_metric, optimizer_step)
        else:
            no_improve += 1
        detail = (f"[{stage_name}] eval epoch={epoch or '-'} step={optimizer_step} dev_acc={metrics['accuracy']:.4f} "
                  f"macro={metrics['macro_f1']:.4f} f1_E={metrics['f1_E']:.4f} f1_C={metrics['f1_C']:.4f} "
                  f"f1_N={metrics['f1_N']:.4f} loss={dev_loss:.4f} "
                  f"{'NEW BEST' if improved else 'no improve'} -> best_macro={best_metric:.4f}@step{best_eval_step} "
                  f"patience_left={max(0, (stage['early_stopping_patience_evals'] or 0) - no_improve)}")
        print(detail, flush=True)
        model.train()
        patience = stage["early_stopping_patience_evals"]
        min_complete_epochs = stage.get("early_stopping_min_complete_epochs")
        if is_m25_balance and min_complete_epochs and (
            current_epoch < min_complete_epochs
            or (current_epoch == min_complete_epochs and not current_epoch_complete)
        ):
            return False
        return patience is not None and no_improve >= patience

    sam_window: list[list[dict]] = []

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, max_epochs + 1):
        current_epoch = epoch
        current_epoch_complete = False
        epoch_source_counts.clear()
        epoch_aux_counts.clear()
        epoch_jtt_counts.clear()
        model.train()
        batches = mixer.batches()
        for batch_index, raw_rows in enumerate(
            tqdm(batches, total=steps_per_epoch, desc=f"{stage_name} epoch {epoch}"), 1
        ):
            if is_m25_balance:
                native_key = stage["mt_vi_augmentation"]["native_source"]
                expected_mix = {
                    native_key: stage["batch_mix"][native_key],
                    LAURER_MTVI_DOMAIN: stage["batch_mix"][LAURER_MTVI_DOMAIN],
                }
                observed_domains = Counter(row.get("domain") for row in raw_rows)
                if observed_domains != expected_mix:
                    raise RuntimeError(
                        f"M25 physical batch {batch_index} composition mismatch: "
                        f"{dict(observed_domains)} != {expected_mix}"
                    )
                epoch_source_counts[native_key] += observed_domains[native_key]
                epoch_source_counts[LAURER_MTVI_DOMAIN] += observed_domains[
                    LAURER_MTVI_DOMAIN
                ]
                epoch_aux_counts.update(
                    row["aux_split"]
                    for row in raw_rows
                    if row.get("domain") == LAURER_MTVI_DOMAIN
                )
            if jtt_mixer is not None:
                epoch_jtt_counts["total"] += len(raw_rows)
                epoch_jtt_counts["hard"] += sum(
                    row["id"] in jtt_mixer.hard_example_ids for row in raw_rows
                )
            batch = collate_rows(raw_rows, tokenizer, cfg["model"]["max_length"],
                                 max_lengths=cfg["data"].get("max_lengths"))
            tensors = {
                key: value.to(device) for key, value in batch.items()
                if key in ("input_ids", "attention_mask", "token_type_ids", "labels")
            }
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                if freelb_params is not None:
                    # FreeLB (arXiv:1909.11764): K adversarial passes on the word
                    # embedding output, gradients accumulated across passes, ONE
                    # optimizer update per batch. Delta init -> grad ascent with
                    # L2 per-example normalization -> projection back to radius.
                    embed_layer = model.backbone.get_input_embeddings()
                    delta = None
                    pass_losses = []
                    for _ in range(freelb_params["steps"]):
                        # Fresh embedding forward every pass: backward frees the
                        # previous pass's graph; gradients into the embedding
                        # weights accumulate exactly once per pass (FreeLB).
                        embeds_init = embed_layer(tensors["input_ids"])
                        if delta is None:
                            delta = init_freelb_delta(
                                embeds_init, tensors["attention_mask"],
                                freelb_params["init_mag"],
                            )
                        delta = delta.requires_grad_()
                        adv_output = model(
                            input_ids=tensors["input_ids"],
                            attention_mask=tensors["attention_mask"],
                            token_type_ids=tensors.get("token_type_ids"),
                            labels=tensors["labels"],
                            inputs_embeds=(embeds_init + delta).float(),
                        )
                        pass_loss = adv_output["loss"] / (
                            grad_accum * freelb_params["steps"]
                        )
                        pass_losses.append(float(pass_loss.item() * grad_accum))
                        scaler.scale(pass_loss).backward()
                        if delta.grad is not None:
                            delta = freelb_delta_step(
                                delta,
                                delta.grad.detach(),
                                freelb_params["step_size"],
                                freelb_params["max_norm"],
                            )
                    loss = sum(pass_losses) / freelb_params["steps"]
                    consistency_scalar = None
                    consistency_key = None
                    freelb_skip_backward = True
                else:
                    freelb_skip_backward = False
                if freelb_skip_backward is False and consistency_mode == "rdrop":
                    # R-Drop: two forward passes of the SAME batch under
                    # independent dropout masks; symmetric KL in FP32.
                    output1 = model(
                        input_ids=tensors["input_ids"],
                        attention_mask=tensors["attention_mask"],
                        token_type_ids=tensors.get("token_type_ids"),
                    )
                    output2 = model(
                        input_ids=tensors["input_ids"],
                        attention_mask=tensors["attention_mask"],
                        token_type_ids=tensors.get("token_type_ids"),
                    )
                    loss_parts = rdrop_loss(
                        logits1=output1["logits"], logits2=output2["logits"],
                        labels=tensors["labels"],
                        alpha=consistency_params["alpha"],
                        label_smoothing=cfg["model"].get("label_smoothing", 0.0),
                    )
                    loss = loss_parts["loss"] / grad_accum
                    consistency_scalar = float(loss_parts["kl_sym"].item())
                    consistency_key = "kl_sym"
                elif consistency_mode == "r3f":
                    # R3F: eps ~ N(0, sigma^2) on word embeddings ONCE per step;
                    # single backward through CE(noisy) + lambda * SKL(p, q).
                    loss_parts = r3f_loss(
                        model,
                        input_ids=tensors["input_ids"],
                        attention_mask=tensors["attention_mask"],
                        token_type_ids=tensors.get("token_type_ids"),
                        labels=tensors["labels"],
                        sigma=consistency_params["sigma"],
                        lambda_=consistency_params["lambda"],
                    )
                    loss = loss_parts["loss"] / grad_accum
                    consistency_scalar = float(loss_parts["skl"].item())
                    consistency_key = "skl"
                elif consistency_mode == "mirror_symmetry":
                    # Li et al. (EMNLP 2019) BERT+M: use exactly the same real
                    # pair in reversed order, without assigning it a label or
                    # persisting an augmented row. CE remains on the original
                    # ordering; only log p(contradiction) is symmetry-constrained.
                    mirrored_rows = [
                        {**row, "premise": row["hypothesis"], "hypothesis": row["premise"]}
                        for row in raw_rows
                    ]
                    mirrored_batch = collate_rows(
                        mirrored_rows,
                        tokenizer,
                        cfg["model"]["max_length"],
                        max_lengths=cfg["data"].get("max_lengths"),
                    )
                    mirrored_tensors = {
                        key: value.to(device) for key, value in mirrored_batch.items()
                        if key in ("input_ids", "attention_mask", "token_type_ids")
                    }
                    original_output = model(
                        input_ids=tensors["input_ids"],
                        attention_mask=tensors["attention_mask"],
                        token_type_ids=tensors.get("token_type_ids"),
                    )
                    mirrored_output = model(
                        input_ids=mirrored_tensors["input_ids"],
                        attention_mask=mirrored_tensors["attention_mask"],
                        token_type_ids=mirrored_tensors.get("token_type_ids"),
                    )
                    loss_parts = mirror_contradiction_symmetry_loss(
                        original_output["logits"],
                        mirrored_output["logits"],
                        tensors["labels"],
                        lambda_=consistency_params["lambda"],
                        label_smoothing=cfg["model"].get("label_smoothing", 0.0),
                    )
                    loss = loss_parts["loss"] / grad_accum
                    consistency_scalar = float(loss_parts["mirror_logprob_l1"].item())
                    consistency_key = "mirror_logprob_l1"
                else:
                    output = model(
                        input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"],
                        token_type_ids=tensors.get("token_type_ids"), labels=tensors["labels"],
                    )
                    loss = output["loss"] / grad_accum
                    consistency_scalar = None
                    consistency_key = None
            if not freelb_skip_backward:
                scaler.scale(loss).backward()
            if sam_params is not None:
                sam_window.append(raw_rows)
            if batch_index % grad_accum != 0 and batch_index != steps_per_epoch:
                continue
            scaler.unscale_(optimizer)
            if child_tune_p is not None:
                # Child-Tuning-F (Xu et al., EMNLP 2021): mask gradients with
                # Bernoulli(p_F) drawn fresh per optimizer step; kept entries
                # are rescaled by 1/p_F to stay unbiased.
                for param in model.parameters():
                    if param.grad is None:
                        continue
                    mask = (
                        torch.rand_like(param.grad) < child_tune_p
                    ).to(param.grad.dtype) / child_tune_p
                    param.grad.mul_(mask)
            if sam_params is not None:
                # SAM second pass: perturb to the sharpness point, re-forward the
                # SAME window batches, restore, then ONE base optimizer update.
                if not sam_window:
                    raise RuntimeError("SAM window cache is empty at step boundary")
                second_scaler = sam_second_scaler
                if second_scaler is None:
                    raise RuntimeError("SAM second-pass scaler was not initialized")
                first_grads_finite = gradients_are_finite(model.parameters())
                if not first_grads_finite:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    sam_window.clear()
                    print(f"[{stage_name}] SAM skipped non-finite first pass", flush=True)
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                optimizer.first_step(zero_grad=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_fp16
                ):
                    for window_rows in sam_window:
                        window_batch = collate_rows(
                            window_rows, tokenizer, cfg["model"]["max_length"],
                            max_lengths=cfg["data"].get("max_lengths"),
                        )
                        window_tensors = {
                            key: value.to(device)
                            for key, value in window_batch.items()
                            if key in ("input_ids", "attention_mask", "token_type_ids", "labels")
                        }
                        sam_out = model(
                            input_ids=window_tensors["input_ids"],
                            attention_mask=window_tensors["attention_mask"],
                            token_type_ids=window_tensors.get("token_type_ids"),
                            labels=window_tensors["labels"],
                        )
                        second_scaler.scale(sam_out["loss"] / grad_accum).backward()
                second_scaler.unscale_(optimizer)
                second_grads_finite = gradients_are_finite(model.parameters())
                if not second_grads_finite:
                    optimizer.restore_step(zero_grad=True)
                    scaler.update()
                    second_scaler.update()
                    sam_window.clear()
                    print(f"[{stage_name}] SAM skipped non-finite second pass", flush=True)
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                optimizer.second_step(zero_grad=True)
                scaler.update()
                second_scaler.update()
                scheduler.step()
                sam_window.clear()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            tracking_step = wandb_step_offset + optimizer_step
            eval_due = optimizer_step % eval_steps == 0
            current_epoch_complete = batch_index == steps_per_epoch
            balance_log = (
                record_balance(epoch, complete=True) if current_epoch_complete else {}
            )
            if optimizer_step % training["logging_steps"] == 0 or balance_log:
                log_payload = {
                    f"{stage_name}/train_loss": float(loss.item() * grad_accum),
                    f"{stage_name}/learning_rate": scheduler.get_last_lr()[0],
                    f"{stage_name}/epoch": epoch,
                }
                log_payload.update(balance_log)
                if consistency_scalar is not None:
                    log_payload[f"{stage_name}/{consistency_key}"] = consistency_scalar
                final_eval_will_reuse_step = (
                    current_epoch_complete and epoch == max_epochs and not eval_due
                )
                wandb_helper.log_step(
                    run,
                    log_payload,
                    step=tracking_step,
                    commit=not (eval_due or final_eval_will_reuse_step),
                )
            if eval_due and evaluate():
                if (is_m25_balance or jtt_mixer is not None) and not current_epoch_complete:
                    record_balance(epoch, complete=False)
                stopped = True
                break
        if stopped:
            break
    if not stopped and optimizer_step % eval_steps != 0:
        evaluate()
    if best_metric < 0 or not (checkpoint_dir / "pytorch_model.bin").exists():
        raise RuntimeError(f"{stage_name} completed without a selected checkpoint")
    return StageResult(best_metric, optimizer_step, best_eval_step, tuple(balance_history))


def load_best(model, checkpoint_dir: pathlib.Path, device) -> None:
    state = torch.load(checkpoint_dir / "pytorch_model.bin", map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)


def persist_selected_dev(model, rows: list[dict], tokenizer, cfg: dict, device,
                         path: pathlib.Path, expected_metric: float | None) -> dict:
    metrics, _, frame = run_inference(
        model, rows, tokenizer, inference_max_length(cfg, rows), device,
        cfg["training"]["physical_batch_size"] * 2, compute_loss=False,
    )
    verification = save_predictions(frame, path, rows)
    if expected_metric is not None and abs(metrics["macro_f1"] - expected_metric) > 1e-12:
        raise RuntimeError("reloaded selected checkpoint metric does not match selection metric")
    return {"metrics": metrics, **verification, "path": str(path)}
