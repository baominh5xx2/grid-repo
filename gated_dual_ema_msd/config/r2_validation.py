"""Validation logic and constants for EXP-001-R2 configurations."""
from __future__ import annotations

import pathlib
from typing import Any
from gated_dual_ema_msd.data import laurer_vi_nli
from gated_dual_ema_msd.training.consistency_losses import (
    R3F_DEFAULT_SIGMA,
    freelb_params_from_cfg,
    sam_params_from_cfg,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/experiments/exp001_r2_full_stage1_stage2_maxlen512_seed42.yaml"
MODEL_REVISION = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d"
VINLI_REVISION = "47bd78ac5d075bd00a3cb4cdd3ede4eec4acf8c2"
VIANLI_REVISION = "0fec8d6ecb043a61c609f9b51f80401fdf1e84d3"
VIMEDNLI_REVISION = "2cd94305ba48ae1ccf8782c1df9819ddad7f035f"
M25_RUN_NAME = "exp-001-r2-m25-mtvi-multinli-macrobi-maxlen512-seed42"
M32_JTT_RUN_NAME = "exp-001-r2-m32-jtt-cm-w5-id2-final5-maxlen512-seed42"
M39_LORA_RUN_NAME = "exp-001-r2-m39-lora-maxlen512-seed42"
M41_MNLI_BRIDGE_RUN_NAME = "exp-001-r2-m41-mnli-bridge-maxlen512-seed42"
M42_ANLI_BRIDGE_RUN_NAME = "exp-001-r2-m42-anli-bridge-maxlen512-seed42"
M45_SIGNED_DELTA_RUN_NAME = "exp-001-r2-m45-gated-signed-delta-maxlen512-seed42"
M46_HIER_E_FIRST_RUN_NAME = "exp-001-r2-m46-hier-e-first-maxlen512-seed42"
M47_MIRROR_SYMMETRY_RUN_NAME = "exp-001-r2-m47-mirror-symmetry-stage2-maxlen512-seed42"
M48_VIMEDNLI_CE_RUN_NAME = "exp-001-r2-m48-vimednli-ce-maxlen256-seed42"
M49_VIMEDNLI_MIRROR_RUN_NAME = "exp-001-r2-m49-vimednli-mirror-symmetry-maxlen256-seed42"
VIMEDNLI_TARGET_RUNS = {M48_VIMEDNLI_CE_RUN_NAME, M49_VIMEDNLI_MIRROR_RUN_NAME}
BRIDGE_RUN_DATASETS = {
    M41_MNLI_BRIDGE_RUN_NAME: "mnli",
    M42_ANLI_BRIDGE_RUN_NAME: "anli",
}
LAURER_MTVI_DOMAIN = laurer_vi_nli.DOMAIN
LAURER_MTVI_REVISION = laurer_vi_nli.REVISION
LAURER_MTVI_SPLITS = laurer_vi_nli.SPLIT_ORDER

AUX_RUNS = {
    M25_RUN_NAME: {
        "text_mode": "mt_vi",
        "aux_splits": list(LAURER_MTVI_SPLITS),
        "code_switch_fraction": 0.0,
        "epoch_anchor": "full_105k_pass",
        "stage1_dataset": "vinli_laurer_mtvi",
        "train_split": "vinli_train+laurer_vi_*",
        "experiment_mode": "fresh_mtvi_macrobi_then_vianli_gated_small",
        "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m25-mtvi-multinli-macrobi-maxlen512-seed42",
    },
    "exp-001-r2-m26-mtvi-anli-only-macrobi-maxlen512-seed42": {
        "text_mode": "mt_vi",
        "aux_splits": ["vi_anli"],
        "code_switch_fraction": 0.0,
        "epoch_anchor": "anli_25k_pass",
        "stage1_dataset": "vinli_laurer_mtvi",
        "train_split": "vinli_train+laurer_vi_anli",
        "experiment_mode": "fresh_mtvi_anli_only_macrobi_then_vianli_gated_small",
        "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m26-mtvi-anli-only-macrobi-maxlen512-seed42",
    },
    "exp-001-r2-m27-mtvi-full-code-switch10-macrobi-maxlen512-seed42": {
        "text_mode": "code_switch_10",
        "aux_splits": list(LAURER_MTVI_SPLITS),
        "code_switch_fraction": 0.1,
        "epoch_anchor": "full_105k_pass",
        "stage1_dataset": "vinli_laurer_mtvi",
        "train_split": "vinli_train+laurer_vi_*",
        "experiment_mode": "fresh_mtvi_codeswitch10_then_vianli_gated_small",
        "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m27-mtvi-full-code-switch10-macrobi-maxlen512-seed42",
    },
    "exp-001-r2-m28-en-anli-only-macrobi-maxlen512-seed42": {
        "text_mode": "en",
        "aux_splits": ["vi_anli"],
        "code_switch_fraction": 0.0,
        "epoch_anchor": "anli_25k_pass",
        "stage1_dataset": "vinli_laurer_mtvi",
        "train_split": "vinli_train+laurer_vi_anli",
        "experiment_mode": "fresh_en_anli_only_macrobi_then_vianli_gated_small",
        "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m28-en-anli-only-macrobi-maxlen512-seed42",
    },
    "exp-001-r2-m29-mtvi-anli-light75-macrobi-maxlen512-seed42": {
        "text_mode": "mt_vi",
        "aux_splits": ["vi_anli"],
        "code_switch_fraction": 0.0,
        "epoch_anchor": "native_full_pass",
        "stage1_dataset": "vinli_laurer_mtvi",
        "train_split": "vinli_train+laurer_vi_anli",
        "experiment_mode": "fresh_mtvi_anli_light75_native_anchor_then_vianli_gated_small",
        "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m29-mtvi-anli-light75-macrobi-maxlen512-seed42",
        "physical_native": 3,
        "physical_aux": 1,
        "stage1_epochs": 2,
    },
    "exp-001-r2-m31-stage2-mtvi-anli-light75-maxlen512-seed42": {
        "text_mode": "mt_vi",
        "aux_splits": ["vi_anli"],
        "code_switch_fraction": 0.0,
        "epoch_anchor": "native_full_pass",
        "stage1_dataset": "vinli",
        "train_split": "vinli_train",
        "experiment_mode": "fresh_vinli_then_vianli_auxlight_gated_small",
        "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m31-stage2-mtvi-anli-light75-maxlen512-seed42",
        "physical_native": 3,
        "physical_aux": 1,
        "stage1_epochs": 5,
    },
}

def validate_custom_config(cfg: dict) -> None:
    project, model, training = cfg["project"], cfg["model"], cfg["training"]
    pool_mode = model.get("pool_mode", "cls")
    if pool_mode not in ("cls", "attn", "dual", "gated_dual", "token_align"):
        raise ValueError(f"unsupported pool_mode: {pool_mode}")
    p_batch = training.get("physical_batch_size", 4)
    g_accum = training.get("gradient_accumulation_steps", 4)
    eff_batch = training.get("effective_batch_size", p_batch * g_accum)
    if p_batch * g_accum != eff_batch:
        raise ValueError(f"effective_batch_size {eff_batch} != {p_batch} * {g_accum}")
    seg_pool = model.get("segment_pooling", "attentive")
    if seg_pool not in ("attentive", "mean"):
        raise ValueError(f"segment_pooling must be 'attentive' or 'mean', got {seg_pool}")


def validate_config(cfg: dict) -> None:
    project, model, training = cfg["project"], cfg["model"], cfg["training"]
    if (
        cfg.get("custom_exploration")
        or project.get("custom_exploration")
        or str(project.get("experiment_id", "")).startswith("CUSTOM")
        or str(project.get("experiment_id", "")).startswith("EXP-001-R2-ABLATION")
        or str(project.get("experiment_id", "")).startswith("EXP-001-R2-BASELINE")
        or project.get("experiment_mode") in ("target_only", "no_stilts", "baseline_flat_cls", "ablation_mean_pooling", "ablation_no_gate", "ablation_h64", "ablation_h256", "ablation_no_stilts")
    ):
        validate_custom_config(cfg)
        return
    run_name = cfg["wandb"]["run_name"]
    hf_repo = cfg["hf_hub"]["repo_id"]
    is_aux_run = run_name in AUX_RUNS
    pool_mode = model.get("pool_mode", "cls")
    gated_mode = pool_mode in ("gated_dual", "token_align")
    if project.get("experiment_id") != "EXP-001-R2" or project.get("seed") not in (7, 17, 23, 42):
        raise ValueError("runner accepts EXP-001-R2 with seeds in {7,17,23,42} only")
    if gated_mode and project.get("seed") != 42:
        raise ValueError("M13/M14 require seed=42")
    # Paper end-to-end requires test inference, so test may be unlocked.
    if gated_mode:
        if model.get("name") != "uitnlp/CafeBERT" or model.get("revision") != MODEL_REVISION:
            raise ValueError("M13/M14 require raw pinned uitnlp/CafeBERT revision")
    else:
        allowed_backbones = {"uitnlp/CafeBERT", "xlm-roberta-base", "vinai/phobert-base"}
        if model.get("name") not in allowed_backbones:
            raise ValueError(f"runner accepts backbones in {sorted(allowed_backbones)} only")
        if model.get("name") == "uitnlp/CafeBERT" and model.get("revision") != MODEL_REVISION:
            raise ValueError("scientific run requires raw pinned uitnlp/CafeBERT revision")
        if model.get("name") != "uitnlp/CafeBERT" and model.get("revision") not in (None, "main"):
            raise ValueError("cross-lingual backbones use the main revision")
    if model.get("max_length") != 512 or model.get("fallback_models"):
        raise ValueError("scientific run requires max_length=512 and no fallback model")
    if pool_mode not in ("cls", "attn", "dual", "gated_dual", "token_align"):
        raise ValueError("pool_mode must be cls, attn, dual, gated_dual, or token_align")
    relation_delta_mode = model.get("relation_delta_mode", "absolute")
    if relation_delta_mode not in {"absolute", "signed"}:
        raise ValueError("relation_delta_mode must be absolute or signed")
    if relation_delta_mode == "signed" and run_name != M45_SIGNED_DELTA_RUN_NAME:
        raise ValueError("signed relation delta is registered for M45 only")
    if run_name == M45_SIGNED_DELTA_RUN_NAME and (
        pool_mode != "gated_dual" or relation_delta_mode != "signed"
    ):
        raise ValueError("M45 requires gated_dual with relation_delta_mode=signed")
    hier = model.get("hierarchical_e_first", False)
    if hier not in (False, True):
        raise ValueError("hierarchical_e_first must be boolean")
    if hier and run_name != M46_HIER_E_FIRST_RUN_NAME:
        raise ValueError("hierarchical_e_first is registered for M46 only")
    if run_name == M46_HIER_E_FIRST_RUN_NAME and (
        pool_mode != "gated_dual" or hier is not True or relation_delta_mode != "absolute"
    ):
        raise ValueError("M46 requires gated_dual with hierarchical_e_first=true and absolute delta")
    hh = model.get("head_hidden")
    if hh is not None and (not isinstance(hh, int) or hh <= 0):
        raise ValueError("head_hidden must be a positive int or omitted")
    if gated_mode:
        if model.get("class_weights") is not None:
            raise ValueError("M13/M14 architecture ablations require plain CE without class weights")
        # M15/M16/M17/M18/M19/M20 small-head/gate ablations allow limited
        # regularization; M21/M22 add exactly one paper-exact consistency loss.
        if run_name in {
            "exp-001-r2-m15-gated-small256-maxlen512-seed42",
            "exp-001-r2-m16-gated-small256-ls02-wd005-maxlen512-seed42",
            "exp-001-r2-m17-gated-small128-maxlen512-seed42",
            "exp-001-r2-m18-gated-bias1-maxlen512-seed42",
            "exp-001-r2-m19-gated-small64-maxlen512-seed42",
            "exp-001-r2-m20-gated-small128-bias1-maxlen512-seed42",
            "exp-001-r2-m21-rdrop-a5-maxlen512-seed42",
            "exp-001-r2-m22-r3f-l1-maxlen512-seed42",
            "exp-001-r2-m24-childtune-f-p20-maxlen512-seed42",
            M25_RUN_NAME,
            "exp-001-r2-m26-mtvi-anli-only-macrobi-maxlen512-seed42",
            "exp-001-r2-m27-mtvi-full-code-switch10-macrobi-maxlen512-seed42",
            "exp-001-r2-m28-en-anli-only-macrobi-maxlen512-seed42",
            "exp-001-r2-m29-mtvi-anli-light75-macrobi-maxlen512-seed42",
            "exp-001-r2-m31-stage2-mtvi-anli-light75-maxlen512-seed42",
            M32_JTT_RUN_NAME,
            "exp-001-r2-m33-freelb-k2-maxlen512-seed42",
            "exp-001-r2-m36-mixout-maxlen512-seed42",
            "exp-001-r2-m37-recadam-maxlen512-seed42",
            "exp-001-r2-m38-sam-maxlen512-seed42",
            M39_LORA_RUN_NAME,
            M41_MNLI_BRIDGE_RUN_NAME,
            M42_ANLI_BRIDGE_RUN_NAME,
            M45_SIGNED_DELTA_RUN_NAME,
            M46_HIER_E_FIRST_RUN_NAME,
            M47_MIRROR_SYMMETRY_RUN_NAME,
            M48_VIMEDNLI_CE_RUN_NAME,
            M49_VIMEDNLI_MIRROR_RUN_NAME,
        }:
            expected_ls = {
                "exp-001-r2-m15-gated-small256-maxlen512-seed42": 0.05,
                "exp-001-r2-m16-gated-small256-ls02-wd005-maxlen512-seed42": 0.02,
                "exp-001-r2-m17-gated-small128-maxlen512-seed42": 0.02,
                "exp-001-r2-m18-gated-bias1-maxlen512-seed42": 0.02,
                "exp-001-r2-m19-gated-small64-maxlen512-seed42": 0.02,
                "exp-001-r2-m20-gated-small128-bias1-maxlen512-seed42": 0.02,
                "exp-001-r2-m21-rdrop-a5-maxlen512-seed42": 0.02,
                "exp-001-r2-m22-r3f-l1-maxlen512-seed42": 0.02,
                "exp-001-r2-m24-childtune-f-p20-maxlen512-seed42": 0.02,
                M25_RUN_NAME: 0.02,
                "exp-001-r2-m26-mtvi-anli-only-macrobi-maxlen512-seed42": 0.02,
                "exp-001-r2-m27-mtvi-full-code-switch10-macrobi-maxlen512-seed42": 0.02,
                "exp-001-r2-m28-en-anli-only-macrobi-maxlen512-seed42": 0.02,
                "exp-001-r2-m29-mtvi-anli-light75-macrobi-maxlen512-seed42": 0.02,
                "exp-001-r2-m31-stage2-mtvi-anli-light75-maxlen512-seed42": 0.02,
                M32_JTT_RUN_NAME: 0.02,
                "exp-001-r2-m33-freelb-k2-maxlen512-seed42": 0.02,
                "exp-001-r2-m36-mixout-maxlen512-seed42": 0.02,
                "exp-001-r2-m37-recadam-maxlen512-seed42": 0.02,
                "exp-001-r2-m38-sam-maxlen512-seed42": 0.02,
                M39_LORA_RUN_NAME: 0.02,
                M41_MNLI_BRIDGE_RUN_NAME: 0.02,
                M42_ANLI_BRIDGE_RUN_NAME: 0.02,
                M45_SIGNED_DELTA_RUN_NAME: 0.02,
                M46_HIER_E_FIRST_RUN_NAME: 0.02,
                M47_MIRROR_SYMMETRY_RUN_NAME: 0.02,
                M48_VIMEDNLI_CE_RUN_NAME: 0.02,
                M49_VIMEDNLI_MIRROR_RUN_NAME: 0.02,
            }[run_name]
            if model.get("label_smoothing") != expected_ls:
                raise ValueError(f"{run_name} requires label_smoothing={expected_ls}")
            expected_gate = -1.0 if run_name in {
                "exp-001-r2-m18-gated-bias1-maxlen512-seed42",
                "exp-001-r2-m20-gated-small128-bias1-maxlen512-seed42",
                "exp-001-r2-m21-rdrop-a5-maxlen512-seed42",
                "exp-001-r2-m22-r3f-l1-maxlen512-seed42",
                "exp-001-r2-m24-childtune-f-p20-maxlen512-seed42",
                M25_RUN_NAME,
                "exp-001-r2-m26-mtvi-anli-only-macrobi-maxlen512-seed42",
                "exp-001-r2-m27-mtvi-full-code-switch10-macrobi-maxlen512-seed42",
                "exp-001-r2-m28-en-anli-only-macrobi-maxlen512-seed42",
                "exp-001-r2-m29-mtvi-anli-light75-macrobi-maxlen512-seed42",
                "exp-001-r2-m31-stage2-mtvi-anli-light75-maxlen512-seed42",
                M32_JTT_RUN_NAME,
                "exp-001-r2-m33-freelb-k2-maxlen512-seed42",
                "exp-001-r2-m36-mixout-maxlen512-seed42",
                "exp-001-r2-m37-recadam-maxlen512-seed42",
                "exp-001-r2-m38-sam-maxlen512-seed42",
                M39_LORA_RUN_NAME,
                M41_MNLI_BRIDGE_RUN_NAME,
                M42_ANLI_BRIDGE_RUN_NAME,
                M45_SIGNED_DELTA_RUN_NAME,
                M46_HIER_E_FIRST_RUN_NAME,
                M47_MIRROR_SYMMETRY_RUN_NAME,
                M48_VIMEDNLI_CE_RUN_NAME,
                M49_VIMEDNLI_MIRROR_RUN_NAME,
            } else -2.0
            if model.get("gate_bias") != expected_gate:
                raise ValueError(f"{run_name} requires gate_bias={expected_gate}")
        else:
            if model.get("label_smoothing") != 0.0:
                raise ValueError("M13/M14 architecture ablations require label_smoothing=0")
            if model.get("gate_bias") != -2.0:
                raise ValueError("M13/M14 require conservative residual gate_bias=-2.0")
        if hh is not None:
            raise ValueError("M13/M14 use their declared fusion head; head_hidden must be omitted")
        small_head_runs = {
            "exp-001-r2-m15-gated-small256-maxlen512-seed42": 256,
            "exp-001-r2-m16-gated-small256-ls02-wd005-maxlen512-seed42": 256,
            "exp-001-r2-m17-gated-small128-maxlen512-seed42": 128,
            "exp-001-r2-m18-gated-bias1-maxlen512-seed42": 256,
            "exp-001-r2-m19-gated-small64-maxlen512-seed42": 64,
            "exp-001-r2-m20-gated-small128-bias1-maxlen512-seed42": 128,
            "exp-001-r2-m21-rdrop-a5-maxlen512-seed42": 128,
            "exp-001-r2-m22-r3f-l1-maxlen512-seed42": 128,
            "exp-001-r2-m24-childtune-f-p20-maxlen512-seed42": 128,
            M25_RUN_NAME: 128,
            "exp-001-r2-m26-mtvi-anli-only-macrobi-maxlen512-seed42": 128,
            "exp-001-r2-m27-mtvi-full-code-switch10-macrobi-maxlen512-seed42": 128,
            "exp-001-r2-m28-en-anli-only-macrobi-maxlen512-seed42": 128,
            "exp-001-r2-m29-mtvi-anli-light75-macrobi-maxlen512-seed42": 128,
            "exp-001-r2-m31-stage2-mtvi-anli-light75-maxlen512-seed42": 128,
            M32_JTT_RUN_NAME: 128,
            "exp-001-r2-m33-freelb-k2-maxlen512-seed42": 128,
            "exp-001-r2-m36-mixout-maxlen512-seed42": 128,
            "exp-001-r2-m37-recadam-maxlen512-seed42": 128,
            "exp-001-r2-m38-sam-maxlen512-seed42": 128,
            M39_LORA_RUN_NAME: 128,
            M41_MNLI_BRIDGE_RUN_NAME: 128,
            M42_ANLI_BRIDGE_RUN_NAME: 128,
            M45_SIGNED_DELTA_RUN_NAME: 128,
            M46_HIER_E_FIRST_RUN_NAME: 128,
            M47_MIRROR_SYMMETRY_RUN_NAME: 128,
            M48_VIMEDNLI_CE_RUN_NAME: 128,
            M49_VIMEDNLI_MIRROR_RUN_NAME: 128,
        }
        if run_name in small_head_runs:
            if model.get("relation_hidden") != small_head_runs[run_name]:
                raise ValueError(f"{run_name} requires relation_hidden={small_head_runs[run_name]}")
        elif model.get("relation_hidden") is not None:
            raise ValueError("only M15/M16/M17 may set relation_hidden")
    if pool_mode == "token_align" and model.get("alignment_dim") != 256:
        raise ValueError("M14 token alignment requires alignment_dim=256")
    # Phase-2 consistency-loss gate (fail closed): exactly ONE method per run.
    rdrop_alpha = model.get("rdrop_alpha")
    r3f_lambda = model.get("r3f_lambda")
    r3f_sigma = model.get("r3f_sigma", R3F_DEFAULT_SIGMA)
    mirror_cfg = model.get("mirror_contradiction_symmetry")
    if sum(value is not None for value in (rdrop_alpha, r3f_lambda, mirror_cfg)) > 1:
        raise ValueError("exactly ONE consistency loss may be active per run")
    if (rdrop_alpha is not None or r3f_lambda is not None) and run_name not in {
        "exp-001-r2-m21-rdrop-a5-maxlen512-seed42",
        "exp-001-r2-m22-r3f-l1-maxlen512-seed42",
    }:
        raise ValueError(
            f"R-Drop/R3F are only registered for m21/m22, got {run_name}"
        )
    expected_m47_mirror = {"lambda": 1.0, "stage": "stage2"}
    if run_name == M48_VIMEDNLI_CE_RUN_NAME and mirror_cfg is not None:
        raise ValueError("M48 ViMedNLI CE baseline must run without mirror symmetry")
    mirror_registered_runs = {M47_MIRROR_SYMMETRY_RUN_NAME, M49_VIMEDNLI_MIRROR_RUN_NAME}
    if mirror_cfg is not None and run_name not in mirror_registered_runs:
        raise ValueError(f"mirror contradiction symmetry is only registered for M47/M49, got {run_name}")
    if run_name == M49_VIMEDNLI_MIRROR_RUN_NAME and mirror_cfg != expected_m47_mirror:
        raise ValueError(f"M49 requires exact mirror symmetry config {expected_m47_mirror}")
    if run_name == M47_MIRROR_SYMMETRY_RUN_NAME:
        if mirror_cfg != expected_m47_mirror:
            raise ValueError(f"M47 requires exact mirror symmetry config {expected_m47_mirror}")
        expected_m47_method = {
            "name": "contradiction_mirror_symmetry",
            "paper": {
                "title": "A Logic-Driven Framework for Consistency of Neural Models",
                "authors": "Li, Gupta, Mehta, and Srikumar",
                "venue": "EMNLP 2019",
                "arxiv": "https://arxiv.org/abs/1909.00126v1",
                "official_code": "https://github.com/utahnlp/consistency/tree/post-camera-ready",
                "variant": "BERT+M",
                "paper_reported": {
                    "learning_rate": 1.0e-05,
                    "epochs": 3,
                    "warmup_epochs": 3,
                    "constraint_index": 6,
                    "lambda": 1.0,
                    "dropout": 0.0,
                },
            },
            "controlled_mapping": {
                "stage": "stage2_only",
                "objective": "CE(P,H,y) + 1.0 * mean(abs(log p_C(P,H) - log p_C(H,P)))",
                "original_pair_supervised": True,
                "mirrored_pair_supervised": False,
                "mirrored_pair_source": "ephemeral_reversal_of_same_vianli_batch",
                "data_rows_modified": False,
                "added_model_heads": False,
                "extra_forwards_per_stage2_batch": 1,
                "target_test_accessed": False,
            },
            "preregistered_stop_go": {
                "baseline_selected_vianli_dev_macro_f1": 0.4879549,
                "promote_if_selected_vianli_dev_macro_f1_at_least": 0.4929549,
                "promote_if_vianli_dev_f1_C_delta_at_least": 0.01,
                "reject_if_vianli_dev_f1_N_delta_below": -0.005,
                "selection_data": "vianli_dev_only",
            },
        }
        if cfg.get("method") != expected_m47_method:
            raise ValueError("M47 requires exact paper provenance and preregistered stop/go")
    if run_name == M48_VIMEDNLI_CE_RUN_NAME:
        expected_m48_method = {
            "name": "vimednli_ce_baseline",
            "paper": {
                "title": (
                    "Enriching Biomedical Knowledge for Low-resource Language "
                    "Through Large-Scale Translation"
                ),
                "authors": "Trang Nguyen, Phong Nguyen-Thuan Do, Tin Van Huynh, "
                           "Kiet Van Nguyen, Ngan Luu-Thuy Nguyen, Ngo Xuan Bach",
                "venue": "EACL 2023",
                "anthology": "2023.eacl-main.228",
                "arxiv": "https://arxiv.org/abs/2210.05598",
                "dataset": "ViMedNLI (translated MedNLI, human-refined)",
                "sota_reference_f1": 0.7820,
                "sota_note": (
                    "owner-confirmed current ViMedNLI test Macro-F1 SOTA; "
                    "the release paper itself reports accuracy (ViPubmedT5 81.65)"
                ),
            },
            "controlled_mapping": {
                "stage1": "vinli_train -> vinli_dev (unchanged M20 family)",
                "stage2_dataset": "vimednli",
                "stage2_max_length": 256,
                "physical_batch": 4,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 16,
                "seed": 42,
                "loss": "CE(label_smoothing=0.02)",
                "data_rows_modified": False,
                "target_test": "vimednli_test (not previously exposed)",
            },
        }
        if cfg.get("method") != expected_m48_method:
            raise ValueError("M48 requires exact CE-baseline provenance contract")
    if run_name == M49_VIMEDNLI_MIRROR_RUN_NAME:
        expected_m49_method = {
            "name": "vimednli_contradiction_mirror_symmetry",
            "paper": {
                "title": "A Logic-Driven Framework for Consistency of Neural Models",
                "authors": "Li, Gupta, Mehta, and Srikumar",
                "venue": "EMNLP 2019",
                "arxiv": "https://arxiv.org/abs/1909.00126v1",
                "official_code": "https://github.com/utahnlp/consistency/tree/post-camera-ready",
                "variant": "BERT+M",
                "dataset": "ViMedNLI (translated MedNLI, human-refined)",
                "sota_reference_f1": 0.7820,
                "sota_note": (
                    "owner-confirmed current ViMedNLI test Macro-F1 SOTA; "
                    "the release paper itself reports accuracy (ViPubmedT5 81.65)"
                ),
            },
            "controlled_mapping": {
                "stage": "stage2_only",
                "objective": "CE(P,H,y) + 1.0 * mean(abs(log p_C(P,H) - log p_C(H,P)))",
                "original_pair_supervised": True,
                "mirrored_pair_supervised": False,
                "mirrored_pair_source": "ephemeral_reversal_of_same_vimednli_batch",
                "data_rows_modified": False,
                "added_model_heads": False,
                "extra_forwards_per_stage2_batch": 1,
                "target_test_accessed": True,
                "target_test_is_vianli": False,
            },
            "preregistered_stop_go": {
                "baseline_run": "exp-001-r2-m48-vimednli-ce-maxlen256-seed42",
                "promote_if_selected_vimednli_dev_macro_f1_delta_at_least": 0.005,
                "promote_if_vimednli_dev_f1_C_delta_at_least": 0.01,
                "reject_if_selected_vimednli_dev_macro_f1_delta_below": 0.0,
                "selection_data": "vimednli_dev_only",
            },
        }
        if cfg.get("method") != expected_m49_method:
            raise ValueError("M49 requires exact mirror + preregistered contract")
    childtune_p = model.get("childtune_p")
    if childtune_p is not None and (
        rdrop_alpha is not None or r3f_lambda is not None or mirror_cfg is not None
    ):
        raise ValueError(
            "childtune_p is mutually exclusive with R-Drop/R3F/mirror symmetry"
        )
    if (childtune_p is not None) and run_name != "exp-001-r2-m24-childtune-f-p20-maxlen512-seed42":
        raise ValueError(
            f"child_tuning is only registered for m24, got {run_name}"
        )
    freelb_params = freelb_params_from_cfg(model)
    if freelb_params is not None and run_name != "exp-001-r2-m33-freelb-k2-maxlen512-seed42":
        raise ValueError(f"freelb is only registered for m33, got {run_name}")
    if run_name == "exp-001-r2-m33-freelb-k2-maxlen512-seed42":
        expected_freelb = {
            "steps": 2, "step_size": 0.1, "max_norm": 0.2, "init_mag": 0.2,
            "padding_perturbation": 0, "optimizer_updates_per_batch": 1,
        }
        if freelb_params != expected_freelb:
            raise ValueError(
                f"m33 requires exact FreeLB params {expected_freelb}, got {freelb_params}"
            )
        if model.get("label_smoothing") != 0.02:
            raise ValueError("m33 requires label_smoothing=0.02")
    sam_params = sam_params_from_cfg(model)
    if sam_params is not None and run_name != "exp-001-r2-m38-sam-maxlen512-seed42":
        raise ValueError(f"sam is only registered for m38, got {run_name}")
    if run_name == "exp-001-r2-m38-sam-maxlen512-seed42":
        if sam_params != {"rho": 0.05, "adaptive": False, "sharpness_factor_m": 1}:
            raise ValueError(f"m38 requires exact SAM params, got {sam_params}")
        if model.get("label_smoothing") != 0.02:
            raise ValueError("m38 requires label_smoothing=0.02")
    mixout_cfg = training.get("stage2", {}).get("mixout")
    if mixout_cfg is not None and run_name != "exp-001-r2-m36-mixout-maxlen512-seed42":
        raise ValueError(f"mixout is only registered for m36, got {run_name}")
    if run_name == "exp-001-r2-m36-mixout-maxlen512-seed42":
        if mixout_cfg != {"p": 0.7}:
            raise ValueError(f"m36 requires exact stage2 mixout p=0.7, got {mixout_cfg}")
        if training.get("weight_decay") != 0.005:
            raise ValueError("m36 requires baseline Stage1 weight_decay=0.005")
        if training["stage2"].get("weight_decay") != 0.0:
            raise ValueError("m36 requires Stage2 weight_decay=0.0 (mixout replaces decay)")
        if model.get("label_smoothing") != 0.02:
            raise ValueError("m36 requires label_smoothing=0.02")
    recadam_cfg = training.get("stage2", {}).get("recadam")
    if recadam_cfg is not None and run_name != "exp-001-r2-m37-recadam-maxlen512-seed42":
        raise ValueError(f"recadam is only registered for m37, got {run_name}")
    if run_name == "exp-001-r2-m37-recadam-maxlen512-seed42":
        recadam_paper = cfg.get("method", {}).get("paper", {})
        expected_paper = {
            "title": "Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting",
            "authors": "Sanyuan Chen et al.",
            "venue": "EMNLP 2020",
            "anthology": "2020.emnlp-main.634",
            "official_code": "https://github.com/Sanyuan-Chen/RecAdam",
            "paper_gamma": 5000.0,
            "paper_anneal": "sigmoid",
            "paper_k": 0.1,
        }
        if recadam_paper != expected_paper:
            raise ValueError("m37 requires exact RecAdam paper provenance metadata")
        expected_recadam = {
            "gamma": 5000.0, "anneal_function": "sigmoid",
            "anneal_k": 0.1, "anneal_t0": 100.0,
        }
        if recadam_cfg != expected_recadam:
            raise ValueError(f"m37 requires exact RecAdam params, got {recadam_cfg}")
        if mixout_cfg is not None:
            raise ValueError("m37 recadam is mutually exclusive with mixout")
        if model.get("label_smoothing") != 0.02:
            raise ValueError("m37 requires label_smoothing=0.02")
    if run_name == "exp-001-r2-m24-childtune-f-p20-maxlen512-seed42" and childtune_p != 0.2:
        raise ValueError("m24 requires exactly childtune_p=0.2")
    if run_name == "exp-001-r2-m21-rdrop-a5-maxlen512-seed42":
        if rdrop_alpha != 5.0 or r3f_lambda is not None:
            raise ValueError(
                "m21 requires exactly rdrop_alpha=5.0 with r3f_lambda=null"
            )
    elif run_name == "exp-001-r2-m22-r3f-l1-maxlen512-seed42" and (
        r3f_lambda != 1.0 or rdrop_alpha is not None
        or float(r3f_sigma) != R3F_DEFAULT_SIGMA
    ):
        raise ValueError(
            "m22 requires exactly r3f_lambda=1.0, rdrop_alpha=null, "
            f"r3f_sigma={R3F_DEFAULT_SIGMA}"
        )
    ml_map = cfg["data"].get("max_lengths") or {}
    if ml_map.get("vinli", 512) != 512 or ml_map.get("vianli", 512) != 512:
        raise ValueError("ViNLI/ViANLI max_length must be 512")
    if ml_map.get("vimednli", 256) != 256:
        raise ValueError("ViMedNLI max_length must be 256")
    if is_aux_run and ml_map.get(LAURER_MTVI_DOMAIN) != 512:
        raise ValueError(f"{run_name} translated/auxiliary NLI max_length must be 512")
    if (training.get("physical_batch_size"), training.get("gradient_accumulation_steps"),
            training.get("effective_batch_size")) != (4, 4, 16):
        raise ValueError("batch contract must be physical=4, grad_accum=4, effective=16")
    if training.get("fp16_train") is not True or training.get("fp32_eval") is not True:
        raise ValueError("precision contract must be FP16 train and FP32 eval")
    if gated_mode:
        wd_map = {
            "exp-001-r2-m15-gated-small256-maxlen512-seed42": 0.01,
            "exp-001-r2-m16-gated-small256-ls02-wd005-maxlen512-seed42": 0.005,
            "exp-001-r2-m17-gated-small128-maxlen512-seed42": 0.005,
            "exp-001-r2-m18-gated-bias1-maxlen512-seed42": 0.005,
            "exp-001-r2-m19-gated-small64-maxlen512-seed42": 0.005,
            "exp-001-r2-m20-gated-small128-bias1-maxlen512-seed42": 0.005,
            "exp-001-r2-m21-rdrop-a5-maxlen512-seed42": 0.005,
            "exp-001-r2-m22-r3f-l1-maxlen512-seed42": 0.005,
            "exp-001-r2-m24-childtune-f-p20-maxlen512-seed42": 0.005,
            M25_RUN_NAME: 0.005,
            "exp-001-r2-m26-mtvi-anli-only-macrobi-maxlen512-seed42": 0.005,
            "exp-001-r2-m27-mtvi-full-code-switch10-macrobi-maxlen512-seed42": 0.005,
            "exp-001-r2-m28-en-anli-only-macrobi-maxlen512-seed42": 0.005,
            "exp-001-r2-m29-mtvi-anli-light75-macrobi-maxlen512-seed42": 0.005,
            "exp-001-r2-m31-stage2-mtvi-anli-light75-maxlen512-seed42": 0.005,
            M32_JTT_RUN_NAME: 0.005,
            "exp-001-r2-m33-freelb-k2-maxlen512-seed42": 0.005,
            "exp-001-r2-m36-mixout-maxlen512-seed42": 0.005,
            "exp-001-r2-m37-recadam-maxlen512-seed42": 0.005,
            "exp-001-r2-m38-sam-maxlen512-seed42": 0.005,
            M39_LORA_RUN_NAME: 0.005,
            M41_MNLI_BRIDGE_RUN_NAME: 0.005,
            M42_ANLI_BRIDGE_RUN_NAME: 0.005,
            M45_SIGNED_DELTA_RUN_NAME: 0.005,
            M46_HIER_E_FIRST_RUN_NAME: 0.005,
            M47_MIRROR_SYMMETRY_RUN_NAME: 0.005,
            M48_VIMEDNLI_CE_RUN_NAME: 0.005,
            M49_VIMEDNLI_MIRROR_RUN_NAME: 0.005,
        }
        expected_wd = wd_map.get(run_name, 0.0)
        if training.get("weight_decay") != expected_wd or training.get("warmup_ratio") != 0.06:
            raise ValueError(f"M13/M14 require weight_decay={expected_wd} and warmup_ratio=0.06")

    stage1, stage2 = training["stage1"], training["stage2"]
    auxiliary_binding = cfg["data"].get("auxiliary_nli")
    stage1_aug = stage1.get("mt_vi_augmentation")
    stage2_aug = stage2.get("mt_vi_augmentation")
    aux_variant = AUX_RUNS.get(run_name)
    is_aux_run = aux_variant is not None
    any_aug = stage1_aug is not None or stage2_aug is not None

    def expected_augmentation(native_source: str, native_count: int,
                              aux_count: int, variant: dict) -> dict:
        total = native_count + aux_count
        return {
            "enabled": True,
            "native_source": native_source,
            "auxiliary_source": LAURER_MTVI_DOMAIN,
            "native_fraction": native_count / total,
            "auxiliary_fraction": aux_count / total,
            "auxiliary_sampling": "row_proportional_without_replacement",
            "epoch_anchor": variant["epoch_anchor"],
            "early_stopping_min_complete_epochs": 1,
            "text_mode": variant["text_mode"],
            "code_switch_fraction": variant["code_switch_fraction"],
            "auxiliary_splits": list(variant["aux_splits"]),
            "seed": 42,
        }

    if not is_aux_run and (auxiliary_binding is not None or any_aug):
        raise ValueError(
            "auxiliary MT-Vietnamese NLI flags are registered for M25/M26/M27/M28/M29/M31 only"
        )
    if is_aux_run:
        if auxiliary_binding is None:
            raise ValueError("auxiliary runs require the pinned dataset binding")
        laurer_vi_nli.validate_config_binding(auxiliary_binding)
        expected_counts = {
            "laurer_vi_anli": 25_000,
            "laurer_vi_fever": 25_000,
            "laurer_vi_ling": 5_000,
            "laurer_vi_mnli": 25_000,
            "laurer_vi_wanli": 25_000,
            "laurer_vi_total": 105_000,
        }
        for key, value in expected_counts.items():
            if cfg["data"].get("expected_counts", {}).get(key) != value:
                raise ValueError(f"{run_name} auxiliary expected count mismatch for {key}")
        if (
            cfg["data"].get("laurer_vi_nli_dir") != "data/external/laurer_vi_nli"
            or cfg["data"].get("manifests", {}).get("laurer_vi_nli")
            != auxiliary_binding["manifest"]
            or cfg["data"].get("revisions", {}).get("laurer_vi_nli")
            != LAURER_MTVI_REVISION
        ):
            raise ValueError(f"{run_name} auxiliary directory/manifest/revision binding mismatch")

        stage1_uses_aux = aux_variant["stage1_dataset"] != "vinli"
        physical_native = int(aux_variant.get("physical_native", 2))
        physical_aux = int(aux_variant.get("physical_aux", 2))
        if stage1_uses_aux:
            if stage1_aug != expected_augmentation(
                "vinli", physical_native, physical_aux, aux_variant
            ):
                raise ValueError(f"{run_name} Stage 1 augmentation contract mismatch")
        elif stage1_aug is not None:
            raise ValueError(f"{run_name} Stage 1 must be ViNLI-only without augmentation")
        if run_name == "exp-001-r2-m31-stage2-mtvi-anli-light75-maxlen512-seed42":
            if stage2_aug != expected_augmentation(
                "vianli", physical_native, physical_aux, aux_variant
            ):
                raise ValueError(f"{run_name} Stage 2 augmentation contract mismatch")
        elif stage2_aug is not None:
            raise ValueError(f"{run_name} Stage 2 must run without auxiliary augmentation")

        expected_method = {
            "dataset_basis": {
                "name": "multilingual-NLI-26lang-2mil7",
                "authors": "Moritz Laurer et al.",
                "release_year": 2022,
                "translated_language": "vi",
            },
            "training_basis": {
                "paper": (
                    "Don't Stop Fine-Tuning: On Training Regimes for Few-Shot "
                    "Cross-Lingual Transfer with Multilingual Language Models"
                ),
                "authors": "Fabian David Schmidt, Ivan Vulic, and Goran Glavas",
                "venue": "EMNLP 2022",
                "anthology_url": "https://aclanthology.org/2022.emnlp-main.736/",
                "regime": "MACRO-BI",
                "paper_source_rows_per_batch": 16,
                "paper_target_rows_per_batch": 16,
                "delta": 0.5,
            },
            "controlled_mapping": {
                "physical_native_rows": physical_native,
                "physical_mt_rows": physical_aux,
                "gradient_accumulation_steps": 4,
                "effective_native_rows": physical_native * 4,
                "effective_mt_rows": physical_aux * 4,
                "effective_batch_size": 16,
                "auxiliary_epoch_anchor": aux_variant["epoch_anchor"],
                "early_stopping_min_complete_epochs": 1,
                "text_mode": aux_variant["text_mode"],
                "code_switch_fraction": aux_variant["code_switch_fraction"],
            },
        }
        if cfg.get("method") != expected_method:
            raise ValueError(f"{run_name} Laurer/MACRO-BI paper-grounding contract mismatch")

    is_jtt_run = run_name == M32_JTT_RUN_NAME
    jtt_identification = training.get("stage2_identification")
    jtt_stage_cfg = stage2.get("jtt")
    if is_jtt_run:
        expected_jtt_method = {
            "name": "compute_matched_jtt",
            "paper": {
                "title": "Just Train Twice: Improving Group Robustness without Training Group Information",
                "authors": "Evan Z. Liu et al.",
                "venue": "ICML 2021",
                "arxiv": "https://arxiv.org/abs/2107.09044",
                "official_code": "https://github.com/anniesch/jtt",
                "direct_nli_evidence": "MultiNLI",
                "paper_multinli_identification_epoch": 2,
                "paper_multinli_error_extra_copies": 4,
                "paper_multinli_final_epochs": 5,
            },
            "controlled_mapping": {
                "error_rule": "argmax_not_equal_gold",
                "error_set_frozen_after_identification": True,
                "hard_example_weight": 5.0,
                "ordinary_example_weight": 1.0,
                "weighted_sampling_with_replacement": True,
                "draws_per_epoch": 8012,
                "physical_batches_per_epoch": 2003,
                "optimizer_steps_per_epoch": 501,
                "compute_matched": True,
                "data_rows_modified": False,
            },
        }
        expected_jtt_identification = {
            "dataset": "vianli",
            "train_split": "vianli_train",
            "selection_split": "vianli_dev_identification_final",
            "selection_metric": "macro_f1",
            "learning_rate": 1e-5,
            "max_epochs": 2,
            "eval_optimizer_steps": 1002,
            "early_stopping_patience_evals": None,
            "replay": False,
            "batch_mix": {"vianli": 4},
        }
        expected_jtt_stage = {
            "enabled": True,
            "identification_phase": "stage2_identification",
            "error_rule": "argmax_not_equal_gold",
            "error_extra_copies_equivalent": 4,
            "hard_example_weight": 5.0,
            "ordinary_example_weight": 1.0,
            "sampling": "weighted_with_replacement",
            "draws_per_epoch": 8012,
            "fixed_error_set": True,
            "compute_matched": True,
            "seed": 42,
        }
        if cfg.get("method") != expected_jtt_method:
            raise ValueError("M32 compute-matched JTT paper/mapping contract mismatch")
        if jtt_identification != expected_jtt_identification:
            raise ValueError("M32 JTT identification phase must be exact two-epoch final model")
        if jtt_stage_cfg != expected_jtt_stage:
            raise ValueError("M32 JTT weighted final-stage contract mismatch")
    elif jtt_identification is not None or jtt_stage_cfg is not None:
        raise ValueError("JTT phases are registered for M32 only")

    is_bridge_run = run_name in BRIDGE_RUN_DATASETS
    is_lora_run = run_name == M39_LORA_RUN_NAME
    if is_lora_run and is_bridge_run:
        raise ValueError("LoRA and bridge phases are mutually exclusive")
    stage1b_cfg = training.get("stage1b")
    if is_bridge_run:
        bname = BRIDGE_RUN_DATASETS[run_name]
        expected_bridge_paper = {
            "name": f"english_{bname}_intermediate_task_bridge",
            "paper": (
                {
                    "title": (
                        "English Intermediate-Task Training Improves Zero-Shot "
                        "Cross-Lingual Transfer Too"
                    ),
                    "authors": "Phang, Fevry, Srinivasan, and Bowman",
                    "venue": "AACL 2020",
                    "arxiv": "https://arxiv.org/abs/2005.13013v2",
                    "intermediate_tasks": ["MNLI"],
                    "finding": (
                        "MNLI is among the best English intermediate tasks for "
                        "zero-shot cross-lingual transfer of XLM-R"
                    ),
                }
                if bname == "mnli"
                else {
                    "title": "Adversarial NLI: A New Benchmark for Natural Language Understanding",
                    "authors": "Nie, Williams, Dinan, Bansal, Weston, and Kiela",
                    "venue": "ACL 2020",
                    "arxiv": "https://arxiv.org/abs/1910.14599v2",
                    "finding": (
                        "Training on adversarially collected NLI data improves "
                        "generalization on standard NLI benchmarks"
                    ),
                }
            ),
            "controlled_mapping": {
                "stage": "between_stage1_and_stage2",
                "selection_split": "vinli_dev",
                "selection_metric": "macro_f1",
                "physical_batch": {"dataset": 4},
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 16,
                "seed": 42,
                "max_length": 512,
                "data_rows_modified": False,
                "target_test_accessed": False,
            },
        }
        if cfg.get("method") != expected_bridge_paper:
            raise ValueError(f"{run_name} EN-bridge paper/mapping contract mismatch")
        expected_bridge_stage = {
            "dataset": bname,
            "dataset_dir": f"data/external/en_nli_bridge/{bname}",
            "dataset_revision": (
                "da70db2af9d09693783c3320c4249840212ee221"
                if bname == "mnli"
                else "8e4813d81f46d313dac7892e1c28076917cfcdf9"
            ),
            "domain": bname,
            "train_split": f"{bname}_bridge_train",
            "selection_split": "vinli_dev",
            "selection_metric": "macro_f1",
            "learning_rate": 1e-5,
            "max_epochs": 3,
            "eval_optimizer_steps": 400,
            "early_stopping_patience_evals": 3,
            "batch_mix": {bname: 4},
        }
        if stage1b_cfg != expected_bridge_stage:
            raise ValueError(f"{run_name} stage1b bridge contract mismatch")
        if cfg["data"].get("expected_counts", {}).get(f"{bname}_bridge_train") != 62406:
            raise ValueError(f"{run_name} expects exactly 62,406 bridge rows")
    elif stage1b_cfg is not None:
        raise ValueError("stage1b bridge phase is registered for M41/M42 only")

    stage2_lora_cfg = stage2.get("lora")
    if is_lora_run:
        expected_lora_method = {
            "name": "lora",
            "paper": {
                "title": "LoRA: Low-Rank Adaptation of Large Language Models",
                "authors": "Hu et al.",
                "venue": "ICLR 2022",
                "arxiv": "https://arxiv.org/abs/2106.09685v2",
                "official_code": "https://github.com/microsoft/LoRA",
                "paper_targets": "query and value projections in all self-attention layers",
                "paper_roberta_large_rank": 8,
                "paper_roberta_large_alpha": 16,
            },
            "controlled_mapping": {
                "apply_stage1": False,
                "apply_stage2": True,
                "target_scope": "backbone.encoder.layer.*.attention.self.{query,value}",
                "expected_target_modules": 48,
                "r": 8,
                "alpha": 16,
                "lora_dropout": 0.0,
                "scaling": "alpha_over_r",
                "bias": "none",
                "init": "microsoft_reference_kaiming_A_zero_B",
                "merge_during_training": False,
            },
        }
        if cfg.get("method") != expected_lora_method:
            raise ValueError("M39 LoRA paper/mapping contract mismatch")
        expected_lora_stage = {
            "r": 8, "alpha": 16, "lora_dropout": 0.0, "expected_target_modules": 48,
        }
        if stage2_lora_cfg != expected_lora_stage:
            raise ValueError(f"M39 requires exact LoRA params, got {stage2_lora_cfg}")
        if mixout_cfg is not None or recadam_cfg is not None or sam_params is not None:
            raise ValueError("M39 LoRA is mutually exclusive with Mixout/RecAdam/SAM")
    elif stage2_lora_cfg is not None:
        raise ValueError("LoRA is registered for M39 only")

    stage1_epochs = int(aux_variant.get("stage1_epochs", 3)) if is_aux_run else 5
    expected_stage1 = (
        (
            aux_variant["stage1_dataset"],
            aux_variant["train_split"],
            "vinli_dev",
            stage1_epochs,
            1e-5,
            400,
            3,
        )
        if is_aux_run
        else ("vinli", "vinli_train", "vinli_dev", 5, 1e-5, 400, 3)
    )
    actual_stage1 = (
        stage1.get("dataset"), stage1.get("train_split"), stage1.get("selection_split"),
        stage1.get("max_epochs"), stage1.get("learning_rate"),
        stage1.get("eval_optimizer_steps"), stage1.get("early_stopping_patience_evals"),
    )
    if actual_stage1 != expected_stage1:
        if is_aux_run:
            raise ValueError(
                f"{run_name} Stage 1 must be {aux_variant['stage1_dataset']} -> ViNLI dev, "
                f"{stage1_epochs} epochs, lr=1e-5, eval=400, patience=3"
            )
        raise ValueError(
            "Stage 1 must be fresh ViNLI train -> ViNLI dev, 5 epochs, "
            "lr=1e-5, eval=400, patience=3"
        )
    is_vimednli_target = run_name in VIMEDNLI_TARGET_RUNS
    expected_stage2 = (
        ("vimednli", "vimednli_train", "vimednli_dev", None, None, 400, None)
        if is_vimednli_target
        else ("vianli", "vianli_train", "vianli_dev", None, None, 400, None)
    )
    actual_stage2 = (
        stage2.get("dataset"), stage2.get("train_split"), stage2.get("selection_split"),
        stage2.get("max_epochs"), stage2.get("learning_rate"),
        stage2.get("eval_optimizer_steps"), stage2.get("early_stopping_patience_evals"),
    )
    is_m31 = run_name == "exp-001-r2-m31-stage2-mtvi-anli-light75-maxlen512-seed42"
    if gated_mode and is_vimednli_target:
        if actual_stage2 != ("vimednli", "vimednli_train", "vimednli_dev", 7, 1e-5, 400, 3):
            raise ValueError(
                "M48/M49 Stage 2 must be ViMedNLI dev selection, 7 epochs, "
                "lr=1e-5, eval=400, patience=3"
            )
        if stage2.get("replay") is not False or stage2.get("batch_mix") != {"vimednli": 4}:
            raise ValueError("M48/M49 Stage 2 must be ViMedNLI-only physical batches (no replay)")
    elif gated_mode and actual_stage2 != (
        "vianli", "vianli_train", "vianli_dev",
        (5 if (is_m31 or is_jtt_run) else 7), 1e-5, 400, 3,
    ):
        raise ValueError(
            f"Stage 2 must be ViANLI-only selection with lr=1e-5, eval=400, patience=3"
            f"{' (5 epochs for M31/M32)' if (is_m31 or is_jtt_run) else ' (7 epochs)'}"
        )
    if actual_stage2[0] != expected_stage2[0] or actual_stage2[1] != expected_stage2[1] \
            or actual_stage2[2] != expected_stage2[2] \
            or actual_stage2[3] not in (5, 7, 10) or actual_stage2[4] not in (1e-5, 5e-6) \
            or actual_stage2[5] != 400 or actual_stage2[6] not in (3, 5):
        raise ValueError(
            f"Stage 2 must be {expected_stage2[0]} dev selection, epochs in (5,7,10), "
            "lr in (1e-5,5e-6), eval=400, patience in (3,5)"
        )
    if not is_vimednli_target and stage2.get("replay"):
        # Replay variant: ViANLI + ViNLI mixed physical batches, stage-2 LR may differ.
        if stage2.get("batch_mix") != {"vianli": 3, "vinli": 1}:
            raise ValueError("Stage 2 replay mix must be vianli:3, vinli:1")
        if stage2.get("learning_rate") not in (1e-5, 5e-6):
            raise ValueError("Stage 2 replay LR must be 1e-5 or 5e-6")
    elif is_m31:
        if stage2.get("batch_mix") != {"vianli": 3, LAURER_MTVI_DOMAIN: 1}:
            raise ValueError("M31 Stage 2 mix must be vianli:3, laurer_mtvi:1")
    elif not is_vimednli_target:
        if stage2.get("batch_mix") != {"vianli": 4}:
            raise ValueError("Stage 2 must be ViANLI-only physical batches (no replay)")
    if is_aux_run and aux_variant["stage1_dataset"] != "vinli":
        allowed_stage1_mixes = [{
            "vinli": int(aux_variant.get("physical_native", 2)),
            LAURER_MTVI_DOMAIN: int(aux_variant.get("physical_aux", 2)),
        }]
    else:
        allowed_stage1_mixes = [{"vinli": 4}, {"vinli": 2, "vimednli": 2}]
    if stage1.get("batch_mix") not in allowed_stage1_mixes:
        raise ValueError("Stage 1 physical batch mix violates the registered run contract")
    if gated_mode and not is_aux_run and stage1.get("batch_mix") != {"vinli": 4}:
        raise ValueError("M13/M14 Stage 1 must be ViNLI-only with batch mix vinli:4")
    if gated_mode and not is_m31 and not is_vimednli_target and (
        stage2.get("replay") is not False or stage2.get("batch_mix") != {"vianli": 4}
    ):
        raise ValueError("M13/M14 Stage 2 must have no replay and batch mix vianli:4")
    if "vimednli" in stage1.get("batch_mix", {}) and stage1["batch_mix"]["vimednli"] != 2:
        raise ValueError("Stage 1 multi-source requires vimednli count 2")
    if stage2.get("selection_metric") != "macro_f1":
        raise ValueError("stage2 must select checkpoints by dev Macro-F1")
    if stage1.get("selection_metric") != "macro_f1":
        raise ValueError("stage1 must select checkpoints by dev Macro-F1")

    if cfg["wandb"].get("run_name") != run_name or cfg["wandb"].get("tracking_only") is not True:
        raise ValueError("W&B run identity/tracking-only contract mismatch")
    hf = cfg["hf_hub"]
    if hf.get("repo_id") != hf_repo or hf.get("private") is not False:
        raise ValueError("public HF artifact contract mismatch")
    if gated_mode:
        if run_name == "exp-001-r2-m15-gated-small256-maxlen512-seed42":
            expected_identity = {
                "run_name": "exp-001-r2-m15-gated-small256-maxlen512-seed42",
                "experiment_mode": "fresh_vinli_then_vianli_gated_small",
                "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m15-gated-small256-maxlen512-seed42",
            }
        elif run_name == "exp-001-r2-m16-gated-small256-ls02-wd005-maxlen512-seed42":
            expected_identity = {
                "run_name": "exp-001-r2-m16-gated-small256-ls02-wd005-maxlen512-seed42",
                "experiment_mode": "fresh_vinli_then_vianli_gated_small",
                "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m16-gated-small256-ls02-wd005-maxlen512-seed42",
            }
        elif run_name == "exp-001-r2-m17-gated-small128-maxlen512-seed42":
            expected_identity = {
                "run_name": "exp-001-r2-m17-gated-small128-maxlen512-seed42",
                "experiment_mode": "fresh_vinli_then_vianli_gated_small",
                "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m17-gated-small128-maxlen512-seed42",
            }
        elif run_name == "exp-001-r2-m18-gated-bias1-maxlen512-seed42":
            expected_identity = {
                "run_name": "exp-001-r2-m18-gated-bias1-maxlen512-seed42",
                "experiment_mode": "fresh_vinli_then_vianli_gated_small",
                "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m18-gated-bias1-maxlen512-seed42",
            }
        elif run_name == "exp-001-r2-m19-gated-small64-maxlen512-seed42":
            expected_identity = {
                "run_name": "exp-001-r2-m19-gated-small64-maxlen512-seed42",
                "experiment_mode": "fresh_vinli_then_vianli_gated_small",
                "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m19-gated-small64-maxlen512-seed42",
            }
        elif run_name == "exp-001-r2-m20-gated-small128-bias1-maxlen512-seed42":
            expected_identity = {
                "run_name": "exp-001-r2-m20-gated-small128-bias1-maxlen512-seed42",
                "experiment_mode": "fresh_vinli_then_vianli_gated_small",
                "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m20-gated-small128-bias1-maxlen512-seed42",
            }
        elif run_name == "exp-001-r2-m21-rdrop-a5-maxlen512-seed42":
            expected_identity = {
                "run_name": "exp-001-r2-m21-rdrop-a5-maxlen512-seed42",
                "experiment_mode": "fresh_vinli_then_vianli_gated_small",
                "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m21-rdrop-a5-maxlen512-seed42",
            }
        elif run_name == "exp-001-r2-m22-r3f-l1-maxlen512-seed42":
            expected_identity = {
                "run_name": "exp-001-r2-m22-r3f-l1-maxlen512-seed42",
                "experiment_mode": "fresh_vinli_then_vianli_gated_small",
                "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m22-r3f-l1-maxlen512-seed42",
            }
        elif run_name == "exp-001-r2-m24-childtune-f-p20-maxlen512-seed42":
            expected_identity = {
                "run_name": "exp-001-r2-m24-childtune-f-p20-maxlen512-seed42",
                "experiment_mode": "fresh_vinli_then_vianli_gated_small",
                "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m24-childtune-f-p20-maxlen512-seed42",
            }
        elif run_name == M32_JTT_RUN_NAME:
            expected_identity = {
                "run_name": M32_JTT_RUN_NAME,
                "experiment_mode": "fresh_vinli_then_vianli_compute_matched_jtt",
                "hf_repo": (
                    "trinhtrantran122/"
                    "vianli-exp-001-r2-m32-jtt-cm-w5-id2-final5-maxlen512-seed42"
                ),
            }
        elif run_name == "exp-001-r2-m33-freelb-k2-maxlen512-seed42":
            expected_identity = {
                "run_name": "exp-001-r2-m33-freelb-k2-maxlen512-seed42",
                "experiment_mode": "fresh_vinli_then_vianli_freelb",
                "hf_repo": (
                    "trinhtrantran122/"
                    "vianli-exp-001-r2-m33-freelb-k2-maxlen512-seed42"
                ),
            }
        elif run_name == "exp-001-r2-m38-sam-maxlen512-seed42":
            expected_identity = {
                "run_name": "exp-001-r2-m38-sam-maxlen512-seed42",
                "experiment_mode": "fresh_vinli_then_vianli_sam",
                "hf_repo": (
                    "trinhtrantran122/vianli-exp-001-r2-m38-sam-maxlen512-seed42"
                ),
            }
        elif run_name == "exp-001-r2-m36-mixout-maxlen512-seed42":
            expected_identity = {
                "run_name": "exp-001-r2-m36-mixout-maxlen512-seed42",
                "experiment_mode": "fresh_vinli_then_vianli_mixout",
                "hf_repo": (
                    "trinhtrantran122/vianli-exp-001-r2-m36-mixout-maxlen512-seed42"
                ),
            }
        elif run_name == "exp-001-r2-m37-recadam-maxlen512-seed42":
            expected_identity = {
                "run_name": "exp-001-r2-m37-recadam-maxlen512-seed42",
                "experiment_mode": "fresh_vinli_then_vianli_recadam",
                "hf_repo": (
                    "trinhtrantran122/vianli-exp-001-r2-m37-recadam-maxlen512-seed42"
                ),
            }
        elif run_name == M39_LORA_RUN_NAME:
            expected_identity = {
                "run_name": M39_LORA_RUN_NAME,
                "experiment_mode": "fresh_vinli_then_vianli_lora_stage2",
                "hf_repo": (
                    "trinhtrantran122/vianli-exp-001-r2-m39-lora-maxlen512-seed42"
                ),
            }
        elif run_name == M41_MNLI_BRIDGE_RUN_NAME:
            expected_identity = {
                "run_name": M41_MNLI_BRIDGE_RUN_NAME,
                "experiment_mode": "fresh_vinli_then_en_mnli_bridge_then_vianli",
                "hf_repo": (
                    "trinhtrantran122/vianli-exp-001-r2-m41-mnli-bridge-maxlen512-seed42"
                ),
            }
        elif run_name == M42_ANLI_BRIDGE_RUN_NAME:
            expected_identity = {
                "run_name": M42_ANLI_BRIDGE_RUN_NAME,
                "experiment_mode": "fresh_vinli_then_en_anli_bridge_then_vianli",
                "hf_repo": (
                    "trinhtrantran122/vianli-exp-001-r2-m42-anli-bridge-maxlen512-seed42"
                ),
            }
        elif run_name == M45_SIGNED_DELTA_RUN_NAME:
            expected_identity = {
                "run_name": M45_SIGNED_DELTA_RUN_NAME,
                "experiment_mode": "fresh_vinli_then_vianli_gated_signed_delta",
                "hf_repo": (
                    "trinhtrantran122/"
                    "vianli-exp-001-r2-m45-gated-signed-delta-maxlen512-seed42"
                ),
            }
        elif run_name == M46_HIER_E_FIRST_RUN_NAME:
            expected_identity = {
                "run_name": M46_HIER_E_FIRST_RUN_NAME,
                "experiment_mode": "fresh_vinli_then_vianli_hier_e_first",
                "hf_repo": (
                    "trinhtrantran122/"
                    "vianli-exp-001-r2-m46-hier-e-first-maxlen512-seed42"
                ),
            }
        elif run_name == M47_MIRROR_SYMMETRY_RUN_NAME:
            expected_identity = {
                "run_name": M47_MIRROR_SYMMETRY_RUN_NAME,
                "experiment_mode": "fresh_vinli_then_vianli_mirror_symmetry_stage2_locked",
                "hf_repo": (
                    "trinhtrantran122/"
                    "vianli-exp-001-r2-m47-mirror-symmetry-stage2-maxlen512-seed42"
                ),
            }
        elif run_name in VIMEDNLI_TARGET_RUNS:
            expected_identity = {
                "run_name": run_name,
                "experiment_mode": (
                    "fresh_vinli_then_vimednli_ce"
                    if run_name == M48_VIMEDNLI_CE_RUN_NAME
                    else "fresh_vinli_then_vimednli_mirror_symmetry"
                ),
                "hf_repo": (
                    "trinhtrantran122/vianli-exp-001-r2-m48-vimednli-ce-maxlen256-seed42"
                    if run_name == M48_VIMEDNLI_CE_RUN_NAME
                    else "trinhtrantran122/"
                         "vianli-exp-001-r2-m49-vimednli-mirror-symmetry-maxlen256-seed42"
                ),
            }
        elif is_aux_run:
            expected_identity = {
                "run_name": run_name,
                "experiment_mode": aux_variant["experiment_mode"],
                "hf_repo": aux_variant["hf_repo"],
            }
        else:
            expected_identity = {
                "gated_dual": {
                    "run_name": "exp-001-r2-m13-gated-global-sentence-maxlen512-seed42",
                    "experiment_mode": "fresh_vinli_then_vianli_gated_global_sentence",
                    "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m13-gated-global-sentence-maxlen512-seed42",
                },
                "token_align": {
                    "run_name": "exp-001-r2-m14-gated-token-align-maxlen512-seed42",
                    "experiment_mode": "fresh_vinli_then_vianli_gated_token_alignment",
                    "hf_repo": "trinhtrantran122/vianli-exp-001-r2-m14-gated-token-align-maxlen512-seed42",
                },
            }[pool_mode]
        wandb = cfg["wandb"]
        test_locked_search = run_name == M47_MIRROR_SYMMETRY_RUN_NAME
        if (
            project.get("experiment_name") != expected_identity["run_name"]
            or project.get("experiment_mode") != expected_identity["experiment_mode"]
            or project.get("target_test_locked") is not test_locked_search
            or wandb.get("enabled") is not True
            or wandb.get("entity") != "trinhtrantran3105-uit"
            or wandb.get("project") != "hierarchical-nli-e-first"
            or wandb.get("run_name") != expected_identity["run_name"]
        ):
            raise ValueError("gated experiment or W&B identity mismatch")
        expected_output = f"outputs/exp001/{expected_identity['run_name']}"
        if cfg["outputs"].get("root") != expected_output:
            raise ValueError("M13/M14 output identity mismatch")
        if hf.get("enabled") is not True or hf.get("repo_id") != expected_identity["hf_repo"]:
            raise ValueError("M13/M14 per-run public HF identity mismatch")
        evaluation = cfg.get("evaluation", {})
        expected_search_splits = {"vinli_dev", "vianli_dev"}
        if not test_locked_search:
            expected_search_splits.add("vianli_test")
        if run_name in VIMEDNLI_TARGET_RUNS:
            expected_search_splits = {"vinli_dev", "vimednli_dev", "vimednli_test"}
        if (
            evaluation.get("target_test_enabled") is not (not test_locked_search)
            or evaluation.get("method_frozen") is not (not test_locked_search)
            or set(evaluation.get("search_splits", []))
            != expected_search_splits
        ):
            raise ValueError("gated evaluation contract mismatch")
    if not {"stage1_best", "stage2_best"}.issubset(set(hf.get("required_checkpoint_dirs", []))):
        raise ValueError("HF contract must require both best stage checkpoints")
    if is_bridge_run and "stage1b_best" not in hf.get("required_checkpoint_dirs", []):
        raise ValueError("bridge runs must require stage1b_best checkpoint")
    if not is_bridge_run and "stage1b_best" in hf.get("required_checkpoint_dirs", []):
        raise ValueError("only bridge runs may require stage1b_best checkpoint")
    required_prediction_splits = set(hf.get("required_prediction_splits", []))
    core_dev_splits = (
        {"vinli_dev", "vimednli_dev"}
        if run_name in VIMEDNLI_TARGET_RUNS
        else {"vinli_dev", "vianli_dev"}
    )
    if not core_dev_splits.issubset(required_prediction_splits):
        raise ValueError("HF contract must require dev prediction/logit files")
    expected_prediction_splits = {"vinli_dev", "vianli_dev"}
    if run_name != M47_MIRROR_SYMMETRY_RUN_NAME:
        expected_prediction_splits.add("vianli_test")
    if run_name in VIMEDNLI_TARGET_RUNS:
        expected_prediction_splits = {"vinli_dev", "vimednli_dev", "vimednli_test"}
    if is_jtt_run:
        expected_prediction_splits.add("jtt_identification_vianli_train")
    if is_bridge_run:
        expected_prediction_splits.add("stage1b_vinli_dev")
    if gated_mode and required_prediction_splits != expected_prediction_splits:
        raise ValueError("gated HF prediction contract mismatch")


def validate_runtime_args(debug: bool, tiny_model: str | None) -> None:
    if tiny_model and not debug:
        raise ValueError("--tiny-model is a debug-only gate and cannot change a scientific run")
