"""W&B helper — TRACKING ONLY (ported from hierarchical-nli-e-first, generalized from
run_type in {"flat","hier"} to an arbitrary experiment_name — the 4 multisource modes
each need their own unique run name: flat-<mode>-seed42).

Logs experiment mode, source datasets, stage1/stage2 mixes, sample counts, ViANLI dev
Macro-F1, final test metrics, audit status, and the HF Hub repo_id/revision where the
checkpoint/predictions actually live. W&B never receives checkpoints or per-sample
predictions — those are mandatory-stored on HF Hub instead (see hf_hub_helper.py).
"""
from __future__ import annotations
import os
from gated_dual_ema_msd.utils.env import load_env  # noqa: F401 (import side-effect: loads .env)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def wandb_enabled(cfg: dict) -> bool:
    return HAS_WANDB and cfg.get("wandb", {}).get("enabled", True) and bool(os.getenv("WANDB_API_KEY"))


def require_wandb(cfg: dict) -> None:
    """Fail closed for scientific runs instead of silently dropping tracking."""
    if not cfg.get("wandb", {}).get("enabled", True):
        raise RuntimeError("W&B is mandatory for real runs but wandb.enabled=false")
    if not HAS_WANDB:
        raise RuntimeError("W&B is mandatory for real runs but the wandb package is unavailable")
    if not os.getenv("WANDB_API_KEY"):
        raise RuntimeError("W&B is mandatory for real runs but WANDB_API_KEY is missing")


def init_run(cfg: dict, experiment_name: str, hparams: dict, tags=None):
    """Start mandatory tracking; initialization failure aborts a real run."""
    require_wandb(cfg)
    wb_cfg = cfg.get("wandb", {})
    try:
        run = wandb.init(
            project=wb_cfg.get("project", "hierarchical-nli-e-first"),
            entity=wb_cfg.get("entity") or None,
            name=experiment_name,
            group=cfg["project"].get("experiment_type", "multisource_flat_transfer"),
            job_type="multisource_flat_transfer",
            tags=tags or [cfg["project"].get("experiment_type", "multisource"), f"seed{cfg['project']['seed']}"],
            config=hparams,
        )
        print(f"[wandb] run started: {run.url}")
        return run
    except Exception as e:
        raise RuntimeError(f"W&B initialization failed for {experiment_name}") from e


def log_step(run, metrics: dict, step: int | None = None, commit: bool = True):
    if run is None:
        return
    try:
        run.log(metrics, step=step, commit=commit)
    except Exception as e:
        raise RuntimeError("W&B metric logging failed") from e


def log_epoch_metrics(run, split: str, metrics: dict, epoch: int):
    if run is None:
        return
    payload = {f"{split}/{k}": v for k, v in metrics.items()}
    payload["epoch"] = epoch
    log_step(run, payload)


def log_confusion_matrix(run, y_true, y_pred, class_names, title: str):
    if run is None:
        return
    try:
        run.log({title: wandb.plot.confusion_matrix(
            preds=list(y_pred), y_true=list(y_true), class_names=class_names)})
    except Exception as e:
        raise RuntimeError("W&B confusion-matrix logging failed") from e


def log_hf_link(run, repo_id: str, revision: str):
    if run is None:
        return
    try:
        run.summary["hf_repo_id"] = repo_id
        run.summary["hf_revision"] = revision
        run.summary["hf_url"] = f"https://huggingface.co/{repo_id}/tree/{revision}"
        run.log({"hf_repo_id": repo_id, "hf_revision": revision})
    except Exception as e:
        raise RuntimeError("W&B HF-link logging failed") from e


def log_audit_status(run, audit_report: dict):
    if run is None:
        return
    try:
        run.summary["audit_leakage_passed"] = audit_report.get("leakage_passed")
        run.summary["audit_all_checks_passed"] = audit_report.get("all_checks_passed")
    except Exception as e:
        raise RuntimeError("W&B audit-status logging failed") from e


def finish(run):
    if run is None:
        return
    try:
        run.finish()
        print("[wandb] run closed")
    except Exception as e:
        raise RuntimeError("W&B run finalization failed") from e
