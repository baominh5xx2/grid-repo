"""Unified tracking session lifecycle and artifact verification."""
from __future__ import annotations

from dataclasses import dataclass
import os
import pathlib
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class VerifiedUpload:
    repo_id: str
    revision: str
    file_sha256: dict[str, str]


def publish_direct_verified(
    cfg: Mapping[str, Any],
    checkpoint_dir: pathlib.Path,
    prediction_paths: Mapping[str, pathlib.Path],
    run_metadata: dict,
) -> VerifiedUpload:
    """Upload and read-back verify single-stage (direct/migrated) artifacts."""
    from gated_dual_ema_msd.tracking.hf import push_run_artifacts

    repo_id, revision = push_run_artifacts(
        dict(cfg),
        pathlib.Path(checkpoint_dir),
        dict(prediction_paths),
        run_metadata,
    )
    file_sha256 = {}
    checkpoint_manifest = run_metadata.get("checkpoint_manifest", {})
    for rel_path, info in checkpoint_manifest.items():
        file_sha256[f"stage2_checkpoint/{rel_path}"] = info["sha256"]
    for split, p in prediction_paths.items():
        file_sha256[f"predictions/{pathlib.Path(p).name}"] = run_metadata.get(
            "prediction_sha256", {}
        ).get(split, "")

    return VerifiedUpload(
        repo_id=repo_id,
        revision=revision,
        file_sha256=file_sha256,
    )


def publish_r2_verified(
    cfg: Mapping[str, Any],
    checkpoint_dirs: Mapping[str, pathlib.Path],
    prediction_paths: Mapping[str, pathlib.Path],
    run_metadata: dict,
) -> VerifiedUpload:
    """Upload and read-back verify dual-stage (R2) artifacts."""
    from gated_dual_ema_msd.tracking.hf import push_dual_stage_artifacts

    repo_id, revision = push_dual_stage_artifacts(
        dict(cfg),
        {k: pathlib.Path(v) for k, v in checkpoint_dirs.items()},
        {k: pathlib.Path(v) for k, v in prediction_paths.items()},
        run_metadata,
    )
    file_sha256 = {}
    checkpoint_manifests = run_metadata.get("checkpoint_manifests", {})
    for stage_name, manifest in checkpoint_manifests.items():
        for rel_path, info in manifest.items():
            file_sha256[f"{stage_name}/{rel_path}"] = info["sha256"]
    for split, p in prediction_paths.items():
        file_sha256[f"predictions/{pathlib.Path(p).name}"] = run_metadata.get(
            "prediction_sha256", {}
        ).get(split, "")

    return VerifiedUpload(
        repo_id=repo_id,
        revision=revision,
        file_sha256=file_sha256,
    )


class RunSession:
    """Context manager unifying W&B tracking and HF artifact lifecycle."""

    def __init__(self, cfg: Mapping[str, Any], *, offline: bool = False):
        self.cfg = dict(cfg)
        self.offline = offline
        self.run = None
        self.artifacts_verified = False
        self.finished = False
        self.current_lifecycle: Optional[str] = None

    def start(self) -> None:
        if self.offline:
            return
        wandb_cfg = self.cfg.get("wandb", {})
        if wandb_cfg.get("enabled", True):
            try:
                import wandb

                self.run = wandb.init(
                    project=wandb_cfg.get("project"),
                    entity=wandb_cfg.get("entity"),
                    name=wandb_cfg.get(
                        "run_name",
                        self.cfg.get("project", {}).get("experiment_name"),
                    ),
                    config=self.cfg,
                )
            except Exception:
                if not self.offline and wandb_cfg.get("tracking_only"):
                    raise

    def preflight(self) -> None:
        if self.offline:
            return
        hf_cfg = self.cfg.get("hf_hub", {})
        if hf_cfg.get("enabled", True):
            from gated_dual_ema_msd.tracking.hf import ensure_repo, require_hf

            require_hf(self.cfg)
            ensure_repo(hf_cfg["repo_id"], private=hf_cfg.get("private"))

    def lifecycle(self, state: str) -> None:
        self.current_lifecycle = state
        if self.run is not None:
            try:
                self.run.log({"lifecycle": state})
                self.run.summary["lifecycle"] = state
            except Exception:
                pass

    def finish(self, state: str) -> None:
        if self.finished:
            return
        self.finished = True
        self.current_lifecycle = state
        if self.run is not None:
            try:
                self.run.summary["lifecycle"] = state
                self.run.finish(exit_code=0 if state == "complete" else 1)
            except Exception:
                pass

    def mark_verified(self, upload: VerifiedUpload) -> None:
        self.artifacts_verified = True
        if self.run is not None:
            try:
                self.run.summary["hf_repo_id"] = upload.repo_id
                self.run.summary["hf_revision"] = upload.revision
            except Exception:
                pass

    def __enter__(self) -> "RunSession":
        try:
            self.start()
            self.lifecycle("initialization")
            self.preflight()
        except Exception:
            self.finish("failed")
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        state = "failed" if exc_type else "complete"
        if (
            state == "complete"
            and self.cfg.get("hf_hub", {}).get("enabled")
            and not self.offline
            and not self.artifacts_verified
        ):
            self.finish("failed")
            raise RuntimeError("Cannot complete before verified artifact read-back")
        self.finish(state)
        return False
