"""Fail-closed Hugging Face artifact storage for scientific runs.

W&B stores tracking only. Checkpoints, tokenizer/config, metadata, and per-sample
predictions live in a HF model repo (public for R2 due to private quota) and are
read back at an immutable commit.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile

from gated_dual_ema_msd.utils.env import load_env  # noqa: F401

try:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    HAS_HF = True
except ImportError:
    HAS_HF = False


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_enabled(cfg: dict) -> bool:
    return HAS_HF and cfg.get("hf_hub", {}).get("enabled", True) and bool(os.getenv("HF_TOKEN"))


def require_hf(cfg: dict) -> None:
    if not cfg.get("hf_hub", {}).get("enabled", True):
        raise RuntimeError("HF Hub is mandatory for real runs but hf_hub.enabled=false")
    if not HAS_HF:
        raise RuntimeError("HF Hub is mandatory for real runs but huggingface_hub is unavailable")
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF Hub is mandatory for real runs but HF_TOKEN is missing")
    if not cfg.get("hf_hub", {}).get("repo_id"):
        raise RuntimeError("HF Hub is mandatory for real runs but hf_hub.repo_id is missing")
    if cfg.get("hf_hub", {}).get("private") not in (True, False):
        raise RuntimeError("hf_hub.private must be explicitly true or false")


def _api():
    return HfApi(token=os.environ["HF_TOKEN"])


def ensure_repo(repo_id: str, private: bool | None = None):
    # private=None means respect existing repo; explicit bool enforces visibility.
    api = _api()
    if private is None:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        return api.repo_info(repo_id=repo_id, repo_type="model")
    api.create_repo(repo_id=repo_id, repo_type="model", private=bool(private), exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if bool(info.private) is not bool(private):
        # Try to reconcile visibility if repo already existed with different setting.
        try:
            api.update_repo_visibility(repo_id=repo_id, private=bool(private), repo_type="model")
            info = api.repo_info(repo_id=repo_id, repo_type="model")
        except Exception as exc:  # noqa: BLE001
            print(f"[hf_hub] visibility reconciliation failed for {repo_id}: {type(exc).__name__}")
    if bool(info.private) is not bool(private):
        raise RuntimeError(f"HF repo visibility mismatch {repo_id}: expected private={private}, got {info.private}")
    return info


def _next_version_tag(api, repo_id: str) -> str:
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
    nums = [int(r.name[1:]) for r in refs.tags if r.name.startswith("v") and r.name[1:].isdigit()]
    return f"v{(max(nums) + 1) if nums else 1}"


def directory_manifest(directory: pathlib.Path) -> dict[str, dict]:
    directory = pathlib.Path(directory)
    return {
        path.relative_to(directory).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _readback_verify(repo_id: str, revision: str, expected_metadata: dict,
                     pred_paths: dict[str, pathlib.Path], checkpoint_manifest: dict,
                     cache_dir: str | None = None) -> None:
    token = os.environ["HF_TOKEN"]
    meta_path = pathlib.Path(hf_hub_download(
        repo_id=repo_id, repo_type="model", filename="run_metadata.json",
        revision=revision, token=token, cache_dir=cache_dir,
    ))
    readback = json.loads(meta_path.read_text(encoding="utf-8"))
    if readback != expected_metadata:
        raise RuntimeError("HF run_metadata.json read-back mismatch")
    api = _api()
    remote_checkpoint_files = {
        name.removeprefix("stage2_checkpoint/")
        for name in api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
        if name.startswith("stage2_checkpoint/")
    }
    if remote_checkpoint_files != set(checkpoint_manifest):
        raise RuntimeError(
            "HF checkpoint file set mismatch: "
            f"remote={sorted(remote_checkpoint_files)}, expected={sorted(checkpoint_manifest)}"
        )
    for relative_path, expected in checkpoint_manifest.items():
        remote = pathlib.Path(hf_hub_download(
            repo_id=repo_id, repo_type="model",
            filename=f"stage2_checkpoint/{relative_path}", revision=revision, token=token, cache_dir=cache_dir,
        ))
        if remote.stat().st_size != expected["size_bytes"] or sha256_file(remote) != expected["sha256"]:
            raise RuntimeError(f"HF checkpoint read-back mismatch: {relative_path}")
    for split, local_path in pred_paths.items():
        remote = pathlib.Path(hf_hub_download(
            repo_id=repo_id, repo_type="model", filename=f"predictions/{local_path.name}",
            revision=revision, token=token, cache_dir=cache_dir,
        ))
        if sha256_file(remote) != sha256_file(local_path):
            raise RuntimeError(f"HF read-back hash mismatch for {split} predictions")


def push_run_artifacts(cfg: dict, checkpoint_dir: pathlib.Path, pred_paths: dict,
                       run_metadata: dict, skip_checkpoint: bool = False) -> tuple[str, str]:
    """Upload artifacts, tag only the final commit, verify read-back, return exact SHA."""
    require_hf(cfg)
    hf_cfg = cfg["hf_hub"]
    repo_id = hf_cfg["repo_id"]
    ensure_repo(repo_id, private=hf_cfg.get("private"))
    api = _api()
    checkpoint_dir = pathlib.Path(checkpoint_dir)

    commit_msg = f"{run_metadata.get('experiment_id', run_metadata.get('experiment_mode'))} seed={run_metadata.get('seed')}"
    if skip_checkpoint:
        base_revision = api.repo_info(repo_id=repo_id, repo_type="model").sha
        checkpoint_manifest = run_metadata.get("checkpoint_manifest")
        if not checkpoint_manifest:
            raise RuntimeError("skip_checkpoint requires the verified checkpoint_manifest")
    else:
        if not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
            raise RuntimeError(f"checkpoint directory is empty: {checkpoint_dir}")
        if (checkpoint_dir / "MOCK_CHECKPOINT.json").exists():
            raise RuntimeError("refusing to upload a mock checkpoint")
        checkpoint_manifest = directory_manifest(checkpoint_dir)
        checkpoint_commit = api.upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=str(checkpoint_dir),
            path_in_repo="stage2_checkpoint", commit_message=f"{commit_msg} checkpoint",
            delete_patterns="stage2_checkpoint/*",
        )
        base_revision = checkpoint_commit.oid

    normalized_preds: dict[str, pathlib.Path] = {}
    for split, raw_path in pred_paths.items():
        path = pathlib.Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"mandatory prediction file missing: {path}")
        if "vianli_test" in split and not run_metadata.get("target_test_accessed", False):
            raise RuntimeError("refusing to upload ViANLI test predictions without target_test_accessed")
        normalized_preds[split] = path

    final_metadata = dict(run_metadata)
    final_metadata["hf_repo_id"] = repo_id
    final_metadata["hf_artifact_base_revision"] = base_revision
    final_metadata["checkpoint_manifest"] = checkpoint_manifest
    final_metadata["prediction_sha256"] = {
        split: sha256_file(path) for split, path in normalized_preds.items()
    }

    with tempfile.TemporaryDirectory() as tmp:
        meta_path = pathlib.Path(tmp) / "run_metadata.json"
        meta_path.write_text(json.dumps(final_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        operations = [CommitOperationAdd(path_in_repo="run_metadata.json", path_or_fileobj=str(meta_path))]
        operations.extend(
            CommitOperationAdd(path_in_repo=f"predictions/{path.name}", path_or_fileobj=str(path))
            for path in normalized_preds.values()
        )
        final_commit = api.create_commit(
            repo_id=repo_id, repo_type="model", operations=operations,
            commit_message=f"{commit_msg} metadata and predictions", parent_commit=base_revision,
        )

    final_revision = final_commit.oid
    version_tag = _next_version_tag(api, repo_id)
    api.create_tag(
        repo_id=repo_id, repo_type="model", tag=version_tag, revision=final_revision,
        tag_message=commit_msg,
    )
    tagged_sha = api.repo_info(repo_id=repo_id, repo_type="model", revision=version_tag).sha
    if tagged_sha != final_revision:
        raise RuntimeError(f"HF tag {version_tag} resolved to stale revision {tagged_sha}")
    _readback_verify(
        repo_id, final_revision, final_metadata, normalized_preds, checkpoint_manifest,
        cache_dir=hf_cfg.get("readback_cache_dir"),
    )
    run_metadata.update(final_metadata)
    print(f"[hf_hub] verified artifact {repo_id}@{final_revision} (tag {version_tag})")
    return repo_id, final_revision


def _canonical_json(obj):
    """JSON-roundtrip-safe normalization: tuples become lists, keys sorted."""
    if isinstance(obj, dict):
        return {str(k): _canonical_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical_json(v) for v in obj]
    return obj


def _readback_verify_dual_stage(repo_id: str, revision: str, expected_metadata: dict,
                                checkpoint_manifests: dict[str, dict],
                                pred_paths: dict[str, pathlib.Path]) -> None:
    """Verify every mandatory dual-stage artifact at the immutable commit SHA."""
    token = os.environ["HF_TOKEN"]
    meta_path = pathlib.Path(hf_hub_download(
        repo_id=repo_id, repo_type="model", filename="run_metadata.json",
        revision=revision, token=token,
    ))
    loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    if _canonical_json(loaded) != _canonical_json(expected_metadata):
        raise RuntimeError("HF dual-stage run_metadata.json read-back mismatch")

    api = _api()
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision))
    for stage_name, manifest in checkpoint_manifests.items():
        remote_stage_files = {
            name.removeprefix(f"{stage_name}/")
            for name in remote_files if name.startswith(f"{stage_name}/")
        }
        if remote_stage_files != set(manifest):
            raise RuntimeError(
                f"HF {stage_name} file set mismatch: remote={sorted(remote_stage_files)}, "
                f"expected={sorted(manifest)}"
            )
        for relative_path, expected in manifest.items():
            remote = pathlib.Path(hf_hub_download(
                repo_id=repo_id, repo_type="model",
                filename=f"{stage_name}/{relative_path}", revision=revision, token=token,
            ))
            if remote.stat().st_size != expected["size_bytes"] or sha256_file(remote) != expected["sha256"]:
                raise RuntimeError(f"HF {stage_name} read-back mismatch: {relative_path}")
    for split, local_path in pred_paths.items():
        remote = pathlib.Path(hf_hub_download(
            repo_id=repo_id, repo_type="model", filename=f"predictions/{local_path.name}",
            revision=revision, token=token,
        ))
        if sha256_file(remote) != sha256_file(local_path):
            raise RuntimeError(f"HF read-back hash mismatch for {split} predictions")


def push_best_checkpoint(repo_id: str, stage_name: str, stage_dir: pathlib.Path,
                         private: bool, metric: float, optimizer_step: int) -> str:
    """Overwrite the experiment's best checkpoint on HF when dev metric improves.

    path_in_repo: best/<stage_name>/ ... The commit deletes the previous best
    files first (delete_patterns), so HF always holds the single best checkpoint
    of this experiment for the stage. Returns the new commit SHA.
    """
    api = _api()
    stage_dir = pathlib.Path(stage_dir)
    if not stage_dir.exists() or not any(stage_dir.iterdir()):
        raise RuntimeError(f"best checkpoint directory is empty: {stage_dir}")
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(stage_dir),
        path_in_repo=f"best/{stage_name}",
        commit_message=f"best {stage_name} dev_macro_f1={metric:.6f} step={optimizer_step}",
        delete_patterns=f"best/{stage_name}/*",
    )
    print(f"[hf_hub] best {stage_name} overwritten @{commit.oid[:12]} (dev_macro_f1={metric:.6f})")
    return commit.oid


def validate_dual_stage_prediction_keys(
    required_predictions: set[str], payload_predictions: set[str],
    target_test_accessed: bool = True,
    target_dataset: str = "vianli",
) -> None:
    """Fail closed for either a frozen-final or explicitly locked search run."""
    if target_dataset not in ("vianli", "vimednli"):
        raise RuntimeError(f"unsupported target dataset: {target_dataset}")
    target_dev = f"{target_dataset}_dev"
    target_test = f"{target_dataset}_test"
    core_predictions = {"vinli_dev", target_dev}
    if target_test_accessed:
        core_predictions.add(target_test)
    elif target_test in required_predictions:
        raise RuntimeError(
            f"locked-search upload must not require {target_test} predictions"
        )
    if not core_predictions.issubset(required_predictions):
        raise RuntimeError(
            "dual-stage upload must require its configured core prediction splits"
        )
    if payload_predictions != required_predictions:
        raise RuntimeError(
            "dual-stage prediction payload must exactly match configured required splits: "
            f"payload={sorted(payload_predictions)} required={sorted(required_predictions)}"
        )


def push_dual_stage_artifacts(cfg: dict, checkpoint_dirs: dict[str, pathlib.Path],
                              pred_paths: dict[str, pathlib.Path],
                              run_metadata: dict) -> tuple[str, str]:
    """Atomically publish and immutably verify both selected stage checkpoints.

    This is the artifact contract for full Stage1->Stage2 runs. It is intentionally
    separate from ``push_run_artifacts``, whose scientifically truthful contract is a
    migrated-source, Stage-2-only checkpoint.
    """
    require_hf(cfg)
    required_stages = set(cfg["hf_hub"].get("required_checkpoint_dirs", []))
    required_predictions = set(cfg["hf_hub"].get("required_prediction_splits", []))
    allowed_stages = {"stage1_best", "stage2_best", "stage1b_best"}
    if not {"stage1_best", "stage2_best"}.issubset(required_stages):
        raise RuntimeError("dual-stage upload must require both stage1_best and stage2_best")
    if not required_stages.issubset(allowed_stages):
        raise RuntimeError(f"unexpected checkpoint stage(s): {required_stages - allowed_stages}")
    if set(checkpoint_dirs) != required_stages:
        raise RuntimeError(
            f"checkpoint payload {sorted(checkpoint_dirs)} != configured required "
            f"{sorted(required_stages)}"
        )
    target_test_accessed = run_metadata.get("target_test_accessed") is True
    validate_dual_stage_prediction_keys(
        required_predictions, set(pred_paths),
        target_test_accessed=target_test_accessed,
        target_dataset=run_metadata.get("target_dataset", "vianli"),
    )

    repo_id = cfg["hf_hub"]["repo_id"]
    ensure_repo(repo_id, private=cfg["hf_hub"].get("private"))
    api = _api()
    checkpoint_manifests: dict[str, dict] = {}
    operations = []
    for stage_name in ("stage1_best", "stage1b_best", "stage2_best"):
        if stage_name not in required_stages:
            continue
        directory = pathlib.Path(checkpoint_dirs[stage_name])
        if not directory.exists() or not any(directory.iterdir()):
            raise RuntimeError(f"checkpoint directory is empty: {directory}")
        if (directory / "MOCK_CHECKPOINT.json").exists():
            raise RuntimeError(f"refusing to upload mock checkpoint: {stage_name}")
        manifest = directory_manifest(directory)
        required_files = {"pytorch_model.bin", "config.json", "checkpoint_metadata.json"}
        if not required_files.issubset(manifest):
            raise RuntimeError(f"{stage_name} lacks model/config/metadata files")
        if not any(name.startswith("tokenizer") or name in {"vocab.txt", "sentencepiece.bpe.model"}
                   for name in manifest):
            raise RuntimeError(f"{stage_name} lacks tokenizer files")
        checkpoint_manifests[stage_name] = manifest
        operations.extend(
            CommitOperationAdd(
                path_in_repo=f"{stage_name}/{path.relative_to(directory).as_posix()}",
                path_or_fileobj=str(path),
            )
            for path in sorted(directory.rglob("*")) if path.is_file()
        )

    normalized_preds: dict[str, pathlib.Path] = {}
    for split in sorted(required_predictions):
        path = pathlib.Path(pred_paths[split])
        if not path.exists():
            raise FileNotFoundError(f"mandatory prediction file missing: {path}")
        normalized_preds[split] = path
        operations.append(CommitOperationAdd(
            path_in_repo=f"predictions/{path.name}", path_or_fileobj=str(path)
        ))

    final_metadata = dict(run_metadata)
    final_metadata["hf_repo_id"] = repo_id
    final_metadata["checkpoint_manifests"] = checkpoint_manifests
    final_metadata["prediction_sha256"] = {
        split: sha256_file(path) for split, path in normalized_preds.items()
    }
    with tempfile.TemporaryDirectory() as tmp:
        metadata_path = pathlib.Path(tmp) / "run_metadata.json"
        metadata_path.write_text(
            json.dumps(final_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        operations.append(CommitOperationAdd(
            path_in_repo="run_metadata.json", path_or_fileobj=str(metadata_path)
        ))
        commit = api.create_commit(
            repo_id=repo_id, repo_type="model", operations=operations,
            commit_message=f"EXP-001-R2 seed={run_metadata.get('seed')} dual-stage artifact",
        )

    revision = commit.oid
    version_tag = _next_version_tag(api, repo_id)
    api.create_tag(
        repo_id=repo_id, repo_type="model", tag=version_tag, revision=revision,
        tag_message="EXP-001-R2 immutable dual-stage artifact",
    )
    tagged_sha = api.repo_info(repo_id=repo_id, repo_type="model", revision=version_tag).sha
    if tagged_sha != revision:
        raise RuntimeError(f"HF tag {version_tag} resolved to stale revision {tagged_sha}")
    _readback_verify_dual_stage(
        repo_id, revision, final_metadata, checkpoint_manifests, normalized_preds
    )
    run_metadata.update(final_metadata)
    print(f"[hf_hub] verified dual-stage artifact {repo_id}@{revision} (tag {version_tag})")
    return repo_id, revision
