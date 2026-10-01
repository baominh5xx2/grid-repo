"""Runtime utilities, environment metadata, and output directory management for R2."""
from __future__ import annotations

import hashlib
import importlib.metadata
import pathlib
import platform
import random
import subprocess
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]

def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "untracked_session"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:
        return False


def fresh_output_root(cfg: dict, debug: bool) -> pathlib.Path:
    root = ROOT / cfg["outputs"]["root"]
    if debug:
        root = root.with_name(root.name + "-debug")
    if root.exists() and any(root.iterdir()):
        # Every run keeps its own artifact directory: append an incrementing
        # suffix instead of wiping or reusing previous run outputs.
        candidate = root
        counter = 2
        while candidate.exists() and any(candidate.iterdir()):
            candidate = root.with_name(f"{root.name}-run{counter}")
            counter += 1
        root = candidate
    root.mkdir(parents=True, exist_ok=True)
    return root


def environment_metadata() -> dict:
    packages = {}
    for name in ("torch", "transformers", "numpy", "pandas", "wandb", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
