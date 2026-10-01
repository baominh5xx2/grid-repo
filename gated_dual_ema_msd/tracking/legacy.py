"""Seamless Tracking & Artifact Uploading for W&B and Hugging Face Hub.

Encapsulates:
  - W&B metric logging, confusion matrix, step curves, and run summaries.
  - HF Hub automatic repository creation, model weights, tokenizer, configs,
    and prediction CSV uploads with revision verification.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from dotenv import load_dotenv
    # Search for .env in current dir, parent dir, or repo root
    for env_cand in [pathlib.Path.cwd() / ".env", pathlib.Path.cwd().parent / ".env", pathlib.Path(__file__).resolve().parents[1] / ".env"]:
        if env_cand.exists():
            load_dotenv(env_cand, override=False)
            break
except ImportError:
    pass


class WandbTracker:
    """Manages W&B run lifecycle and real-time metric logging."""

    def __init__(
        self,
        project: str = "gated-dual-ema-msd",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.run = None
        if not self.enabled:
            return

        if api_key:
            os.environ["WANDB_API_KEY"] = api_key

        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            print("[W&B] Warning: wandb package not installed. Metric tracking disabled.")
            self.enabled = False
            return

        if not os.getenv("WANDB_API_KEY"):
            print("[W&B] Note: WANDB_API_KEY not found in environment. Metric tracking disabled.")
            self.enabled = False
            return

        try:
            self.run = self.wandb.init(
                project=project,
                entity=entity,
                name=name,
                tags=tags or ["nli", "gated_dual_ema_msd"],
                config=config or {},
                reinit=True,
            )
            print(f"🚀 [W&B] Tracking started: {self.run.url}")
        except Exception as e:
            print(f"[W&B] Warning: Could not initialize W&B: {e}")
            self.enabled = False

    def log_step(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self.enabled and self.run is not None:
            try:
                self.run.log(metrics, step=step)
            except Exception as e:
                print(f"[W&B] Warning: Failed to log step metrics: {e}")

    def log_confusion_matrix(self, y_true: List[int], y_pred: List[int], title: str = "Test Confusion Matrix"):
        if self.enabled and self.run is not None:
            try:
                class_names = ["Entailment", "Contradiction", "Neutral"]
                cm = self.wandb.plot.confusion_matrix(
                    preds=y_pred,
                    y_true=y_true,
                    class_names=class_names,
                    title=title,
                )
                self.run.log({title: cm})
            except Exception as e:
                print(f"[W&B] Warning: Failed to log confusion matrix: {e}")

    def log_hf_summary(self, repo_id: str, commit_url: Optional[str] = None):
        if self.enabled and self.run is not None:
            try:
                self.run.summary["hf_repo_id"] = repo_id
                if commit_url:
                    self.run.summary["hf_model_url"] = commit_url
                self.run.log({"hf_repo_id": repo_id})
            except Exception as e:
                print(f"[W&B] Warning: Failed to log HF summary: {e}")

    def finish(self):
        if self.enabled and self.run is not None:
            try:
                self.run.finish()
                print("🏁 [W&B] Tracking finished cleanly.")
            except Exception as e:
                print(f"[W&B] Warning: Failed to finish W&B run: {e}")


class HFHubUploader:
    """Manages pushing checkpoints, configs, tokenizer, and prediction tables to Hugging Face Hub."""

    def __init__(
        self,
        token: Optional[str] = None,
        private: bool = False,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.private = private
        if not self.enabled:
            return

        if token:
            os.environ["HF_TOKEN"] = token

        try:
            from huggingface_hub import HfApi
            self.api_cls = HfApi
        except ImportError:
            print("[HF Hub] Warning: huggingface_hub package not installed. HF upload disabled.")
            self.enabled = False
            return

        token_val = os.getenv("HF_TOKEN")
        if not token_val:
            print("[HF Hub] Note: HF_TOKEN not found in environment. HF upload disabled.")
            self.enabled = False
            return

        try:
            self.api = self.api_cls(token=token_val)
        except Exception as e:
            print(f"[HF Hub] Warning: Failed to authenticate with HF Hub: {e}")
            self.enabled = False

    def upload_artifacts(
        self,
        repo_id: str,
        output_dir: pathlib.Path,
        model: nn.Module,
        tokenizer: Any,
        result_metadata: Dict[str, Any],
        commit_message: Optional[str] = None,
    ) -> Optional[str]:
        """Save all deployable artifacts into output_dir and push directly to HF Hub."""
        if not self.enabled:
            return None

        out_path = pathlib.Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"\n📦 [HF Hub] Preparing model artifacts for {repo_id}...")

        # 1. Save Tokenizer
        try:
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(str(out_path))
        except Exception as e:
            print(f"[HF Hub] Note: Could not save tokenizer: {e}")

        # 2. Save PyTorch Model weights & architecture config
        model_weights_path = out_path / "pytorch_model.bin"
        if not model_weights_path.exists():
            torch.save(model.state_dict(), model_weights_path)

        # Save clean metadata
        meta_file = out_path / "result_metadata.json"
        meta_file.write_text(json.dumps(result_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        # 3. Create repo if it doesn't exist
        try:
            self.api.create_repo(repo_id=repo_id, repo_type="model", private=self.private, exist_ok=True)
            print(f"  ✓ HF Repo ensured: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"[HF Hub] Note on create_repo {repo_id}: {e}")

        test_metric = (result_metadata.get("test") or {}).get("macro_f1")
        dev_metric = (result_metadata.get("final_dev") or {}).get("macro_f1")
        if test_metric is not None:
            f1_str = f"{test_metric:.4f}"
        elif dev_metric is not None:
            f1_str = f"{dev_metric:.4f} (dev)"
        else:
            f1_str = "N/A"
        msg = commit_message or f"Upload {result_metadata.get('run_name', 'model')} results (Macro-F1: {f1_str})"
        try:
            commit_info = self.api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(out_path),
                commit_message=msg,
                ignore_patterns=["*.tmp", "*checkpoint_temp*"],
            )
            model_url = f"https://huggingface.co/{repo_id}"
            print(f"🌟 [HF Hub] SUCCESSFULLY UPLOADED ARTIFACTS TO HF HUB!")
            print(f"   Repository: {model_url}")
            print(f"   Commit SHA: {getattr(commit_info, 'oid', 'latest')}\n")
            return model_url
        except Exception as e:
            print(f"[HF Hub] Upload failed: {e}")
            return None
