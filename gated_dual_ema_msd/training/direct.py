"""Direct, single-stage trainer for the baseline/ablation experiment matrix.

The trainer selects checkpoints exclusively by validation Macro-F1, evaluates
current and EMA weights separately and preserves FP32 evaluation. Test access
requires an explicit frozen-final or clearly labelled exploratory CLI mode.
"""
from __future__ import annotations

from gated_dual_ema_msd.training.precision import bf16_enabled

import json
import math
import os
import pathlib
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from gated_dual_ema_msd.experiment_registry import DATASET_MAX_LENGTHS
from gated_dual_ema_msd.models.flat_cafebert import FlatCafeBERT
from gated_dual_ema_msd.tracking import HFHubUploader, WandbTracker
from gated_dual_ema_msd.training.ema import ModelEMA
from gated_dual_ema_msd.training.ema_context import ema_weights
from gated_dual_ema_msd.training.optimizer_step import optimizer_update
from gated_dual_ema_msd.training.selection import DevSelection

LABEL_MAP = {
    0: 0,
    1: 1,
    2: 2,
    "E": 0,
    "C": 1,
    "N": 2,
    "entailment": 0,
    "contradiction": 1,
    "neutral": 2,
    "e": 0,
    "c": 1,
    "n": 2,
}
LABELS = ["E", "C", "N"]


def _label_id(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    return LABEL_MAP.get(str(value), LABEL_MAP.get(str(value).lower(), 0))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ModelEMA is imported from gated_dual_ema_msd.training.ema


class RawDictDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer: Any, max_len: int = 512):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        enc = self.tokenizer(
            str(row["premise"]),
            str(row["hypothesis"]),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in enc.items()}
        item["labels"] = torch.tensor(_label_id(row["label"]), dtype=torch.long)
        return item


def make_loader(
    rows: List[dict],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        RawDictDataset(rows, tokenizer, max_len=max_length),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def collate_eval_rows(rows: Sequence[dict], tokenizer: Any, max_length: int) -> dict:
    encodings = [
        tokenizer(
            str(row["premise"]),
            str(row["hypothesis"]),
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        for row in rows
    ]
    enc = tokenizer.pad(encodings, padding="longest", return_tensors="pt")
    batch = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor([_label_id(row["label"]) for row in rows], dtype=torch.long),
        "sample_id": [str(row.get("id", f"sample_{idx}")) for idx, row in enumerate(rows)],
        "premise": [str(row["premise"]) for row in rows],
        "hypothesis": [str(row["hypothesis"]) for row in rows],
    }
    if "token_type_ids" in enc:
        batch["token_type_ids"] = enc["token_type_ids"]
    return batch


def eval_batches(rows: Sequence[dict], batch_size: int):
    for index in range(0, len(rows), batch_size):
        yield rows[index : index + batch_size]


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if len(labels) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            **{
                f"{metric}_{label}": 0.0
                for metric in ("precision", "recall", "f1")
                for label in LABELS
            },
        }
    accuracy = accuracy_score(labels, preds)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", labels=[0, 1, 2], zero_division=0
    )
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "precision": float(precision),
        "recall": float(recall),
    }
    for index, label in enumerate(LABELS):
        metrics[f"precision_{label}"] = float(p_cls[index])
        metrics[f"recall_{label}"] = float(r_cls[index])
        metrics[f"f1_{label}"] = float(f1_cls[index])
    return metrics


def _unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    rows: Sequence[dict],
    tokenizer: Any,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 16,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """FP32 evaluation with loss, probabilities, and optional gate diagnostics."""
    was_training = model.training
    model.eval()
    all_logits: List[np.ndarray] = []
    all_gold: List[int] = []
    all_ids: List[str] = []
    all_premises: List[str] = []
    all_hypotheses: List[str] = []
    all_gate_means: List[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0
    try:
        for batch_rows in eval_batches(rows, batch_size):
            batch = collate_eval_rows(batch_rows, tokenizer, max_length)
            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key in ("input_ids", "attention_mask", "token_type_ids", "labels")
            }
            with torch.autocast(device_type=device.type, enabled=False):
                output = model(**model_inputs)
            logits = output["logits"].float()
            all_logits.append(logits.cpu().numpy())
            gold = batch["labels"].numpy().tolist()
            all_gold.extend(gold)
            all_ids.extend(batch["sample_id"])
            all_premises.extend(batch["premise"])
            all_hypotheses.extend(batch["hypothesis"])
            if output.get("loss") is not None:
                total_loss += float(output["loss"].float().item()) * len(batch_rows)
                total_examples += len(batch_rows)

            gate = getattr(_unwrap_model(model), "last_gate", None)
            if isinstance(gate, torch.Tensor) and gate.shape[0] == len(batch_rows):
                all_gate_means.append(gate.float().mean(dim=-1).cpu().numpy())
            else:
                all_gate_means.append(np.full(len(batch_rows), np.nan, dtype=np.float32))
    finally:
        if was_training:
            model.train()

    if not all_logits:
        metrics = compute_metrics(np.array([], dtype=int), np.array([], dtype=int))
        metrics["loss"] = 0.0
        return metrics, pd.DataFrame()

    logits_np = np.concatenate(all_logits, axis=0)
    probabilities = torch.softmax(torch.from_numpy(logits_np), dim=-1).numpy()
    predictions = logits_np.argmax(axis=1)
    gold_np = np.asarray(all_gold, dtype=int)
    metrics = compute_metrics(predictions, gold_np)
    metrics["loss"] = total_loss / max(1, total_examples)
    frame = pd.DataFrame(
        {
            "sample_id": all_ids,
            "premise": all_premises,
            "hypothesis": all_hypotheses,
            "gold_label": [LABELS[index] for index in all_gold],
            "logit_E": logits_np[:, 0],
            "logit_C": logits_np[:, 1],
            "logit_N": logits_np[:, 2],
            "prob_E": probabilities[:, 0],
            "prob_C": probabilities[:, 1],
            "prob_N": probabilities[:, 2],
            "pred_label": [LABELS[index] for index in predictions],
            "gate_mean": np.concatenate(all_gate_means, axis=0),
        }
    )
    return metrics, frame


def parameter_counts(model: nn.Module) -> Dict[str, int]:
    raw_model = _unwrap_model(model)
    total = sum(param.numel() for param in raw_model.parameters())
    trainable = sum(param.numel() for param in raw_model.parameters() if param.requires_grad)
    backbone = getattr(raw_model, "backbone", None)
    backbone_total = sum(param.numel() for param in backbone.parameters()) if backbone else 0
    backbone_trainable = (
        sum(param.numel() for param in backbone.parameters() if param.requires_grad)
        if backbone
        else 0
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "backbone_parameters": int(backbone_total),
        "backbone_trainable_parameters": int(backbone_trainable),
        "additional_head_parameters": int(total - backbone_total),
        "additional_trainable_head_parameters": int(trainable - backbone_trainable),
    }


def msd_metadata(model: nn.Module) -> Dict[str, Any]:
    raw_model = _unwrap_model(model)
    layers = getattr(raw_model, "msd_layers", None)
    classifier = getattr(raw_model, "classifier", None)
    if layers is None and hasattr(raw_model, "head"):
        layers = getattr(raw_model.head, "msd_layers", None)
        classifier = getattr(raw_model.head, "classifier", classifier)
    enabled = layers is not None and len(layers) > 0
    return {
        "msd_enabled": bool(enabled),
        "msd_num_paths": len(layers) if enabled else 0,
        "msd_dropout_probabilities": [float(layer.p) for layer in layers] if enabled else [],
        # The implementation owns exactly one classifier object and reuses it
        # for every dropout path.
        "msd_shared_classifier": bool(enabled and classifier is not None),
    }


def gate_summary(frame: pd.DataFrame) -> Optional[Dict[str, float]]:
    if frame.empty or "gate_mean" not in frame or not frame["gate_mean"].notna().any():
        return None
    valid = frame[frame["gate_mean"].notna()].copy()
    summary: Dict[str, float] = {
        "overall_mean": float(valid["gate_mean"].mean()),
        "overall_std": float(valid["gate_mean"].std(ddof=0)),
    }
    for label in LABELS:
        values = valid.loc[valid["gold_label"] == label, "gate_mean"]
        if not values.empty:
            summary[f"mean_{label}"] = float(values.mean())
    correct = valid["gold_label"] == valid["pred_label"]
    if correct.any():
        summary["mean_correct"] = float(valid.loc[correct, "gate_mean"].mean())
    if (~correct).any():
        summary["mean_incorrect"] = float(valid.loc[~correct, "gate_mean"].mean())
    return summary


def save_confusion_matrix(frame: pd.DataFrame, path: pathlib.Path) -> List[List[int]]:
    matrix = confusion_matrix(
        [_label_id(value) for value in frame["gold_label"]],
        [_label_id(value) for value in frame["pred_label"]],
        labels=[0, 1, 2],
    )
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(path)
    return matrix.astype(int).tolist()


def _load_weights(path: pathlib.Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        return torch.load(path, map_location=device)


def build_gated_dual_model(
    model_name: str = "uitnlp/CafeBERT",
    revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
    relation_hidden: int = 128,
    gate_bias: float = -1.0,
    use_msd: bool = True,
    label_smoothing: float = 0.02,
    dropout: float = 0.1,
    sep_token_id: int = 2,
) -> FlatCafeBERT:
    return FlatCafeBERT(
        model_name=model_name,
        fallback_models=[],
        dropout=dropout,
        num_labels=3,
        label_smoothing=label_smoothing,
        revision=revision,
        pool_mode="gated_dual",
        segment_pooling="attentive",
        relation_hidden=relation_hidden,
        gate_bias=gate_bias,
        head_architecture="standard",
        relation_features_mode="standard",
        use_multi_sample_dropout=use_msd,
        msd_dropouts=[0.1, 0.2, 0.3, 0.4, 0.5],
        classifier_type="linear",
        classifier_hidden=256,
        sep_token_id=sep_token_id,
    )


class DirectTrainer:
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
        output_dir: pathlib.Path = pathlib.Path("outputs/direct_nli"),
        dataset: Optional[str] = None,
        experiment_id: Optional[str] = None,
        model_configuration: Optional[Dict[str, Any]] = None,
        max_length: int = 512,
        lr: float = 1.0e-5,
        weight_decay: float = 0.005,
        warmup_ratio: float = 0.06,
        physical_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_epochs: int = 7,
        eval_steps: int = 100,
        patience: int = 5,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        use_ema: bool = True,
        ema_decay: float = 0.992,
        ema_start_step: int = 100,
        bf16: bool = True,
        evaluate_test: bool = False,
        test_peak_exploratory: bool = False,
        use_wandb: bool = False,
        wandb_project: str = "gated-dual-ema-msd",
        wandb_entity: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
        use_hf: bool = False,
        hf_repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_private: bool = False,
    ):
        self.seed = int(seed)
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset.lower() if dataset else None
        self.experiment_id = experiment_id
        self.model_configuration = model_configuration or {}
        self.max_length = int(max_length)
        if self.dataset:
            expected = DATASET_MAX_LENGTHS[self.dataset]
            if self.max_length != expected:
                raise ValueError(
                    f"max_length contract violation for {self.dataset}: "
                    f"expected {expected}, got {self.max_length}"
                )
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_ratio = float(warmup_ratio)
        self.physical_batch_size = int(physical_batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.max_epochs = int(max_epochs)
        self.eval_steps = int(eval_steps)
        self.patience = int(patience)
        self.max_grad_norm = float(max_grad_norm)
        self.bf16 = bf16_enabled(self.device, requested=bf16)
        self.evaluate_test = bool(evaluate_test)
        self.test_peak_exploratory = bool(test_peak_exploratory)
        if self.test_peak_exploratory and not self.evaluate_test:
            raise ValueError("Exploratory test scans require explicit test access")
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema_start_step = int(ema_start_step)
        self.use_wandb = bool(use_wandb)
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_api_key = wandb_api_key
        self.use_hf = bool(use_hf)
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.hf_private = bool(hf_private)

    def _evaluate_pair(
        self,
        ema: Optional[ModelEMA],
        rows: Sequence[dict],
    ) -> Tuple[Dict[str, float], pd.DataFrame, Optional[Dict[str, float]], Optional[pd.DataFrame]]:
        current_metrics, current_frame = evaluate_model(
            self.model,
            rows,
            self.tokenizer,
            self.device,
            self.max_length,
            self.physical_batch_size * 2,
        )
        ema_metrics: Optional[Dict[str, float]] = None
        ema_frame: Optional[pd.DataFrame] = None
        if ema is not None and getattr(ema, "ready", False):
            with ema_weights(self.model, ema):
                ema_metrics, ema_frame = evaluate_model(
                    self.model,
                    rows,
                    self.tokenizer,
                    self.device,
                    self.max_length,
                    self.physical_batch_size * 2,
                )
        return current_metrics, current_frame, ema_metrics, ema_frame

    def train(
        self,
        train_rows: List[dict],
        dev_rows: List[dict],
        test_rows: Optional[List[dict]] = None,
        run_name: str = "direct_nli",
    ) -> Dict[str, Any]:
        if not train_rows or not dev_rows:
            raise ValueError("train_rows and dev_rows must both be non-empty")
        if self.evaluate_test and not test_rows:
            raise ValueError("evaluate_test=True requires a non-empty test split")
        if not self.evaluate_test and test_rows:
            raise ValueError(
                "Search passing non-empty test_rows with evaluate_test=False is forbidden; "
                "test split is locked."
            )
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required")

        set_seed(self.seed)
        if self.model is None:
            self.model = build_gated_dual_model(
                sep_token_id=getattr(self.tokenizer, "sep_token_id", 2) or 2
            )
        self.model = self.model.to(self.device)
        counts = parameter_counts(self.model)
        msd_info = msd_metadata(self.model)
        ema_tracker = ModelEMA(self.model, decay=self.ema_decay) if self.use_ema else None

        hparams: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "model_configuration": self.model_configuration,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "effective_batch_size": self.physical_batch_size * self.gradient_accumulation_steps,
            "physical_batch_size": self.physical_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_length": self.max_length,
            "max_epochs": self.max_epochs,
            "eval_steps": self.eval_steps,
            "patience": self.patience,
            "seed": self.seed,
            "bf16_train": self.bf16,
            "fp16_train": False,
            "train_precision": "bf16" if self.bf16 else "fp32",
            "fp32_eval": True,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "ema_start_step": self.ema_start_step,
            "test_access_enabled": self.evaluate_test,
            "test_peak_exploratory": self.test_peak_exploratory,
            "train_samples": len(train_rows),
            "dev_samples": len(dev_rows),
            "test_samples": len(test_rows or []),
            **counts,
            **msd_info,
        }
        wandb_tracker = WandbTracker(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=run_name,
            config=hparams,
            api_key=self.wandb_api_key,
            enabled=self.use_wandb,
        )
        hf_uploader = HFHubUploader(
            token=self.hf_token,
            private=self.hf_private,
            enabled=self.use_hf and bool(self.hf_repo_id),
        )

        train_loader = make_loader(
            train_rows,
            self.tokenizer,
            self.physical_batch_size,
            self.max_length,
            True,
            self.seed,
        )
        steps_per_epoch = math.ceil(len(train_loader) / self.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.max_epochs
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        optimizer_groups = [
            {
                "params": [
                    param
                    for name, param in self.model.named_parameters()
                    if param.requires_grad and not any(term in name for term in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in self.model.named_parameters()
                    if param.requires_grad and any(term in name for term in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_groups, lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.warmup_ratio),
            num_training_steps=total_steps,
        )
        scaler = torch.amp.GradScaler(self.device.type, enabled=False)

        best_path = self.output_dir / "best_model.pt"
        best_current_path = self.output_dir / "best_current_model.pt"
        peak_test_path = self.output_dir / "best_test_model.pt"
        peak_test_metrics = None
        peak_test_step = None
        test_curve = []
        test_evaluations = 0
        log_file = self.output_dir / "train.log"
        dev_selection = DevSelection(patience=self.patience, mode="max")
        optimizer_step = 0
        started = time.time()
        if self.device.type == "cuda":
            # This is emitted with the immutable run result so the matrix
            # supervisor can tune its five-process packing from real, not
            # guessed, VRAM use on the selected GPU.
            torch.cuda.reset_peak_memory_stats(self.device)
        optimizer.zero_grad(set_to_none=True)

        header = (
            f"[{run_name}] STARTED | dataset={self.dataset} seed={self.seed} "
            f"max_length={self.max_length} steps={total_steps} "
            f"train_precision={'bf16' if self.bf16 else 'fp32'} eval_precision=fp32 "
            f"EMA={self.use_ema} MSD={msd_info['msd_enabled']} "
            f"test_access={bool(test_rows)}"
        )
        print(header, flush=True)
        log_file.write_text(header + "\n", encoding="utf-8")
        wandb_tracker.log_step({"lifecycle": "training"}, step=0)

        stop_training = False
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for micro_index, raw_batch in enumerate(train_loader, start=1):
                batch = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in raw_batch.items()
                    if key in ("input_ids", "attention_mask", "token_type_ids", "labels")
                }
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.bf16,
                    dtype=torch.bfloat16,
                ):
                    output = self.model(**batch)
                    loss = output["loss"]
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at step {optimizer_step}: {loss}")
                loss_value = float(loss.detach().item())
                scaler.scale(loss / self.gradient_accumulation_steps).backward()
                is_update = (
                    micro_index % self.gradient_accumulation_steps == 0
                    or micro_index == len(train_loader)
                )
                if not is_update:
                    continue

                optimizer_stepped = optimizer_update(
                    self.model,
                    optimizer,
                    scaler,
                    scheduler,
                    max_grad_norm=self.max_grad_norm,
                )
                if optimizer_stepped:
                    optimizer_step += 1
                    if ema_tracker is not None and optimizer_step >= self.ema_start_step:
                        ema_tracker.update(self.model)

                if optimizer_step % 10 == 0:
                    wandb_tracker.log_step(
                        {
                            "train/loss": loss_value,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        },
                        step=optimizer_step,
                    )

                if optimizer_step % self.eval_steps != 0 and optimizer_step != total_steps:
                    continue

                current_metrics, _, ema_metrics, ema_frame = self._evaluate_pair(
                    ema_tracker, dev_rows
                )
                selected_metrics = ema_metrics or current_metrics
                selected_source = "ema" if ema_metrics is not None else "current"

                log_values: Dict[str, Any] = {
                    "dev/current_macro_f1": current_metrics["macro_f1"],
                    "dev/current_accuracy": current_metrics["accuracy"],
                    "dev/current_loss": current_metrics["loss"],
                    "dev/selected_macro_f1": selected_metrics["macro_f1"],
                    "epoch": epoch,
                }
                if ema_metrics is not None:
                    log_values.update(
                        {
                            "dev/ema_macro_f1": ema_metrics["macro_f1"],
                            "dev/ema_accuracy": ema_metrics["accuracy"],
                            "dev/ema_loss": ema_metrics["loss"],
                        }
                    )
                wandb_tracker.log_step(log_values, step=optimizer_step)

                # Dense evaluation may precede EMA's first update. Track current
                # dev diagnostics, but M3/M2 selection must use active EMA weights.
                if self.use_ema and ema_metrics is None:
                    print(f"[{run_name}] Step {optimizer_step}: dev diagnostic only; waiting for EMA at step {self.ema_start_step}", flush=True)
                    continue

                improved = dev_selection.observe(
                    selected_metrics["macro_f1"],
                    optimizer_step,
                    source=selected_source,
                    epoch=epoch,
                )
                if improved:
                    torch.save(self.model.state_dict(), best_current_path)
                    with ema_weights(self.model, ema_tracker if selected_source == "ema" else None):
                        torch.save(self.model.state_dict(), best_path)
                    status = "NEW DEV BEST"
                else:
                    status = f"no-imp ({dev_selection.no_improve}/{self.patience})"

                if self.test_peak_exploratory:
                    with ema_weights(self.model, ema_tracker if selected_source == "ema" else None):
                        scan_metrics, scan_frame = evaluate_model(
                            self.model, test_rows, self.tokenizer, self.device,
                            self.max_length, self.physical_batch_size * 2,
                        )
                        test_evaluations += 1
                        if peak_test_metrics is None or scan_metrics["macro_f1"] > peak_test_metrics["macro_f1"]:
                            peak_test_metrics = dict(scan_metrics)
                            peak_test_step = optimizer_step
                            torch.save(self.model.state_dict(), peak_test_path)
                            scan_frame.to_csv(self.output_dir / "test_predictions_peak.csv", index=False)
                    test_curve.append({"optimizer_step": optimizer_step, "epoch": epoch,
                                       "weight_source": selected_source, "test_macro_f1": scan_metrics["macro_f1"],
                                       "test_accuracy": scan_metrics["accuracy"], "dev_macro_f1": selected_metrics["macro_f1"]})
                    pd.DataFrame(test_curve).to_csv(self.output_dir / "test_curve.csv", index=False)
                    wandb_tracker.log_step({"exploratory/test_macro_f1": scan_metrics["macro_f1"],
                                            "exploratory/peak_test_macro_f1": peak_test_metrics["macro_f1"],
                                            "exploratory/peak_test_step": peak_test_step}, step=optimizer_step)
                    print(f"[{run_name}] EXPLORATORY TEST step={optimizer_step} f1={scan_metrics['macro_f1']:.4f} peak={peak_test_metrics['macro_f1']:.4f} at step={peak_test_step}", flush=True)

                log_line = (
                    f"[{run_name}] Ep {epoch}/{self.max_epochs} | Step {optimizer_step:4d}/{total_steps} | "
                    f"Dev F1: {selected_metrics['macro_f1']:.4f} ({status})"
                )
                print(log_line, flush=True)
                with log_file.open("a", encoding="utf-8") as handle:
                    handle.write(log_line + "\n")
                if dev_selection.should_stop:
                    stop_training = True
                    break
            if stop_training:
                break

        if not best_path.exists():
            torch.save(self.model.state_dict(), best_current_path)
            torch.save(self.model.state_dict(), best_path)

        # Evaluate the two checkpoints from the same dev-selected optimizer step.
        self.model.load_state_dict(_load_weights(best_current_path, self.device), strict=True)
        final_dev_current, dev_current_frame = evaluate_model(
            self.model,
            dev_rows,
            self.tokenizer,
            self.device,
            self.max_length,
            self.physical_batch_size * 2,
        )
        dev_current_frame.to_csv(self.output_dir / "dev_predictions_current.csv", index=False)

        self.model.load_state_dict(_load_weights(best_path, self.device), strict=True)
        final_dev_selected, dev_selected_frame = evaluate_model(
            self.model,
            dev_rows,
            self.tokenizer,
            self.device,
            self.max_length,
            self.physical_batch_size * 2,
        )
        dev_selected_frame.to_csv(self.output_dir / "dev_predictions.csv", index=False)
        dev_confusion = save_confusion_matrix(
            dev_selected_frame, self.output_dir / "dev_confusion_matrix.csv"
        )

        test_current: Optional[Dict[str, float]] = None
        test_selected: Optional[Dict[str, float]] = None
        test_selected_frame = pd.DataFrame()
        if self.evaluate_test and test_rows:
            self.model.load_state_dict(_load_weights(best_path, self.device), strict=True)
            test_selected, test_selected_frame = evaluate_model(
                self.model,
                test_rows,
                self.tokenizer,
                self.device,
                self.max_length,
                self.physical_batch_size * 2,
            )
            test_selected_frame.to_csv(
                self.output_dir / "test_predictions.csv", index=False
            )
            test_evaluations += 1
        test_confusion = (
            save_confusion_matrix(
                test_selected_frame, self.output_dir / "test_confusion_matrix.csv"
            )
            if not test_selected_frame.empty
            else None
        )

        result: Dict[str, Any] = {
            "run_name": run_name,
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "seed": self.seed,
            "model_name": getattr(self.model, "model_name_used", self.model_configuration.get("model_name", "uitnlp/CafeBERT")),
            "best_dev_macro_f1": float(dev_selection.best_metric if dev_selection.best_metric > float("-inf") else -1.0),
            "best_dev_step": int(dev_selection.best_step),
            "best_epoch": int(dev_selection.best_epoch),
            "selected_weight_source": dev_selection.weight_source,
            "final_dev": final_dev_selected,
            "final_dev_current": final_dev_current,
            "final_dev_ema": final_dev_selected if dev_selection.weight_source == "ema" else None,
            "test": test_selected,
            "test_current": None,
            "test_ema": test_selected if dev_selection.weight_source == "ema" else None,
            "peak_test_macro_f1": peak_test_metrics["macro_f1"] if peak_test_metrics else None,
            "peak_test_step": peak_test_step,
            "peak_test_checkpoint": str(peak_test_path) if peak_test_metrics else None,
            "peak_test_metrics": peak_test_metrics,
            "test_peak_exploratory": self.test_peak_exploratory,
            "test_selection_policy": "exploratory_test_macro_f1" if self.test_peak_exploratory else None,
            "selection_policy": "dev_macro_f1",
            "test_evaluations": test_evaluations,
            "schema_version": 2,
            "dev_confusion_matrix": dev_confusion,
            "test_confusion_matrix": test_confusion,
            "dev_gate_statistics": gate_summary(dev_selected_frame),
            "test_gate_statistics": gate_summary(test_selected_frame),
            "elapsed_seconds": time.time() - started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else None
            ),
            "checkpoint": str(best_path),
            "current_checkpoint": str(best_current_path),
            "hparams": hparams,
            "wandb_run_path": (
                wandb_tracker.run.path
                if wandb_tracker.run is not None else None
            ),
        }
        result_path = self.output_dir / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        final_log: Dict[str, Any] = {
            "dev/final_selected_macro_f1": final_dev_selected["macro_f1"],
            "dev/final_current_macro_f1": final_dev_current["macro_f1"],
            "best_dev_macro_f1": float(dev_selection.best_metric if dev_selection.best_metric > float("-inf") else -1.0),
            "lifecycle": "artifact_upload" if self.use_hf else "complete",
        }
        if test_selected is not None:
            final_log.update(
                {
                    "test/selected_macro_f1": test_selected["macro_f1"],
                    "test/selected_accuracy": test_selected["accuracy"],
                }
            )
            y_true = [_label_id(value) for value in test_selected_frame["gold_label"]]
            y_pred = [_label_id(value) for value in test_selected_frame["pred_label"]]
            wandb_tracker.log_confusion_matrix(
                y_true, y_pred, title=f"Test Confusion Matrix ({run_name})"
            )
        wandb_tracker.log_step(final_log)

        if self.use_hf and self.hf_repo_id:
            hf_url = hf_uploader.upload_artifacts(
                repo_id=self.hf_repo_id,
                output_dir=self.output_dir,
                model=self.model,
                tokenizer=self.tokenizer,
                result_metadata=result,
            )
            if hf_url:
                result["hf_hub_url"] = hf_url
                wandb_tracker.log_hf_summary(self.hf_repo_id, commit_url=hf_url)
                result_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        wandb_tracker.finish()

        test_text = (
            f" test_f1={test_selected['macro_f1']:.4f}" if test_selected else " test=LOCKED"
        )
        print(
            f"[{run_name}] FINISHED dev_f1={final_dev_selected['macro_f1']:.4f}{test_text}",
            flush=True,
        )
        return result
