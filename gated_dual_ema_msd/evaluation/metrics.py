"""NLI Evaluation Metrics & Diagnostic Reporting."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from gated_dual_ema_msd.config.contracts import LABELS, LABEL2ID


def compute_metrics(
    predictions: Union[np.ndarray, Sequence[int]],
    references: Union[np.ndarray, Sequence[int]],
) -> Dict[str, float]:
    """Compute Macro-F1, Accuracy, and per-class metrics."""
    preds = np.asarray(predictions)
    refs = np.asarray(references)

    acc = float(accuracy_score(refs, preds))
    p, r, f1, s = precision_recall_fscore_support(
        refs, preds, labels=[0, 1, 2], average=None, zero_division=0
    )
    macro_f1 = float(np.mean(f1))

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_entailment": float(f1[0]),
        "f1_contradiction": float(f1[1]),
        "f1_neutral": float(f1[2]),
        "f1_E": float(f1[0]),
        "f1_C": float(f1[1]),
        "f1_N": float(f1[2]),
        "precision_entailment": float(p[0]),
        "precision_contradiction": float(p[1]),
        "precision_neutral": float(p[2]),
        "recall_entailment": float(r[0]),
        "recall_contradiction": float(r[1]),
        "recall_neutral": float(r[2]),
    }
    return metrics



def compute_confusion_matrix(
    predictions: Union[np.ndarray, Sequence[int]],
    references: Union[np.ndarray, Sequence[int]],
) -> np.ndarray:
    """Compute standard 3x3 confusion matrix for E, C, N."""
    return confusion_matrix(references, predictions, labels=[0, 1, 2])


def diagnostic_report(
    predictions: Union[np.ndarray, Sequence[int]],
    references: Union[np.ndarray, Sequence[int]],
    target_names: Optional[Sequence[str]] = ("entailment", "contradiction", "neutral"),
) -> str:
    """Generate scikit-learn classification report for diagnostics."""
    from sklearn.metrics import classification_report

    preds = np.asarray(predictions)
    refs = np.asarray(references)
    return classification_report(
        refs,
        preds,
        labels=[0, 1, 2],
        target_names=list(target_names) if target_names else None,
        zero_division=0,
    )
