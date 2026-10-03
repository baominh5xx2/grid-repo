"""CPU-only evidence checks for dev-selected architecture screening."""
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from gated_dual_ema_msd.training import direct
from verify_source import TinyTokenizer, rows


class DiagnosticModel(nn.Module):
    """One tiny shared classifier; synthetic diagnostics depend on row labels."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(1, 3)

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.classifier(torch.ones(input_ids.shape[0], 1))
        index = labels.float() if labels is not None else torch.zeros(len(input_ids))
        self.last_gate = torch.stack([0.1 + index * 0.1, 0.3 + index * 0.1], dim=-1)
        self.last_architecture_diagnostics = {
            "gate_channel_std": self.last_gate.std(dim=-1, correction=0),
            "gate_fraction_below_005": torch.zeros_like(index),
            "gate_fraction_above_095": torch.zeros_like(index),
            "relation_to_cls_norm_ratio": index + 1,
            "premise_attention_entropy": torch.full_like(index, 0.5),
            "hypothesis_attention_entropy": torch.full_like(index, 0.25),
        }
        return {"logits": logits, "loss": nn.functional.cross_entropy(logits, labels)}


class ArchitectureHistoryTests(unittest.TestCase):
    def trainer(self, directory, **kwargs):
        configuration = dict(
            model=DiagnosticModel(), tokenizer=TinyTokenizer(), device=torch.device("cpu"),
            output_dir=pathlib.Path(directory), dataset="vinli", max_length=512,
            max_epochs=1, physical_batch_size=4, gradient_accumulation_steps=1,
            eval_steps=1, patience=5, use_ema=True, ema_start_step=2,
            bf16=False, evaluate_test=False, use_wandb=False, use_hf=False,
        )
        configuration.update(kwargs)
        return direct.DirectTrainer(**configuration)

    def test_two_evaluations_record_pre_ema_and_both_active_sources(self):
        actual_evaluate = direct.evaluate_model
        evaluations = []

        def evaluate(*args, **kwargs):
            metric, frame = actual_evaluate(*args, **kwargs)
            evaluations.append(args[1])
            metric["macro_f1"] = (0.99, 0.6, 0.5)[min(len(evaluations) - 1, 2)]
            metric["loss"] = float(len(evaluations))
            return metric, frame

        with tempfile.TemporaryDirectory() as directory, patch.object(direct, "evaluate_model", side_effect=evaluate):
            trainer = self.trainer(directory)
            result = trainer.train(rows("train", 8), rows("dev", 3))
            history = pd.read_csv(pathlib.Path(directory) / "dev_history.csv")
            self.assertEqual(history["optimizer_step"].tolist(), [1, 2])
            self.assertEqual(history["epoch"].tolist(), [1, 1])
            self.assertEqual(history["selection_eligible"].tolist(), [False, True])
            self.assertEqual(history["is_best"].tolist(), [False, True])
            self.assertEqual(history["patience_counter"].tolist(), [0, 0])
            self.assertTrue(pd.isna(history.loc[0, "ema_macro_f1"]))
            self.assertTrue(pd.isna(history.loc[0, "selected_source"]))
            self.assertTrue(pd.isna(history.loc[0, "selected_macro_f1"]))
            self.assertEqual(history["current_macro_f1"].tolist(), [0.99, 0.6])
            self.assertEqual(history.loc[1, "ema_macro_f1"], 0.5)
            self.assertEqual(history.loc[1, "current_loss"], 2.0)
            self.assertEqual(history.loc[1, "ema_loss"], 3.0)
            self.assertEqual(history.loc[1, "selected_source"], "ema")
            self.assertEqual(int(history.loc[history.is_best, "optimizer_step"].iloc[-1]), result["best_dev_step"])
            self.assertEqual(result["best_dev_step"], 2)
            self.assertEqual(result["final_dev_current"]["macro_f1"], 0.6)
            self.assertEqual(result["final_dev"]["macro_f1"], 0.5)
            self.assertEqual(result["optimizer_steps"], 2)
            self.assertGreaterEqual(result["train_seconds_per_optimizer_step"], 0)
            self.assertGreaterEqual(result["evaluation_seconds"], 0)
            for column in ("learning_rate", "training_seconds", "evaluation_seconds", "elapsed_seconds"):
                self.assertTrue(np.isfinite(history[column]).all(), column)
            self.assertEqual(result["test_evaluations"], 0)
            self.assertEqual(len(evaluations), 3, "reuse selected dev outputs instead of repeating dev forwards")
            for filename in ("dev_predictions.csv", "dev_predictions_current.csv", "dev_confusion_matrix.csv", "architecture_diagnostics.json"):
                self.assertTrue((pathlib.Path(directory) / filename).is_file(), filename)
            evidence = json.loads((pathlib.Path(directory) / "architecture_diagnostics.json").read_text())
            self.assertEqual(evidence["optimizer_step"], result["best_dev_step"])
            self.assertEqual(evidence["selected_source"], "ema")
            self.assertEqual(evidence["selected"], result["dev_architecture_diagnostics"])

    def test_non_ema_run_selects_current_and_keeps_ema_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.trainer(directory, use_ema=False).train(rows("train", 4), rows("dev", 3))
            history = pd.read_csv(pathlib.Path(directory) / "dev_history.csv")
        self.assertEqual(result["selected_weight_source"], "current")
        self.assertEqual(history["selected_source"].tolist(), ["current"])
        self.assertTrue(history["selection_eligible"].all())
        self.assertTrue(history["ema_macro_f1"].isna().all())

    def test_ema_run_without_eligible_evaluation_cannot_publish_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory, ema_start_step=100)
            with self.assertRaisesRegex(RuntimeError, "eligible EMA"):
                trainer.train(rows("train", 4), rows("dev", 3))
            history = pd.read_csv(pathlib.Path(directory) / "dev_history.csv")
            self.assertFalse(history["selection_eligible"].any())
            self.assertFalse((pathlib.Path(directory) / "best_model.pt").exists())

    def test_non_ema_zero_update_fallback_evaluation_is_explicitly_unselected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.trainer(directory, use_ema=False, max_epochs=0).train(rows("train", 4), rows("dev", 3))
            history = pd.read_csv(pathlib.Path(directory) / "dev_history.csv")
            self.assertTrue((pathlib.Path(directory) / "best_model.pt").exists())
        self.assertEqual(history["optimizer_step"].tolist(), [0])
        self.assertFalse(history["selection_eligible"].any())
        self.assertFalse(history["is_best"].any())
        self.assertTrue(history["selected_source"].isna().all())
        self.assertEqual(result["best_dev_macro_f1"], -1.0)

    def test_patience_tracks_eligible_ema_evaluations_in_history(self):
        actual_evaluate = direct.evaluate_model
        calls = []

        def evaluate(*args, **kwargs):
            metric, frame = actual_evaluate(*args, **kwargs)
            calls.append(1)
            metric["macro_f1"] = [0.9, 0.5, 0.9, 0.5, 0.9, 0.4][min(len(calls) - 1, 5)]
            return metric, frame

        with tempfile.TemporaryDirectory() as directory, patch.object(direct, "evaluate_model", side_effect=evaluate):
            result = self.trainer(directory, ema_start_step=1, patience=2).train(rows("train", 16), rows("dev", 3))
            history = pd.read_csv(pathlib.Path(directory) / "dev_history.csv")
        self.assertEqual(history["optimizer_step"].tolist(), [1, 2, 3])
        self.assertEqual(history["patience_counter"].tolist(), [0, 1, 2])
        self.assertEqual(history["is_best"].tolist(), [True, False, False])
        self.assertEqual(result["best_dev_step"], 1)
        self.assertEqual(result["optimizer_steps"], 3)

    def test_training_step_timing_excludes_evaluation_time(self):
        clock = SimpleNamespace(value=0.0)

        def tick():
            value = clock.value
            clock.value += 1.0
            return value

        actual_evaluate = direct.evaluate_model

        def evaluate(*args, **kwargs):
            output = actual_evaluate(*args, **kwargs)
            clock.value += 100.0
            return output

        fake_time = SimpleNamespace(time=lambda: clock.value, perf_counter=tick)
        with tempfile.TemporaryDirectory() as directory, patch.object(direct, "time", fake_time), patch.object(direct, "evaluate_model", side_effect=evaluate):
            result = self.trainer(directory).train(rows("train", 8), rows("dev", 3))
        self.assertGreaterEqual(result["evaluation_seconds"], 300.0)
        self.assertLess(result["training_seconds"], 100.0)
        self.assertEqual(result["train_seconds_per_optimizer_step"], result["training_seconds"] / 2)

    def test_diagnostic_errors_restore_original_training_mode(self):
        class InvalidDiagnosticModel(DiagnosticModel):
            def forward(self, **kwargs):
                output = super().forward(**kwargs)
                self.last_architecture_diagnostics["gate_channel_std"] = torch.ones(1, 2)
                return output

        model = InvalidDiagnosticModel()
        model.train()
        with self.assertRaisesRegex(ValueError, "Tensor\\[3\\]"):
            direct.evaluate_model(model, rows("dev", 3), TinyTokenizer(), torch.device("cpu"))
        self.assertTrue(model.training)

    def test_diagnostics_reduce_samples_and_channels_separately(self):
        model = DiagnosticModel()
        model.train()
        _, frame = direct.evaluate_model(model, rows("dev", 3), TinyTokenizer(), torch.device("cpu"), batch_size=2)
        self.assertTrue(model.training)
        summary = direct.architecture_summary(frame)
        ratio = summary["statistics"]["relation_to_cls_norm_ratio"]
        self.assertEqual(ratio["count"], 3)
        self.assertAlmostEqual(ratio["mean"], 2.0)
        self.assertAlmostEqual(ratio["std"], np.std([1, 2, 3]))
        self.assertIn("sample", ratio["reduction_axes"])
        gate = summary["gate_channels"]
        self.assertEqual(gate["sample_count"], 3)
        self.assertEqual(gate["channel_count"], 2)
        np.testing.assert_allclose(gate["mean_across_samples"], [0.2, 0.4], atol=1e-7)
        np.testing.assert_allclose(gate["std_across_samples"], [np.std([0.1, 0.2, 0.3])] * 2, atol=1e-7)
        self.assertIn("normalized", summary["definitions"]["premise_attention_entropy"])
        self.assertEqual(frame["diag_relation_to_cls_norm_ratio"].tolist(), [1.0, 2.0, 3.0])

    def test_token_alignment_retains_direction_and_unavailable_pool_entropy(self):
        frame = pd.DataFrame({
            "diag_premise_alignment_entropy": [0.0, 0.4, 0.8],
            "diag_hypothesis_alignment_entropy": [0.1, 0.2, 0.3],
        })
        summary = direct.architecture_summary(frame)
        self.assertAlmostEqual(summary["statistics"]["premise_alignment_entropy"]["mean"], 0.4)
        self.assertAlmostEqual(summary["statistics"]["hypothesis_alignment_entropy"]["mean"], 0.2)
        self.assertNotIn("premise_attention_entropy", summary["statistics"])
        self.assertIn("valid query", summary["definitions"]["premise_alignment_entropy"])


if __name__ == "__main__":
    unittest.main()
