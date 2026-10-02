"""Check M3 against paper dimensions, parameter count and averaging equations."""
from types import SimpleNamespace
import unittest
import tempfile
import pathlib
from unittest.mock import patch

import torch
from torch import nn

from gated_dual_ema_msd.config.experiments import get_experiment, model_kwargs_for_experiment
from gated_dual_ema_msd.models import create_nli_model
from gated_dual_ema_msd.training.direct import parameter_counts
from gated_dual_ema_msd.training.ema import ModelEMA


class PaperRecipeTests(unittest.TestCase):
    def test_m3_has_paper_head_dimensions_and_2761859_parameters(self):
        backbone = nn.Module()
        backbone.config = SimpleNamespace(hidden_size=1024)
        backbone.register_parameter("dummy", nn.Parameter(torch.zeros(1)))
        recipe = get_experiment("M3_FULL")
        with patch("transformers.AutoModel.from_pretrained", return_value=backbone):
            model = create_nli_model(recipe.model_type, **model_kwargs_for_experiment(recipe, 2))
        self.assertTrue(recipe.use_msd and recipe.use_ema)
        self.assertEqual(parameter_counts(model)["additional_trainable_head_parameters"], 2761859)
        self.assertEqual([(m.in_features, m.out_features) for m in model.relation_projection if isinstance(m, nn.Linear)], [(4096, 128), (128, 1024)])
        self.assertEqual([m.p for m in model.msd_layers], [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertIsNone(model.premise_attention.bias)
        self.assertIsNone(model.hypothesis_attention.bias)
        self.assertEqual(model.label_smoothing, 0.02)
        model.eval()
        current = torch.randn(2, 1024)
        relation = torch.randn(2, 1024)
        expected_gate = torch.sigmoid(torch.nn.functional.linear(torch.cat([current, relation], dim=-1), model.fusion_gate.weight, model.fusion_gate.bias))
        expected_relation = torch.nn.functional.layer_norm(relation, (1024,), model.fusion_norm.weight, model.fusion_norm.bias, model.fusion_norm.eps)
        torch.testing.assert_close(model._residual_fuse(current, relation), current + expected_gate * expected_relation)

    def test_ema_initializes_at_current_weights_and_uses_paper_decay(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        ema = ModelEMA(model, decay=0.992)
        self.assertFalse(ema.ready)
        with torch.no_grad():
            model.weight.fill_(3)
        torch.testing.assert_close(ema.shadow["weight"], torch.ones(1, 1))
        ema.update(model)
        torch.testing.assert_close(ema.shadow["weight"], torch.tensor([[0.992 + 0.008 * 3]]))
        self.assertTrue(ema.ready)

    def test_m3_selection_waits_until_ema_is_active(self):
        from gated_dual_ema_msd.training import direct
        from verify_source import TinyBackbone, TinyTokenizer, rows
        recipe = get_experiment("M3_FULL")
        with patch("transformers.AutoModel.from_pretrained", return_value=TinyBackbone()):
            model = create_nli_model(recipe.model_type, **model_kwargs_for_experiment(recipe, 2))
        actual_evaluate = direct.evaluate_model
        evaluation_count = 0
        def evaluate(*args, **kwargs):
            nonlocal evaluation_count
            evaluation_count += 1
            metrics, frame = actual_evaluate(*args, **kwargs)
            metrics["macro_f1"] = 0.99 if evaluation_count == 1 else 0.5
            return metrics, frame
        with tempfile.TemporaryDirectory() as directory, patch.object(direct, "evaluate_model", side_effect=evaluate):
            trainer = direct.DirectTrainer(model=model, tokenizer=TinyTokenizer(), device=torch.device("cpu"),
                                          output_dir=pathlib.Path(directory), dataset="vianli", experiment_id="M3_FULL",
                                          max_epochs=1, physical_batch_size=4, gradient_accumulation_steps=1,
                                          eval_steps=1, patience=5, use_ema=True, ema_start_step=2,
                                          use_wandb=False, evaluate_test=False)
            result = trainer.train(rows("train", 8), rows("dev", 3))
        self.assertEqual(result["selected_weight_source"], "ema")
        self.assertEqual(result["best_dev_step"], 2)


if __name__ == "__main__":
    unittest.main()
