"""Offline contracts for the isolated ViNLI architecture candidates."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from gated_dual_ema_msd.config.experiments import (
    EXPERIMENTS,
    MAIN_SEEDS,
    get_experiment,
    model_kwargs_for_experiment,
)
from gated_dual_ema_msd.models import create_nli_model
from gated_dual_ema_msd.models.flat_cafebert import FlatCafeBERT
from gated_dual_ema_msd.training.direct import parameter_counts


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=1024)
        self.embeddings = nn.Embedding(64, 1024)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        hidden = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=hidden)


class ArchitectureSearchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_ids = torch.tensor([
            [0, 10, 11, 2, 2, 12, 13, 2, 1, 1],
            [0, 14, 15, 2, 2, 16, 17, 2, 1, 1],
        ])
        self.mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2)

    def make_model(self, experiment_id):
        self.assertIn(experiment_id, EXPERIMENTS, "architecture must be registered")
        recipe = get_experiment(experiment_id)
        with patch("transformers.AutoModel.from_pretrained", return_value=DummyBackbone()):
            return create_nli_model(recipe.model_type, **model_kwargs_for_experiment(recipe, 2))

    def test_registered_candidates_construct_the_planned_head_capacity(self):
        counts = {
            "ARCH_REL256": 3417347,
            "ARCH_CONDPOOL128": 3284099,
            "ARCH_ALIGN256": 3677699,
            "ARCH_REL230": 3284201,
            "ARCH_REL304": 3663155,
            "ARCH_REL512": 4728323,
        }
        for experiment_id, expected_count in counts.items():
            with self.subTest(experiment=experiment_id):
                model = self.make_model(experiment_id)
                recipe = get_experiment(experiment_id)
                self.assertTrue(recipe.use_msd and recipe.use_ema)
                self.assertEqual(recipe.seeds, MAIN_SEEDS)
                self.assertEqual(parameter_counts(model)["additional_trainable_head_parameters"], expected_count)
                self.assertEqual(model.classifier.out_features, 3)

    def test_candidates_have_finite_logits_gradients_and_shared_msd_classifier(self):
        for experiment_id in ("ARCH_REL256", "ARCH_CONDPOOL128", "ARCH_ALIGN256"):
            with self.subTest(experiment=experiment_id):
                model = self.make_model(experiment_id)
                path_logits = []
                handle = model.classifier.register_forward_hook(
                    lambda module, inputs, output: path_logits.append(output)
                )
                model.train()
                result = model(self.input_ids, self.mask, labels=torch.tensor([0, 2]))
                self.assertEqual(result["logits"].shape, (2, 3))
                self.assertEqual(len(path_logits), 5)
                torch.testing.assert_close(result["logits"], torch.stack(path_logits).mean(0))
                self.assertTrue(torch.isfinite(result["loss"]))
                self.assertTrue(((result["logits"].argmax(-1) >= 0) & (result["logits"].argmax(-1) < 3)).all())
                result["loss"].backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                path_logits.clear()
                model.eval()
                with torch.no_grad():
                    first = model(self.input_ids, self.mask)["logits"]
                    second = model(self.input_ids, self.mask)["logits"]
                self.assertEqual(len(path_logits), 2)
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                handle.remove()

    def test_conditioned_attention_ignores_special_and_padded_tokens(self):
        model = self.make_model("ARCH_CONDPOOL128").eval()
        hidden = torch.randn(2, 10, 1024)
        premise_mask, hypothesis_mask = model._segment_masks(self.input_ids, self.mask)
        excluded = ~(premise_mask | hypothesis_mask)
        excluded[:, 0] = False  # CLS is intentionally the global fusion input.
        changed = hidden.clone()
        changed[excluded] = torch.randn_like(changed[excluded]) * 1e5
        with torch.no_grad():
            first = model._pool_gated_dual(hidden, self.input_ids, self.mask)
            p_weights = model.last_premise_attention.clone()
            h_weights = model.last_hypothesis_attention.clone()
            second = model._pool_gated_dual(changed, self.input_ids, self.mask)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for weights, valid in ((p_weights, premise_mask), (h_weights, hypothesis_mask)):
            self.assertTrue(weights[~valid].eq(0).all())
            torch.testing.assert_close(weights.sum(1), torch.ones(2))
            self.assertEqual(weights.dtype, torch.float32)

    def test_conditioned_pool_changes_with_opposite_summary(self):
        model = self.make_model("ARCH_CONDPOOL128").eval()
        self.assertIsNot(model.premise_query, model.hypothesis_query)
        self.assertIsNot(model.premise_key, model.hypothesis_key)
        for projection in (model.premise_query, model.hypothesis_query, model.premise_key, model.hypothesis_key):
            self.assertIsNone(projection.bias)
        hidden = torch.zeros(1, 2, 1024)
        hidden[0, :, 0] = torch.tensor([-1.0, 1.0])
        valid = torch.ones(1, 2, dtype=torch.bool)
        with torch.no_grad():
            model.premise_key.weight.zero_()
            model.hypothesis_query.weight.zero_()
            model.premise_key.weight[0, 0] = 128 ** 0.5
            model.hypothesis_query.weight[0, 0] = 1
        summary = torch.zeros(1, 1024)
        summary[0, 0] = 1
        positive, positive_weights = model._conditioned_attention_pool(
            hidden, valid, summary, model.premise_key, model.hypothesis_query
        )
        negative, negative_weights = model._conditioned_attention_pool(
            hidden, valid, -summary, model.premise_key, model.hypothesis_query
        )
        self.assertGreater(positive[0, 0].item(), 0.7)
        self.assertLess(negative[0, 0].item(), -0.7)
        torch.testing.assert_close(positive_weights, negative_weights.flip(1))
        # The complete pooling branch must actually use the opposite segment,
        # rather than passing its own summary into an otherwise correct helper.
        pair_hidden = torch.zeros(1, 8, 1024)
        pair_hidden[0, 1:3, 0] = torch.tensor([-1.0, 1.0])
        pair_hidden[0, 5:7, 0] = 1
        pair_ids = self.input_ids[:1, :8]
        pair_mask = self.mask[:1, :8]
        with torch.no_grad():
            model._pool_gated_dual(pair_hidden, pair_ids, pair_mask)
            actual_positive = model.last_premise_attention[:, 1:3].clone()
            pair_hidden[0, 5:7, 0] = -1
            model._pool_gated_dual(pair_hidden, pair_ids, pair_mask)
            actual_negative = model.last_premise_attention[:, 1:3].clone()
        torch.testing.assert_close(actual_positive, positive_weights)
        torch.testing.assert_close(actual_negative, negative_weights)

    def test_conditioned_pool_is_fp32_and_empty_segment_is_zero_under_autocast(self):
        model = self.make_model("ARCH_CONDPOOL128").eval()
        hidden = torch.randn(2, 3, 1024)
        valid = torch.tensor([[False, False, False], [True, True, False]])
        summary = torch.randn(2, 1024)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            pooled, weights = model._conditioned_attention_pool(
                hidden, valid, summary, model.premise_key, model.hypothesis_query
            )
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(pooled.dtype, torch.float32)
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(pooled[0].eq(0).all())
        self.assertTrue(weights[0].eq(0).all())
        with torch.no_grad():
            empty_ids = torch.tensor([[0, 2, 2, 2, 1]])
            result = model(empty_ids, torch.tensor([[1, 1, 1, 1, 0]]))
        self.assertTrue(torch.isfinite(result["logits"]).all())

    def test_alignment_wrapper_matches_the_existing_token_alignment_path_exactly(self):
        wrapped = self.make_model("ARCH_ALIGN256").eval()
        with patch("transformers.AutoModel.from_pretrained", return_value=DummyBackbone()):
            reference = FlatCafeBERT(
                "uitnlp/CafeBERT", revision="af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
                pool_mode="token_align", alignment_dim=256, gate_bias=-1.0,
                dropout=0.1, label_smoothing=0.02, sep_token_id=2,
                use_multi_sample_dropout=True, msd_dropouts=[0.1, 0.2, 0.3, 0.4, 0.5],
            ).eval()
        reference.load_state_dict(wrapped.state_dict(), strict=True)
        with torch.no_grad():
            actual = wrapped(self.input_ids, self.mask)["logits"]
            expected = reference(self.input_ids, self.mask)["logits"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(wrapped.relation_projection[0].in_features, 1024)
        self.assertEqual(wrapped.relation_projection[0].out_features, 1024)


if __name__ == "__main__":
    unittest.main()
