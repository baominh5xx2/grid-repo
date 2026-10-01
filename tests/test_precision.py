"""Precision regression checks without model downloads or benchmark data."""
import contextlib
import inspect
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from verify_source import TinyBackbone, TinyTokenizer, rows
from gated_dual_ema_msd.config.experiments import EXPERIMENTS, model_kwargs_for_experiment
from gated_dual_ema_msd.config.r2_validation import validate_config
from gated_dual_ema_msd.models import create_nli_model
from gated_dual_ema_msd.training.direct import DirectTrainer
from gated_dual_ema_msd.training.trainer import BaseTrainer

ROOT = pathlib.Path(__file__).resolve().parents[1]


class PrecisionTests(unittest.TestCase):
    def test_trainers_default_to_bf16(self):
        for trainer in (DirectTrainer, BaseTrainer):
            parameters = inspect.signature(trainer).parameters
            self.assertIn("bf16", parameters)
            self.assertIs(parameters["bf16"].default, True)

    def test_configs_and_r2_contract_require_bf16(self):
        for path in (ROOT / "configs").rglob("*.yaml"):
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertIs(cfg["training"]["bf16_train"], True)
            self.assertIs(cfg["training"]["fp16_train"], False)
            self.assertIs(cfg["training"]["fp32_eval"], True)
        cfg = yaml.safe_load((ROOT / "configs/experiments/exp001_r2_full_stage1_stage2_maxlen512_seed42.yaml").read_text())
        validate_config(cfg)
        cfg["training"]["bf16_train"] = False
        cfg["training"]["fp16_train"] = True
        with self.assertRaisesRegex(ValueError, "BF16"):
            validate_config(cfg)

    def test_gpu_capability_gate_and_cpu_fallback(self):
        from gated_dual_ema_msd.training.precision import bf16_enabled
        self.assertFalse(bf16_enabled(torch.device("cpu")))
        with patch("torch.cuda.device", return_value=contextlib.nullcontext()):
            with patch("torch.cuda.is_bf16_supported", return_value=True):
                self.assertTrue(bf16_enabled(torch.device("cuda:0")))
            with patch("torch.cuda.is_bf16_supported", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "BF16"):
                    bf16_enabled(torch.device("cuda:0"))
                self.assertFalse(bf16_enabled(torch.device("cuda:0"), requested=False))

    def test_all_architectures_bf16_forward_backward_and_fp32_eval(self):
        ids = torch.tensor([[0, 3, 2, 2, 4, 2], [0, 5, 2, 2, 6, 2]])
        with patch("transformers.AutoModel.from_pretrained", side_effect=lambda *a, **k: TinyBackbone()):
            for experiment in EXPERIMENTS.values():
                with self.subTest(experiment=experiment.experiment_id):
                    torch.manual_seed(42)
                    model = create_nli_model(experiment.model_type, **model_kwargs_for_experiment(experiment, sep_token_id=2))
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        output = model(ids, torch.ones_like(ids), labels=torch.tensor([0, 1]))
                    self.assertEqual(output["logits"].dtype, torch.bfloat16)
                    self.assertTrue(torch.isfinite(output["loss"]))
                    output["loss"].backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    model.eval()
                    with torch.no_grad(), torch.autocast("cpu", enabled=False):
                        self.assertEqual(model(ids, torch.ones_like(ids))["logits"].dtype, torch.float32)

    def test_direct_bf16_train_and_fp32_eval_with_disabled_scaler(self):
        observations, scalers = [], []
        original_scaler = torch.amp.GradScaler
        def create_scaler(*args, **kwargs):
            scaler = original_scaler(*args, **kwargs)
            scalers.append(scaler.is_enabled())
            return scaler
        with patch("transformers.AutoModel.from_pretrained", side_effect=lambda *a, **k: TinyBackbone()):
            experiment = EXPERIMENTS["M3_FULL"]
            torch.manual_seed(42)
            model = create_nli_model(experiment.model_type, **model_kwargs_for_experiment(experiment, sep_token_id=2))
            model.register_forward_hook(lambda model, args, output: observations.append((model.training, output["logits"].dtype)))
            with tempfile.TemporaryDirectory(prefix="bf16-check-") as directory:
                # Exercise actual BF16 ops on CPU; the production capability gate
                # enables BF16 on supported CUDA and leaves CPU runs in FP32.
                with patch("gated_dual_ema_msd.training.direct.bf16_enabled", return_value=True), patch("torch.amp.GradScaler", side_effect=create_scaler):
                    trainer = DirectTrainer(model=model, tokenizer=TinyTokenizer(), device=torch.device("cpu"), output_dir=pathlib.Path(directory), dataset="vianli", max_length=512, max_epochs=1, physical_batch_size=2, gradient_accumulation_steps=2, eval_steps=1, bf16=True, evaluate_test=False, use_wandb=False, use_hf=False)
                    result = trainer.train(rows("train", 8), rows("dev", 3), run_name="synthetic-bf16-check")
                self.assertEqual(result["test_evaluations"], 0)
                self.assertTrue((pathlib.Path(directory) / "best_model.pt").is_file())
        self.assertIn((True, torch.bfloat16), observations)
        self.assertIn((False, torch.float32), observations)
        self.assertEqual(scalers, [False])

    def test_base_trainer_bf16_optimizer_step(self):
        with patch("transformers.AutoModel.from_pretrained", side_effect=lambda *a, **k: TinyBackbone()):
            experiment = EXPERIMENTS["M3_FULL"]
            model = create_nli_model(experiment.model_type, **model_kwargs_for_experiment(experiment, sep_token_id=2))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            observations = []
            model.register_forward_hook(lambda model, args, output: observations.append(output["logits"].dtype))
            with patch("gated_dual_ema_msd.training.trainer.bf16_enabled", return_value=True):
                trainer = BaseTrainer(model, optimizer, data_module=None, device=torch.device("cpu"), grad_accum_steps=1)
            original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            ids = torch.tensor([[0, 3, 2, 2, 4, 2], [0, 5, 2, 2, 6, 2]])
            trainer.training_step({"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": torch.tensor([0, 1])}, 1)
            trainer.optimizer_step()
            self.assertEqual(observations, [torch.bfloat16])
            self.assertFalse(trainer.scaler.is_enabled())
            self.assertTrue(any(not torch.equal(original[name], parameter) for name, parameter in model.named_parameters()))

    def test_r2_stage_bf16_train_fp32_eval_and_checkpoint(self):
        from gated_dual_ema_msd.training.r2_stage import train_stage
        cfg = yaml.safe_load((ROOT / "configs/experiments/exp001_r2_full_stage1_stage2_maxlen512_seed42.yaml").read_text())
        cfg["debug"]["max_epochs_per_stage"] = 1
        cfg["debug"]["eval_optimizer_steps"] = 1
        with patch("transformers.AutoModel.from_pretrained", side_effect=lambda *a, **k: TinyBackbone()):
            experiment = EXPERIMENTS["M3_FULL"]
            model = create_nli_model(experiment.model_type, **model_kwargs_for_experiment(experiment, sep_token_id=2))
            observations = []
            model.register_forward_hook(lambda model, args, output: observations.append((model.training, output["logits"].dtype)))
            with tempfile.TemporaryDirectory(prefix="bf16-r2-check-") as directory:
                with patch("gated_dual_ema_msd.training.r2_stage.bf16_enabled", return_value=True):
                    result = train_stage(model, {"vianli": rows("train", 8)}, rows("dev", 3), TinyTokenizer(), cfg, "stage2", torch.device("cpu"), pathlib.Path(directory), run=None, debug=True)
                self.assertGreater(result.optimizer_steps, 0)
                self.assertTrue((pathlib.Path(directory) / "pytorch_model.bin").is_file())
        self.assertIn((True, torch.bfloat16), observations)
        self.assertIn((False, torch.float32), observations)


if __name__ == "__main__":
    unittest.main()
