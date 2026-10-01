"""Verify the extracted package without network, GPU or benchmark data."""
from __future__ import annotations

import ast
import importlib
import pathlib
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parent


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, num_hidden_layers=1)
        self.embedding = nn.Embedding(64, 8)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                inputs_embeds=None, output_hidden_states=False):
        hidden = inputs_embeds if inputs_embeds is not None else self.embedding(input_ids)
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=(hidden, hidden))

    def get_input_embeddings(self):
        return self.embedding


class TinyTokenizer:
    sep_token_id = 2

    def __call__(self, premise, hypothesis, truncation=True, max_length=512,
                 padding=False, return_tensors=None, **kwargs):
        assert max_length == 512, "This verification fixture represents ViANLI"
        ids = [0, 3, 2, 2, 4, 2]
        mask = [1] * len(ids)
        if padding == "max_length":
            ids += [1] * (max_length - len(ids))
            mask += [0] * (max_length - len(mask))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}
        return {"input_ids": ids, "attention_mask": mask}

    def pad(self, records, padding="longest", return_tensors="pt", **kwargs):
        length = max(len(row["input_ids"]) for row in records)
        return {
            "input_ids": torch.tensor([row["input_ids"] + [1] * (length-len(row["input_ids"])) for row in records]),
            "attention_mask": torch.tensor([row["attention_mask"] + [0] * (length-len(row["attention_mask"])) for row in records]),
        }


def rows(prefix, count):
    return [dict(id=f"{prefix}-{i}", premise="P", hypothesis="H",
                 label=("E", "C", "N")[i % 3], domain="vianli") for i in range(count)]


def main():
    for path in (ROOT / "gated_dual_ema_msd").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            assert not any(name == "src" or name.startswith("src.") or name.startswith("scripts.") for name in names), path
    for command in ("train", "matrix", "evaluate", "r2"):
        importlib.import_module(f"gated_dual_ema_msd.cli.{command}")
    for module, attribute in (
        ("training.r2_runtime", "ROOT"),
        ("training.train_exp001", "ROOT"),
        ("config.r2_validation", "ROOT"),
        ("utils.env", "_ROOT"),
    ):
        assert getattr(importlib.import_module(f"gated_dual_ema_msd.{module}"), attribute) == ROOT

    from gated_dual_ema_msd.cli.train import build_parser
    from gated_dual_ema_msd.config.contracts import DATASET_MAX_LENGTHS
    from gated_dual_ema_msd.config.experiments import EXPERIMENTS, model_kwargs_for_experiment
    from gated_dual_ema_msd.config.test_policy import allow_final_test
    from gated_dual_ema_msd.models import create_nli_model
    from gated_dual_ema_msd.training.direct import DirectTrainer

    assert dict(DATASET_MAX_LENGTHS) == {"vinli":512, "vianli":512, "vimednli":256}
    assert not build_parser().parse_args([]).frozen_final
    assert not allow_final_test(frozen_final=True, no_test=True)

    ids = torch.tensor([[0,3,2,2,4,2], [0,5,2,2,6,2]])
    labels = torch.tensor([0,1])
    with patch("transformers.AutoModel.from_pretrained", side_effect=lambda *a, **k: TinyBackbone()):
        for experiment in EXPERIMENTS.values():
            torch.manual_seed(42)
            model = create_nli_model(experiment.model_type,
                **model_kwargs_for_experiment(experiment, sep_token_id=2))
            output = model(ids, torch.ones_like(ids), labels=labels)
            assert tuple(output["logits"].shape) == (2,3)
            assert torch.isfinite(output["loss"])
            output["loss"].backward()
            model.eval()
            with torch.no_grad():
                assert torch.isfinite(model(ids, torch.ones_like(ids))["logits"]).all()

        with tempfile.TemporaryDirectory(prefix="main-source-check-") as temporary:
            torch.manual_seed(42)
            experiment = EXPERIMENTS["M3_FULL"]
            model = create_nli_model(experiment.model_type,
                **model_kwargs_for_experiment(experiment, sep_token_id=2))
            trainer = DirectTrainer(model=model, tokenizer=TinyTokenizer(),
                device=torch.device("cpu"), output_dir=pathlib.Path(temporary),
                dataset="vianli", max_length=512, max_epochs=1,
                physical_batch_size=2, gradient_accumulation_steps=2,
                eval_steps=1, patience=3, bf16=False, use_ema=True,
                ema_start_step=1, evaluate_test=False, use_wandb=False, use_hf=False)
            result = trainer.train(rows("train",8), rows("dev",3), run_name="synthetic-source-check")
            assert result["test"] is None
            assert result["test_evaluations"] == 0
            assert result["selection_policy"] == "dev_macro_f1"
            assert (pathlib.Path(temporary) / "best_model.pt").is_file()
    print("PASS: syntax, isolated CLI imports, contracts, 11 architectures and synthetic CPU training/checkpoint reload")


if __name__ == "__main__":
    main()
