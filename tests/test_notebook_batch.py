import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import hashlib
import os
import sys

from gated_dual_ema_msd.cli.matrix import build_jobs
from gated_dual_ema_msd.config.experiments import EXPERIMENTS

ROOT = pathlib.Path(__file__).resolve().parents[1]


class NotebookBatchTests(unittest.TestCase):
    def test_all_methods_allow_three_registered_seeds(self):
        for experiment in EXPERIMENTS.values():
            self.assertEqual(experiment.seeds, (42, 2024, 3407))

    def test_multiseed_cohort_has_99_unique_jobs(self):
        jobs = build_jobs("multiseed", ["vinli", "vianli", "vimednli"])
        self.assertEqual(len(jobs), 99)
        self.assertEqual(len({(j.dataset, j.experiment_id, j.seed) for j in jobs}), 99)
        self.assertTrue(all(j.seed == 42 for j in jobs[:33]))

    def test_resume_rejects_changed_protocol(self):
        from gated_dual_ema_msd.operations.notebook_batch import bind_manifest
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            manifest = {"git_sha": "a" * 40, "seeds": [42, 2024, 3407], "precision": "bf16"}
            bind_manifest(root, manifest)
            bind_manifest(root, manifest)
            with self.assertRaisesRegex(ValueError, "manifest"):
                bind_manifest(root, dict(manifest, precision="fp16"))

    def test_notebook_is_clean_and_all_code_cells_compile(self):
        path = ROOT / "notebooks/bf16_multiseed_main_method.ipynb"
        self.assertTrue(path.is_file())
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), f"cell-{index}", "exec")

    def test_batch_resumes_verified_jobs_and_rejects_unverified_markers(self):
        from gated_dual_ema_msd.operations import notebook_batch as batch
        from gated_dual_ema_msd.cli.matrix import MatrixJob
        job = MatrixJob("vianli", "B0_CLS", 2024)
        result = {"dataset": "vianli", "experiment_id": "B0_CLS", "seed": 2024,
                  "hparams": {"train_precision": "bf16", "fp32_eval": True, "max_length": 512},
                  "test": None, "test_evaluations": 0, "final_dev": {"macro_f1": 0.5, "accuracy": 0.5},
                  "hf_revision": "b" * 40, "artifact_readback_verified": True}
        manifest = {"git_sha": "a" * 40, "seeds": [42, 2024, 3407]}
        result["batch_manifest_sha256"] = batch.manifest_digest(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            def fake_train(command, repo, log_path):
                run_dir = job.output_dir(root / "work")
                batch.write_json(run_dir / "result.json", result)
                (run_dir / "best_model.pt").write_bytes(b"synthetic-test-only")
            with patch.object(batch, "run_logged", side_effect=fake_train) as train, patch.object(batch, "publish_run", return_value=result) as publish:
                config = dict(wandb_project="offline", epochs=1, eval_steps=1, patience=1, lr=1e-5,
                              weight_decay=0, warmup_ratio=0, label_smoothing=0, dropout=0.1, physical_batch_size=4, grad_accum=4)
                batch.run_batch([job], root, root / "work", root / "stored", config, manifest)
                batch.run_batch([job], root, root / "work", root / "stored", config, manifest)
                self.assertEqual(train.call_count, 1)
                self.assertEqual(publish.call_count, 1)
                self.assertFalse((job.output_dir(root / "work") / "best_model.pt").exists())
                marker = job.output_dir(root / "stored") / "verified_run.json"
                unverified = dict(result, artifact_readback_verified=False)
                batch.write_json(marker, unverified)
                with self.assertRaisesRegex(ValueError, "resume marker"):
                    batch.run_batch([job], root, root / "work", root / "stored", config, manifest)

    def test_batch_commands_keep_test_locked_and_seed_explicit(self):
        from gated_dual_ema_msd.operations.notebook_batch import train_command
        from gated_dual_ema_msd.cli.matrix import MatrixJob
        config = dict(wandb_project="offline", epochs=7, eval_steps=100, patience=5, lr=1e-5,
                      weight_decay=0.005, warmup_ratio=0.06, label_smoothing=0.02, dropout=0.1, physical_batch_size=4, grad_accum=4)
        command = train_command(MatrixJob("vinli", "B0_CLS", 3407), config, ROOT)
        self.assertIn("--no_test", command)
        self.assertIn("--require_cuda", command)
        self.assertIn("--no_wandb", command)
        self.assertNotIn("--frozen_final", command)
        self.assertEqual(command[command.index("--seed") + 1], "3407")

    def test_frozen_final_command_and_result_require_one_dev_selected_test(self):
        from gated_dual_ema_msd.operations.notebook_batch import train_command, validate_run
        from gated_dual_ema_msd.cli.matrix import MatrixJob
        job = MatrixJob("vianli", "M3_FULL", 42)
        config = dict(wandb_project="offline", epochs=7, eval_steps=50, patience=5, lr=1e-5,
                      weight_decay=0.005, warmup_ratio=0.06, label_smoothing=0.02, dropout=0.1,
                      physical_batch_size=4, grad_accum=4, frozen_final=True)
        command = train_command(job, config, ROOT)
        self.assertIn("--frozen_final", command)
        self.assertNotIn("--no_test", command)
        result = dict(dataset="vianli", experiment_id="M3_FULL", seed=42,
                      hparams=dict(train_precision="bf16", fp32_eval=True, max_length=512),
                      selection_policy="dev_macro_f1", test_evaluations=1,
                      test=dict(macro_f1=0.5, accuracy=0.5), peak_test_step=None)
        validate_run(result, job, frozen_final=True)
        for changes in (dict(test_evaluations=2), dict(test=None), dict(selection_policy="test_macro_f1"), dict(peak_test_step=50)):
            with self.assertRaises(ValueError):
                validate_run(dict(result, **changes), job, frozen_final=True)

    def test_notebook_configuration_and_plan_execute_offline(self):
        notebook = json.loads((ROOT / "notebooks/bf16_multiseed_main_method.ipynb").read_text(encoding="utf-8"))
        sources = {cell["metadata"]["tags"][0]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
        namespace = {"display": lambda value: None}
        exec(compile(sources["configuration"], "configuration", "exec"), namespace)
        tracked = SimpleNamespace(path="offline/project/preflight", state="finished", log=lambda value: None, finish=lambda: None)
        wandb = SimpleNamespace(init=lambda **kwargs: tracked, Api=lambda: SimpleNamespace(run=lambda path: tracked))
        class FakeHub:
            def __init__(self, **kwargs):
                self.created = []
            def whoami(self):
                return {"name": "offline-test"}
            def create_repo(self, **kwargs):
                self.created.append(kwargs["repo_id"])
            def repo_info(self, **kwargs):
                return SimpleNamespace(private=False)
        with tempfile.TemporaryDirectory() as temporary:
            namespace.update(previous=None, SOURCE_SHA="a" * 40, OUTPUT_ROOT=pathlib.Path(temporary), REPO_DIR=ROOT)
            with patch.dict(sys.modules, {"wandb": wandb}), patch("huggingface_hub.HfApi", FakeHub), patch.dict(os.environ, {"HF_TOKEN": "offline-test-only"}):
                exec(compile(sources["tracking-preflight"], "tracking-preflight", "exec"), namespace)
            self.assertEqual(namespace["METHODS"], ["M3_FULL"])
            self.assertEqual(namespace["HYPERPARAMS"]["eval_steps"], 50)
            self.assertEqual(namespace["HYPERPARAMS"]["ema_decay"], 0.992)
            self.assertEqual(namespace["HYPERPARAMS"]["ema_start_step"], 100)
            from gated_dual_ema_msd.operations.notebook_batch import train_command
            command = train_command(namespace["JOBS"][0], namespace["CONFIG"], ROOT)
            for flag, value in (("--eval_steps", "50"), ("--ema_decay", "0.992"), ("--ema_start_step", "100")):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertEqual(len(namespace["JOBS"]), 9)
            self.assertTrue(all(job.experiment_id == "M3_FULL" for job in namespace["JOBS"]))
            self.assertEqual(len(set(namespace["api"].created)), 9)
            self.assertNotIn("preflight_path", namespace)

    def test_notebook_does_not_import_wandb_or_require_its_secret(self):
        notebook = json.loads((ROOT / "notebooks/bf16_multiseed_main_method.ipynb").read_text(encoding="utf-8"))
        code = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
        self.assertNotIn('import wandb', code)
        self.assertNotIn('WANDB_API_KEY', code)

    def test_hf_readback_is_immutable_and_detects_modified_weights(self):
        from gated_dual_ema_msd.tracking import hf
        metadata = {"verification_fixture": True}
        revision = "a" * 40
        payloads = {"run_metadata.json": json.dumps(metadata).encode(), "stage2_checkpoint/weights.bin": b"test-weights", "predictions/dev.csv": b"synthetic-prediction-fixture"}
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            predictions = root / "dev.csv"
            predictions.write_bytes(payloads["predictions/dev.csv"])
            checkpoint = {"weights.bin": {"sha256": hashlib.sha256(payloads["stage2_checkpoint/weights.bin"]).hexdigest(), "size_bytes": len(payloads["stage2_checkpoint/weights.bin"])}}
            def download(repo_id, repo_type, filename, revision, token, cache_dir):
                self.assertEqual(revision, "a" * 40)
                self.assertEqual(cache_dir, str(root / "cache"))
                path = pathlib.Path(cache_dir) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payloads[filename])
                return str(path)
            api = SimpleNamespace(list_repo_files=lambda **kwargs: list(payloads))
            with patch.dict(os.environ, {"HF_TOKEN": "offline-test-only"}), patch.object(hf, "_api", return_value=api), patch.object(hf, "hf_hub_download", side_effect=download):
                hf._readback_verify("offline/test", revision, metadata, {"dev": predictions}, checkpoint, cache_dir=str(root / "cache"))
                payloads["stage2_checkpoint/weights.bin"] = b"modified"
                with self.assertRaisesRegex(RuntimeError, "checkpoint read-back mismatch"):
                    hf._readback_verify("offline/test", revision, metadata, {"dev": predictions}, checkpoint, cache_dir=str(root / "cache"))

    def test_single_run_cli_accepts_baseline_seed2024_and_records_wandb_path(self):
        from unittest.mock import MagicMock
        from gated_dual_ema_msd.cli import train
        from verify_source import TinyBackbone, TinyTokenizer, rows
        tracker = MagicMock()
        tracker.run.path = "offline/project/synthetic-run"
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(train, "load_nli_dataset", return_value=({"train": rows("train", 8), "dev": rows("dev", 3)}, TinyTokenizer())) as loader, patch("transformers.AutoModel.from_pretrained", side_effect=lambda *args, **kwargs: TinyBackbone()), patch("gated_dual_ema_msd.training.direct.WandbTracker", return_value=tracker):
                train.main(["--experiment_id", "B0_CLS", "--dataset", "vianli", "--seed", "2024", "--epochs", "1", "--eval_steps", "1", "--no_test", "--output_dir", directory])
            result = json.loads((pathlib.Path(directory) / "vianli/B0_CLS/seed2024/result.json").read_text())
            self.assertEqual(result["seed"], 2024)
            self.assertEqual(result["wandb_run_path"], "offline/project/synthetic-run")
            self.assertEqual(result["test_evaluations"], 0)
            self.assertIs(loader.call_args.kwargs["include_test"], False)

    def test_frozen_final_m3_tests_selected_checkpoint_exactly_once(self):
        from gated_dual_ema_msd.cli import train
        from gated_dual_ema_msd.training import direct
        from verify_source import TinyBackbone, TinyTokenizer, rows
        actual_evaluate = direct.evaluate_model
        calls = []
        def evaluate(model, evaluation_rows, *args, **kwargs):
            calls.append(evaluation_rows[0]["id"].split("-")[0])
            return actual_evaluate(model, evaluation_rows, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(train, "load_nli_dataset", return_value=({"train": rows("train", 8), "dev": rows("dev", 3), "test": rows("test", 3)}, TinyTokenizer())) as loader, patch("transformers.AutoModel.from_pretrained", side_effect=lambda *args, **kwargs: TinyBackbone()), patch.object(direct, "evaluate_model", side_effect=evaluate):
                train.main(["--experiment_id", "M3_FULL", "--dataset", "vianli", "--seed", "42",
                            "--epochs", "1", "--eval_steps", "1", "--ema_start_step", "1",
                            "--frozen_final", "--no_wandb", "--output_dir", directory])
            run_dir = pathlib.Path(directory) / "vianli/M3_FULL/seed42"
            result = json.loads((run_dir / "result.json").read_text())
            self.assertEqual(calls.count("test"), 1)
            self.assertEqual(calls[-1], "test")
            self.assertEqual(result["selection_policy"], "dev_macro_f1")
            self.assertEqual(result["selected_weight_source"], "ema")
            self.assertEqual(result["test_evaluations"], 1)
            self.assertIsNone(result["peak_test_step"])
            self.assertTrue((run_dir / "test_predictions.csv").is_file())
            self.assertIs(loader.call_args.kwargs["include_test"], True)


if __name__ == "__main__":
    unittest.main()
