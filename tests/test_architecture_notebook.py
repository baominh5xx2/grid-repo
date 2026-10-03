"""Execute architecture notebook protocol cells without network or GPU work."""
import contextlib
import copy
import io
import json
import pathlib
import runpy
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/build_vinli_architecture_notebook.py"
TEMPLATE = ROOT / "notebooks/bf16_multiseed_main_method.ipynb"
ENVIRONMENT = {"python": "3.11.13", "platform": "Linux", "cuda_available": True,
               "cuda_version": "12.4", "gpu": "NVIDIA L4",
               "packages": {"torch": "2.5.1", "transformers": "4.45.2", "numpy": "2.1.1",
                            "pandas": "2.2.3", "huggingface-hub": "0.25.0", "wandb": None}}
ENVIRONMENT_CONTRACT = {"python": "3.11.13", "cuda_version": "12.4", "gpu": "NVIDIA L4",
                        "packages": {"torch": "2.5.1", "transformers": "4.45.2", "numpy": "2.1.1",
                                     "pandas": "2.2.3", "huggingface-hub": "0.25.0"}}


def contract_stub(metadata):
    """The runtime helper owns this reduction; notebook tests consume its payload."""
    return {"python": metadata["python"], "cuda_version": metadata["cuda_version"],
            "gpu": metadata["gpu"], "packages": {k: v for k, v in metadata["packages"].items() if k != "wandb"}}


class ArchitectureNotebookTests(unittest.TestCase):
    def notebook(self):
        self.assertTrue(GENERATOR.is_file(), "Architecture notebook generator is missing")
        namespace = runpy.run_path(str(GENERATOR), run_name="notebook_generator_test")
        return namespace["build_notebook"]()

    def cells(self):
        return {cell["metadata"]["tags"][0]: "".join(cell["source"])
                for cell in self.notebook()["cells"] if cell["cell_type"] == "code"}

    def config(self):
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(self.cells()["configuration"], namespace)
        return namespace

    def test_screening_is_exactly_four_frozen_seed42_jobs(self):
        config = self.config()
        self.assertEqual(config["JOB_SPECS"], [
            {"dataset": "vinli", "experiment_id": "M3_FULL", "seed": 42},
            {"dataset": "vinli", "experiment_id": "ARCH_ALIGN256", "seed": 42},
            {"dataset": "vinli", "experiment_id": "ARCH_REL256", "seed": 42},
            {"dataset": "vinli", "experiment_id": "ARCH_CONDPOOL128", "seed": 42},
        ])
        self.assertEqual(config["VINLI_MAX_LENGTH"], 512)
        self.assertEqual(config["TRAIN_PRECISION"], "bf16")
        self.assertEqual(config["EVALUATION_PRECISION"], "fp32")
        self.assertEqual(config["HYPERPARAMS"], dict(
            epochs=7, eval_steps=30, patience=50, lr=1e-5, weight_decay=0.005,
            warmup_ratio=0.06, label_smoothing=0.02, dropout=0.1,
            physical_batch_size=4, grad_accum=4, ema_decay=0.992, ema_start_step=100,
        ))
        self.assertTrue(config["USE_EMA"])
        self.assertTrue(config["USE_MSD"])
        self.assertEqual(config["MSD_PROBABILITIES"], [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertFalse(config["WANDB_ENABLED"])
        self.assertFalse(config["FROZEN_FINAL"])
        self.assertFalse(config["TEST_PEAK_EXPLORATORY"])
        self.assertNotEqual(config["RUN_GROUP"], "m3-bf16-clean-testpeak50-hf-only-2026-10-03")

    def test_generator_preserves_original_notebook_and_has_clean_compilable_cells(self):
        original = TEMPLATE.read_bytes()
        notebook = self.notebook()
        self.assertEqual(TEMPLATE.read_bytes(), original)
        self.assertEqual(notebook["metadata"]["colab"]["name"], "vinli_architecture_bf16_seed42.ipynb")
        tags = list(self.cells())
        self.assertLess(tags.index("github-auth"), tags.index("bootstrap"))
        self.assertLess(tags.index("bootstrap"), tags.index("installation"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), cell["id"], "exec")
        template = json.loads(original)
        reusable = {c["metadata"]["tags"][0]: c for c in template["cells"] if c["cell_type"] == "code"}
        for tag in ("github-auth", "bootstrap", "installation", "gpu-preflight", "secrets", "training"):
            self.assertEqual(self.cells()[tag], "".join(reusable[tag]["source"]))

    def test_saved_notebook_matches_generator(self):
        saved = json.loads((ROOT / "notebooks/vinli_architecture_bf16_seed42.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(saved, self.notebook(), "Regenerate the architecture notebook after editing its builder")

    def test_data_binding_requests_train_dev_only(self):
        namespace = self.config()
        calls = []
        module = types.ModuleType("gated_dual_ema_msd.operations.notebook_batch")
        module.write_json = lambda path, value: calls.append((path, value))
        architecture_module = types.ModuleType("gated_dual_ema_msd.operations.architecture_search")
        architecture_module.environment_contract = contract_stub
        fingerprints = {"vinli": {"revision": "47bd78ac5d075bd00a3cb4cdd3ede4eec4acf8c2",
                        "max_length": 512, "splits": {"train": {"row_count": 18282}, "dev": {"row_count": 2255}}}}
        def prepare(repo, datasets, *, frozen_final):
            self.assertFalse(frozen_final)
            self.assertEqual(datasets, ["vinli"])
            return fingerprints
        namespace.update(REPO_DIR=pathlib.Path("repo"), OUTPUT_ROOT=pathlib.Path("results"),
                         SOURCE_SHA="a" * 40, MODEL_NAME="uitnlp/CafeBERT",
                         MODEL_REVISION="af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
                         JOBS=[types.SimpleNamespace(**spec) for spec in namespace["JOB_SPECS"]],
                         CONFIG={**namespace["HYPERPARAMS"], "frozen_final": False, "test_peak_exploratory": False},
                         prepare_data=prepare, bind_manifest=lambda path, value: "signature", write_json=module.write_json,
                         environment_metadata=lambda: copy.deepcopy(ENVIRONMENT))
        with patch.dict("sys.modules", {module.__name__: module, architecture_module.__name__: architecture_module}), contextlib.redirect_stdout(io.StringIO()):
            exec(self.cells()["data-preparation"], namespace)
        manifest = namespace["BATCH_MANIFEST"]
        self.assertTrue(manifest["test_locked"])
        self.assertEqual(manifest["protocol"], "vinli_architecture_screening_train_dev")
        self.assertNotIn("test", manifest["data_fingerprints"]["vinli"]["splits"])
        self.assertEqual(manifest.get("environment_contract"), ENVIRONMENT_CONTRACT)
        self.assertEqual(manifest["screening_rule"]["maximum_entailment_drop_pp"], 0.30)
        self.assertTrue(manifest["screening_rule"]["require_positive_mean_cn_delta"])

    def test_preflight_rejects_disabled_msd_wrong_counts_or_wrong_model(self):
        namespace = self.config()
        report = {"train_precision": "bf16", "eval_precision": "fp32", "max_length": 512,
                  "split_sizes": {"train": 18282, "dev": 2255},
                  "experiment": {"experiment_id": "M3_FULL", "use_ema": True, "use_msd": True},
                  "parameters": {"additional_trainable_head_parameters": 2761859},
                  "msd": {"msd_enabled": True, "msd_num_paths": 5, "msd_shared_classifier": True,
                          "msd_dropout_probabilities": [0.1, 0.2, 0.3, 0.4, 0.5]}}
        namespace.update(JOBS=[types.SimpleNamespace(dataset="vinli", experiment_id="M3_FULL", seed=42)],
                         CONFIG={}, LOCAL_RUN_ROOT=pathlib.Path("local"), REPO_DIR=pathlib.Path("repo"),
                         OUTPUT_ROOT=pathlib.Path("outputs"), write_json=lambda *args: None,
                         display=lambda frame: None, pd=types.SimpleNamespace(DataFrame=lambda rows: rows),
                         train_command=lambda *args: ["python", "--no_test", "--no_wandb"])
        changed_reports = []
        disabled_msd = copy.deepcopy(report)
        disabled_msd["msd"]["msd_enabled"] = False
        changed_reports.append(disabled_msd)
        wrong_counts = copy.deepcopy(report)
        wrong_counts["split_sizes"]["dev"] = 1000
        changed_reports.append(wrong_counts)
        wrong_model = copy.deepcopy(report)
        wrong_model["experiment"]["experiment_id"] = "ARCH_ALIGN256"
        changed_reports.append(wrong_model)
        for changed in changed_reports:
            with self.subTest(report=changed), patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, json.dumps(changed))):
                with self.assertRaisesRegex(ValueError, "preflight"):
                    exec(self.cells()["parameter-preflight"], namespace)
        saved = []
        namespace["write_json"] = lambda path, value: saved.append(value)
        stdout = "[FlatCafeBERT] trying uitnlp/CafeBERT @ pinned...\n" + json.dumps(report, indent=2) + "\n"
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout)):
            try:
                exec(self.cells()["parameter-preflight"], namespace)
            except json.JSONDecodeError:
                self.fail("Preflight must parse the final JSON report after normal backbone loader messages")
        self.assertEqual(saved, [{"jobs": [report]}])

    def test_confirmation_is_inert_and_requires_candidate_gate_and_explicit_authorization(self):
        module = types.ModuleType("gated_dual_ema_msd.operations.architecture_search")
        selected = "ARCH_CONDPOOL128"
        module.select_candidate = lambda root: selected
        module.environment_contract = contract_stub
        module.confirmation_jobs = lambda root, candidate: [
            types.SimpleNamespace(dataset="vinli", experiment_id=method, seed=seed)
            for seed in (2024, 3407) for method in ("M3_FULL", selected)]
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            namespace = self.config()
            launches = []
            manifests = []
            namespace.update(REPO_DIR=root / "repo", OUTPUT_ROOT=root / "screening", SIGNATURE="b" * 64,
                             environment_metadata=lambda: copy.deepcopy(ENVIRONMENT),
                             LOCAL_RUN_ROOT=root / "training" / namespace["RUN_GROUP"],
                             CONFIG={**namespace["HYPERPARAMS"], "hf_prefix": namespace["RUN_GROUP"],
                                     "frozen_final": False, "test_peak_exploratory": False},
                             BATCH_MANIFEST={"git_sha": "a" * 40, "methods": namespace["METHODS"],
                                             "seeds": [42], "config": {}, "test_locked": True,
                                             "environment_contract": copy.deepcopy(ENVIRONMENT_CONTRACT),
                                             "data_fingerprints": {"vinli": {"splits": {"train": {}, "dev": {}}}}},
                             bind_manifest=lambda path, value: manifests.append(copy.deepcopy(value)),
                             verify_hf_write_access=lambda jobs, config: None,
                             run_batch=lambda *args: launches.append(args))
            with patch.dict("sys.modules", {module.__name__: module}):
                exec(self.cells()["confirmation"], namespace)
                self.assertEqual(launches, [], "Run All must not launch confirmation")
                launch = namespace["launch_confirmation"]
                with self.assertRaisesRegex(ValueError, "authorize=True"):
                    launch(selected)
                with self.assertRaisesRegex(ValueError, "selected"):
                    launch("ARCH_ALIGN256", authorize=True)
                self.assertEqual(launches, [])
                launch(selected, authorize=True)
                different_gpu = {**ENVIRONMENT, "gpu": "NVIDIA A100"}
                namespace["environment_metadata"] = lambda: different_gpu
                with self.assertRaisesRegex(ValueError, "environment"):
                    launch(selected, authorize=True)
            self.assertEqual(len(launches), 1)
            jobs, _, _, output_root, config, manifest = launches[0]
            self.assertEqual({job.seed for job in jobs}, {2024, 3407})
            self.assertEqual({job.experiment_id for job in jobs}, {"M3_FULL", selected})
            self.assertEqual(len(jobs), 4)
            self.assertNotEqual(output_root, namespace["OUTPUT_ROOT"])
            self.assertNotEqual(config["hf_prefix"], namespace["RUN_GROUP"])
            for job in jobs:
                hf_name = f"{config['hf_prefix']}-{job.dataset}-{job.experiment_id.lower().replace('_', '-')}-seed{job.seed}"
                self.assertLessEqual(len(hf_name), 96, "Derived confirmation HF repository name is too long")
            self.assertFalse(config["frozen_final"])
            self.assertFalse(config["test_peak_exploratory"])
            self.assertTrue(manifest["test_locked"])
            self.assertEqual(manifest["environment_contract"], ENVIRONMENT_CONTRACT)
            self.assertEqual(namespace["BATCH_MANIFEST"]["seeds"], [42])

    def test_capacity_control_is_inert_and_isolated_from_pilot_selection(self):
        module = types.ModuleType("gated_dual_ema_msd.operations.architecture_search")
        module.select_candidate = lambda root: "ARCH_ALIGN256"
        module.environment_contract = contract_stub
        module.capacity_control_jobs = lambda root: [types.SimpleNamespace(
            dataset="vinli", experiment_id="ARCH_REL304", seed=42)]
        namespace = self.config()
        launches = []
        namespace.update(OUTPUT_ROOT=pathlib.Path("runs") / namespace["RUN_GROUP"],
                         LOCAL_RUN_ROOT=pathlib.Path("local") / namespace["RUN_GROUP"],
                         REPO_DIR=pathlib.Path("repo"), SIGNATURE="b" * 64,
                         environment_metadata=lambda: copy.deepcopy(ENVIRONMENT),
                         CONFIG={"hf_prefix": namespace["RUN_GROUP"], "frozen_final": False,
                                 "test_peak_exploratory": False},
                         BATCH_MANIFEST={"git_sha": "a" * 40, "seeds": [42],
                                         "methods": namespace["METHODS"], "config": {}, "test_locked": True,
                                         "environment_contract": copy.deepcopy(ENVIRONMENT_CONTRACT)},
                         bind_manifest=lambda *args: None, verify_hf_write_access=lambda *args: None,
                         run_batch=lambda *args: launches.append(args))
        cells = self.cells()
        self.assertIn("capacity-control", cells, "Safe capacity-control launcher is missing")
        with patch.dict("sys.modules", {module.__name__: module}):
            exec(cells["capacity-control"], namespace)
            self.assertEqual(launches, [], "Run All must not launch capacity controls")
            with self.assertRaisesRegex(ValueError, "authorize=True"):
                namespace["launch_capacity_control"]()
            self.assertEqual(launches, [])
            output = namespace["launch_capacity_control"](authorize=True)
            different_version = copy.deepcopy(ENVIRONMENT)
            different_version["packages"]["transformers"] = "4.50.0"
            namespace["environment_metadata"] = lambda: different_version
            with self.assertRaisesRegex(ValueError, "environment"):
                namespace["launch_capacity_control"](authorize=True)
        self.assertEqual(len(launches), 1)
        jobs, _, _, output_root, config, manifest = launches[0]
        self.assertEqual([(j.experiment_id, j.seed) for j in jobs], [("ARCH_REL304", 42)])
        self.assertEqual(output, output_root)
        self.assertNotEqual(output_root, namespace["OUTPUT_ROOT"])
        self.assertNotEqual(config["hf_prefix"], namespace["RUN_GROUP"])
        self.assertFalse(config["frozen_final"])
        self.assertFalse(config["test_peak_exploratory"])
        self.assertEqual(manifest["methods"], ["ARCH_REL304"])
        self.assertTrue(manifest["test_locked"])
        self.assertEqual(manifest["environment_contract"], ENVIRONMENT_CONTRACT)
        self.assertEqual(manifest["selected_candidate"], "ARCH_ALIGN256")
        self.assertFalse(manifest["replaces_selected_candidate"])
        self.assertEqual(namespace["BATCH_MANIFEST"]["methods"], namespace["METHODS"])

    def test_final_test_is_inert_until_explicit_authorization(self):
        module = types.ModuleType("gated_dual_ema_msd.operations.architecture_final")
        evaluations = []
        module.evaluate_final_batch = lambda **kwargs: evaluations.append(kwargs) or {"complete": True}
        namespace = self.config()
        namespace.update(OUTPUT_ROOT=pathlib.Path("runs") / namespace["RUN_GROUP"],
                         REPO_DIR=pathlib.Path("repo"), select_candidate=lambda root: "ARCH_REL256")
        cells = self.cells()
        self.assertIn("final-evaluation", cells, "Safe final inference cell is missing")
        exec(cells["final-evaluation"], namespace)
        self.assertEqual(evaluations, [], "Run All must never open the final test path")
        with self.assertRaisesRegex(ValueError, "authorize=True"):
            namespace["run_final_test"]("ARCH_REL256")
        with self.assertRaisesRegex(ValueError, "selected"):
            namespace["run_final_test"]("ARCH_ALIGN256", authorize=True)
        with patch.dict("sys.modules", {module.__name__: module}):
            result = namespace["run_final_test"]("ARCH_REL256", authorize=True)
        self.assertTrue(result["complete"])
        self.assertEqual(len(evaluations), 1)
        call = evaluations[0]
        self.assertEqual(call["candidate"], "ARCH_REL256")
        self.assertEqual(call["screening_root"], namespace["OUTPUT_ROOT"])
        self.assertEqual(call["confirmation_root"], namespace["OUTPUT_ROOT"].parent / (namespace["RUN_GROUP"] + "-confirm"))
        self.assertEqual(call["repo_dir"], namespace["REPO_DIR"])
        self.assertEqual(call["output_root"], namespace["OUTPUT_ROOT"] / "final-evaluations")
        self.assertFalse(call["hf_private"])


if __name__ == "__main__":
    unittest.main()
