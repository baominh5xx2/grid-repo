"""Generate the dev-only ViNLI architecture notebook; never execute its cells."""
from __future__ import annotations

import copy
import json
import pathlib
import textwrap


ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/vinli_architecture_bf16_seed42.ipynb"


def build_notebook(template_path: pathlib.Path | None = None) -> dict:
    """Reuse established credential/bootstrap cells without importing their writer."""
    template_path = template_path or ROOT / "notebooks/bf16_multiseed_main_method.ipynb"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    reusable = {cell["metadata"]["tags"][0]: cell for cell in template["cells"]
                if cell["cell_type"] == "code"}
    cells = []

    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": textwrap.dedent(source).strip().splitlines(keepends=True)})

    def code(source, tag):
        cells.append({"cell_type": "code", "metadata": {"tags": [tag]},
                      "execution_count": None, "outputs": [],
                      "source": (textwrap.dedent(source).strip() + "\n").splitlines(keepends=True)})

    def reuse(tag):
        cell = copy.deepcopy(reusable[tag])
        cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    markdown("""
        # ViNLI · one-model architecture screening · BF16 · seed 42

        Four sequential jobs: **M3_FULL → ARCH_ALIGN256 → ARCH_REL256 → ARCH_CONDPOOL128**.
        Each uses one pinned CafeBERT encoder, one shared three-label classifier,
        EMA and five multi-sample dropout paths. Inference uses one dev-selected
        checkpoint. There is no ensemble. Training is BF16, evaluation is FP32,
        ViNLI max length is **512**, physical batch 4 × accumulation 4 = effective 16.

        The frozen recipe matches the existing HF evidence: LR 1e-5, weight decay
        0.005, warmup 0.06, smoothing 0.02, dropout 0.1, seven epochs, dev every
        **30 optimizer steps**, patience **50 evaluations**, EMA 0.992 from step 100.
        Screening and confirmation use **train/dev only** and **HF-only artifacts**.
        No test split is loaded or evaluated by a training job in this notebook.
        Existing test-aware M3 runs are previously exposed exploratory evidence;
        this protocol does not make the old test set untouched again.

        Choose a BF16-capable CUDA GPU and add `HF_TOKEN` to Colab Secrets with
        model-repository write access. A private GitHub checkout also needs a
        `GITHUB_TOKEN` with Contents: read permission. Enable notebook access.
        Run the setup, screening and summary cells in order. Each child is supervised;
        interrupting its cell terminates it. Completed jobs resume only with the same
        source SHA and manifest. Use a **new RUN_GROUP** when changing configuration.

        **Run All launches only the four screening jobs.** Later confirmation and
        final-inference cells define functions; they do not launch additional work.
        Invoke those functions explicitly after reviewing the recorded dev decisions.
        Generating or opening this notebook does not launch a GPU job.
    """)
    code("""
        import os
        import pathlib
        import subprocess
        import sys
        import json

        REPO_URL = "https://github.com/baominh5xx2/grid-repo.git"
        SOURCE_REF = "main"  # First session freezes its SHA; resume reuses that SHA.
        REQUIRE_GITHUB_TOKEN = False
        RUN_GROUP = "vinli-architecture-bf16-seed42-2026-10-03"
        DATASETS = ["vinli"]
        METHODS = ["M3_FULL", "ARCH_ALIGN256", "ARCH_REL256", "ARCH_CONDPOOL128"]
        SEEDS = [42]
        JOB_SPECS = [dict(dataset="vinli", experiment_id=method, seed=42) for method in METHODS]
        VINLI_MAX_LENGTH = 512
        PINNED_MODEL_NAME = "uitnlp/CafeBERT"
        PINNED_MODEL_REVISION = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d"
        PINNED_VINLI_REVISION = "47bd78ac5d075bd00a3cb4cdd3ede4eec4acf8c2"
        TRAIN_PRECISION = "bf16"
        EVALUATION_PRECISION = "fp32"
        USE_EMA = True
        USE_MSD = True
        MSD_PROBABILITIES = [0.1, 0.2, 0.3, 0.4, 0.5]
        WANDB_ENABLED = False
        TEST_PEAK_EXPLORATORY = False
        FROZEN_FINAL = False
        USE_DRIVE = True
        HF_PRIVATE = False
        HYPERPARAMS = dict(epochs=7, eval_steps=30, patience=50, lr=1e-5,
                          weight_decay=0.005, warmup_ratio=0.06, label_smoothing=0.02,
                          dropout=0.1, physical_batch_size=4, grad_accum=4,
                          ema_decay=0.992, ema_start_step=100)
        assert DATASETS == ["vinli"] and SEEDS == [42]
        assert METHODS == ["M3_FULL", "ARCH_ALIGN256", "ARCH_REL256", "ARCH_CONDPOOL128"]
        assert VINLI_MAX_LENGTH == 512 and USE_EMA and USE_MSD
        assert not WANDB_ENABLED and not TEST_PEAK_EXPLORATORY and not FROZEN_FINAL
        print(f"Configured: {len(JOB_SPECS)} ViNLI seed-42 screening jobs; test locked")
    """, "configuration")
    markdown("""
        ## GitHub access

        [Create a fine-grained read token](https://github.com/settings/personal-access-tokens/new?name=grid-colab-read&target_name=baominh5xx2&expires_in=30&contents=read)
        for **grid-repo → Contents: Read-only** if needed. Put it in Colab Secrets
        as `GITHUB_TOKEN`. Public checkout works without a token. Credentials stay
        in the Git child environment and never enter remote URLs or manifests.
        For private notebook access, upload the `.ipynb` through File → Upload notebook.
    """)
    reuse("github-auth")
    markdown("""
        ## Clean checkout and persistent evidence

        Drive holds the batch manifest, small evidence files and immutable HF
        references. HF holds verified checkpoints/tokenizer/config/predictions.
        Resume refuses source or configuration changes under an existing RUN_GROUP.
    """)
    reuse("bootstrap")
    reuse("installation")
    markdown("""
        ## CUDA BF16 and HF credentials

        Unsupported hardware stops before training. Secrets are read from Colab
        Secrets/environment or a hidden prompt; their values are never printed.
    """)
    reuse("gpu-preflight")
    reuse("secrets")
    code("""
        import pandas as pd
        from huggingface_hub import HfApi
        from gated_dual_ema_msd.cli.matrix import MatrixJob
        from gated_dual_ema_msd.config.experiments import EXPERIMENTS, DATASET_MAX_LENGTHS, MODEL_NAME, MODEL_REVISION
        from gated_dual_ema_msd.operations.notebook_batch import prepare_data, bind_manifest, run_batch, train_command, write_json
        from gated_dual_ema_msd.training.r2_runtime import environment_metadata

        if MODEL_NAME != PINNED_MODEL_NAME or MODEL_REVISION != PINNED_MODEL_REVISION:
            raise ValueError("Backbone differs from the frozen CafeBERT source")
        if DATASET_MAX_LENGTHS["vinli"] != VINLI_MAX_LENGTH:
            raise ValueError("ViNLI max length must be 512")
        for method in METHODS:
            experiment = EXPERIMENTS[method]
            if not experiment.use_ema or not experiment.use_msd or 42 not in experiment.seeds:
                raise ValueError(f"Architecture must preserve EMA/MSD/seed42: {method}")
        JOBS = [MatrixJob(**spec) for spec in JOB_SPECS]
        display(pd.DataFrame([dict(**spec, max_length=VINLI_MAX_LENGTH) for spec in JOB_SPECS]))
        api = HfApi(token=os.environ["HF_TOKEN"])
        HF_NAMESPACE = api.whoami()["name"]
        CONFIG = {**HYPERPARAMS, "hf_namespace": HF_NAMESPACE, "hf_prefix": RUN_GROUP,
                  "hf_private": HF_PRIVATE, "keep_local_checkpoints": False,
                  "frozen_final": False, "test_peak_exploratory": False}
        if previous and previous["config"] != CONFIG:
            raise ValueError("Settings differ from this RUN_GROUP. Choose a new RUN_GROUP.")

        def verify_hf_write_access(jobs, config):
            for job in jobs:
                repo_id = (f"{config['hf_namespace']}/{config['hf_prefix']}-{job.dataset}-"
                           f"{job.experiment_id.lower().replace('_', '-')}-seed{job.seed}")
                api.create_repo(repo_id=repo_id, repo_type="model", private=config["hf_private"], exist_ok=True)
                if api.repo_info(repo_id=repo_id, repo_type="model").private != config["hf_private"]:
                    raise RuntimeError(f"HF repository visibility differs: {repo_id}")
        verify_hf_write_access(JOBS, CONFIG)
        print("HF write access verified for", len(JOBS), "repositories; W&B disabled")
    """, "tracking-preflight")
    markdown("""
        ## Freeze pinned train/dev data and manifest

        This binds the code SHA, model/data revisions, train/dev hashes, label order,
        jobs and hyperparameters. The test split is excluded from fingerprint reads.
        Local split preparation preserves established cleaning rules. No test metric
        participates in screening or checkpoint selection.
        The runtime contract freezes Python, CUDA, GPU and installed package versions
        (excluding optional W&B), and must match before capacity/confirmation launches.
    """)
    code("""
        from gated_dual_ema_msd.operations.architecture_search import environment_contract
        DATA_FINGERPRINTS = prepare_data(REPO_DIR, DATASETS, frozen_final=False)
        if DATA_FINGERPRINTS["vinli"]["revision"] != PINNED_VINLI_REVISION:
            raise ValueError("ViNLI differs from its frozen data revision")
        SESSION_ENVIRONMENT = environment_metadata()
        ENVIRONMENT_CONTRACT = environment_contract(SESSION_ENVIRONMENT)
        BATCH_MANIFEST = {"git_sha": SOURCE_SHA, "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
                          "precision": "bf16", "evaluation_precision": "fp32", "test_locked": True,
                          "environment_contract": ENVIRONMENT_CONTRACT,
                          "label_order": ["E", "C", "N"], "seeds": SEEDS, "datasets": DATASETS, "methods": METHODS,
                          "jobs": [dict(dataset=j.dataset, experiment_id=j.experiment_id, seed=j.seed) for j in JOBS],
                          "config": CONFIG, "data_fingerprints": DATA_FINGERPRINTS,
                          "protocol": "vinli_architecture_screening_train_dev",
                          "prior_test_exposure": "Existing M3 test-aware runs were exploratory; test is previously exposed.",
                          "screening_rule": {"minimum_gain_pp": 0.30, "slow_time_ratio": 1.5,
                                             "slow_minimum_gain_pp": 0.50, "maximum_entailment_drop_pp": 0.30,
                                             "require_positive_mean_cn_delta": True},
                          "selection_policy": "best_active_ema_dev_macro_f1"}
        SIGNATURE = bind_manifest(OUTPUT_ROOT, BATCH_MANIFEST)
        write_json(OUTPUT_ROOT / "session_environment.json", SESSION_ENVIRONMENT)
        print("Manifest signature:", SIGNATURE)
        print("Prepared train/dev counts:", {s: x['row_count'] for s, x in DATA_FINGERPRINTS['vinli']['splits'].items()})
    """, "data-preparation")
    markdown("""
        ## Measure head capacity before training

        The CLI preflight reconstructs each registered model with seed 42 and the
        pinned backbone, measures actual trainable head parameters and checks split
        counts/precision/MSD. These are measured counts, not GPU time estimates.
    """)
    code("""
        HEAD_PREFLIGHT = []
        for job in JOBS:
            completed = subprocess.run(train_command(job, CONFIG, LOCAL_RUN_ROOT) + ["--preflight"],
                                       cwd=REPO_DIR, text=True, capture_output=True, check=True)
            # The backbone prints loader status before the final pretty JSON report.
            report_start = completed.stdout.rfind("\\n{")
            report = json.loads(completed.stdout[report_start + 1:] if report_start >= 0 else completed.stdout)
            experiment = report["experiment"]
            msd = report["msd"]
            if (report["train_precision"] != "bf16" or report["eval_precision"] != "fp32"
                    or report["max_length"] != 512 or report["split_sizes"] != {"train": 18282, "dev": 2255}
                    or experiment["experiment_id"] != job.experiment_id
                    or experiment["use_ema"] is not True or experiment["use_msd"] is not True
                    or msd["msd_enabled"] is not True or msd["msd_num_paths"] != 5
                    or msd["msd_shared_classifier"] is not True
                    or msd["msd_dropout_probabilities"] != MSD_PROBABILITIES
                    or report["parameters"]["additional_trainable_head_parameters"] <= 0):
                raise ValueError(f"Architecture preflight protocol mismatch: {job.experiment_id}")
            HEAD_PREFLIGHT.append(report)
        write_json(OUTPUT_ROOT / "head_preflight.json", {"jobs": HEAD_PREFLIGHT})
        display(pd.DataFrame([dict(experiment_id=r["experiment"]["experiment_id"],
                                   head_parameters=r["parameters"]["additional_trainable_head_parameters"],
                                   **r["split_sizes"]) for r in HEAD_PREFLIGHT]))
    """, "parameter-preflight")
    markdown("""
        ## Four supervised screening children

        Each job uses `--no_test --no_wandb`. The HF publication verifies file hashes,
        prediction row count/ID order/labels/finite logits/argmax at an immutable revision;
        history and diagnostic evidence travel with the selected checkpoint.
        Completed verified jobs are skipped. Interrupting this cell stops its child.
        During the first token-alignment interval, inspect optimizer-step throughput
        before committing to its complete run. Compare recorded timing and VRAM
        with the control; do not infer speed from head parameter counts.
    """)
    reuse("training")
    markdown("""
        ## Dev screening decision

        The primary endpoint is best active-EMA dev Macro-F1. Advance at +0.30
        percentage points; a candidate exceeding 1.5× control training time needs
        +0.50 points. The mean C/N change must be positive and the E decrease must
        be at most 0.30 points. All four immutable screening runs must exist before selection.
        These are practical compute gates, not significance tests.
    """)
    code("""
        from gated_dual_ema_msd.operations.architecture_search import screening_table, select_candidate
        SCREENING_TABLE = screening_table(OUTPUT_ROOT)
        display(SCREENING_TABLE)
        try:
            SELECTED_CANDIDATE = select_candidate(OUTPUT_ROOT)
        except ValueError as error:
            SELECTED_CANDIDATE = None
            print("Confirmation is blocked:", error)
        else:
            print("Dev-selected candidate:", SELECTED_CANDIDATE)
        print("Verified HF revisions and evidence:", OUTPUT_ROOT)
    """, "summary")
    markdown("""
        ## Optional seed-42 capacity controls

        Do not run these automatically. Maximum two Round-2 jobs, still `--no_test`:
        if alignment wins, compare pooled `ARCH_REL304` (parameter match);
        if only conditioned pooling wins, compare pooled `ARCH_REL230`;
        if only width 256 wins, try `ARCH_REL512` and retain 256 unless 512 improves.
        Use a fresh manifest/run group for each applicable branch. Do not treat
        extra capacity as evidence for a token-comparison mechanism. A conditioned
        global/local combination remains conditional future work only after both
        individual components pass; there is no enabled hybrid stub here.

        This cell defines a launcher; after reviewing the pilot decision, invoke
        `launch_capacity_control(authorize=True)` explicitly. It runs one applicable
        control and records the pilot winner separately. Width 512 does not replace
        the selected width-256 candidate automatically; a stronger width-512 result
        motivates a later study with an extended dev-selection/confirmation protocol.
    """)
    code("""
        import copy
        from gated_dual_ema_msd.operations.architecture_search import capacity_control_jobs, select_candidate, environment_contract

        def launch_capacity_control(*, authorize=False):
            if authorize is not True:
                raise ValueError("Explicit authorize=True is required after reviewing the pilot dev decision")
            if environment_contract(environment_metadata()) != BATCH_MANIFEST["environment_contract"]:
                raise ValueError("Current runtime environment differs from the frozen screening contract")
            selected_candidate = select_candidate(OUTPUT_ROOT)
            jobs = capacity_control_jobs(OUTPUT_ROOT)
            if len(jobs) != 1 or jobs[0].dataset != "vinli" or jobs[0].seed != 42:
                raise ValueError("Capacity control must be exactly one ViNLI seed-42 job")
            group = RUN_GROUP + "-capacity"
            output_root = OUTPUT_ROOT.parent / group
            work_root = LOCAL_RUN_ROOT.parent / group
            config = {**CONFIG, "hf_prefix": group, "frozen_final": False, "test_peak_exploratory": False}
            manifest = copy.deepcopy(BATCH_MANIFEST)
            manifest.update(protocol="vinli_architecture_capacity_control_train_dev", seeds=[42],
                            methods=[jobs[0].experiment_id], config=config,
                            jobs=[dict(dataset=j.dataset, experiment_id=j.experiment_id, seed=j.seed) for j in jobs],
                            screening_manifest_sha256=SIGNATURE, selected_candidate=selected_candidate,
                            replaces_selected_candidate=False, test_locked=True)
            bind_manifest(output_root, manifest)
            verify_hf_write_access(jobs, config)
            run_batch(jobs, REPO_DIR, work_root, output_root, config, manifest)
            return output_root
    """, "capacity-control")
    markdown("""
        ## Confirmation: explicit invocation after dev review

        This cell **defines** a launcher and performs no training. After the dev gate
        and applicable capacity review, manually invoke, for example:
        `launch_confirmation("ARCH_REL256", authorize=True)` using the actual selected ID.
        It rejects a candidate that differs from the gate, creates a separate manifest
        and HF prefix, and runs only M3 plus that candidate with seeds 2024/3407.
        Test remains locked. Confirmation requires mean dev gain ≥0.20 points and
        improvement on at least two of three paired seeds before recommending replacement.
    """)
    code("""
        import copy
        from gated_dual_ema_msd.operations.architecture_search import confirmation_jobs, select_candidate, environment_contract

        def launch_confirmation(selected_candidate, *, authorize=False):
            if authorize is not True:
                raise ValueError("Explicit authorize=True is required after reviewing the dev gate")
            if environment_contract(environment_metadata()) != BATCH_MANIFEST["environment_contract"]:
                raise ValueError("Current runtime environment differs from the frozen screening contract")
            if selected_candidate != select_candidate(OUTPUT_ROOT):
                raise ValueError("Candidate differs from the dev-selected architecture")
            jobs = confirmation_jobs(OUTPUT_ROOT, selected_candidate)
            group = RUN_GROUP + "-confirm"
            output_root = OUTPUT_ROOT.parent / group
            work_root = LOCAL_RUN_ROOT.parent / group
            config = {**CONFIG, "hf_prefix": group, "frozen_final": False, "test_peak_exploratory": False}
            manifest = copy.deepcopy(BATCH_MANIFEST)
            manifest.update(protocol="vinli_architecture_confirmation_train_dev", seeds=[2024, 3407],
                            methods=["M3_FULL", selected_candidate], config=config,
                            jobs=[dict(dataset=j.dataset, experiment_id=j.experiment_id, seed=j.seed) for j in jobs],
                            screening_manifest_sha256=SIGNATURE, selected_candidate=selected_candidate,
                            test_locked=True)
            bind_manifest(output_root, manifest)
            verify_hf_write_access(jobs, config)
            run_batch(jobs, REPO_DIR, work_root, output_root, config, manifest)
            return output_root
    """, "confirmation")
    markdown("""
        ## Final inference after confirmation and a frozen decision

        Select the single architecture using all three paired dev results before any
        final test evaluation. Evaluate the six dev-selected checkpoints (M3/candidate,
        three seeds) exactly once with the inference-only CLI, `--split test --frozen_final`.
        The executor downloads paths from verified immutable HF manifests, strictly loads
        each registered architecture, and writes a checkpoint-SHA evaluation ledger and
        predictions into a new immutable revision without changing screening evidence.
        Do not relaunch training or select architecture by an exploratory test peak.
        After confirmation review, explicitly invoke
        `run_final_test("ARCH_REL256", authorize=True)` with the actual selected ID.
        The final inference helper enforces confirmation before opening test.
    """)
    code("""
        def run_final_test(candidate, *, authorize=False):
            if authorize is not True:
                raise ValueError("Explicit authorize=True is required after freezing the confirmed dev decision")
            if candidate != select_candidate(OUTPUT_ROOT):
                raise ValueError("Candidate differs from the dev-selected architecture")
            from gated_dual_ema_msd.operations.architecture_final import evaluate_final_batch
            confirmation_group = RUN_GROUP + "-confirm"
            return evaluate_final_batch(screening_root=OUTPUT_ROOT,
                                        confirmation_root=OUTPUT_ROOT.parent / confirmation_group,
                                        candidate=candidate, repo_dir=REPO_DIR,
                                        output_root=OUTPUT_ROOT / "final-evaluations", hf_private=HF_PRIVATE)
    """, "final-evaluation")
    metadata = copy.deepcopy(template["metadata"])
    metadata["colab"] = {"name": TARGET.name, "provenance": []}
    for index, cell in enumerate(cells):
        cell["id"] = f"arch-{index:02d}"
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes((json.dumps(build_notebook(), ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(TARGET)


if __name__ == "__main__":
    main()
