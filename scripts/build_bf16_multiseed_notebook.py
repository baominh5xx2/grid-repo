"""Generate the executable BF16 multi-seed Colab notebook without saved outputs."""
import json
import pathlib
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
cells = []


def markdown(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(keepends=True)})


def code(source, tag):
    cells.append({"cell_type": "code", "metadata": {"tags": [tag]}, "execution_count": None,
                  "outputs": [], "source": (textwrap.dedent(source).strip() + "\n").splitlines(keepends=True)})


markdown("""
    # BF16 · all registered methods · three seeds

    **Default: 11 configurations × 3 datasets × seeds 42/2024/3407 = 99 independent runs.**
    CafeBERT and datasets use pinned revisions. Training is BF16, evaluation FP32;
    ViNLI/ViANLI max length 512, ViMedNLI 256. Checkpoints are selected on dev.
    **Test stays locked throughout this notebook.**

    Choose a CUDA GPU with BF16 support in **Runtime → Change runtime type**.
    Add `HF_TOKEN` (model-repo write access) and `WANDB_API_KEY` in Colab Secrets
    and enable notebook access. Run the cells in order. This batch can span several
    sessions: reuse the same `RUN_GROUP` to skip verified completed runs; an
    interrupted training run starts again from its seed. Completed weights,
    tokenizer/config, metadata and predictions are verified on HF at an immutable
    revision. Google Drive holds progress, per-run results and mean/std tables.
    Local generated weight copies are removed only after HF verification.

    This matrix contains the repository's three baselines, four main methods and
    four ablations. Select fewer datasets/methods in the first cell if needed.
""")
code("""
    import os
    import pathlib
    import subprocess
    import sys
    import json

    REPO_URL = "https://github.com/baominh5xx2/grid-repo.git"
    SOURCE_REF = "main"  # First session captures its SHA; resume reuses that SHA.
    RUN_GROUP = "bf16-multiseed-2026-10-03"
    DATASETS = ["vinli", "vianli", "vimednli"]
    METHODS = ["B0_CLS", "B1_PARAM_MATCHED_CLS", "B2_SIMPLE_RELATION",
               "M0_RELATION_GATE", "M1_RELATION_MSD", "M2_RELATION_EMA", "M3_FULL",
               "A1_MEAN_POOL", "A2_NO_INTERACTION", "A3_NO_BOTTLENECK", "A4_NO_GATE"]
    SEEDS = [42, 2024, 3407]
    USE_DRIVE = True
    HF_PRIVATE = False
    WANDB_ENTITY = "trinhtrantran3105-uit"  # Change to your own accessible entity if needed.
    WANDB_PROJECT = "gated-relation-cafebert-bf16-multiseed"
    HYPERPARAMS = dict(epochs=7, eval_steps=100, patience=5, lr=1e-5,
                      weight_decay=0.005, warmup_ratio=0.06, label_smoothing=0.02,
                      dropout=0.1, physical_batch_size=4, grad_accum=4)
    assert len(SEEDS) == len(set(SEEDS)) and all(s in (42, 2024, 3407) for s in SEEDS)
    assert DATASETS and METHODS and SEEDS
    print(f"Configured: {len(METHODS)} methods × {len(DATASETS)} datasets × {len(SEEDS)} seeds = {len(METHODS)*len(DATASETS)*len(SEEDS)} runs")
""", "configuration")
markdown("""
    ## Workspace and persistent progress
    The code checkout stays clean. Data is prepared under its ignored `data/`
    folder; temporary checkpoints stay on the runtime disk. Keep the Drive output
    directory and its batch manifest to resume. With `USE_DRIVE=False`, point
    `OUTPUT_ROOT` at a persistent disk before running a long batch.
""")
code("""
    try:
        from google.colab import drive
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    WORKSPACE = pathlib.Path("/content") if IN_COLAB else pathlib.Path.cwd() / "bf16_workspace"
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if IN_COLAB and USE_DRIVE:
        drive.mount("/content/drive")
        OUTPUT_ROOT = pathlib.Path("/content/drive/MyDrive/grid-repo-runs") / RUN_GROUP
    else:
        OUTPUT_ROOT = WORKSPACE / "results" / RUN_GROUP
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPO_DIR = WORKSPACE / "grid-repo"
    LOCAL_RUN_ROOT = WORKSPACE / "training" / RUN_GROUP
    os.environ["HF_HOME"] = str(WORKSPACE / "hf-cache")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    previous_manifest = OUTPUT_ROOT / "batch_manifest.json"
    previous = json.loads(previous_manifest.read_text()) if previous_manifest.exists() else None
    target_ref = previous["git_sha"] if previous else SOURCE_REF
    if not (REPO_DIR / ".git").exists():
        if REPO_DIR.exists() and any(REPO_DIR.iterdir()):
            raise RuntimeError(f"Choose an empty checkout directory: {REPO_DIR}")
        subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(REPO_DIR)], check=True)
    else:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_DIR, text=True).strip()
        if dirty:
            raise RuntimeError("Source checkout is dirty; preserve those edits before proceeding.")
    subprocess.run(["git", "fetch", "--depth", "1", "origin", target_ref], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPO_DIR, check=True)
    SOURCE_SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_DIR, text=True).strip()
    print("Frozen source SHA:", SOURCE_SHA)
    print("Persistent results:", OUTPUT_ROOT)
""", "bootstrap")
code("""
    if "_LOADED_SOURCE_SHA" in globals() and _LOADED_SOURCE_SHA != SOURCE_SHA:
        raise RuntimeError("Source SHA changed in this kernel. Restart the runtime kernel and rerun the setup cells.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(REPO_DIR) + "[tracking]"], check=True)
    sys.path.insert(0, str(REPO_DIR))
    os.chdir(REPO_DIR)
    _LOADED_SOURCE_SHA = SOURCE_SHA
""", "installation")
markdown("""
    ## GPU and credentials preflight
    Unsupported GPUs stop here. Secrets are read from environment/Colab Secrets
    or entered through a hidden prompt; they are never saved in notebook output.
""")
code("""
    import torch
    from gated_dual_ema_msd.training.precision import bf16_enabled
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU runtime before starting this notebook.")
    assert bf16_enabled(torch.device("cuda:0"))
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__, "| Train: BF16 | Eval: FP32")
""", "gpu-preflight")
code("""
    from getpass import getpass
    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        value = os.environ.get(key)
        if not value and IN_COLAB:
            from google.colab import userdata
            try:
                value = userdata.get(key)
            except Exception:
                value = None
        if not value:
            value = getpass(f"{key}: ")
        if not value:
            raise RuntimeError(f"Missing {key}")
        os.environ[key] = value
    del value
    os.environ["WANDB_MODE"] = "online"
    os.environ.pop("WANDB_DISABLED", None)
    print("HF_TOKEN and WANDB_API_KEY are present.")
""", "secrets")
markdown("""
    ## Jobs and tracking preflight
    One isolated train process and W&B run per dataset/method/seed. Each job gets
    its own HF model repository. The default HF visibility is public; change
    `HF_PRIVATE` in the first cell before creating the batch if desired.
""")
code("""
    import pandas as pd
    import wandb
    from huggingface_hub import HfApi
    from gated_dual_ema_msd.cli.matrix import MatrixJob
    from gated_dual_ema_msd.config.experiments import EXPERIMENTS, DATASET_MAX_LENGTHS, MODEL_NAME, MODEL_REVISION
    from gated_dual_ema_msd.operations.notebook_batch import prepare_data, bind_manifest, run_batch
    from gated_dual_ema_msd.training.r2_runtime import environment_metadata

    if set(DATASETS) - set(DATASET_MAX_LENGTHS) or set(METHODS) - set(EXPERIMENTS):
        raise ValueError("Unknown dataset or method in the configuration cell")
    if len(DATASETS) != len(set(DATASETS)) or len(METHODS) != len(set(METHODS)):
        raise ValueError("Datasets and methods must be unique")
    for method in METHODS:
        if set(SEEDS) - set(EXPERIMENTS[method].seeds):
            raise ValueError(f"Unsupported seeds for {method}")
    JOBS = [MatrixJob(dataset, method, seed) for seed in SEEDS for dataset in DATASETS for method in METHODS]
    display(pd.DataFrame([dict(dataset=j.dataset, method=j.experiment_id, seed=j.seed,
                               max_length=DATASET_MAX_LENGTHS[j.dataset]) for j in JOBS]))
    api = HfApi(token=os.environ["HF_TOKEN"])
    HF_NAMESPACE = api.whoami()["name"]
    CONFIG = {**HYPERPARAMS, "hf_namespace": HF_NAMESPACE, "hf_prefix": RUN_GROUP,
              "hf_private": HF_PRIVATE, "wandb_project": WANDB_PROJECT,
              "wandb_entity": WANDB_ENTITY, "keep_local_checkpoints": False}
    if previous and previous["config"] != CONFIG:
        raise ValueError("Settings differ from this RUN_GROUP. Restore them or select a new RUN_GROUP.")
    # Establish real write access before downloading a model or training.
    for job in JOBS:
        repo_id = f"{HF_NAMESPACE}/{RUN_GROUP}-{job.dataset}-{job.experiment_id.lower().replace('_', '-')}-seed{job.seed}"
        api.create_repo(repo_id=repo_id, repo_type="model", private=HF_PRIVATE, exist_ok=True)
        if api.repo_info(repo_id=repo_id, repo_type="model").private != HF_PRIVATE:
            raise RuntimeError(f"HF repository visibility differs: {repo_id}")
    preflight = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
                          name=RUN_GROUP + "-preflight", config={"source_git_sha": SOURCE_SHA,
                          "seeds": SEEDS, "train_precision": "bf16", "test_locked": True})
    preflight_path = preflight.path
    preflight.log({"lifecycle": "preflight_complete", "jobs": len(JOBS)})
    preflight.finish()
    assert wandb.Api().run(preflight_path).state == "finished"
    print("Verified HF repository access and queryable W&B preflight:", len(JOBS), "jobs")
""", "tracking-preflight")
markdown("""
    ## Prepare pinned data and bind the protocol
    Preparation applies the established cleaning rules, including deterministic
    ViANLI decontamination. Training only loads train/dev. The manifest records
    source SHA, immutable model/data revisions, train/dev hashes, job list and
    hyperparameters. Resume refuses to mix different settings or source versions.
""")
code("""
    DATA_FINGERPRINTS = prepare_data(REPO_DIR, DATASETS)
    BATCH_MANIFEST = {"git_sha": SOURCE_SHA, "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
                      "precision": "bf16", "evaluation_precision": "fp32", "test_locked": True,
                      "seeds": SEEDS, "datasets": DATASETS, "methods": METHODS,
                      "jobs": [dict(dataset=j.dataset, experiment_id=j.experiment_id, seed=j.seed) for j in JOBS],
                      "config": CONFIG, "data_fingerprints": DATA_FINGERPRINTS}
    SIGNATURE = bind_manifest(OUTPUT_ROOT, BATCH_MANIFEST)
    from gated_dual_ema_msd.operations.notebook_batch import write_json
    write_json(OUTPUT_ROOT / "session_environment.json", environment_metadata())
    print("Manifest signature:", SIGNATURE)
    print("Prepared train/dev counts:", {d: {s: x['row_count'] for s, x in f['splits'].items()} for d, f in DATA_FINGERPRINTS.items()})
""", "data-preparation")
markdown("""
    ## Run / resume the batch
    Run this cell again after reconnecting and rerunning the setup cells.
    Completed and verified jobs are skipped. Failed artifact publication can be
    retried from the local selected checkpoint while the same runtime survives.
    After a runtime reset, incomplete jobs train again. Interrupting this cell
    terminates its current child process; no detached training launch is used.
""")
code("""
    try:
        run_batch(JOBS, REPO_DIR, LOCAL_RUN_ROOT, OUTPUT_ROOT, CONFIG, BATCH_MANIFEST)
    except KeyboardInterrupt:
        write_json(OUTPUT_ROOT / "progress.json", {"state": "interrupted", "resume": "rerun setup and batch cells"})
        raise
    except Exception as error:
        write_json(OUTPUT_ROOT / "progress.json", {"state": "failed", "error_type": type(error).__name__,
                                                  "resume": "fix the logged failure, then rerun the batch cell"})
        raise
""", "training")
markdown("""
    ## Results across seeds
    `all_runs.csv` contains each verified run; `paper_summary.csv` contains dev
    Macro-F1 mean/sample standard deviation and the number of completed seeds.
    Check the seed count before comparing methods. Test columns remain empty.
""")
code("""
    from gated_dual_ema_msd.cli.matrix import collect_results, write_summaries
    write_summaries(OUTPUT_ROOT)
    runs = collect_results(OUTPUT_ROOT)
    if runs.empty:
        print("No verified runs completed yet.")
    else:
        display(runs.sort_values(["dataset", "experiment_id", "seed"]))
        summary = pd.read_csv(OUTPUT_ROOT / "paper_summary.csv")
        summary["complete"] = summary["seeds"] == len(SEEDS)
        display(summary.sort_values(["dataset", "dev_macro_f1_mean"], ascending=[True, False]))
        print("Verified runs:", len(runs), "/", len(JOBS))
        print("Tables and per-run immutable HF revisions:", OUTPUT_ROOT)
""", "summary")
markdown("""
    References: [Colab file/Drive workflow](https://colab.research.google.com/notebooks/io.ipynb),
    [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html),
    [HF immutable artifact APIs](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api).
""")

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
             "language_info": {"name": "python", "version": "3.11"}, "accelerator": "GPU",
             "colab": {"name": "bf16_multiseed_all_methods.ipynb", "provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells):
    cell["id"] = f"bf16-{index:02d}"
target = ROOT / "notebooks/bf16_multiseed_all_methods.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes((json.dumps(notebook, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
print(target)
