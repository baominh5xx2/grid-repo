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
    # M3_FULL · BF16 · three seeds

    **M3_FULL × 3 datasets × seeds 42/2024/3407 = 9 independent runs.**
    CafeBERT and datasets use pinned revisions. Training is BF16, evaluation FP32;
    ViNLI/ViANLI max length 512, ViMedNLI 256. Checkpoints are selected on dev.
    **Current mode: test-aware exploratory checkpoint scan, requested by the user.**
    Test is checked at each active-EMA dev evaluation and its highest-scoring
    checkpoint is saved separately. These peak scores are **exploratory**, not
    independent held-out test scores for a paper reproduction.

    This is the paper's **M3 training recipe on the approved cleaned splits**,
    with BF16 replacing FP16. ViANLI train has 8,010 rows (paper: 8,012),
    ViMedNLI 11,217 (paper: 11,232). These data differences are preserved in
    the run manifest; scores are a BF16 reproduction on cleaned splits.
    Dev evaluation is every **50 optimizer steps** (paper: 100), as requested
    for denser checkpoint checks. Patience remains 5 evaluations (250 steps).

    Choose a CUDA GPU with BF16 support in **Runtime → Change runtime type**.
    Add `GITHUB_TOKEN` (repository Contents: read access), `HF_TOKEN`
    (model-repo write access) in Colab Secrets. W&B is disabled.
    and enable notebook access. Run the cells in order. This batch can span several
    sessions: reuse the same `RUN_GROUP` to skip verified completed runs; an
    interrupted training run starts again from its seed. Completed weights,
    tokenizer/config, metadata and predictions are verified on HF at an immutable
    revision. Google Drive holds progress, per-run results and mean/std tables.
    Local generated weight copies are removed only after HF verification.

    The main method is **M3_FULL: Gated-Dual CafeBERT + Multi-Sample Dropout + EMA**.
    Select datasets and hyperparameters in the first cell if needed.
""")
code("""
    import os
    import pathlib
    import subprocess
    import sys
    import json

    REPO_URL = "https://github.com/baominh5xx2/grid-repo.git"
    SOURCE_REF = "main"  # First session captures its SHA; resume reuses that SHA.
    REQUIRE_GITHUB_TOKEN = False  # Public clone works without it; private clone prompts if absent.
    RUN_GROUP = "m3-bf16-clean-testpeak50-hf-only-2026-10-03"
    TEST_PEAK_EXPLORATORY = True  # Explicitly requested: explore test curve and save its peak.
    FROZEN_FINAL = False  # For paper protocol: set True, exploratory False, and choose a new RUN_GROUP.
    DATASETS = ["vinli", "vianli", "vimednli"]
    METHODS = ["M3_FULL"]
    SEEDS = [42, 2024, 3407]
    USE_DRIVE = True
    HF_PRIVATE = False
    HYPERPARAMS = dict(epochs=7, eval_steps=50, patience=5, lr=1e-5,
                      weight_decay=0.005, warmup_ratio=0.06, label_smoothing=0.02,
                  dropout=0.1, physical_batch_size=4, grad_accum=4,
                  ema_decay=0.992, ema_start_step=100)
    assert len(SEEDS) == len(set(SEEDS)) and all(s in (42, 2024, 3407) for s in SEEDS)
    assert DATASETS and METHODS and SEEDS
    assert not (FROZEN_FINAL and TEST_PEAK_EXPLORATORY)
    print(f"Configured: {len(METHODS)} methods × {len(DATASETS)} datasets × {len(SEEDS)} seeds = {len(METHODS)*len(DATASETS)*len(SEEDS)} runs")
""", "configuration")
markdown("""
    ## GitHub access (before cloning)

    [Create a fine-grained token](https://github.com/settings/personal-access-tokens/new?name=grid-colab-read&target_name=baominh5xx2&expires_in=30&contents=read).
    Choose **Only select repositories → grid-repo**, with **Contents: Read-only**.
    Put its value in **Colab Secrets → GITHUB_TOKEN**, enable notebook access.
    A public repository can be read without a token; a private repository needs it.
    If the Colab GitHub link cannot open a private notebook, upload the downloaded
    `.ipynb` via **File → Upload notebook**; this cell authenticates the code clone.

    Credentials stay in memory and in the Git child environment. The temporary
    askpass script contains no token and is removed after clone/fetch. Remote URLs,
    command arguments, git config, Drive manifests and notebook output contain no token.
""")
code('''
    import contextlib
    import tempfile
    from getpass import getpass
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlparse

    parsed_repo = urlparse(REPO_URL)
    if (parsed_repo.scheme != "https" or parsed_repo.netloc != "github.com"
            or parsed_repo.query or parsed_repo.fragment):
        raise ValueError("Use a plain HTTPS github.com repository URL without credentials")
    repo_parts = parsed_repo.path.removesuffix(".git").strip("/").split("/")
    if len(repo_parts) != 2 or not all(repo_parts):
        raise ValueError("Expected https://github.com/OWNER/REPO.git")
    GITHUB_REPOSITORY = "/".join(repo_parts)

    def github_repo_info(token):
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "grid-colab-bootstrap"}
        if token:
            headers["Authorization"] = "Bearer " + token
        request = Request("https://api.github.com/repos/" + GITHUB_REPOSITORY, headers=headers)
        try:
            with urlopen(request, timeout=30) as response:
                info = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"GitHub access failed (HTTP {error.code}); check token, expiry and selected repository") from None
        except URLError:
            raise RuntimeError("GitHub is unreachable; retry the access cell") from None
        if info.get("full_name", "").lower() != GITHUB_REPOSITORY.lower():
            raise RuntimeError("GitHub repository identity mismatch")
        return info

    @contextlib.contextmanager
    def github_git_environment(token):
        env = os.environ.copy()
        env.pop("GITHUB_TOKEN", None)
        env.update(GIT_TERMINAL_PROMPT="0", GIT_CONFIG_COUNT="3",
                   GIT_CONFIG_KEY_0="credential.helper", GIT_CONFIG_VALUE_0="",
                   GIT_CONFIG_KEY_1="credential.useHttpPath", GIT_CONFIG_VALUE_1="true",
                   GIT_CONFIG_KEY_2="credential.username", GIT_CONFIG_VALUE_2="x-access-token")
        with tempfile.TemporaryDirectory(prefix="grid-git-auth-") as directory:
            helper = pathlib.Path(directory) / "askpass.py"
            helper.write_text(chr(10).join([
                "#!" + sys.executable, "import os, sys",
                'print("x-access-token" if "username" in sys.argv[1].lower() else os.environ.get("GRID_GITHUB_TOKEN", ""))',
                ""]), encoding="utf-8")
            helper.chmod(0o700)
            env["GIT_ASKPASS"] = str(helper)
            if token:
                env["GRID_GITHUB_TOKEN"] = token
            else:
                env.pop("GRID_GITHUB_TOKEN", None)
            yield env

    def github_git(arguments, env, cwd=None):
        result = subprocess.run(["git", *arguments], cwd=cwd, env=env,
                                text=True, capture_output=True)
        if result.returncode:
            # Avoid echoing credential-bearing remote errors or subprocess arguments.
            raise RuntimeError(f"Git failed ({arguments[0]}, exit {result.returncode}); check GITHUB_TOKEN Contents: read permission and SOURCE_REF")
        return result.stdout.strip()

    _github_token = os.environ.get("GITHUB_TOKEN")
    if not _github_token:
        try:
            from google.colab import userdata
            _github_token = userdata.get("GITHUB_TOKEN")
        except Exception:
            _github_token = None
    if not _github_token and REQUIRE_GITHUB_TOKEN:
        _github_token = getpass("GITHUB_TOKEN (Contents: read): ").strip()
        if not _github_token:
            raise RuntimeError("GITHUB_TOKEN is required by configuration")
    if _github_token:
        _github_info = github_repo_info(_github_token)
    else:
        try:
            _github_info = github_repo_info(None)
        except RuntimeError as error:
            if "HTTP 404" not in str(error) and "HTTP 401" not in str(error):
                raise
            _github_token = getpass("GITHUB_TOKEN for private repository (Contents: read): ").strip()
            if not _github_token:
                raise RuntimeError("Private/unavailable repository: add GITHUB_TOKEN to Colab Secrets") from None
            _github_info = github_repo_info(_github_token)
    print("GitHub repository access verified:", GITHUB_REPOSITORY,
          "| authenticated" if _github_token else "| public read")
''', "github-auth")
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
    with github_git_environment(_github_token) as git_env:
        if not (REPO_DIR / ".git").exists():
            if REPO_DIR.exists() and any(REPO_DIR.iterdir()):
                raise RuntimeError(f"Choose an empty checkout directory: {REPO_DIR}")
            github_git(["clone", "--no-checkout", REPO_URL, str(REPO_DIR)], git_env)
        else:
            if github_git(["remote", "get-url", "origin"], git_env, REPO_DIR) != REPO_URL:
                raise RuntimeError("Checkout origin differs; choose a fresh workspace directory")
            if github_git(["status", "--porcelain"], git_env, REPO_DIR):
                raise RuntimeError("Source checkout is dirty; preserve those edits before proceeding.")
        github_git(["fetch", "--depth", "1", "origin", target_ref], git_env, REPO_DIR)
        github_git(["checkout", "--detach", "FETCH_HEAD"], git_env, REPO_DIR)
        SOURCE_SHA = github_git(["rev-parse", "HEAD"], git_env, REPO_DIR)
        assert not github_git(["status", "--porcelain"], git_env, REPO_DIR)
    del git_env, _github_token
    print("Frozen source SHA:", SOURCE_SHA)
    print("Persistent results:", OUTPUT_ROOT)
""", "bootstrap")
code("""
    if "_LOADED_SOURCE_SHA" in globals() and _LOADED_SOURCE_SHA != SOURCE_SHA:
        raise RuntimeError("Source SHA changed in this kernel. Restart the runtime kernel and rerun the setup cells.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(REPO_DIR) + "[hf]"], check=True)
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
    for key in ("HF_TOKEN",):
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
    print("HF_TOKEN is present. Artifacts and metadata use HF only.")
""", "secrets")
markdown("""
    ## Jobs and tracking preflight
    One isolated train process per dataset/method/seed. Each job gets
    its own HF model repository. The default HF visibility is public; change
    `HF_PRIVATE` in the first cell before creating the batch if desired.
""")
code("""
    import pandas as pd
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
              "hf_private": HF_PRIVATE, "keep_local_checkpoints": False,
              "frozen_final": FROZEN_FINAL, "test_peak_exploratory": TEST_PEAK_EXPLORATORY}
    if previous and previous["config"] != CONFIG:
        raise ValueError("Settings differ from this RUN_GROUP. Restore them or select a new RUN_GROUP.")
    # Establish real write access before downloading a model or training.
    for job in JOBS:
        repo_id = f"{HF_NAMESPACE}/{RUN_GROUP}-{job.dataset}-{job.experiment_id.lower().replace('_', '-')}-seed{job.seed}"
        api.create_repo(repo_id=repo_id, repo_type="model", private=HF_PRIVATE, exist_ok=True)
        if api.repo_info(repo_id=repo_id, repo_type="model").private != HF_PRIVATE:
            raise RuntimeError(f"HF repository visibility differs: {repo_id}")
    print("Verified HF repository access:", len(JOBS), "jobs; W&B disabled")
""", "tracking-preflight")
markdown("""
    ## Prepare pinned data and bind the protocol
    Preparation applies the established cleaning rules, including deterministic
    ViANLI decontamination. Gradient updates and dev checkpoint selection use train/dev.
    In exploratory mode, test is scanned throughout training from the first active
    EMA evaluation. Dev still controls early stopping and `best_model.pt`;
    `best_test_model.pt` holds the separate exploratory peak. The manifest records
    source SHA, immutable model/data revisions, split hashes, job list and
    hyperparameters. Resume refuses to mix different settings or source versions.
""")
code("""
    DATA_FINGERPRINTS = prepare_data(REPO_DIR, DATASETS, frozen_final=FROZEN_FINAL or TEST_PEAK_EXPLORATORY)
    BATCH_MANIFEST = {"git_sha": SOURCE_SHA, "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
                      "precision": "bf16", "evaluation_precision": "fp32", "test_locked": not (FROZEN_FINAL or TEST_PEAK_EXPLORATORY),
                      "seeds": SEEDS, "datasets": DATASETS, "methods": METHODS,
                      "jobs": [dict(dataset=j.dataset, experiment_id=j.experiment_id, seed=j.seed) for j in JOBS],
                  "config": CONFIG, "data_fingerprints": DATA_FINGERPRINTS,
                      "protocol": "exploratory_m3_bf16_test_peak" if TEST_PEAK_EXPLORATORY else
                                  ("paper_m3_recipe_bf16_clean_frozen_final" if FROZEN_FINAL else "paper_m3_recipe_bf16_clean_train_dev"),
                  "paper_differences": {"training_precision": "FP16 -> BF16",
                    "eval_optimizer_steps": {"paper": 100, "run": HYPERPARAMS["eval_steps"]},
                    "train_counts": {"vianli": {"paper": 8012, "run": 8010},
                                     "vimednli": {"paper": 11232, "run": 11217}},
                        "test": "test-aware exploratory checkpoint scan" if TEST_PEAK_EXPLORATORY else
                                ("one evaluation after dev selection" if FROZEN_FINAL else "locked; dev metrics only")}}
    SIGNATURE = bind_manifest(OUTPUT_ROOT, BATCH_MANIFEST)
    from gated_dual_ema_msd.operations.notebook_batch import write_json
    write_json(OUTPUT_ROOT / "session_environment.json", environment_metadata())
    print("Manifest signature:", SIGNATURE)
    print("Prepared split counts:", {d: {s: x['row_count'] for s, x in f['splits'].items()} for d, f in DATA_FINGERPRINTS.items()})
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
    `all_runs.csv` contains each verified run. In exploratory mode,
    `exploratory_test_summary.csv` contains dev-selected scores and the separate
    peak test mean/sample SD across seeds; `test_curve.csv` records each scanned
    optimizer step. Peak scores reflect test-aware selection. Frozen-final mode
    instead writes `paper_summary.csv` without test-peak selection.
""")
code("""
    from gated_dual_ema_msd.cli.matrix import collect_results, write_summaries
    write_summaries(OUTPUT_ROOT)
    runs = collect_results(OUTPUT_ROOT)
    if runs.empty:
        print("No verified runs completed yet.")
    else:
        display(runs.sort_values(["dataset", "experiment_id", "seed"]))
        summary_file = "exploratory_test_summary.csv" if TEST_PEAK_EXPLORATORY else "paper_summary.csv"
        summary = pd.read_csv(OUTPUT_ROOT / summary_file)
        summary["complete"] = summary["seeds"] == len(SEEDS)
        display(summary.sort_values(["dataset", "dev_macro_f1_mean"], ascending=[True, False]))
        if TEST_PEAK_EXPLORATORY:
            peaks = runs.sort_values("peak_test_macro_f1", ascending=False).drop_duplicates("dataset")
            print("Exploratory highest test checkpoint per dataset (test-aware selection):")
            display(peaks[["dataset", "seed", "peak_test_macro_f1", "peak_test_step", "hf_repo_id", "hf_revision", "hf_exploratory_peak_checkpoint_path"]])
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
             "colab": {"name": "bf16_multiseed_main_method.ipynb", "provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells):
    cell["id"] = f"bf16-{index:02d}"
target = ROOT / "notebooks/bf16_multiseed_main_method.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes((json.dumps(notebook, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
print(target)
