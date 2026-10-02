# Gated-Dual CafeBERT — main source

Standalone source for Vietnamese NLI: Gated-Dual models, multi-sample dropout,
parameter EMA, training, evaluation, experiment registry and dataset preparation.
The main package was already present in `refactored_clean`; this extraction keeps
the dependency closure of its training/evaluation entry points.

## Install

Python 3.10 or newer:

```sh
python -m pip install -e ".[hf]"
```

## Run

Inspect available arguments without loading a model or dataset:

```sh
python -m gated_dual_ema_msd.cli.train --help
python -m gated_dual_ema_msd.cli.matrix --help
python -m gated_dual_ema_msd.cli.evaluate --help
python -m gated_dual_ema_msd.cli.r2 --help
```

Train the full direct method on ViANLI using train/dev only:

```sh
python -m gated_dual_ema_msd.cli.train --dataset vianli --experiment_id M3_FULL --seed 42 --no_test
```

Dataset sequence limits: **ViNLI=512, ViANLI=512, ViMedNLI=256**.
Training defaults to **BF16 autocast on supported CUDA GPUs**, with loss scaling
disabled; evaluation stays **FP32**. Unsupported CUDA GPUs fail before model/data
loading. Local CPU verification uses FP32.
The direct CLI locks test by default. Use `--frozen_final` only for an explicitly
approved final evaluation. The supplied R2 config is the separately approved
end-to-end protocol; it includes final test inference and requires prepared data
and provenance manifests.

Run the eleven registered configurations on all three datasets with seed 42:

```sh
python -m gated_dual_ema_msd.cli.matrix --cohort seed42 --datasets vinli vianli vimednli --output_dir outputs/bf16_seed42 --parallel_jobs 1
```

This command selects checkpoints on dev and keeps test locked. Increase
`--parallel_jobs` only when GPU memory can fit multiple independent runs.
Use a fresh output directory when comparing with previous FP16 runs; completed
results in an existing directory are skipped unless `--rerun` is supplied.

## BF16 multi-seed notebook

Open [bf16_multiseed_main_method.ipynb in Colab](https://colab.research.google.com/github/baominh5xx2/grid-repo/blob/main/notebooks/bf16_multiseed_main_method.ipynb).
The notebook runs **M3_FULL × 3 datasets × seeds 42/2024/3407 = 9 runs**.
M3_FULL is Gated-Dual CafeBERT with Multi-Sample Dropout and EMA.
The first cell controls datasets, seeds and hyperparameters. Add
`GITHUB_TOKEN` and `HF_TOKEN` in Colab Secrets, select a BF16-capable CUDA GPU,
and run the cells in order. `HF_PRIVATE=False` is explicit in the configuration.

The notebook prepares pinned datasets, verifies tracking/artifact access, and
records an immutable source SHA and train/dev/test hashes. Drive holds the progress
ledger and mean/std CSVs; HF holds the selected checkpoint, tokenizer/config,
metadata and dev/test predictions, read back at an immutable revision. Reuse
`RUN_GROUP` to resume completed verified runs; interrupted training starts again.
The notebook now defaults to the explicitly requested **test-aware exploratory**
mode: dev/test evaluation every **50 optimizer steps** after EMA starts at step
100, a test curve, and a separate peak-test checkpoint. Dev still selects
`best_model.pt` and controls early stopping; `best_test_model.pt` stores the
exploratory peak. Both checkpoints and their predictions are verified on HF.
`exploratory_test_summary.csv` labels the test-aware scores separately.
To use the independent **frozen-final** protocol, set `TEST_PEAK_EXPLORATORY=False`,
`FROZEN_FINAL=True`, and a fresh `RUN_GROUP`: only the dev-selected checkpoint is
tested once, with no test-peak selection.
Patience remains 5 evaluations; maximum epochs remain 7. Local generated checkpoint copies are cleaned after HF
verification to keep runtime disk usage bounded.

The paper's M3 architecture and main optimizer settings match. This run uses BF16
instead of the paper's FP16, a 50-step instead of 100-step eval interval, and the
approved cleaned training sets (ViANLI 8,010 vs 8,012; ViMedNLI 11,217 vs 11,232).
See [the config audit](docs/paper_m3_bf16_audit.md) for the comparison and the
24-layer CafeBERT correction to the paper's Figure 1.

### GitHub authentication on Colab

[Create a fine-grained read token](https://github.com/settings/personal-access-tokens/new?name=grid-colab-read&target_name=baominh5xx2&expires_in=30&contents=read).
Choose resource owner `baominh5xx2`, **Only select repositories → grid-repo**, and
**Contents: Read-only**. Save the token under **Colab Secrets → GITHUB_TOKEN** and
enable notebook access. GitHub documents these settings in
[Managing personal access tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

The access cell runs before clone/install. It verifies repository identity and
uses a temporary Git askpass helper; no token is embedded in URLs, command
arguments, Git config, notebook output or Drive manifests. Clone/fetch verifies
actual Git read access and freezes the exact source commit. If the repository is
public, a token is optional. A missing token for a private repo triggers a hidden
prompt; an invalid supplied token fails without falling back to public access.
If Colab cannot open the private GitHub notebook link, download this `.ipynb` and
use **File → Upload notebook**, then run its GitHub access cell.

Authentication with your real token will be verified on Colab when you add it;
the local auth tests use synthetic credentials and mocked HTTP responses.

## Source layout

- `gated_dual_ema_msd/models/`: models and classification heads.
- `gated_dual_ema_msd/training/`: direct trainer, EMA and R2 stage execution.
- `gated_dual_ema_msd/data/`: loading, tokenization and dataset preparation.
- `gated_dual_ema_msd/evaluation/`: inference and prediction validation.
- `gated_dual_ema_msd/cli/`: train, matrix, evaluate and R2 commands.
- `gated_dual_ema_msd/config/`: pinned sources and experiment definitions.
- `gated_dual_ema_msd/tracking/`: W&B and Hugging Face integrations.
- `gated_dual_ema_msd/compatibility/`: existing model/checkpoint variants.
- `scripts/`: dataset preparation and audit entry points.
- `configs/experiments/`: four main experiment recipes.

The notebook uses Hugging Face only: checkpoint, config, metrics, predictions and
immutable file verification. Batch training passes `--no_wandb`; W&B is not required.
Legacy optional tracking entry points remain available for older callers.
For an already completed Colab run blocked on W&B, use
[HF-only recovery](docs/colab_hf_only_recovery.md) in the existing runtime.
Set credentials through environment variables. Dataset files and trained weights
are downloaded/prepared separately; they are not included in this source repo.

`SOURCE_MANIFEST.json` records source paths and SHA-256 hashes. Extraction changes
package import paths, workspace roots, matrix subprocess working directory and compatibility CLI
forwarding. Model/training arithmetic is retained from the existing source.

## Offline verification

```sh
python verify_source.py
python -m unittest discover -s tests
```

This check parses the source, verifies imports/sequence limits/test defaults, runs
all eleven architectures against a tiny synthetic CPU backbone, and exercises
direct training/checkpoint reload with synthetic train/dev records. It does not
download a model, access benchmark test data, or publish artifacts.
Precision regression tests additionally run actual BF16 forward/backward for all
eleven architectures and a synthetic BF16 trainer loop on CPU, check FP32
evaluation and disabled loss scaling, and mock the CUDA capability checks.
