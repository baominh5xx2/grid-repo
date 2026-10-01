# Gated-Dual CafeBERT — main source

Standalone source for Vietnamese NLI: Gated-Dual models, multi-sample dropout,
parameter EMA, training, evaluation, experiment registry and dataset preparation.
The main package was already present in `refactored_clean`; this extraction keeps
the dependency closure of its training/evaluation entry points.

## Install

Python 3.10 or newer:

```sh
python -m pip install -e ".[tracking]"
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

Open [bf16_multiseed_all_methods.ipynb in Colab](https://colab.research.google.com/github/baominh5xx2/grid-repo/blob/main/notebooks/bf16_multiseed_all_methods.ipynb).
The default is **11 configurations × 3 datasets × seeds 42/2024/3407 = 99 runs**.
The first cell controls methods, datasets, seeds and hyperparameters. Add
`HF_TOKEN` and `WANDB_API_KEY` in Colab Secrets, select a BF16-capable CUDA GPU,
and run the cells in order. `HF_PRIVATE=False` is explicit in the configuration.

The notebook prepares pinned datasets, verifies tracking/artifact access, and
records an immutable source SHA and train/dev hashes. Drive holds the progress
ledger and mean/std CSVs; HF holds the selected checkpoint, tokenizer/config,
metadata and dev predictions, read back at an immutable revision. Reuse
`RUN_GROUP` to resume completed verified runs; interrupted training starts again.
Test stays locked. Local generated checkpoint copies are cleaned after HF
verification to keep runtime disk usage bounded.

CLI equivalent for the explicitly requested multi-seed protocol:

```sh
python -m gated_dual_ema_msd.cli.matrix --cohort multiseed --datasets vinli vianli vimednli --output_dir outputs/bf16_multiseed --parallel_jobs 1
```

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

W&B tracks metrics; Hugging Face stores scientific model artifacts.
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
