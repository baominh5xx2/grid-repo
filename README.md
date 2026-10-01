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
The direct CLI locks test by default. Use `--frozen_final` only for an explicitly
approved final evaluation. The supplied R2 config is the separately approved
end-to-end protocol; it includes final test inference and requires prepared data
and provenance manifests.

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
```

This check parses the source, verifies imports/sequence limits/test defaults, runs
all eleven architectures against a tiny synthetic CPU backbone, and exercises
direct training/checkpoint reload with synthetic train/dev records. It does not
download a model, access benchmark test data, or publish artifacts.
