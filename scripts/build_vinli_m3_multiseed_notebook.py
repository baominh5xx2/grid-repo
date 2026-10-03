"""Generate the M3-only ViNLI notebook without launching any training."""
from __future__ import annotations

import copy
import json
import pathlib
import textwrap

from build_vinli_architecture_notebook import build_notebook as build_template

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/vinli_m3_bf16_multiseed.ipynb"


def build_notebook() -> dict:
    template = build_template()
    reusable = {c["metadata"]["tags"][0]: c for c in template["cells"] if c["cell_type"] == "code"}
    cells = []

    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": (textwrap.dedent(source).strip() + "\n").splitlines(keepends=True)})

    def reuse(tag, replacements=()):
        cell = copy.deepcopy(reusable[tag])
        source = "".join(cell["source"])
        for before, after in replacements:
            if before not in source:
                raise ValueError(f"Shared cell changed: {tag}/{before}")
            source = source.replace(before, after)
        cell["source"] = source.splitlines(keepends=True)
        cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    markdown("""
        # ViNLI · M3_FULL only · BF16 · seeds 42 / 2024 / 3407

        **Run All runs exactly three sequential jobs**, all using the existing M3_FULL
        method. No architecture search or ensemble. Same settings as the current run:
        pinned CafeBERT, ViNLI max length **512**, BF16 training / FP32 evaluation,
        physical batch 4 × accumulation 4, LR 1e-5, weight decay 0.005, warmup 0.06,
        label smoothing 0.02, dropout 0.1, EMA 0.992 from step 100 and five MSD paths.
        Train the full **7 epochs**, **patience=0**, evaluate every **30 optimizer steps**.
        EMA test scans start at step 120 and also run at the last training step.

        Every higher test Macro-F1 saves its exact EMA checkpoint and predictions;
        ties retain the earliest step. Dev is reference only. These are explicitly
        **test-selected exploratory peaks**. Report all three seed peaks and mean/std.
        The dev-selected final test score is shown separately in the summary.

        Add `HF_TOKEN` with model-repo write permission to Colab Secrets and enable
        notebook access. `GITHUB_TOKEN` (Contents: read) is needed for private GitHub
        access. Select a BF16-capable CUDA GPU, then Run All. W&B is disabled.
        Drive holds resume manifests and evidence; immutable HF revisions hold both
        checkpoints, predictions and test curves after read-back verification.

        This fresh RUN_GROUP reruns all three seeds, including 42. It does not reuse
        the old architecture pilot. Resume this group only with identical settings
        and runtime; completed verified jobs are skipped. Interrupting training stops
        its supervised child. Generating this notebook does not start a GPU job.
    """)
    reuse("configuration", [
        ('vinli-arch-testpeak30-bf16-2026-10-03', 'vinli-m3-testpeak30-bf16-3seeds-2026-10-03'),
        ('METHODS = ["M3_FULL", "ARCH_ALIGN256", "ARCH_REL256", "ARCH_CONDPOOL128"]', 'METHODS = ["M3_FULL"]'),
        ('SEEDS = [42]', 'SEEDS = [42, 2024, 3407]'),
        ('JOB_SPECS = [dict(dataset="vinli", experiment_id=method, seed=42) for method in METHODS]',
         'JOB_SPECS = [dict(dataset="vinli", experiment_id="M3_FULL", seed=seed) for seed in SEEDS]'),
        ('and SEEDS == [42]', 'and SEEDS == [42, 2024, 3407]'),
        ('assert METHODS == ["M3_FULL", "ARCH_ALIGN256", "ARCH_REL256", "ARCH_CONDPOOL128"]', 'assert METHODS == ["M3_FULL"]'),
        ('ViNLI seed-42 test-aware exploratory jobs', 'ViNLI M3_FULL multi-seed test-aware exploratory jobs'),
    ])
    for tag in ("github-auth", "bootstrap", "installation", "gpu-preflight", "secrets", "tracking-preflight"):
        reuse(tag)
    reuse("data-preparation", [
        ('vinli_architecture_test_peak_exploratory', 'vinli_m3_multiseed_test_peak_exploratory'),
    ])
    markdown("""
        ## Validate and train M3_FULL on all three seeds
        Pinned split counts: train 18,282 / dev 2,255 / test 2,264. HF access and
        model/precision/EMA/MSD preflight must pass before training. Local checkpoint
        copies are removed only after immutable HF read-back verifies publication.
    """)
    reuse("parameter-preflight")
    reuse("training")
    markdown("""
        ## Verified results: per-seed test peak and mean/std
        This cell reads saved evidence and performs no extra inference. Check the
        checkpoint step and immutable HF revision for each seed. Standard deviation
        is the sample standard deviation across seeds; partial tables are labeled.
    """)
    source = textwrap.dedent("""
        import pandas as pd
        from gated_dual_ema_msd.operations.architecture_test_peak import verified_peak_result

        rows = []
        for seed in SEEDS:
            job = next(j for j in JOBS if j.seed == seed)
            if not (job.output_dir(OUTPUT_ROOT) / "verified_run.json").exists():
                print(f"Pending seed {seed}; not included in summary")
                continue
            result = verified_peak_result(OUTPUT_ROOT, "M3_FULL", seed)
            rows.append(dict(dataset="vinli", experiment_id="M3_FULL", seed=seed,
                peak_test_macro_f1=result["peak_test_macro_f1"],
                peak_test_accuracy=result["peak_test_metrics"]["accuracy"],
                peak_test_step=result["peak_test_step"],
                dev_macro_f1_at_test_peak=result["peak_test_dev_metrics"]["macro_f1"],
                dev_selected_test_macro_f1=result["test"]["macro_f1"],
                hf_repo_id=result["hf_repo_id"], hf_revision=result["hf_revision"],
                hf_peak_checkpoint=result["hf_exploratory_peak_checkpoint_path"],
                selection_policy="exploratory_test_macro_f1"))
        RUNS = pd.DataFrame(rows)
        print(f"Verified runs: {len(RUNS)} / {len(SEEDS)}")
        if not RUNS.empty:
            SUMMARY = RUNS.groupby(["dataset", "experiment_id"]).agg(
                seeds=("seed", "nunique"),
                peak_test_macro_f1_mean=("peak_test_macro_f1", "mean"),
                peak_test_macro_f1_std=("peak_test_macro_f1", "std"),
                peak_test_accuracy_mean=("peak_test_accuracy", "mean"),
                peak_test_accuracy_std=("peak_test_accuracy", "std")).reset_index()
            SUMMARY["complete"] = len(RUNS) == len(SEEDS)
            SUMMARY["selection_policy"] = "exploratory_test_macro_f1"
            RUNS.to_csv(OUTPUT_ROOT / "m3_exploratory_peak_runs.csv", index=False)
            SUMMARY.to_csv(OUTPUT_ROOT / "m3_exploratory_peak_summary.csv", index=False)
            display(RUNS)
            display(SUMMARY)
        print("Verified evidence and resume directory:", OUTPUT_ROOT)
    """).strip() + "\n"
    cells.append({"cell_type": "code", "metadata": {"tags": ["summary"]},
                  "execution_count": None, "outputs": [], "source": source.splitlines(keepends=True)})
    for index, cell in enumerate(cells):
        cell["id"] = f"m3-multiseed-{index:02d}"
    return {**template, "cells": cells}


if __name__ == "__main__":
    TARGET.write_bytes((json.dumps(build_notebook(), ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(TARGET)
