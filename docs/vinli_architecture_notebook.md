# ViNLI single-model architecture test-peak notebook

Open `notebooks/vinli_architecture_bf16_seed42.ipynb` in Colab. Run All trains
M3_FULL, ARCH_ALIGN256, ARCH_REL256 and ARCH_CONDPOOL128 sequentially, seed 42.
Each run uses one pinned CafeBERT encoder/classifier, EMA/MSD and HF-only storage.

## Current protocol

The user explicitly requested selecting the best observed test checkpoint during
training. The default notebook enables `TEST_PEAK_EXPLORATORY=True`, keeps
`FROZEN_FINAL=False`, and uses a fresh group:
`vinli-arch-testpeak30-bf16-2026-10-03`. It cannot resume the earlier dev-only group.

ViNLI max length is 512. Training is BF16; dev/test evaluation is FP32. Physical
batch 4 x accumulation 4, LR 1e-5, weight decay .005, warmup .06, smoothing .02,
dropout .1, EMA .992 from step 100, and five MSD paths .1/.2/.3/.4/.5 remain fixed.
Dev/test are evaluated every 30 optimizer steps once EMA is active (first scan
at step 120), plus the final step. **Patience 0 disables dev early stopping**:
each run uses the full seven-epoch budget. Dev is logged as reference evidence.

This is a test-aware exploratory search: the test set is used to select both
checkpoint and architecture. Report that protocol explicitly. The peak score
is not an independent held-out test estimate, and extra seeds do not undo that
selection. There is no ensemble or gradient training on test examples.

## Checkpoints and evidence

Every strictly higher test Macro-F1 saves `best_test_model.pt`,
`test_predictions_peak.csv` and `test_peak.json`. Exact ties keep the earliest
step. Metadata records the step, epoch, EMA source, all test metrics and dev
metrics at that same step. `test_curve.csv` contains every scan, per-class F1,
peak status and running peak. `best_model.pt` remains the dev-selected reference;
its final test score is a separate column and is not the reported peak.

HF publication stores both selected checkpoints, tokenizer/config, dev/test
predictions, history, diagnostics and the complete test curve. The peak weights
are `stage2_checkpoint/exploratory_best_test_model.pt`; peak predictions are
`predictions/test_predictions_peak.csv`. All files are read back at an
immutable HF revision and validated for hashes, row count, IDs, legal labels,
finite logits and argmax. Drive retains small evidence files and HF references.
Local weights are cleaned only after successful HF verification.

The notebook's summary uses `test_peak_table()` to recompute peak metrics from
saved predictions, check curve/prediction hashes, and match the recorded peak to
the maximum of the full curve. Peak metadata hashes and step/epoch/EMA source
must also match the saved run evidence. It exports
`exploratory_architecture_test_peaks.csv` with exact peak scores, step, class
metrics, reference dev F1, head parameters, throughput/VRAM and immutable HF paths.
All four verified pilot results are required before selecting the winner.
Highest peak wins; equal architecture peaks prefer M3, then registry order.
No dev improvement/class/throughput gate blocks test-peak selection.

## Colab execution

Add `HF_TOKEN` with model-repo write access to Colab Secrets. Private GitHub
checkout also needs `GITHUB_TOKEN` with Contents: read access. Choose a BF16-capable
GPU. Setup freezes the source SHA, model/data hashes and Python/package/CUDA/GPU
contract; changes require a fresh group. Train/dev/test fingerprints are included.
Training children receive `--test_peak_exploratory --no_wandb`. Interrupting the
cell terminates its supervised child; completed verified jobs are skipped.

Run All launches only the four pilot jobs. After reviewing peak results,
explicitly call `launch_confirmation("<winner>", authorize=True)` to run only M3
and the selected architecture on seeds 2024/3407 under the same scan/budget.
If M3 wins, only its two additional seeds run. The separate `-confirm` manifest
binds to the original source, recipe, data and runtime.

After those jobs finish, call `show_robustness("<winner>")` to export every seed
peak and mean/std as `exploratory_architecture_peak_runs.csv` and
`exploratory_architecture_peak_summary.csv`. This reads saved evidence and runs
no additional test inference. All seeds remain reported.

Optional `launch_capacity_control(authorize=True)` runs one seed-42 control:
ALIGN256 -> REL304, CONDPOOL128 -> REL230, or REL256 -> REL512. M3 needs no added
architecture control. This uses a separate `-capacity` group with test scanning.
It does not silently replace the pilot winner or launch additional seeds.

## Rebuild and verify

From the repository root:

```powershell
python scripts/build_vinli_architecture_notebook.py
python -m unittest discover -s tests
```

The generator writes a clean, output-free notebook. Its legacy
`build_dev_notebook()` function and dev-only helpers remain available for
regression verification, but the saved notebook uses the requested test-peak
workflow. The original multi-dataset M3 notebook is unchanged.
