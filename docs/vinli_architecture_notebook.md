# ViNLI single-model architecture notebook

`notebooks/vinli_architecture_bf16_seed42.ipynb` screens four heads in order:
`M3_FULL`, `ARCH_ALIGN256`, `ARCH_REL256`, `ARCH_CONDPOOL128`. Each uses one
pinned CafeBERT encoder and one shared classifier, with EMA and MSD. The
original multi-dataset notebook remains unchanged.

The fixed recipe is ViNLI max length 512, BF16 training / FP32 evaluation,
physical batch 4 × accumulation 4, LR 1e-5, weight decay 0.005, warmup 0.06,
smoothing 0.02, dropout 0.1, seven epochs, dev evaluation every 30 optimizer
steps, patience 50 evaluations, EMA 0.992 from step 100 and five MSD dropout
rates 0.1/0.2/0.3/0.4/0.5. Screening uses seed 42 only.

## Rebuild and verify

Run from the repository root:

```powershell
python scripts/build_vinli_architecture_notebook.py
python -m unittest discover -s tests -p test_architecture_notebook.py
```

The generator reads the existing notebook to reuse the tested GitHub
authentication, clean-checkout bootstrap, installation, CUDA preflight, HF
secret handling and supervised child-process cells. Importing the generator
or calling `build_notebook()` writes no files. Its command writes only the
new architecture notebook, without executed cells or saved outputs.

## Screening

Use a BF16-capable Colab GPU. Add `HF_TOKEN` with model-repository write access;
private GitHub checkout also needs `GITHUB_TOKEN` with Contents: read access.
Choose a fresh `RUN_GROUP` before the first run. Setup freezes the source SHA
and rejects changed configuration when resuming that group. The manifest also
freezes Python, CUDA and GPU identity, plus tracked package versions for torch,
transformers, numpy, pandas and huggingface-hub. Optional W&B, platform text
and CUDA-availability flags are excluded from the stable comparison.

The head preflight measures each model's actual parameter count and rejects
incorrect precision, split counts, EMA/MSD settings or dropout paths. Every
training child uses `--no_test --no_wandb`. Training runs sequentially, and
interrupting the cell terminates its child. No detached launch is used.

HF is the source of truth for the selected checkpoint, tokenizer/config,
dev predictions, history and diagnostics. Publication verifies file hashes
and prediction IDs/labels/logits at an immutable revision. Drive retains
the manifest, small evidence files and verified HF references. Local selected
weights are removed only after HF verification.

Review the dev summary, per-class F1, measured head count, throughput,
evaluation time and peak VRAM. Advancement requires at least +0.30 dev
percentage points; a candidate above 1.5× control training time needs +0.50.
The mean C/N change must be positive, and the E decrease must be at most 0.30
points. Missing or invalid evidence blocks advancement. These are practical
compute gates, not significance tests.

## Conditional work

Run All launches the four screening jobs only. The later cells define
functions and remain inert, including when an existing kernel has a selected
candidate from a previous execution.

After reviewing the dev gate and applicable capacity control, invoke
`launch_confirmation("<selected ID>", authorize=True)` explicitly. It rejects
any ID other than the dev-selected winner, uses its own manifest/output/HF
prefix and runs exactly M3 plus the candidate on seeds 2024 and 3407. Test
remains locked. Its derived run group is `<RUN_GROUP>-confirm`.
The confirmation manifest keeps the pilot's frozen environment contract, and
the launcher rejects a changed current environment before creating HF repos
or starting a child.
Use `confirmation_decision(screening_root, confirmation_root,
candidate)` from `gated_dual_ema_msd.operations.architecture_search` to review
the three paired seeds. Replacement requires mean dev gain ≥0.20 points and
positive gain on at least two of three seeds.

Optional seed-42 capacity controls have distinct IDs: `ARCH_REL304` if
alignment wins, `ARCH_REL230` if only conditioned pooling wins and
`ARCH_REL512` if only bottleneck widening wins. Explicitly invoke
`launch_capacity_control(authorize=True)` after reviewing the pilot decision.
It derives the applicable single seed-42 job and uses its own
`<RUN_GROUP>-capacity` manifest/output/HF prefix with the same recipe and test
lock. The result is mechanism/capacity evidence, and does not change the
pilot-selected architecture. Its manifest inherits the same frozen environment
contract and checks the current environment before launching. A stronger width-512 result cannot automatically
replace width 256; it motivates a later study with an extended dev-selection
and confirmation protocol. The current selected candidate remains width 256.
The conditional global/local combination remains future work after both
components pass individually; this notebook contains no hybrid stub.

Only after the candidate is frozen and confirmation passes, invoke
`run_final_test("<selected ID>", authorize=True)`. The helper verifies the
decision and downloaded checkpoint hashes, reconstructs registered models,
and uses inference-only evaluation once for each of the six dev-selected
checkpoints. It writes predictions and evaluation ledgers to new immutable
HF revisions while preserving the original screening evidence. It does not
relaunch training. It requires the same frozen runtime contract. Each final
output is bound to its source/checkpoint/test hashes before inference. A setup
failure with zero test evaluations may retry; completed inference recovers
publication without rerunning test. A started but incomplete evaluation stays
blocked for manual inspection. Run only one final launcher at a time.
The existing test set was previously used in exploratory
M3 runs; report that exposure and do not describe it as untouched.
