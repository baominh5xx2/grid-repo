# ViNLI M3 / REL256 G4 pilot

[Open the notebook in Colab](https://colab.research.google.com/github/baominh5xx2/grid-repo/blob/main/notebooks/vinli_m3_rel256_bf16_g4_parallel_seed42.ipynb).

Run All trains **M3_FULL and ARCH_REL256, seed 42**, in two independent processes
on the same CUDA:0 GPU. REL256 increases only the relation bottleneck 128 → 256.
The existing pinned encoder, max length 512, BF16 training, FP32 evaluation,
batch 4 × accumulation 4, optimizer, EMA/MSD and seven-epoch recipe are preserved.
Dev/test are evaluated every 30 optimizer steps. Patience 0 keeps the full budget.
The summary displays dev-selected test results and exploratory test peaks separately.
No ensemble or extra methods/seeds are launched.

## Scheduling on the 96 GB G4

The first configuration cell contains:

```python
EXECUTION = dict(parallel_jobs=2, per_job_vram_gib=24, reserve_vram_gib=8,
                 cpu_threads_per_job=2, poll_seconds=2)
```

The runner accepts at most three trainers. It checks actual free GPU memory and
admits `min(requested, unfinished_jobs, floor((free_GiB - reserve_GiB) / budget_GiB))`.
24 GiB per job is an admission **estimate**, not a calibrated activation peak or
allocator limit. The per-run measured GPU peak is recorded by the trainer. The
two-job pilot remains capped at two workers even if parallel_jobs is set to 3.
Use three only for a later batch with at least three approved jobs, after observing
the measured memory and throughput. Leave physical batch and accumulation unchanged.
On OOM, all workers stop and logs/checkpoints remain; choose a fresh RUN_GROUP
with fewer workers or a larger estimated per-job budget. No automatic batch shrinking.

CPU BLAS/tokenizer parallelism is bounded per child. GPU compute and CPU/disk/HF
bandwidth remain shared; higher concurrency is not a guarantee of faster execution.
Training time from this pilot is measured under contention. The manifest freezes
requested scheduling settings; parallel_execution_plan.json records the admitted
slots and starting memory. Model preparation preflights run sequentially before
the concurrent trainers start.

## Artifacts, logs and recovery

Add HF_TOKEN (model-repository write) to Colab Secrets; GITHUB_TOKEN is optional
for private checkout. Every run has its own output folder, worker.log and HF repo.
HF publication/read-back and progress/summary writes are serialized by the notebook
supervisor. Both the dev-selected checkpoint and peak-test checkpoint are published
at an immutable revision before generated local weights are removed.

Interrupting the training cell terminates/reaps all its active child processes.
Resume the same RUN_GROUP with the same source/recipe/runtime/execution settings:

- Verified markers are validated and skipped.
- A completed local result retries HF publication without retraining.
- An interrupted training job starts again from scratch; optimizer state is not resumed.
- If the Colab VM disappears, unfinished local weights disappear with it; verified
  Drive evidence and immutable HF artifacts survive.

The pair comparison includes dev F1, dev-selected test F1/accuracy, exploratory
peak test F1/accuracy, per-class peak F1, peak step and both checkpoint references.
REL256 minus M3 deltas in percentage points appear only after both verified runs.
The next seeds/architecture experiment should be chosen after inspecting that pair;
this notebook contains no automatic follow-up launcher.

Regenerate without running training:

```sh
python scripts/build_vinli_relation_width_notebook.py
```
