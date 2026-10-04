"""Build the two-model, seed-42 G4 pilot; never execute its cells."""
from __future__ import annotations

import copy
import json
import pathlib
import textwrap

from build_vinli_architecture_notebook import build_notebook as build_template

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/vinli_m3_rel256_bf16_g4_parallel_seed42.ipynb"


def build_notebook() -> dict:
    template = build_template()
    shared = {c["metadata"]["tags"][0]: c for c in template["cells"] if c["cell_type"] == "code"}
    cells = []

    def markdown(source):
        cells.append(dict(cell_type="markdown", metadata={},
                          source=(textwrap.dedent(source).strip()+"\n").splitlines(keepends=True)))

    def code(source, tag):
        cells.append(dict(cell_type="code", metadata={"tags": [tag]}, execution_count=None,
                          outputs=[], source=(textwrap.dedent(source).strip()+"\n").splitlines(keepends=True)))

    def reuse(tag, replacements=(), append=""):
        cell = copy.deepcopy(shared[tag]); source = "".join(cell["source"])
        for before, after in replacements:
            if before not in source:
                raise ValueError(f"Shared notebook cell changed: {tag}/{before}")
            source = source.replace(before, after)
        cell["source"] = (source + textwrap.dedent(append)).splitlines(keepends=True)
        cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    markdown("""
        # ViNLI · M3 vs REL256 · BF16 · seed 42 · G4 parallel pilot

        **Run All launches exactly two independent trainers on CUDA:0:**
        **M3_FULL (relation width 128)** and **ARCH_REL256 (width 256)**, both seed 42.
        Each run uses one CafeBERT encoder and classifier. No ensemble. Only the
        relation branch width changes; all remaining settings match the current run.

        Pinned CafeBERT; ViNLI max length **512**; BF16 train / FP32 evaluation;
        physical batch **4 × accumulation 4 = effective 16 per process**; LR 1e-5;
        weight decay .005; warmup .06; smoothing .02; dropout .1;
        EMA .992 from step 100; five MSD paths; **7 epochs, patience 0**.
        Dev/test scans remain every **30 optimizer steps**, with test scans after
        EMA activation (first interval step 120), plus the last training step.
        HF stores both dev-selected and exploratory peak-test checkpoints. The
        comparison shows both endpoints separately, with accuracy and per-class F1.

        Select the **96 GB G4** runtime. Default: **two concurrent processes**.
        `EXECUTION` accepts 1–3 workers; this two-job pilot never launches a third
        model. GPU admission checks actual free VRAM, using an **estimated** 24 GiB
        per trainer and 8 GiB spare. This is not a measured peak or a memory cap.
        Fewer slots are admitted when free VRAM is lower. Each child gets two CPU
        threads. Increasing concurrency shares GPU compute and may not increase speed.

        Add `HF_TOKEN` (model-repo write) and optional `GITHUB_TOKEN` (private
        checkout, Contents: read) to Colab Secrets. Drive keeps evidence/progress;
        HF keeps immutable verified weights. Source, recipe, runtime and requested
        execution settings are frozen under RUN_GROUP. Changing them needs a new group.
        Interrupting the training cell stops all active children. Completed verified
        jobs are skipped; completed local results retry publication; interrupted
        training restarts that job from scratch. Run All adds no other methods/seeds.
        Test peaks are explicitly exploratory scores selected on the test set.
    """)
    reuse("configuration", [
        ('vinli-arch-testpeak30-bf16-2026-10-03', 'vinli-rel256-g4-bf16-pilot-2026-10-04'),
        ('["M3_FULL", "ARCH_ALIGN256", "ARCH_REL256", "ARCH_CONDPOOL128"]', '["M3_FULL", "ARCH_REL256"]'),
    ], append="""

        EXECUTION = dict(parallel_jobs=2, per_job_vram_gib=24, reserve_vram_gib=8,
                         cpu_threads_per_job=2, poll_seconds=2)
        assert 1 <= EXECUTION["parallel_jobs"] <= 3
        print("Independent trainers requested:", EXECUTION["parallel_jobs"], "| jobs:", len(JOB_SPECS))
    """)
    markdown("""
        ## Authenticate, freeze source and check GPU
        Public checkout works without GITHUB_TOKEN. Secrets are never put in remote
        URLs or manifests. Resume checks out the original frozen SHA.
    """)
    for tag in ("github-auth", "bootstrap", "installation"):
        reuse(tag)
    reuse("gpu-preflight", append="""

        from gated_dual_ema_msd.operations.parallel_notebook_batch import gpu_slot_plan
        GPU_PLAN = gpu_slot_plan(*torch.cuda.mem_get_info(0), requested=EXECUTION["parallel_jobs"],
                                job_count=len(JOB_SPECS), per_job_gib=EXECUTION["per_job_vram_gib"],
                                reserve_gib=EXECUTION["reserve_vram_gib"])
        print("GPU scheduling estimate:", GPU_PLAN)
    """)
    for tag in ("secrets", "tracking-preflight"):
        reuse(tag)
    markdown("""
        ## Freeze the two-job recipe and pinned data
        HF write access is checked for each run before training. The manifest binds
        train/dev/test hashes, label order, code/model/data revisions and scheduler
        settings. The helper's preparation flag includes test fingerprints; this
        study keeps the existing exploratory test scans.
    """)
    reuse("data-preparation", [
        ('"protocol": "vinli_architecture_test_peak_exploratory"', '"protocol": "vinli_relation_width_parallel_pilot"'),
        ('"config": CONFIG, "data_fingerprints": DATA_FINGERPRINTS', '"config": CONFIG, "execution": EXECUTION, "data_fingerprints": DATA_FINGERPRINTS'),
    ])
    markdown("""
        ## Preflight each head, then train both models concurrently
        Preflights run one at a time and exit before training starts. Independent
        processes own their seeds, CUDA allocator, optimizer, EMA, log and checkpoints.
        Trainer logs stream below and remain in each Drive run folder as worker.log.
        Publication runs one at a time; local weights are removed only after HF read-back.
        The scheduler rechecks VRAM before launch and records its actual slot count.
    """)
    reuse("parameter-preflight", append="""

        EXPECTED_HEAD_COUNTS = {"M3_FULL": 2761859, "ARCH_REL256": 3417347}
        for report in HEAD_PREFLIGHT:
            method = report["experiment"]["experiment_id"]
            if report["parameters"]["additional_trainable_head_parameters"] != EXPECTED_HEAD_COUNTS[method]:
                raise ValueError(f"Unexpected relation-width head size: {method}")
    """)
    code("""
        from gated_dual_ema_msd.operations.parallel_notebook_batch import run_parallel_batch
        if BATCH_MANIFEST["execution"] != EXECUTION:
            raise ValueError("Execution settings changed after freeze. Use a new RUN_GROUP.")
        run_parallel_batch(JOBS, REPO_DIR, LOCAL_RUN_ROOT, OUTPUT_ROOT, CONFIG,
                           BATCH_MANIFEST, **EXECUTION)
    """, "training")
    markdown("""
        ## Compare verified M3 and REL256 evidence
        This reads saved predictions/curves and immutable HF references; it runs no
        additional inference. Deltas appear only after both runs are verified.
        Compare dev-selected test scores separately from exploratory test peaks.
        Training times below include contention on the shared GPU; they are not
        isolated architecture throughput benchmarks. No extra seeds launch here.
    """)
    code("""
        from gated_dual_ema_msd.operations.relation_width_pilot import relation_width_table
        COMPARISON = relation_width_table(OUTPUT_ROOT)
        print(f"Verified runs: {len(COMPARISON)} / 2")
        if not COMPARISON.empty:
            COMPARISON.to_csv(OUTPUT_ROOT / "m3_rel256_seed42_comparison.csv", index=False)
            display(COMPARISON)
        if len(COMPARISON) != 2:
            print("Comparison pending: finish both verified seed-42 runs before choosing the next experiment")
        print("Evidence, worker logs and resume directory:", OUTPUT_ROOT)
    """, "summary")
    for index, cell in enumerate(cells):
        cell["id"] = f"rel-width-g4-{index:02d}"
    return {**template, "cells": cells}


if __name__ == "__main__":
    TARGET.write_bytes((json.dumps(build_notebook(), ensure_ascii=False, indent=2)+"\n").encode("utf-8"))
    print(TARGET)
