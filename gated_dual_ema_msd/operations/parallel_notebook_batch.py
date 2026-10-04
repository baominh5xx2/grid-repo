"""Supervise independent CUDA trainers; serialize verified HF publication."""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import time
from collections import deque

from gated_dual_ema_msd.operations import notebook_batch as batch

GIB = 1024 ** 3


def gpu_slot_plan(free_bytes: int, total_bytes: int, *, requested: int,
                  job_count: int, per_job_gib: float = 24, reserve_gib: float = 8) -> dict:
    """Admission estimate, not a measured peak or allocator memory limit."""
    if not isinstance(requested, int) or isinstance(requested, bool) or not 1 <= requested <= 3:
        raise ValueError("parallel_jobs must be an integer from 1 to 3")
    if (job_count < 0 or not math.isfinite(per_job_gib) or per_job_gib <= 0
            or not math.isfinite(reserve_gib) or reserve_gib < 0
            or not 0 <= free_bytes <= total_bytes or total_bytes <= 0):
        raise ValueError("Invalid GPU memory budget or job count")
    capacity = max(0, math.floor((free_bytes / GIB - reserve_gib) / per_job_gib))
    slots = min(requested, job_count, capacity)
    if job_count and not slots:
        raise RuntimeError("Insufficient free VRAM for one estimated job plus reserve; free the GPU first")
    return dict(requested_parallel_jobs=requested, parallel_jobs=slots,
                free_vram_gib=free_bytes/GIB, total_vram_gib=total_bytes/GIB,
                estimated_per_job_vram_gib=per_job_gib, reserve_vram_gib=reserve_gib,
                budget_is_estimate=True)


def _cuda_memory_info() -> tuple[int, int]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU before launching parallel trainers")
    return torch.cuda.mem_get_info(0)


def run_parallel_batch(jobs, repo: pathlib.Path, work_root: pathlib.Path,
                       output_root: pathlib.Path, config: dict, batch_manifest: dict, *,
                       parallel_jobs: int = 2, per_job_vram_gib: float = 24,
                       reserve_vram_gib: float = 8, cpu_threads_per_job: int = 2,
                       poll_seconds: float = 2) -> None:
    """Launch at most three processes on CUDA:0 without changing each job's recipe.

    A cell interruption/failure stops all active trainers. A finished result can
    resume HF publication without retraining; only HF-verified jobs are skipped.
    Interrupted training restarts that job from scratch (no optimizer-state resume).
    """
    if not isinstance(parallel_jobs, int) or isinstance(parallel_jobs, bool) or not 1 <= parallel_jobs <= 3:
        raise ValueError("parallel_jobs must be an integer from 1 to 3")
    if cpu_threads_per_job < 1 or not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("CPU threads and poll interval must be positive")
    if len(set(jobs)) != len(jobs):
        raise ValueError("Duplicate jobs would overwrite checkpoints")
    repo, work_root, output_root = map(pathlib.Path, (repo, work_root, output_root))
    if work_root.resolve() == output_root.resolve():
        raise ValueError("Work and persistent output roots must be separate")
    signature = batch.bind_manifest(output_root, batch_manifest)
    batch.bind_manifest(work_root, batch_manifest)
    pending, ready = deque(), deque()
    running = {}  # MatrixJob -> (Popen, binary log handle, read offset)
    verified = []

    def validate_marker(result, job):
        if (result.get("batch_manifest_sha256") != signature
                or result.get("artifact_readback_verified") is not True
                or not re.fullmatch(r"[0-9a-f]{40}", result.get("hf_revision", ""))):
            raise ValueError(f"Invalid resume/publication marker: {job.run_name}")
        batch.validate_run(result, job, frozen_final=bool(config.get("frozen_final", False)),
                           test_peak_exploratory=bool(config.get("test_peak_exploratory", False)))

    for job in jobs:
        marker = job.output_dir(output_root) / "verified_run.json"
        if marker.exists():
            validate_marker(json.loads(marker.read_text(encoding="utf-8")), job)
            verified.append(job.run_name)
            print(f"[parallel] SKIP verified {job.run_name}", flush=True)
        elif (job.output_dir(work_root) / "result.json").exists():
            ready.append(job)
        else:
            pending.append(job)

    plan = (gpu_slot_plan(*_cuda_memory_info(), requested=parallel_jobs, job_count=len(pending),
                          per_job_gib=per_job_vram_gib, reserve_gib=reserve_vram_gib)
            if pending else dict(requested_parallel_jobs=parallel_jobs, parallel_jobs=0))
    plan.update(cpu_threads_per_job=cpu_threads_per_job, batch_manifest_sha256=signature,
                wall_clock_start=time.time())
    batch.write_json(output_root / "parallel_execution_plan.json", plan)
    slots = plan["parallel_jobs"]
    if pending:
        print(f"[parallel] {slots} CUDA:0 workers; {plan['free_vram_gib']:.1f} GiB free; "
              f"{per_job_vram_gib} GiB/job is an estimate, {reserve_vram_gib} GiB reserved", flush=True)
    child_env = os.environ.copy()
    child_env.update(TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS=str(cpu_threads_per_job),
                     MKL_NUM_THREADS=str(cpu_threads_per_job), OPENBLAS_NUM_THREADS=str(cpu_threads_per_job),
                     NUMEXPR_NUM_THREADS=str(cpu_threads_per_job), PYTHONUNBUFFERED="1")

    def progress(state, **extra):
        batch.write_json(output_root / "progress.json", dict(state=state, total=len(jobs),
                         active_jobs=[j.run_name for j in running], pending_jobs=[j.run_name for j in pending],
                         publication_pending=[j.run_name for j in ready], verified_jobs=list(verified), **extra))

    def fill_slots():
        while pending and len(running) < slots:
            job = pending.popleft()
            job.output_dir(work_root).mkdir(parents=True, exist_ok=True)
            log_path = job.output_dir(output_root) / "worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("ab")
            log.write(b"\n--- supervised training attempt ---\n"); log.flush()
            offset = log.tell()
            try:
                child = subprocess.Popen(batch.train_command(job, config, work_root), cwd=repo,
                                         env=child_env, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            running[job] = (child, log, offset)
            print(f"[parallel {len(running)}/{slots}] START {job.run_name} pid={child.pid}", flush=True)
            progress("training")

    def tail(job):
        child, log, offset = running[job]
        with (job.output_dir(output_root) / "worker.log").open("rb") as reader:
            reader.seek(offset)
            content = reader.read(65536)
            offset = reader.tell()
        if content:
            print(content.decode("utf-8", errors="replace"), end="", flush=True)
        running[job] = (child, log, offset)

    def publish(job):
        run_dir, stored_dir = job.output_dir(work_root), job.output_dir(output_root)
        progress("artifact_upload", uploading_job=job.run_name)
        result = batch.publish_run(run_dir, repo, job, config, batch_manifest)
        validate_marker(result, job)
        stored_dir.mkdir(parents=True, exist_ok=True)
        for path in run_dir.iterdir():
            if path.is_file() and path.suffix in (".csv", ".json", ".log") and path.name != "worker.log":
                shutil.copy2(path, stored_dir / path.name)
        batch.write_json(stored_dir / "verified_run.json", result)
        verified.append(job.run_name)
        batch.write_summaries(output_root)
        if not config.get("keep_local_checkpoints", False):
            for name in ("best_model.pt", "best_current_model.pt", "best_test_model.pt", "pytorch_model.bin"):
                (run_dir / name).unlink(missing_ok=True)
        print(f"[parallel] HF VERIFIED {job.run_name} @ {result['hf_revision']}", flush=True)
        progress("training" if running else "publication")

    try:
        while pending or running or ready:
            fill_slots()
            for job in list(running):
                tail(job)
                child, log, _ = running[job]
                code = child.poll()
                if code is None:
                    continue
                tail(job)
                if code != 0:
                    raise RuntimeError(f"{job.run_name} exited {code}; inspect {job.output_dir(output_root)/'worker.log'}")
                log.close(); del running[job]
                if not (job.output_dir(work_root) / "result.json").exists():
                    raise RuntimeError(f"{job.run_name} exited without result.json")
                ready.append(job)
            # Fill a newly freed slot before a potentially slow HF upload.
            fill_slots()
            if ready:
                publish(ready.popleft())
            elif running:
                time.sleep(poll_seconds)
        batch.write_summaries(output_root)
        progress("complete")
    except BaseException as error:
        progress("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                 error_type=type(error).__name__, resume="Rerun setup and training cells with the same RUN_GROUP")
        raise
    finally:
        # Signal all workers first; then reap them, including on Ctrl-C.
        stopped_jobs = [job.run_name for job in running]
        for child, _, _ in running.values():
            if child.poll() is None:
                child.terminate()
        for child, log, _ in running.values():
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            finally:
                log.close()
        if stopped_jobs:
            state_path = output_root / "progress.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state.update(active_jobs=[], stopped_jobs=stopped_jobs)
            batch.write_json(state_path, state)
