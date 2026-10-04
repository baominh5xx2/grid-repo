"""Exercise scheduling with real short child processes, without GPU/network work."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gated_dual_ema_msd.cli.matrix import MatrixJob
from gated_dual_ema_msd.operations import notebook_batch as batch
from gated_dual_ema_msd.operations import parallel_notebook_batch as parallel


class ParallelBatchTests(unittest.TestCase):
    def jobs(self):
        return [MatrixJob("vinli", method, 42) for method in
                ("M3_FULL", "ARCH_REL256", "ARCH_ALIGN256")]

    def result(self, job):
        return dict(dataset=job.dataset, experiment_id=job.experiment_id, seed=42,
                    hparams=dict(train_precision="bf16", fp32_eval=True, max_length=512),
                    test=None, test_evaluations=0,
                    final_dev=dict(macro_f1=.5, accuracy=.5))

    def command(self, job, root, *, fail=False, wait=False):
        # Both first-wave children must start before either can finish. This
        # distinguishes concurrent execution from a sequential implementation.
        code = """
import json, pathlib, sys, time
folder, events = map(pathlib.Path, sys.argv[1:3])
name, result, fail, wait = sys.argv[3:]
folder.mkdir(parents=True, exist_ok=True)
start = time.monotonic()
(events / (name + '.start')).write_text(str(start))
deadline = time.monotonic() + 10
while len(list(events.glob('*.start'))) < 2:
    if time.monotonic() > deadline: raise RuntimeError('second worker never started')
    time.sleep(.01)
if fail == 'True': sys.exit(17)
if wait == 'True': time.sleep(60)
time.sleep(.08)
(folder / 'best_model.pt').write_bytes(b'synthetic-test-only')
(folder / 'result.json').write_text(result)
(events / (name + '.end')).write_text(str(time.monotonic()))
print('child completed', name, flush=True)
"""
        return [sys.executable, "-u", "-c", code, str(job.output_dir(root / "work")),
                str(root / "events"), job.experiment_id, json.dumps(self.result(job)),
                str(fail), str(wait)]

    def publish(self, run_dir, repo, job, config, manifest):
        result = json.loads((run_dir / "result.json").read_text())
        result.update(artifact_readback_verified=True, hf_revision="b" * 40,
                      batch_manifest_sha256=batch.manifest_digest(manifest))
        return result

    def setup_root(self, root):
        (root / "events").mkdir()
        return dict(git_sha="a" * 40, config={}), {}

    def test_memory_admission_handles_96gb_three_slots_and_limited_free_memory(self):
        gib = 1024 ** 3
        self.assertEqual(parallel.gpu_slot_plan(88*gib, 89*gib, requested=3,
                         job_count=9, per_job_gib=24, reserve_gib=8)["parallel_jobs"], 3)
        self.assertEqual(parallel.gpu_slot_plan(50*gib, 89*gib, requested=3,
                         job_count=9, per_job_gib=24, reserve_gib=8)["parallel_jobs"], 1)
        self.assertEqual(parallel.gpu_slot_plan(88*gib, 89*gib, requested=3,
                         job_count=2, per_job_gib=24, reserve_gib=8)["parallel_jobs"], 2)
        with self.assertRaisesRegex(RuntimeError, "VRAM"):
            parallel.gpu_slot_plan(20*gib, 89*gib, requested=2,
                                   job_count=2, per_job_gib=24, reserve_gib=8)
        for invalid in (0, 4):
            with self.assertRaises(ValueError):
                parallel.gpu_slot_plan(88*gib, 89*gib, requested=invalid, job_count=2)

    def test_two_slots_overlap_and_third_waits_then_verified_resume_skips_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); manifest, config = self.setup_root(root)
            jobs = self.jobs()
            with patch.object(parallel, "_cuda_memory_info", return_value=(88*1024**3, 89*1024**3)), \
                 patch.object(batch, "train_command", side_effect=lambda j,c,w:self.command(j,root)), \
                 patch.object(batch, "publish_run", side_effect=self.publish) as publish:
                parallel.run_parallel_batch(jobs, root, root/"work", root/"stored",
                                            config, manifest, parallel_jobs=2, poll_seconds=.01)
                intervals = [(float((root/"events"/(j.experiment_id+".start")).read_text()),
                              float((root/"events"/(j.experiment_id+".end")).read_text())) for j in jobs]
                self.assertLess(max(intervals[0][0], intervals[1][0]), min(intervals[0][1], intervals[1][1]))
                self.assertGreaterEqual(intervals[2][0], min(intervals[0][1], intervals[1][1]))
                for job in jobs:
                    self.assertIn("child completed", (job.output_dir(root/"stored")/"worker.log").read_text())
                    self.assertTrue((job.output_dir(root/"stored")/"verified_run.json").exists())
                    self.assertFalse((job.output_dir(root/"work")/"best_model.pt").exists())
                with patch.object(parallel.subprocess, "Popen", side_effect=AssertionError("resume launched a child")):
                    parallel.run_parallel_batch(jobs, root, root/"work", root/"stored",
                                                config, manifest, parallel_jobs=2, poll_seconds=.01)
                self.assertEqual(publish.call_count, 3)
                self.assertEqual(json.loads((root/"stored/progress.json").read_text())["state"], "complete")

    def test_worker_failure_stops_survivor_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); manifest, config = self.setup_root(root)
            jobs = self.jobs()[:2]; children = []; original = subprocess.Popen
            def launch(*args, **kwargs):
                child = original(*args, **kwargs); children.append(child); return child
            with patch.object(parallel, "_cuda_memory_info", return_value=(88*1024**3,89*1024**3)), \
                 patch.object(batch, "train_command", side_effect=lambda j,c,w:self.command(j,root,fail=j==jobs[0],wait=j==jobs[1])), \
                 patch.object(parallel.subprocess, "Popen", side_effect=launch), \
                 patch.object(batch, "publish_run") as publish:
                with self.assertRaisesRegex(RuntimeError, "17"):
                    parallel.run_parallel_batch(jobs,root,root/"work",root/"stored",config,manifest,
                                                parallel_jobs=2,poll_seconds=.01)
                self.assertEqual(len(children), 2)
                self.assertTrue(all(p.poll() is not None for p in children))
                publish.assert_not_called()
            self.assertEqual(json.loads((root/"stored/progress.json").read_text())["state"], "failed")

    def test_upload_failure_retains_weights_and_resumes_publication_without_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); manifest, config = self.setup_root(root)
            job = self.jobs()[0]; folder = job.output_dir(root/"work"); folder.mkdir(parents=True)
            batch.write_json(folder/"result.json", self.result(job))
            (folder/"best_model.pt").write_bytes(b"synthetic-test-only")
            with patch.object(parallel, "_cuda_memory_info", return_value=(88*1024**3,89*1024**3)), \
                 patch.object(parallel.subprocess, "Popen", side_effect=AssertionError("retraining")), \
                 patch.object(batch, "publish_run", side_effect=RuntimeError("HF unavailable")):
                with self.assertRaisesRegex(RuntimeError, "HF unavailable"):
                    parallel.run_parallel_batch([job],root,root/"work",root/"stored",config,manifest)
                self.assertTrue((folder/"best_model.pt").exists())
                self.assertFalse((job.output_dir(root/"stored")/"verified_run.json").exists())
            with patch.object(parallel, "_cuda_memory_info", return_value=(88*1024**3,89*1024**3)), \
                 patch.object(parallel.subprocess, "Popen", side_effect=AssertionError("retraining")), \
                 patch.object(batch, "publish_run", side_effect=self.publish):
                parallel.run_parallel_batch([job],root,root/"work",root/"stored",config,manifest)
                self.assertFalse((folder/"best_model.pt").exists())

    def test_cell_interrupt_during_publication_reaps_other_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); manifest, config = self.setup_root(root)
            jobs = self.jobs()[:2]; children = []; original = subprocess.Popen
            def launch(*args, **kwargs):
                child = original(*args, **kwargs); children.append(child); return child
            with patch.object(parallel, "_cuda_memory_info", return_value=(88*1024**3,89*1024**3)), \
                 patch.object(batch, "train_command", side_effect=lambda j,c,w:self.command(j,root,wait=j==jobs[1])), \
                 patch.object(parallel.subprocess, "Popen", side_effect=launch), \
                 patch.object(batch, "publish_run", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    parallel.run_parallel_batch(jobs,root,root/"work",root/"stored",config,manifest,
                                                parallel_jobs=2,poll_seconds=.01)
                self.assertTrue(all(p.poll() is not None for p in children))
            state = json.loads((root/"stored/progress.json").read_text())
            self.assertEqual(state["state"], "interrupted")
            self.assertEqual(state["active_jobs"], [])
            self.assertTrue((jobs[0].output_dir(root/"work")/"best_model.pt").exists())

    def test_foreign_resume_marker_and_duplicate_jobs_stop_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); manifest, config = self.setup_root(root)
            job = self.jobs()[0]
            marker = dict(self.result(job), artifact_readback_verified=True,
                          hf_revision="b"*40, batch_manifest_sha256="foreign")
            batch.write_json(job.output_dir(root/"stored")/"verified_run.json", marker)
            with patch.object(parallel.subprocess, "Popen", side_effect=AssertionError("launched")):
                with self.assertRaisesRegex(ValueError, "marker"):
                    parallel.run_parallel_batch([job],root,root/"work",root/"stored",config,manifest)
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    parallel.run_parallel_batch([job,job],root,root/"work",root/"stored",config,manifest)


if __name__ == "__main__":
    unittest.main()
