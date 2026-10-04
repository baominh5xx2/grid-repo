import json
import pathlib
import runpy
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]


class RelationWidthNotebookTests(unittest.TestCase):
    def notebook(self):
        with patch.object(sys, "path", [str(ROOT/"scripts"), *sys.path]):
            return runpy.run_path(str(ROOT/"scripts/build_vinli_relation_width_notebook.py"))["build_notebook"]()

    def test_pilot_has_two_seed42_jobs_and_invokes_parallel_training(self):
        notebook = self.notebook()
        cells = {c["metadata"]["tags"][0]: "".join(c["source"])
                 for c in notebook["cells"] if c["cell_type"] == "code"}
        namespace = {}; exec(cells["configuration"], namespace)
        self.assertEqual([(j["experiment_id"],j["seed"]) for j in namespace["JOB_SPECS"]],
                         [("M3_FULL",42),("ARCH_REL256",42)])
        self.assertEqual(namespace["EXECUTION"]["parallel_jobs"], 2)
        self.assertEqual(namespace["HYPERPARAMS"]["physical_batch_size"], 4)
        self.assertEqual(namespace["HYPERPARAMS"]["grad_accum"], 4)
        self.assertEqual(namespace["HYPERPARAMS"]["eval_steps"], 30)
        self.assertEqual(namespace["HYPERPARAMS"]["patience"], 0)
        self.assertEqual(namespace["VINLI_MAX_LENGTH"], 512)
        calls = []
        namespace.update(JOBS=namespace["JOB_SPECS"], REPO_DIR="repo", LOCAL_RUN_ROOT="work",
                         OUTPUT_ROOT="stored", CONFIG={}, BATCH_MANIFEST={"execution":namespace["EXECUTION"]})
        with patch("gated_dual_ema_msd.operations.parallel_notebook_batch.run_parallel_batch",
                   side_effect=lambda *a,**kw:calls.append((a,kw))):
            exec(cells["training"], namespace)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["parallel_jobs"], 2)
        self.assertNotIn("confirmation", cells)
        for i, c in enumerate(notebook["cells"]):
            if c["cell_type"] == "code":
                self.assertIsNone(c["execution_count"]); self.assertEqual(c["outputs"], [])
                compile("".join(c["source"]), f"cell-{i}", "exec")

    def test_generated_notebook_matches_builder(self):
        saved = json.loads((ROOT/"notebooks/vinli_m3_rel256_bf16_g4_parallel_seed42.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(saved, self.notebook())


if __name__ == "__main__":
    unittest.main()
