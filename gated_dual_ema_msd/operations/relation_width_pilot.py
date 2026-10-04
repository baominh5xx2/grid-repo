"""Report the two verified seed-42 width experiments without launching inference."""
import math
import pathlib

import pandas as pd

from gated_dual_ema_msd.cli.matrix import MatrixJob
from gated_dual_ema_msd.operations.architecture_test_peak import verified_peak_result

METHODS = ("M3_FULL", "ARCH_REL256")


def relation_width_table(output_root: pathlib.Path) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        if not (MatrixJob("vinli", method, 42).output_dir(output_root) / "verified_run.json").exists():
            continue
        result = verified_peak_result(output_root, method, 42)
        selected = result["test"]
        for key in ("macro_f1", "accuracy"):
            if not isinstance(selected.get(key), (int, float)) or not math.isfinite(selected[key]) or not 0 <= selected[key] <= 1:
                raise ValueError(f"Missing dev-selected test metric: {method}/{key}")
        peak = result["peak_test_metrics"]
        rows.append(dict(experiment_id=method, seed=42,
                         dev_macro_f1=result["final_dev"]["macro_f1"],
                         dev_selected_test_macro_f1=selected["macro_f1"], dev_selected_test_accuracy=selected["accuracy"],
                         exploratory_peak_test_macro_f1=peak["macro_f1"], exploratory_peak_test_accuracy=peak["accuracy"],
                         exploratory_peak_step=result["peak_test_step"],
                         peak_f1_E=peak["f1_E"], peak_f1_C=peak["f1_C"], peak_f1_N=peak["f1_N"],
                         dev_at_peak_macro_f1=result["peak_test_dev_metrics"]["macro_f1"],
                         head_parameters=result["hparams"].get("additional_head_parameters"),
                         train_seconds_per_optimizer_step=result.get("train_seconds_per_optimizer_step"),
                         peak_gpu_memory_bytes=result.get("peak_gpu_memory_bytes"),
                         hf_repo_id=result["hf_repo_id"], hf_revision=result["hf_revision"],
                         hf_dev_checkpoint=result["hf_checkpoint_path"],
                         hf_peak_checkpoint=result["hf_exploratory_peak_checkpoint_path"]))
    table = pd.DataFrame(rows)
    if len(table) == 2:
        baseline = table.loc[table.experiment_id == "M3_FULL"].iloc[0]
        for key in ("dev_macro_f1", "dev_selected_test_macro_f1", "exploratory_peak_test_macro_f1"):
            table[f"{key}_delta_pp"] = 100 * (table[key] - baseline[key])
    return table
