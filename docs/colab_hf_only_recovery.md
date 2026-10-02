# Resume a completed run with HF-only publication

The user removed W&B after a completed ViNLI run failed during publication because
`wandb_run_path` was missing. This fix removes that requirement; checkpoint and
prediction validation plus immutable HF read-back remain mandatory.

Use the **existing Colab runtime and notebook variables**. Do not restart the
runtime or rerun clone/bootstrap: the completed weights are under `LOCAL_RUN_ROOT`.
Add this code in a new cell after the failed batch cell. It fetches the repaired
publication helper outside the checkout, preserving the training source SHA and
original batch manifest. The publication helper commit is recorded separately in
HF/result metadata. Completed training is skipped because `result.json` already
exists; after verification the batch continues with W&B disabled.

```python
import os, types
from google.colab import userdata

PATCH_REF = "main"  # Resolve once and record the exact fetched helper SHA.
token = os.environ.get("GITHUB_TOKEN") or userdata.get("GITHUB_TOKEN")
with github_git_environment(token) as git_env:
    github_git(["fetch", "origin", PATCH_REF], git_env, REPO_DIR)
    patch_sha = github_git(["rev-parse", "FETCH_HEAD"], git_env, REPO_DIR)
    helper_source = github_git([
        "show", patch_sha + ":gated_dual_ema_msd/operations/notebook_batch.py"
    ], git_env, REPO_DIR)
del token, git_env

hf_batch = types.ModuleType("hf_batch_hotfix")
exec(compile(helper_source, "hf_batch_hotfix", "exec"), hf_batch.__dict__)
hf_batch.BATCH_HELPER_PATCH_SHA = patch_sha
run_batch = hf_batch.run_batch
print("HF-only publication helper:", patch_sha)
run_batch(JOBS, REPO_DIR, LOCAL_RUN_ROOT, OUTPUT_ROOT, CONFIG, BATCH_MANIFEST)
```

Retain the original `CONFIG`, `BATCH_MANIFEST` and output roots while recovering.
The helper ignores old W&B settings, publishes existing weights, and explicitly
records `wandb_enabled=False`. It does not pretend missing W&B history was stored.
The next runs use the original training source/config with `--no_wandb`; only the
publication/batch helper is patched. A new session using the updated notebook
starts in a separate HF-only run group, which will not reuse the old batch.

If the Colab VM has already reset, this cannot restore unuploaded weights from
the terminated VM. If HF publication fails for another reason, keep the runtime
and fix that error before rerunning this same batch cell.
