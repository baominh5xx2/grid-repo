# ViNLI Single-Model Architecture Search Implementation Plan

> **Protocol superseded by the user's subsequent instruction:** the default
> notebook now scans test every 30 eligible optimizer steps, disables dev early
> stopping (patience 0), and selects the test-aware exploratory peak. Dev-only
> gates below describe the original plan, retained as historical documentation.
> See `docs/vinli_architecture_notebook.md` for the current executable workflow.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. The user subsequently approved updating source/notebook and pushing GitHub; GPU execution is deferred to the user's Colab run.

**Goal:** Improve ViNLI Macro-F1 with one CafeBERT model by testing how its head preserves and combines sentence-level and token-level comparison evidence.

**Architecture:** Keep the pinned CafeBERT pair encoder, EMA, MSD and one shared three-label classifier. Screen three head changes against M3: a wider relation bottleneck, segment attention conditioned on the opposite segment, and the existing token-alignment head. Only combine successful components after their individual effects have been measured.

**Tech Stack:** PyTorch, Transformers, existing Python model registry/CLI, Colab BF16 training and FP32 evaluation, HF-only artifacts.

**Spec:** User request on 2026-10-03: no ensemble; prioritize model architecture. Evidence is the three immutable HF runs linked at the end of this document. This plan was accepted for source/notebook implementation and GitHub publication.

## Implementation decisions and verification

Tasks 1–3 are implemented, with CPU tests and mocked HF/CUDA verification. No
GPU experiment or real HF artifact upload was performed. Task 4 remains the
future Colab experiment sequence; scientific gains are not claimed here.

The isolated `main_source` repository uses branch `codex/vinli-architecture-search`
from main `5659b2917cb9e86182ac1b9c5fd17946d2a98171`. The enclosing legacy checkout
is preserved. The user requested a complete source/notebook push, so these
coupled changes are integrated together after verification.

The automatic pilot gate enforces positive mean C/N change and an E loss of at
most 0.30 points, alongside macro-F1 and throughput gates. This conservative
rule is explicit in notebook manifests. Confirmation binds the screening SHA,
recipe, data/model fingerprints and Python/package/CUDA/GPU contract.

One capacity-control launcher is implemented for the selected pilot winner.
The global/local hybrid remains conditional future work after individual
signals exist. Width 512 is capacity evidence and cannot replace selected width
256 in this notebook without an extended selection/confirmation study.

Final inference stores a source binding before loading weights. Matching setup
failures with zero test evaluations may retry. Completed CLI output recovers
wrapper provenance without inference; started but incomplete test evaluation
never reruns automatically. Local and immutable HF resume are covered offline.

## Global Constraints

- One encoder, one classifier, one selected checkpoint at inference; no seed ensemble or checkpoint ensemble.
- CafeBERT: `uitnlp/CafeBERT@af76fcf2a04096b2b54b348a3e4eb48253c93c5d`, hidden 1024, 24 layers.
- ViNLI: `trantranuit/ViHLM_NLI_Project@47bd78ac5d075bd00a3cb4cdd3ede4eec4acf8c2`; E/C/N label order; train/dev/test 18,282/2,255/2,264.
- ViNLI = 512; ViANLI = 512; ViMedNLI = 256 for every train/evaluation call.
- Physical batch 4, accumulation 4, effective batch 16; BF16 train, FP32 evaluation.
- Freeze LR 1e-5, weight decay 0.005, warmup 0.06, dropout 0.1, label smoothing 0.02, epoch cap 7.
- Match the actual HF runs: evaluate dev every 30 optimizer steps, patience 50 evaluations. Do not simultaneously change this recipe during architecture screening.
- EMA decay 0.992 from step 100; five MSD paths 0.1/0.2/0.3/0.4/0.5; same EMA selection rule across variants.
- Seed 42 for screening; seeds 2024 and 3407 only for the chosen candidate and its M3 control.
- Proposed search protocol: select architecture/checkpoint on dev; use `--no_test` while screening. This proposal reduces further test-driven decisions; it does not erase the test exposure in existing runs.
- Final comparison evaluates the preselected candidate and M3 control once per seed. Record previous exploratory test use and do not call the old test set untouched.
- HF is the artifact backend; no W&B requirement. Use a new run group and immutable code/config/data fingerprints.
- Preserve `M3_FULL`, legacy wrappers and old checkpoints. New architectures get distinct registry IDs and model types.

## Evidence and hypotheses

The existing dev-selected M3 test mean is 81.7729%; the corresponding dev mean is 83.0683%. Mean per-class dev/test F1 is E 84.06/83.37, C 83.53/81.04, N 81.62/80.90. Contradiction accounts for the largest dev/test drop; Neutral has the lowest absolute score. This does not establish which linguistic phenomenon causes the errors.

EMA dev F1 exceeds current-weight dev F1 at the selected training step for all three seeds. Keep EMA for this round; this observation is not proof that EMA wins at every step.

Current head: independently scored segment pools of already pair-contextualized tokens; `[u;v;u*v;abs(u-v)]`; `4096 -> 128 -> 1024`; residual gated into CLS. CafeBERT already permits premise/hypothesis token interaction. We are testing an explicit head inductive bias, not repairing a missing cross-attention encoder.

The scalar `gate_mean` across 1024 channels is approximately 0.267–0.269. That average cannot establish gate collapse. No gate redesign is justified until channel-level behavior and branch contribution are measured.

## Round 1: four seed-42 jobs

| ID | Difference from M3 | Hypothesis | Estimated trainable head parameters |
|---|---|---|---:|
| `M3_FULL` | New control run with the frozen recipe | Same-environment reference | 2,761,859 |
| `ARCH_REL256` | Relation bottleneck 128 -> 256 | Compression is too restrictive | 3,417,347 |
| `ARCH_CONDPOOL128` | Replace scalar segment scorers with opposite-segment-conditioned attention, attention dimension 128 | A token's relevance depends on the other sentence | 3,284,099 |
| `ARCH_ALIGN256` | Existing `pool_mode=token_align`, dimension 256; retain CLS fusion/MSD/EMA | Explicit local comparison retains evidence lost in pooled head representations | 3,677,699 |

Parameter totals are analytical estimates from the current 1024-dimensional head and must be measured in preflight before GPU work. Candidate accuracy must be interpreted alongside capacity and throughput.

Recommended launch order: M3 control -> token alignment -> wider bottleneck -> conditioned pooling. If an existing M3 control is reused, require identical recipe, sources, environment and initialization policy; otherwise run the control again. Do not claim GPU time estimates from parameter counts.

### Exact candidate definitions

**Wider bottleneck:** only `relation_hidden=256`; leave segment pooling, feature operators, gate and classifier identical to M3.

**Conditioned pooling:** for premise tokens `P`, hypothesis tokens `H`, use masked mean summaries `mP`, `mH`; four independent bias-free projections from 1024 to 128:

```text
qH = WqH * mH; kP_i = WkP * P_i
sP_i = dot(kP_i, qH) / sqrt(128)
u = sum_i masked_softmax(sP)_i * P_i

qP = WqP * mP; kH_j = WkH * H_j
sH_j = dot(kH_j, qP) / sqrt(128)
v = sum_j masked_softmax(sH)_j * H_j
```

Keep `[u;v;u*v;abs(u-v)]`, bottleneck 128, gate and classifier identical to M3. Score/softmax/mask normalization are FP32; empty segments produce zero vectors. Initial random linear weights follow the existing PyTorch initialization, with a frozen seed before model construction.

**Token alignment:** reuse `_pool_token_alignment` and its existing initialization. Project contextual tokens to 256; bidirectional masked similarity; local `[x;aligned;x-aligned;x*aligned]`; masked mean/max aggregation; project relation to 1024; existing gated CLS fusion. Similarity/softmax stay FP32. This replaces the pooled relation branch, not the CafeBERT backbone. Do not silently change its relation projection to the M3 bottleneck in this first experiment.

The local-comparison proposal is based on [ESIM](https://aclanthology.org/P17-1152/) and [Decomposable Attention](https://aclanthology.org/D16-1244/). Reusing these ideas is not by itself a novelty claim or a guarantee of higher ViNLI F1.

## Decision rules

- Primary screening endpoint: best EMA dev Macro-F1, selected by the same rule as M3.
- Advance a candidate with at least +0.30 percentage points over the new seed-42 M3 control, with no non-finite gradients/metrics. This is a practical compute gate, not a significance test.
- Require a positive mean of C/N changes and an E decrease no greater than 0.30 points before automatic advancement.
- Report head count, peak VRAM, seconds per optimizer step and evaluation time. A candidate taking more than 1.5 times M3 training time needs a larger practical benefit (at least +0.50 dev points) to advance.
- If no candidate passes, do not launch all seeds. Examine attention diagnostics and capacity controls first; stop this head-search round if those controls offer no reproducible gain.
- Confirmation: run only the best candidate and M3 with seeds 2024/3407. Require a mean dev improvement of at least +0.20 points and improvement on at least two of three paired seeds before recommending replacement. These thresholds do not establish statistical significance.
- Select the final candidate using all three dev results before the final test comparison. Never switch to a different architecture because its exploratory test peak is larger.

## Round 2: conditional follow-ups, maximum two seed-42 jobs

**If token alignment wins:** first run a pooled M3 capacity control with `relation_hidden=304` (3,663,155 head parameters, within 0.5% of the alignment head). If alignment fails to beat this capacity control on dev, frame the gain as head capacity until further evidence exists.

**If both conditioned pooling and token alignment pass:** the second optional job combines their relation vectors in one model:

```text
r_global = conditioned pooled relation, dimension 1024
r_local  = token-alignment relation, dimension 1024
r = Dropout(GELU(Linear_256_to_1024(
      Dropout(GELU(Linear_2048_to_256([r_global;r_local]))))))
f = CLS + sigmoid(Wg[CLS;r] + bg) * LN(r)
logits = shared_MSD_classifier(f)
```

This still has one encoder and one classifier. It tests complementary global/local evidence. Do not combine components that failed individually; do not introduce separate expert heads, voting or distillation in this round. If a combined model wins, its component ablations and parameter-matched comparison remain required before a paper claim.

**If only bottleneck widening wins:** test width 512 as the second capacity point; if it does not improve over width 256, retain 256. Do not run the global/local combination in this branch.

**If only conditioned pooling wins:** use the first follow-up for a capacity control, `relation_hidden=230` with original pooling (3,284,201 head parameters), nearly identical to conditioned pooling. Do not add token alignment just because it is available.

## Files and implementation tasks

All paths below are relative to `main_source`. Implement after the architecture proposal is accepted; do not modify compatibility copies of `flat_cafebert.py`.

### Task 1: isolated architecture wrappers and registrations

**Files:** create `gated_dual_ema_msd/models/architecture_search.py`; modify `models/factory.py` and `config/experiments.py`; create `tests/test_architecture_search.py`.

**Interfaces:** `ConditionedPoolCafeBERT` and `TokenAlignmentCafeBERT` accept the common `model_kwargs_for_experiment` kwargs (model name/revision, dropout, label smoothing, separator ID, MSD). New registry definitions retain `use_msd=True`, `use_ema=True` and the three allowed seeds; the screening launcher chooses seed 42 only.

- [x] Write a failing registry test showing the three architecture IDs are unknown. Add a dummy backbone with hidden size 1024, as in `tests/test_paper_recipe.py`, so no download is necessary.
- [x] Implement `ARCH_REL256` via existing `GatedDualCafeBERT(relation_hidden=256)`. Implement `TokenAlignmentCafeBERT` by delegating to `FlatCafeBERT(pool_mode='token_align', alignment_dim=256, gate_bias=-1.0, use_multi_sample_dropout=True)` and forwarding the common kwargs.
- [x] Implement conditioned pooling in the new module: subclass `FlatCafeBERT` with the unchanged gated-dual relation/fusion; replace the two original scorer modules with `Identity` placeholders and override `_pool_gated_dual` to compute the equations above, retaining the parent relation projection and `_residual_fuse`.
- [x] Test invariance to padded/special-token changes, attention mass only on valid segment tokens, finite zero output for empty segments, sensitivity to the opposite segment summary and nonzero gradients through query/key projections.
- [x] Test head counts and `(batch,3)` logits, legal labels, training/eval dropout behavior and parameter gradients. Preserve the existing exact M3 parameter/equation tests.
- [x] Run `python -m unittest discover -s tests -p 'test_*.py'`; commit this task only when it passes.

### Task 2: export evidence that separates architecture behavior from checkpoint noise

**Files:** modify `training/direct.py`; create `tests/test_architecture_history.py`.

**Interfaces:** append one row per dev evaluation to `dev_history.csv`: optimizer step, epoch, current/EMA dev F1 and loss, selected source, selected F1, learning rate, best-step flag and patience counter. Missing EMA before activation is explicit, not substituted silently. Existing result schema fields continue to work.

- [x] Add a two-evaluation dummy-trainer test: pre-EMA diagnostic does not select a checkpoint, EMA rows record both sources, and checkpoint/history steps agree.
- [x] Export the history using values already computed in `_evaluate_pair`; do not add extra forward passes for F1/loss logging.
- [x] At dev-selected checkpoint record valid-token attention entropy, gate channel variability, and `norm(g * LN(r)) / max(norm(CLS), 1e-8)`. Record each statistic's reduction axes so channel/sample means are not confused. Token alignment records each direction's masked attention entropy, not the scalar segment-attention statistic.
- [x] Save dev predictions for each selected source and the three-class confusion matrix. No test inference occurs in this task.
- [x] Run the dummy-trainer history test and all existing tests; commit the export changes.

### Task 3: a dedicated architecture notebook and HF publication path

**Files:** create `scripts/build_vinli_architecture_notebook.py`, `notebooks/vinli_architecture_bf16_seed42.ipynb`; modify `operations/notebook_batch.py` for small additional evidence files and `cli/evaluate.py` for inference-only final evaluation; create `tests/test_architecture_notebook.py` and `tests/test_architecture_final_eval.py`.

**Interfaces:** the generator writes a new notebook configured for the four Round-1 ViNLI seed-42 jobs and frozen recipe. Reuse GitHub authentication, HF token setup, source/manifest binding and sequential process execution from the existing notebook. Do not overwrite `bf16_multiseed_main_method.ipynb` or resume an old run group with changed configuration.

- [x] Write a notebook test that executes its configuration cell and asserts exactly four jobs, all seed 42, ViNLI max length 512, EMA/MSD active, eval 30/patience 50, no W&B, no exploratory test scan and no frozen-final test in screening.
- [x] Add the four registered jobs and a separate confirmation cell containing only M3 plus the selected candidate with seeds 2024/3407. Leave confirmation unexecuted until the dev gate passes.
- [x] Extend HF publication to include `dev_history.csv` and diagnostic metadata with hash/read-back validation at immutable revisions. Notebook summaries include head parameters, per-class dev metrics, timing and VRAM. The HF backend already supports runs without test predictions; preserve that behavior.
- [x] Fix the existing inference-only CLI's split loading: `load_nli_dataset(args.dataset, include_test=(args.split == 'test'))` after its test authorization check. Test with a mocked loader that dev requests never load test and final test requests include it, reconstruct the registered architecture, strictly load the selected checkpoint and call FP32 `evaluate_model` exactly once. Add checkpoint SHA, source/data revision and metrics to a JSON evaluation ledger; include final predictions/ledger in a new immutable HF revision without overwriting the original screening evidence.
- [x] Run all notebook, protocol and HF publication tests; regenerate source hashes; verify clean Git status after commit and push under the user's existing authorization when implementation is complete.

### Task 4: screen, make the decision, confirm the chosen single model

- [ ] Capture code SHA and manifest; verify BF16 CUDA support, HF access and unique output dirs before any training. Use the notebook's supervised sequential children; if changing to detached execution, apply repo launch/cleanup requirements first.
- [ ] Run the four seed-42 jobs; compare candidates using the fixed decision rules and dev outputs. Measure a short initial throughput interval before committing to the complete token-alignment run.
- [ ] Run only the applicable Round-2 branch, maximum two jobs. Record the decision and capacity-control outcome; do not claim language-phenomenon improvements from aggregate labels alone.
- [ ] Run the chosen candidate and M3 on seeds 2024/3407; retain all seeds, not only the best one. If confirmation fails, keep M3 and document the negative finding.
- [ ] Freeze the candidate ID/config/checkpoint selection rule, then run inference-only final evaluation once per dev-selected checkpoint for the paired comparison: `python -m gated_dual_ema_msd.cli.evaluate --dataset vinli --split test --experiment_id M3_FULL --checkpoint <verified-local-checkpoint> --frozen_final --output_csv <unique-final-predictions.csv>`. Substitute the registered candidate ID and its own verified checkpoint for candidate evaluation. The executor resolves the two file paths from the frozen HF download manifest; no training job is relaunched. Report all seed metrics, mean/std, per-class results and prior test exposure; never present a test-peak score as the chosen-model result.

## Budget and completion criteria

Round 1: four jobs. Round 2: zero to two jobs. Confirmation: four jobs. Total: eight to ten training jobs if a candidate advances; stop earlier on failed screening. Final inference uses the same six dev-selected checkpoints, without retraining.

Deliverables: new isolated architecture registrations; a seed-42 notebook; verified HF checkpoints/predictions/history; one decision record identifying the selected single-model architecture or retaining M3. A positive publication claim additionally needs the eventual candidate's component ablations, capacity/compute comparison and cross-dataset evidence. ViANLI/ViMedNLI expansion happens only after the ViNLI signal, using their mandated max lengths.

## Immutable evidence

- [Seed 42 metadata](https://huggingface.co/trinhtrantran122/m3-bf16-clean-testpeak50-hf-only-2026-10-03-vinli-m3-full-seed42/blob/4cff90e50b01b8ff162d923091c9a141d8a8f77c/run_metadata.json)
- [Seed 2024 metadata](https://huggingface.co/trinhtrantran122/m3-bf16-clean-testpeak50-hf-only-2026-10-03-vinli-m3-full-seed2024/blob/3f511d20f29ff0e4b9aafcb08db57ec4eb3eff90/run_metadata.json)
- [Seed 3407 metadata](https://huggingface.co/trinhtrantran122/m3-bf16-clean-testpeak50-hf-only-2026-10-03-vinli-m3-full-seed3407/blob/a49329fe094732f3c85a0dd54adc0eb05894f04c/run_metadata.json)

## Plan self-review

The trials change architecture only; all training values match the actual runs. Existing token alignment is reused rather than recreated. Conditioned attention uses already pair-contextualized tokens. Gate mean is not treated as collapse evidence. Head counts are tested with a 1024-dimensional mock backbone and measured again during Colab preflight. Screening thresholds are practical, not statistical claims. Single-model inference and legacy M3 behavior are preserved. Implementation is verified offline; GPU experiments remain pending.
