# M3 / GRIP BF16 configuration audit

Reviewed 2026-10-02 against the user-provided 9-page paper,
"GRIP: A Controlled Study of Explicit Segment Comparison for Vietnamese Natural Language Inference".
Source PDF SHA-256: `aae48e3a0e3473f48a41c9aa55573952c3fd453508dd7f707d2ccf92c333e586`. The PDF is not redistributed in this source repo.

M3 is the highest observed mean Macro-F1 configuration within the paper's controlled
study (Tables 3-4). It does not exceed every published result: Table 2 gives M3
82.59 versus NLIMoE Dynamic 82.83 on ViNLI. No new benchmark result is claimed here.

| Setting | Paper | Notebook / implementation |
| --- | --- | --- |
| Scope | Separate model per dataset, no mixing | M3_FULL only, 3 datasets x 3 seeds = 9 independent runs |
| Backbone | uitnlp/CafeBERT | Same, immutable revision af76fcf2a04096b2b54b348a3e4eb48253c93c5d |
| Segment pools | Separate bias-free learned attention | Matches for native XLM-R paired input |
| Relation features | [u; v; u*v; abs(u-v)] | Matches |
| Relation map | 4096 -> 128 -> 1024; GELU + dropout 0.1 after both maps | Matches; no LayerNorm in map |
| Fusion | c + sigmoid(Wg[c;r] + bg) * LN(r) | Matches; gate reads unnormalized r |
| Trainable head | 2,761,859 parameters | Verified offline at hidden dimension 1024 |
| MSD | 5 paths, probabilities 0.1/0.2/0.3/0.4/0.5; mean CE loss | Matches, one shared linear classifier |
| Label smoothing | 0.02 | Matches |
| EMA | All trainable parameters, initialized from model; decay 0.992 from optimizer step 100 | Matches; pre-100 dev checks are diagnostics only, not selection candidates |
| Max lengths | ViANLI / ViNLI 512; ViMedNLI 256 | Matches in every training/evaluation path |
| Optimizer | AdamW; LR 1e-5; weight decay 0.005; linear schedule; warmup 0.06 | Matches; gradients clipped at existing 1.0 default |
| Batch | Physical 4, accumulation 4, effective 16 | Matches |
| Epoch cap / patience | 7 epochs / 5 dev evaluations | Matches; 50-step cadence now means 250 steps without improvement instead of 500 |
| Seeds | 42, 2024, 3407 | Matches |
| Train / eval precision | FP16 / FP32 (Table 6) | BF16 / FP32, explicitly requested; loss scaling disabled |
| Evaluation interval | 100 optimizer steps (Table 6) | 50, explicitly requested |
| Train counts ViANLI | 8,012 | Approved cleaned split 8,010; remove 2 normalized exact-pair train/holdout overlaps |
| Train counts ViMedNLI | 11,232 | Approved cleaned split 11,217; remove 15 conflicting-label train rows |
| Train/dev/test ViNLI | 18,282 / 2,255 / 2,264; exclude Other | Matches three-label pinned mirror |
| Dev/test ViANLI | 1,000 / 1,000 | Preserved |
| Dev/test ViMedNLI | 1,395 / 1,422 | Preserved |
| Paper test protocol | Select on dev, test once after selection | Available via FROZEN_FINAL=True with exploratory mode off |
| Latest user-selected mode | Not part of paper protocol | TEST_PEAK_EXPLORATORY=True: scan EMA test every 50 steps from step 100 and retain its peak separately |

## Encoder figure discrepancy

Figure 1 labels CafeBERT as a 12-layer encoder. The actual immutable model
[config](https://huggingface.co/uitnlp/CafeBERT/blob/af76fcf2a04096b2b54b348a3e4eb48253c93c5d/config.json)
uses **24 layers, hidden size 1024, 16 attention heads**. The code loads this actual
checkpoint; the paper figure should be corrected if revised. Special-token masks
exclude the initial position, separator IDs and padding by attention mask; the code
does not dynamically exclude every tokenizer-specific special ID, unlike the broader
wording of Appendix B. This matters if special literals occur inside segment content.

## Current run protocol and provenance

The latest explicit user request selects test-aware exploration. Dev Macro-F1
continues to select `best_model.pt` and control early stopping. `best_test_model.pt`
and `test_predictions_peak.csv` hold the exploratory test maximum; `test_curve.csv`
records its optimizer-step history. The dev-selected checkpoint is also tested at
the end. All test evaluations are counted. The scan does not alter optimizer updates.

Peak-test scores are labelled exploratory in CLI results, the batch manifest,
HF metadata and `exploratory_test_summary.csv`. They are not an independent
held-out test reproduction and must not be described as such. Test-aware results
remain test-aware even if later re-evaluated once on the same test set.

For the independent frozen-final code path, set `TEST_PEAK_EXPLORATORY=False`,
`FROZEN_FINAL=True`, and a fresh `RUN_GROUP`. That path evaluates test exactly once
per dev-selected checkpoint and rejects test-peak metadata. The cleaned data,
BF16 precision and 50-step eval cadence still differ from the paper recipe.

GitHub stores code with a frozen commit SHA. The user subsequently removed W&B;
the notebook and publication pipeline now require only HF. HF stores
the selected weights plus exploratory peak weights (when enabled), tokenizer,
config and predictions at an immutable revision with complete file read-back.
Drive stores resume state and tables. Existing batches refuse changed source/config.

## GitHub / Colab access

The notebook reads `GITHUB_TOKEN` from Colab Secrets, environment, or a hidden
prompt before cloning. A fine-grained token needs only the `grid-repo` repository
and Contents: Read-only. [GitHub token setup](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
No token is embedded in clone URLs, notebook output, Git config or saved manifests.
Public access works without a token; private access is checked with the supplied
credential. Real user-token authentication remains to be verified when provided.

## Validation limits

Offline checks cover all model architectures/BF16 arithmetic, M3 head size and
fusion equation, EMA initialization/update/selection, test lock, exactly-once final
evaluation, exploratory peak capture, clean notebook syntax, synthetic credential
handling, immutable artifact read-back and protocol-preserving resume. No real GPU
training, user-token GitHub clone, or new W&B/HF experiment upload has been run.
