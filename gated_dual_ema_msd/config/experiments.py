"""Canonical registry for the 11 baseline/ablation configurations.

This module is the single source of truth shared by the single-run CLI and the
multi-seed matrix launcher. It contains configuration only; importing it never
loads CafeBERT or touches a dataset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Tuple

MODEL_NAME = "uitnlp/CafeBERT"
MODEL_REVISION = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d"
MSD_DROPOUTS = (0.1, 0.2, 0.3, 0.4, 0.5)
EMA_DECAY = 0.992
EMA_START_STEP = 100
DATASET_MAX_LENGTHS: Mapping[str, int] = {
    "vinli": 512,
    "vianli": 512,
    "vimednli": 256,
}
MAIN_SEEDS = (42, 2024, 3407)
SINGLE_SEED = (42,)


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    description: str
    model_type: str
    use_msd: bool = False
    use_ema: bool = False
    seeds: Tuple[int, ...] = MAIN_SEEDS
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    family: str = "ablation"

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["seeds"] = list(self.seeds)
        value["model_kwargs"] = dict(self.model_kwargs)
        return value


EXPERIMENTS: Dict[str, ExperimentDefinition] = {
    "B0_CLS": ExperimentDefinition(
        "B0_CLS",
        "Standard CafeBERT CLS + dropout + linear classifier",
        "vanilla",
        family="baseline",
    ),
    "B1_PARAM_MATCHED_CLS": ExperimentDefinition(
        "B1_PARAM_MATCHED_CLS",
        "CLS-only residual MLP parameter-matched to the M0 relation head",
        "parameter_matched_cls",
        model_kwargs={"relation_hidden": 128},
        family="baseline",
    ),
    "B2_SIMPLE_RELATION": ExperimentDefinition(
        "B2_SIMPLE_RELATION",
        "Attentive four-way relation features with direct projection and no gate",
        "simple_relation",
        family="baseline",
    ),
    "M0_RELATION_GATE": ExperimentDefinition(
        "M0_RELATION_GATE",
        "Attentive relation bottleneck and learned residual gate",
        "gated_dual",
        seeds=MAIN_SEEDS,
        model_kwargs={"relation_hidden": 128, "gate_bias": -1.0},
        family="main",
    ),
    "M1_RELATION_MSD": ExperimentDefinition(
        "M1_RELATION_MSD",
        "M0 with five-path Multi-Sample Dropout",
        "gated_dual",
        use_msd=True,
        seeds=MAIN_SEEDS,
        model_kwargs={"relation_hidden": 128, "gate_bias": -1.0},
        family="main",
    ),
    "M2_RELATION_EMA": ExperimentDefinition(
        "M2_RELATION_EMA",
        "M0 with parameter EMA",
        "gated_dual",
        use_ema=True,
        seeds=MAIN_SEEDS,
        model_kwargs={"relation_hidden": 128, "gate_bias": -1.0},
        family="main",
    ),
    "M3_FULL": ExperimentDefinition(
        "M3_FULL",
        "M0 with five-path MSD and parameter EMA",
        "gated_dual",
        use_msd=True,
        use_ema=True,
        seeds=MAIN_SEEDS,
        model_kwargs={"relation_hidden": 128, "gate_bias": -1.0},
        family="main",
    ),
    "A1_MEAN_POOL": ExperimentDefinition(
        "A1_MEAN_POOL",
        "M0 with masked segment means instead of attentive pooling",
        "gated_dual_mean_pool",
        model_kwargs={"relation_hidden": 128, "gate_bias": -1.0},
    ),
    "A2_NO_INTERACTION": ExperimentDefinition(
        "A2_NO_INTERACTION",
        "M0 with [u,v] only (no product or absolute-difference operators)",
        "gated_dual_no_interaction",
        model_kwargs={"relation_hidden": 128, "gate_bias": -1.0},
    ),
    "A3_NO_BOTTLENECK": ExperimentDefinition(
        "A3_NO_BOTTLENECK",
        "M0 with direct 4096-to-1024 relation projection",
        "gated_dual_no_bottleneck",
        model_kwargs={"gate_bias": -1.0},
    ),
    "A4_NO_GATE": ExperimentDefinition(
        "A4_NO_GATE",
        "M0 with unconditional residual relation addition",
        "gated_dual_no_gate",
        model_kwargs={"relation_hidden": 128},
    ),
}

SIGNAL_EXPERIMENTS = ("B0_CLS", "M0_RELATION_GATE", "M3_FULL")
MAIN_EXPERIMENTS = tuple(k for k, v in EXPERIMENTS.items() if v.family == "main")
SINGLE_SEED_EXPERIMENTS = tuple(k for k, v in EXPERIMENTS.items() if v.seeds == SINGLE_SEED)
CANONICAL_EXPERIMENTS = tuple(EXPERIMENTS)

# Explicit opt-in candidates; default legacy matrices retain their 11 methods.
ARCHITECTURE_EXPERIMENTS: Dict[str, ExperimentDefinition] = {
    "ARCH_REL256": ExperimentDefinition(
        "ARCH_REL256", "M3 with relation bottleneck width 256", "gated_dual_rel256",
        use_msd=True, use_ema=True, seeds=MAIN_SEEDS, family="architecture",
        model_kwargs={"relation_hidden": 256, "gate_bias": -1.0},
    ),
    "ARCH_CONDPOOL128": ExperimentDefinition(
        "ARCH_CONDPOOL128", "M3 with opposite-segment-conditioned pooling", "conditioned_pool",
        use_msd=True, use_ema=True, seeds=MAIN_SEEDS, family="architecture",
        model_kwargs={"attention_dim": 128, "gate_bias": -1.0},
    ),
    "ARCH_ALIGN256": ExperimentDefinition(
        "ARCH_ALIGN256", "Bidirectional token alignment with M3 MSD and EMA", "token_alignment",
        use_msd=True, use_ema=True, seeds=MAIN_SEEDS, family="architecture",
        model_kwargs={"alignment_dim": 256, "gate_bias": -1.0},
    ),
    "ARCH_REL230": ExperimentDefinition(
        "ARCH_REL230", "M3 capacity control for conditioned pooling", "gated_dual_rel230",
        use_msd=True, use_ema=True, seeds=MAIN_SEEDS, family="architecture",
        model_kwargs={"relation_hidden": 230, "gate_bias": -1.0},
    ),
    "ARCH_REL304": ExperimentDefinition(
        "ARCH_REL304", "M3 capacity control for token alignment", "gated_dual_rel304",
        use_msd=True, use_ema=True, seeds=MAIN_SEEDS, family="architecture",
        model_kwargs={"relation_hidden": 304, "gate_bias": -1.0},
    ),
    "ARCH_REL512": ExperimentDefinition(
        "ARCH_REL512", "M3 conditional relation-width follow-up", "gated_dual_rel512",
        use_msd=True, use_ema=True, seeds=MAIN_SEEDS, family="architecture",
        model_kwargs={"relation_hidden": 512, "gate_bias": -1.0},
    ),
}
EXPERIMENTS.update(ARCHITECTURE_EXPERIMENTS)


def normalize_experiment_id(experiment_id: str) -> str:
    key = str(experiment_id).strip().upper()
    if key not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment_id={experiment_id!r}. "
            f"Expected one of: {', '.join(EXPERIMENTS)}"
        )
    return key


def get_experiment(experiment_id: str) -> ExperimentDefinition:
    return EXPERIMENTS[normalize_experiment_id(experiment_id)]


def model_kwargs_for_experiment(
    experiment: ExperimentDefinition,
    sep_token_id: int,
    *,
    model_name: Optional[str] = None,
    revision: Optional[str] = None,
    label_smoothing: float = 0.02,
    dropout: float = 0.1,
) -> Dict[str, Any]:
    chosen_model = model_name or MODEL_NAME
    # Default to pinned revision only if using the default CafeBERT model
    if revision is not None:
        chosen_rev = revision
    elif chosen_model == MODEL_NAME:
        chosen_rev = MODEL_REVISION
    else:
        chosen_rev = None

    kwargs: Dict[str, Any] = {
        "model_name": chosen_model,
        "revision": chosen_rev,
        "num_labels": 3,
        "dropout": dropout,
        "label_smoothing": label_smoothing,
        "sep_token_id": sep_token_id,
        "use_multi_sample_dropout": experiment.use_msd,
        "msd_dropouts": list(MSD_DROPOUTS),
    }
    kwargs.update(dict(experiment.model_kwargs))
    return kwargs


def iter_matrix(
    datasets: Iterable[str] = DATASET_MAX_LENGTHS,
    experiment_ids: Iterable[str] = CANONICAL_EXPERIMENTS,
) -> List[Tuple[str, str, int]]:
    """Return deterministic (dataset, experiment_id, seed) jobs.

    Seed 42 jobs always precede robustness seeds, enforcing the feasibility
    pilot before multi-seed work when callers iterate this list in order.
    """
    normalized_datasets = [str(d).lower() for d in datasets]
    for dataset in normalized_datasets:
        if dataset not in DATASET_MAX_LENGTHS:
            raise ValueError(f"Unknown dataset={dataset!r}")
    normalized_experiments = [normalize_experiment_id(e) for e in experiment_ids]

    jobs: List[Tuple[str, str, int]] = []
    for seed_group in ((42,), (2024, 3407)):
        for dataset in normalized_datasets:
            for experiment_id in normalized_experiments:
                spec = EXPERIMENTS[experiment_id]
                for seed in seed_group:
                    if seed in spec.seeds:
                        jobs.append((dataset, experiment_id, seed))
    return jobs
