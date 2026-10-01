"""Architecture-only ablations for the gated relation-aware CafeBERT head.

All variants use the same FlatCafeBERT execution path as the full method and
change exactly one registered component. MSD remains configurable so the
experiment registry can force it off for A1--A4 as required by the paper spec.
"""
from __future__ import annotations

from typing import List, Optional

from gated_dual_ema_msd.models.flat_cafebert import FlatCafeBERT


class _RelationAblationBase(FlatCafeBERT):
    """Shared constructor that keeps every non-ablated setting identical."""

    variant_name = "relation_ablation"

    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.02,
        relation_hidden: Optional[int] = 128,
        gate_bias: float = -1.0,
        use_multi_sample_dropout: bool = False,
        msd_dropouts: Optional[List[float]] = None,
        class_weights: Optional[List[float]] = None,
        sep_token_id: int = 2,
        segment_pooling: str = "attentive",
        relation_features_mode: str = "standard",
        use_gate: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            fallback_models=[],
            dropout=dropout,
            num_labels=num_labels,
            label_smoothing=label_smoothing,
            revision=revision,
            class_weights=class_weights,
            pool_mode="gated_dual",
            segment_pooling=segment_pooling,
            relation_hidden=relation_hidden,
            gate_bias=gate_bias,
            head_architecture="standard",
            relation_features_mode=relation_features_mode,
            use_gate=use_gate,
            use_multi_sample_dropout=use_multi_sample_dropout,
            msd_dropouts=msd_dropouts or [0.1, 0.2, 0.3, 0.4, 0.5],
            classifier_type="linear",
            classifier_hidden=256,
            sep_token_id=sep_token_id,
        )

    @property
    def model_type(self) -> str:
        return self.variant_name


class GatedDualMeanPoolCafeBERT(_RelationAblationBase):
    """A1: replace learned segment attention with masked segment means."""

    variant_name = "gated_dual_mean_pool"

    def __init__(self, **kwargs):
        super().__init__(segment_pooling="mean", **kwargs)


class GatedDualNoInteractionCafeBERT(_RelationAblationBase):
    """A2: use [u,v] rather than [u,v,u*v,|u-v|]."""

    variant_name = "gated_dual_no_interaction"

    def __init__(self, **kwargs):
        super().__init__(relation_features_mode="concat", **kwargs)


class GatedDualNoBnCafeBERT(_RelationAblationBase):
    """A3: replace 4d->128->d with the direct 4d->d projection."""

    variant_name = "gated_dual_no_bottleneck"

    def __init__(self, **kwargs):
        kwargs.pop("relation_hidden", None)
        super().__init__(relation_hidden=None, **kwargs)


class GatedDualNoGateCafeBERT(_RelationAblationBase):
    """A4: use unconditional h_cls + LayerNorm(r) residual fusion."""

    variant_name = "gated_dual_no_gate"

    def __init__(self, **kwargs):
        kwargs.pop("use_gate", None)
        super().__init__(use_gate=False, **kwargs)
