"""Clean Baseline Implementations for Benchmarking & Ablation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from gated_dual_ema_msd.models.base import BaseNLIModel
from gated_dual_ema_msd.models.heads import MultiSampleDropoutHead, StandardClassifierHead
from gated_dual_ema_msd.models.poolers import (
    AttentiveSegmentPooler,
    ClsPooler,
    MeanPooler,
)


def relation_head_parameter_count(
    hidden_size: int, relation_hidden: int = 128, num_labels: int = 3
) -> int:
    """Parameter count of the canonical M0 relation head (backbone excluded)."""
    d = int(hidden_size)
    b = int(relation_hidden)
    c = int(num_labels)
    # Two bias-free attention scorers, 4d->b->d relation projection,
    # 2d->d gate, LayerNorm(d), and d->c classifier.
    return 2 * d + (4 * d * b + b) + (b * d + d) + (2 * d * d + d) + 2 * d + (d * c + c)


def parameter_matched_width(
    hidden_size: int, relation_hidden: int = 128, num_labels: int = 3
) -> int:
    """Choose the closest CLS-MLP width to the canonical relation-head size."""
    d = int(hidden_size)
    c = int(num_labels)
    target = relation_head_parameter_count(d, relation_hidden, c)
    # CLS MLP: d->w->d, LayerNorm(d), and d->c classifier.
    fixed = d + 2 * d + d * c + c
    return max(1, round((target - fixed) / (2 * d + 1)))


class VanillaCafeBERT(BaseNLIModel):
    """Standard Baseline (B1): Fine-tuning CafeBERT with [CLS] token classification."""
    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.02,
        class_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None

        self.backbone = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        self.pooler = ClsPooler()
        self.head = StandardClassifierHead(self.backbone.config.hidden_size, num_labels=num_labels, dropout=dropout)

    @property
    def model_type(self) -> str:
        return "vanilla_cls"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        h = self.pooler(outputs.last_hidden_state, input_ids, attention_mask)
        logits, loss = self.head(h, labels=labels, label_smoothing=self.label_smoothing, class_weights=self.class_weights)
        return {"logits": logits, "loss": loss, "hidden": h}


class MeanPoolCafeBERT(BaseNLIModel):
    """Ablation Baseline (B2): Fine-tuning CafeBERT with mean sequence pooling."""
    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        self.pooler = MeanPooler()
        self.head = StandardClassifierHead(self.backbone.config.hidden_size, num_labels=num_labels, dropout=dropout)
        self.label_smoothing = label_smoothing

    @property
    def model_type(self) -> str:
        return "mean_pool"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        h = self.pooler(outputs.last_hidden_state, input_ids, attention_mask)
        logits, loss = self.head(h, labels=labels, label_smoothing=self.label_smoothing)
        return {"logits": logits, "loss": loss, "hidden": h}


class VanillaMSDCafeBERT(BaseNLIModel):
    """Baseline (B4) & Ablation (A5): CafeBERT [CLS] with Multi-Sample Dropout head (no relation branch)."""
    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        label_smoothing: float = 0.02,
        msd_dropouts: Optional[List[float]] = None,
        class_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None

        self.backbone = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        self.pooler = ClsPooler()
        self.head = MultiSampleDropoutHead(
            in_features=self.backbone.config.hidden_size,
            num_labels=num_labels,
            dropout_rates=msd_dropouts or [0.1, 0.2, 0.3, 0.4, 0.5],
        )

    @property
    def model_type(self) -> str:
        return "vanilla_msd"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        h = self.pooler(outputs.last_hidden_state, input_ids, attention_mask)
        logits, loss = self.head(
            h,
            labels=labels,
            label_smoothing=self.label_smoothing,
            class_weights=self.class_weights,
        )
        return {"logits": logits, "loss": loss, "hidden": h}


class ParameterMatchedCLSCafeBERT(BaseNLIModel):
    """B1: CLS-only MLP with a head parameter count matched to M0.

    For CafeBERT-Large (hidden size 1024) and a 128-d relation bottleneck,
    the automatically selected width is 1345; its head differs from M0 by
    only 193 parameters while using no premise/hypothesis-specific features.
    """

    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.02,
        relation_hidden: int = 128,
        matched_hidden: Optional[int] = None,
        class_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.backbone = AutoModel.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )
        hidden = int(self.backbone.config.hidden_size)
        self.matched_hidden = matched_hidden or parameter_matched_width(
            hidden, relation_hidden=relation_hidden, num_labels=num_labels
        )
        self.pooler = ClsPooler()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, self.matched_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.matched_hidden, hidden),
        )
        self.fusion_norm = nn.LayerNorm(hidden)
        self.head = StandardClassifierHead(hidden, num_labels=num_labels, dropout=dropout)

    @property
    def model_type(self) -> str:
        return "parameter_matched_cls"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        h_cls = self.pooler(outputs.last_hidden_state, input_ids, attention_mask)
        h = h_cls + self.fusion_norm(self.mlp(h_cls))
        logits, loss = self.head(
            h,
            labels=labels,
            label_smoothing=self.label_smoothing,
            class_weights=self.class_weights,
        )
        return {"logits": logits, "loss": loss, "hidden": h}


class SimpleRelationCafeBERT(BaseNLIModel):
    """B2: explicit relation features with direct projection and no gate.

    This implements the specification exactly: attentive P/H pooling,
    [u,v,u*v,|u-v|], Linear(4d,d)+GELU, then LayerNorm(h_cls+r).
    """

    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.02,
        class_weights: Optional[List[float]] = None,
        sep_token_id: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.backbone = AutoModel.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )
        hidden = int(self.backbone.config.hidden_size)
        self.pooler = AttentiveSegmentPooler(hidden, sep_token_id=sep_token_id)
        self.relation_projection = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_norm = nn.LayerNorm(hidden)
        self.head = StandardClassifierHead(hidden, num_labels=num_labels, dropout=dropout)
        self.last_gate = None

    @property
    def model_type(self) -> str:
        return "simple_relation"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state
        h_cls = hidden[:, 0]
        u, v = self.pooler(hidden, input_ids, attention_mask)
        relation = self.relation_projection(
            torch.cat([u, v, u * v, (u - v).abs()], dim=-1)
        )
        self.last_relation = relation
        h = self.fusion_norm(h_cls + relation)
        logits, loss = self.head(
            h,
            labels=labels,
            label_smoothing=self.label_smoothing,
            class_weights=self.class_weights,
        )
        return {"logits": logits, "loss": loss, "hidden": h}
