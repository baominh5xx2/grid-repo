"""Ablation Models for Evaluating Architecture Components."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from gated_dual_ema_msd.compatibility.root_models.base import BaseNLIModel
from gated_dual_ema_msd.compatibility.root_models.heads import StandardClassifierHead
from gated_dual_ema_msd.compatibility.root_models.output import NLIModelOutput
from gated_dual_ema_msd.compatibility.root_models.poolers import AttentiveSegmentPooler, MeanPooler


class GatedDualMeanPoolCafeBERT(BaseNLIModel):
    """A1: Replace attentive segment pooling with uniform mean segment pooling."""
    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.02,
        relation_hidden: int = 128,
        gate_bias: float = -1.0,
        class_weights: Optional[List[float]] = None,
        sep_token_id: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.sep_token_id = sep_token_id
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None

        self.backbone = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        hidden = int(self.backbone.config.hidden_size)
        self.hidden_size = hidden

        self.relation_projection = nn.Sequential(
            nn.Linear(hidden * 4, relation_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(relation_hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_gate = nn.Linear(hidden * 2, hidden)
        nn.init.constant_(self.fusion_gate.bias, gate_bias)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.head = StandardClassifierHead(
            hidden,
            num_labels=num_labels,
            dropout=dropout,
            label_smoothing=label_smoothing,
            class_weights=weights_tensor,
        )

    @property
    def model_type(self) -> str:
        return "ablation_mean_pool"

    def _mean_pool_segment(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)

    def _get_segment_masks(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        valid = attention_mask.bool()
        is_sep = input_ids.eq(self.sep_token_id)
        has_sep = is_sep.any(dim=1)
        first_sep = is_sep.int().argmax(dim=1).clamp(min=1)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        non_special = valid & ~is_sep & positions.gt(0)
        premise = non_special & positions.lt(first_sep.unsqueeze(1))
        hypothesis = non_special & positions.gt(first_sep.unsqueeze(1))
        premise = torch.where(has_sep.unsqueeze(1), premise, non_special)
        hypothesis = torch.where(has_sep.unsqueeze(1), hypothesis, non_special)
        return premise, hypothesis

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> NLIModelOutput:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = outputs.last_hidden_state
        h_cls = hidden[:, 0]
        p_mask, h_mask = self._get_segment_masks(input_ids, attention_mask)
        u = self._mean_pool_segment(hidden, p_mask)
        v = self._mean_pool_segment(hidden, h_mask)
        r = self.relation_projection(torch.cat([u, v, u * v, (u - v).abs()], dim=-1))
        gate = torch.sigmoid(self.fusion_gate(torch.cat([h_cls, r], dim=-1)))
        h_fused = h_cls + gate * self.fusion_norm(r)
        logits, loss = self.head(h_fused, labels=labels)
        return NLIModelOutput(logits=logits, loss=loss, hidden_states=h_fused, gate_weights=gate)


class GatedDualNoInteractionCafeBERT(BaseNLIModel):
    """A2: Feed concatenation [u, v] into relation projection without u*v and |u-v|."""
    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.02,
        relation_hidden: int = 128,
        gate_bias: float = -1.0,
        class_weights: Optional[List[float]] = None,
        sep_token_id: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.backbone = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        hidden = int(self.backbone.config.hidden_size)
        self.pooler = AttentiveSegmentPooler(hidden, sep_token_id=sep_token_id)
        self.relation_projection = nn.Sequential(
            nn.Linear(hidden * 2, relation_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(relation_hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_gate = nn.Linear(hidden * 2, hidden)
        nn.init.constant_(self.fusion_gate.bias, gate_bias)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.head = StandardClassifierHead(
            hidden,
            num_labels=num_labels,
            dropout=dropout,
            label_smoothing=label_smoothing,
            class_weights=weights_tensor,
        )

    @property
    def model_type(self) -> str:
        return "ablation_no_interaction"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> NLIModelOutput:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = outputs.last_hidden_state
        h_cls = hidden[:, 0]
        u, v = self.pooler(hidden, input_ids, attention_mask)
        r = self.relation_projection(torch.cat([u, v], dim=-1))
        gate = torch.sigmoid(self.fusion_gate(torch.cat([h_cls, r], dim=-1)))
        h_fused = h_cls + gate * self.fusion_norm(r)
        logits, loss = self.head(h_fused, labels=labels)
        return NLIModelOutput(logits=logits, loss=loss, hidden_states=h_fused, gate_weights=gate)


class GatedDualNoGateCafeBERT(BaseNLIModel):
    """A4: Unconditional residual fusion h_cls + LayerNorm(r)."""
    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.02,
        relation_hidden: int = 128,
        class_weights: Optional[List[float]] = None,
        sep_token_id: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.backbone = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        hidden = int(self.backbone.config.hidden_size)
        self.pooler = AttentiveSegmentPooler(hidden, sep_token_id=sep_token_id)
        self.relation_projection = nn.Sequential(
            nn.Linear(hidden * 4, relation_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(relation_hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_norm = nn.LayerNorm(hidden)
        self.head = StandardClassifierHead(
            hidden,
            num_labels=num_labels,
            dropout=dropout,
            label_smoothing=label_smoothing,
            class_weights=weights_tensor,
        )

    @property
    def model_type(self) -> str:
        return "ablation_no_gate"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> NLIModelOutput:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = outputs.last_hidden_state
        h_cls = hidden[:, 0]
        u, v = self.pooler(hidden, input_ids, attention_mask)
        r = self.relation_projection(torch.cat([u, v, u * v, (u - v).abs()], dim=-1))
        h_fused = h_cls + self.fusion_norm(r)
        logits, loss = self.head(h_fused, labels=labels)
        return NLIModelOutput(logits=logits, loss=loss, hidden_states=h_fused)


class GatedDualNoBnCafeBERT(BaseNLIModel):
    """A3: Wide single-layer projection without bottleneck."""
    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.02,
        gate_bias: float = -1.0,
        class_weights: Optional[List[float]] = None,
        sep_token_id: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.backbone = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        hidden = int(self.backbone.config.hidden_size)
        self.pooler = AttentiveSegmentPooler(hidden, sep_token_id=sep_token_id)
        self.relation_projection = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_gate = nn.Linear(hidden * 2, hidden)
        nn.init.constant_(self.fusion_gate.bias, gate_bias)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.head = StandardClassifierHead(
            hidden,
            num_labels=num_labels,
            dropout=dropout,
            label_smoothing=label_smoothing,
            class_weights=weights_tensor,
        )

    @property
    def model_type(self) -> str:
        return "ablation_no_bottleneck"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> NLIModelOutput:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = outputs.last_hidden_state
        h_cls = hidden[:, 0]
        u, v = self.pooler(hidden, input_ids, attention_mask)
        r = self.relation_projection(torch.cat([u, v, u * v, (u - v).abs()], dim=-1))
        gate = torch.sigmoid(self.fusion_gate(torch.cat([h_cls, r], dim=-1)))
        h_fused = h_cls + gate * self.fusion_norm(r)
        logits, loss = self.head(h_fused, labels=labels)
        return NLIModelOutput(logits=logits, loss=loss, hidden_states=h_fused, gate_weights=gate)
