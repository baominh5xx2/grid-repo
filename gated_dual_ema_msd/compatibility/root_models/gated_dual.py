"""Gated-Dual Attentive Relation Bottleneck Architecture with Multi-Sample Dropout."""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel

from gated_dual_ema_msd.compatibility.root_models.base import BaseNLIModel
from gated_dual_ema_msd.compatibility.root_models.heads import MultiSampleDropoutHead, StandardClassifierHead
from gated_dual_ema_msd.compatibility.root_models.output import NLIModelOutput
from gated_dual_ema_msd.compatibility.root_models.poolers import AttentiveSegmentPooler


class GatedDualCafeBERT(BaseNLIModel):
    """CafeBERT-Large with Attentive Segment Pooling, Relation Bottleneck (128),
    Learned Residual Gate (bias = -1.0), and Multi-Sample Dropout (MSD).
    """
    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.02,
        relation_hidden: int = 128,
        gate_bias: float = -1.0,
        use_multi_sample_dropout: bool = True,
        msd_dropouts: Optional[List[float]] = None,
        class_weights: Optional[List[float]] = None,
        focal_gamma: float = 0.0,
        sep_token_id: int = 2,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.relation_hidden = relation_hidden
        self.gate_bias = gate_bias
        self.use_multi_sample_dropout = use_multi_sample_dropout
        self.focal_gamma = focal_gamma
        self.sep_token_id = sep_token_id

        weights_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None

        # 1. CafeBERT-Large Backbone
        self.backbone = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )
        self.hidden_size = self.backbone.config.hidden_size  # 1024

        # 2. Attentive Segment Pooler
        self.segment_pooler = AttentiveSegmentPooler(
            hidden_size=self.hidden_size,
            sep_token_id=sep_token_id,
        )

        # 3. Relation Bottleneck: 4096 -> 128 -> 1024
        rel_in_dim = self.hidden_size * 4
        self.relation_projection = nn.Sequential(
            nn.Linear(rel_in_dim, relation_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(relation_hidden, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. Learned Residual Fusion Gate with Gate Bias
        self.fusion_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        nn.init.constant_(self.fusion_gate.bias, gate_bias)
        self.fusion_norm = nn.LayerNorm(self.hidden_size)

        # 5. Classifier Head
        if use_multi_sample_dropout:
            self.head = MultiSampleDropoutHead(
                in_features=self.hidden_size,
                num_labels=num_labels,
                dropout_rates=msd_dropouts,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
                class_weights=weights_tensor,
            )
        else:
            self.head = StandardClassifierHead(
                in_features=self.hidden_size,
                num_labels=num_labels,
                dropout=dropout,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
                class_weights=weights_tensor,
            )

    @property
    def model_type(self) -> str:
        return "gated_dual_msd" if self.use_multi_sample_dropout else "gated_dual"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> NLIModelOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state  # [B, L, 1024]
        h_cls = hidden[:, 0]  # [B, 1024]

        # 1. Segment Pooling
        u, v = self.segment_pooler(hidden, input_ids, attention_mask)

        # 2. Relational Interaction & Bottleneck
        feats = torch.cat([u, v, u * v, (u - v).abs()], dim=-1)  # [B, 4096]
        r = self.relation_projection(feats)  # [B, 1024]

        # 3. Learned Residual Fusion Gate
        gate = torch.sigmoid(self.fusion_gate(torch.cat([h_cls, r], dim=-1)))  # [B, 1024]
        h_fused = h_cls + gate * self.fusion_norm(r)  # [B, 1024]

        # 4. Classification
        logits, loss = self.head(h_fused, labels=labels)

        return NLIModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=h_fused,
            gate_weights=gate,
            relation_vectors=r,
        )
