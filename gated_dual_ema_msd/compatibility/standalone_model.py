"""Gated-Dual Attentive Relation Bottleneck Architecture with Multi-Sample Dropout."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

LABELS = ["E", "C", "N"]
LABEL2ID = {"E": 0, "C": 1, "N": 2}


class GatedDualCafeBERT(nn.Module):
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

        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        )

        # 1. CafeBERT-Large Backbone (562M parameters, hidden=1024)
        self.backbone = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )
        self.hidden_size = self.backbone.config.hidden_size  # 1024

        # 2. Attentive Segment Scorers for Premise and Hypothesis
        self.premise_attention = nn.Linear(self.hidden_size, 1, bias=False)
        self.hypothesis_attention = nn.Linear(self.hidden_size, 1, bias=False)

        # 3. Relation Bottleneck: 4096 -> 128 -> 1024
        # Relational interaction vector: [u, v, u * v, |u - v|] (4 * 1024 = 4096)
        rel_in_dim = self.hidden_size * 4
        self.relation_projection = nn.Sequential(
            nn.Linear(rel_in_dim, relation_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(relation_hidden, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. Learned Residual Fusion Gate with Gate Bias = -1.0
        # g = Sigmoid(Linear([h_cls, r]) - 1.0)
        self.fusion_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        nn.init.constant_(self.fusion_gate.bias, gate_bias)
        self.fusion_norm = nn.LayerNorm(self.hidden_size)

        # 5. Classifier Head & Multi-Sample Dropout (MSD)
        self.dropout = nn.Dropout(dropout)
        if use_multi_sample_dropout:
            drop_rates = msd_dropouts if msd_dropouts is not None else [0.1, 0.2, 0.3, 0.4, 0.5]
            self.msd_layers = nn.ModuleList([nn.Dropout(p) for p in drop_rates])
        else:
            self.msd_layers = None

        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def _segment_masks(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract exact premise and hypothesis masks from <s> P </s></s> H </s> token structure."""
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

    def _attentive_pool(
        self, hidden: torch.Tensor, mask: torch.Tensor, scorer: nn.Module
    ) -> torch.Tensor:
        """Weighted attentive average over token hidden states."""
        logits = scorer(hidden).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1) * mask.to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state  # [B, L, 1024]
        h_cls = hidden[:, 0]  # [B, 1024]

        # 1. Attentive Premise & Hypothesis Pooling
        p_mask, h_mask = self._segment_masks(input_ids, attention_mask)
        u = self._attentive_pool(hidden, p_mask, self.premise_attention)  # [B, 1024]
        v = self._attentive_pool(hidden, h_mask, self.hypothesis_attention)  # [B, 1024]

        # 2. Relational Interaction & Bottleneck
        feats = torch.cat([u, v, u * v, (u - v).abs()], dim=-1)  # [B, 4096]
        r = self.relation_projection(feats)  # [B, 1024]

        # 3. Learned Residual Fusion Gate
        gate = torch.sigmoid(self.fusion_gate(torch.cat([h_cls, r], dim=-1)))  # [B, 1024]
        h_fused = h_cls + gate * self.fusion_norm(r)  # [B, 1024]

        # 4. Multi-Sample Dropout (MSD) Classification
        if self.use_multi_sample_dropout and self.training and self.msd_layers is not None:
            msd_outs = [self.classifier(drop(h_fused)) for drop in self.msd_layers]
            logits = torch.stack(msd_outs, dim=0).mean(dim=0)
        else:
            logits = self.classifier(self.dropout(h_fused))

        loss = None
        if labels is not None:
            if self.use_multi_sample_dropout and self.training and self.msd_layers is not None:
                losses = []
                for drop in self.msd_layers:
                    l_out = self.classifier(drop(h_fused))
                    crit = nn.CrossEntropyLoss(
                        label_smoothing=self.label_smoothing,
                        weight=(self.class_weights.to(l_out.device) if self.class_weights is not None else None),
                    )
                    l_val = crit(l_out, labels)
                    if self.focal_gamma > 0.0:
                        with torch.no_grad():
                            probs = torch.softmax(l_out, dim=-1)
                            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                            factor = ((1.0 - pt) ** self.focal_gamma).clamp(min=1e-5)
                        l_val = l_val * factor.mean()
                    losses.append(l_val)
                loss = torch.stack(losses).mean()
            else:
                crit = nn.CrossEntropyLoss(
                    label_smoothing=self.label_smoothing,
                    weight=(self.class_weights.to(logits.device) if self.class_weights is not None else None),
                )
                loss = crit(logits, labels)
                if self.focal_gamma > 0.0:
                    with torch.no_grad():
                        probs = torch.softmax(logits, dim=-1)
                        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                        factor = ((1.0 - pt) ** self.focal_gamma).clamp(min=1e-5)
                    loss = loss * factor.mean()

        return {"logits": logits, "loss": loss, "hidden": h_fused}
