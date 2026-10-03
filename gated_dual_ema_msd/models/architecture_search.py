"""Isolated single-model heads for the ViNLI architecture screening round."""
from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from gated_dual_ema_msd.models.flat_cafebert import FlatCafeBERT


class ConditionedPoolCafeBERT(FlatCafeBERT):
    """Pool each segment using a query from the opposite segment's mean."""

    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.02,
        attention_dim: int = 128,
        gate_bias: float = -1.0,
        use_multi_sample_dropout: bool = True,
        msd_dropouts: Optional[List[float]] = None,
        class_weights: Optional[List[float]] = None,
        sep_token_id: int = 2,
    ):
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        super().__init__(
            model_name=model_name, fallback_models=[], revision=revision,
            num_labels=num_labels, dropout=dropout, label_smoothing=label_smoothing,
            class_weights=class_weights, sep_token_id=sep_token_id,
            pool_mode="gated_dual", segment_pooling="attentive", relation_hidden=128,
            gate_bias=gate_bias, use_multi_sample_dropout=use_multi_sample_dropout,
            msd_dropouts=msd_dropouts if msd_dropouts is not None else [0.1, 0.2, 0.3, 0.4, 0.5],
        )
        # Remove the original scalar scorers without changing the M3 relation,
        # residual-fusion or shared-classifier modules.
        self.premise_attention = nn.Identity()
        self.hypothesis_attention = nn.Identity()
        self.attention_dim = attention_dim
        self.premise_query = nn.Linear(self.hidden_size, attention_dim, bias=False)
        self.hypothesis_query = nn.Linear(self.hidden_size, attention_dim, bias=False)
        self.premise_key = nn.Linear(self.hidden_size, attention_dim, bias=False)
        self.hypothesis_key = nn.Linear(self.hidden_size, attention_dim, bias=False)

    @property
    def model_type(self) -> str:
        return "conditioned_pool_msd"

    def _conditioned_attention_pool(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        opposite_summary: torch.Tensor,
        key_projection: nn.Linear,
        query_projection: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Queries, keys, scores and normalization stay FP32 under BF16/FP16.
        # Mask before projecting to keep excluded token values out of arithmetic.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            valid_hidden = hidden.float().masked_fill(~mask.unsqueeze(-1), 0.0)
            keys = F.linear(valid_hidden, key_projection.weight.float())
            query = F.linear(opposite_summary.float(), query_projection.weight.float())
            scores = (keys * query.unsqueeze(1)).sum(-1) / math.sqrt(self.attention_dim)
            scores = scores.masked_fill(~mask, torch.finfo(torch.float32).min)
            weights = torch.softmax(scores, dim=1) * mask.float()
            weights = weights / weights.sum(1, keepdim=True).clamp(min=1e-8)
            pooled = torch.bmm(weights.unsqueeze(1), valid_hidden).squeeze(1)
        return pooled, weights

    def _pool_gated_dual(
        self, hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        premise_mask, hypothesis_mask = self._segment_masks(input_ids, attention_mask)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            premise_mean = self._masked_mean(
                hidden.float().masked_fill(~premise_mask.unsqueeze(-1), 0.0), premise_mask
            )
            hypothesis_mean = self._masked_mean(
                hidden.float().masked_fill(~hypothesis_mask.unsqueeze(-1), 0.0), hypothesis_mask
            )
        premise, premise_weights = self._conditioned_attention_pool(
            hidden, premise_mask, hypothesis_mean, self.premise_key, self.hypothesis_query
        )
        hypothesis, hypothesis_weights = self._conditioned_attention_pool(
            hidden, hypothesis_mask, premise_mean, self.hypothesis_key, self.premise_query
        )
        self.last_premise_attention = premise_weights.detach()
        self.last_hypothesis_attention = hypothesis_weights.detach()
        self._record_pool_diagnostics(
            premise_weights, hypothesis_weights, premise_mask, hypothesis_mask
        )
        relation_features = torch.cat(
            [premise, hypothesis, premise * hypothesis, (premise - hypothesis).abs()], dim=-1
        )
        relation = self.relation_projection(relation_features)
        self.last_relation = relation
        return self._residual_fuse(hidden[:, 0], relation)


class TokenAlignmentCafeBERT(FlatCafeBERT):
    """Expose the existing 256-dimensional token alignment with the M3 recipe."""

    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.02,
        alignment_dim: int = 256,
        gate_bias: float = -1.0,
        use_multi_sample_dropout: bool = True,
        msd_dropouts: Optional[List[float]] = None,
        class_weights: Optional[List[float]] = None,
        sep_token_id: int = 2,
    ):
        super().__init__(
            model_name=model_name, fallback_models=[], revision=revision,
            num_labels=num_labels, dropout=dropout, label_smoothing=label_smoothing,
            class_weights=class_weights, sep_token_id=sep_token_id,
            pool_mode="token_align", alignment_dim=alignment_dim, gate_bias=gate_bias,
            use_multi_sample_dropout=use_multi_sample_dropout,
            msd_dropouts=msd_dropouts if msd_dropouts is not None else [0.1, 0.2, 0.3, 0.4, 0.5],
        )

    @property
    def model_type(self) -> str:
        return "token_alignment_msd"
