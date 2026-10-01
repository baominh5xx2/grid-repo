"""Modular Pooling Strategies for NLI Token Sequences."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BasePooler(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class ClsPooler(BasePooler):
    """Standard Baseline: extracts <s> / [CLS] token at position 0."""
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states[:, 0]


class MeanPooler(BasePooler):
    """Baseline: Average over all active non-padding tokens."""
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        sum_hidden = (hidden_states * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-8)
        return sum_hidden / sum_mask


class AttentiveSegmentPooler(nn.Module):
    """Attentive Segment Pooling for Premise and Hypothesis separation."""
    def __init__(self, hidden_size: int = 1024, sep_token_id: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.sep_token_id = sep_token_id
        self.premise_scorer = nn.Linear(hidden_size, 1, bias=False)
        self.hypothesis_scorer = nn.Linear(hidden_size, 1, bias=False)

    def _get_segment_masks(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _pool_segment(
        self, hidden: torch.Tensor, mask: torch.Tensor, scorer: nn.Module
    ) -> torch.Tensor:
        logits = scorer(hidden).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1) * mask.to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p_mask, h_mask = self._get_segment_masks(input_ids, attention_mask)
        u = self._pool_segment(hidden_states, p_mask, self.premise_scorer)
        v = self._pool_segment(hidden_states, h_mask, self.hypothesis_scorer)
        return u, v
