"""LoRA for EXP-001-R2 Stage 2 (Hu et al. 2021, arXiv:2106.09685v2; microsoft/LoRA).

Controlled mapping for M39:
- Applied ONLY in Stage 2, after reloading the exact Stage-1 best full state.
- Target scope: every self-attention ``query`` and ``value`` projection in the
  CafeBERT backbone (XLM-R-large geometry: 24 layers -> 48 modules).
- RoBERTa-large paper values: r=8, alpha=16, LoRA-path dropout not reported
  (official loralib.Linear default is 0.0), scaling = alpha / r = 2.
- Init: Kaiming-uniform A, zero B (official code; initial adapter is a no-op).
- Frozen Stage-1 backbone weights remain the W0 reference; the gated task head
  stays fully trainable (official run_glue trains everything outside the
  frozen ``roberta`` prefix, i.e. the task head).

Exactly ONE method per run; never combined with Mixout/RecAdam/SAM.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    """Linear layer with a low-rank adapter: y = x W^T + (x A^T) B^T * (alpha/r).

    The wrapped pretrained weight is frozen; only A and B are trainable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        r: int,
        alpha: float,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        if not isinstance(weight, torch.Tensor):
            raise TypeError("LoRALinear requires an explicit pretrained weight tensor")
        self.in_features = in_features
        self.out_features = out_features
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        # Official microsoft/LoRA initialization: Kaiming-uniform A, zero B.
        # Parameters MUST inherit the wrapped weight's device/dtype: the wrap may
        # happen after model.to(device), so torch.empty defaults (CPU/fp32) would
        # put adapters on the wrong device and crash the CUDA forward pass.
        self.lora_A = nn.Parameter(
            torch.empty(r, in_features, device=weight.device, dtype=weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, r, device=weight.device, dtype=weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        adapter = F.linear(self.lora_dropout(x), self.lora_A)
        result = result + F.linear(adapter, self.lora_B) * self.scaling
        return result


def _is_target(module: nn.Module, module_name: str, target_names: set[str]) -> bool:
    return isinstance(module, nn.Linear) and module_name.split(".")[-1] in target_names


def apply_lora_to_model(
    model: nn.Module,
    r: int,
    alpha: float,
    lora_dropout: float = 0.0,
    target_names: set[str] | None = None,
) -> int:
    """Replace target ``nn.Linear`` modules in-place with LoRALinear.

    Freezes every non-adapter param of the wrapped module (only lora_A/lora_B
    remain trainable). Returns the number of replaced modules.
    """
    if target_names is None:
        target_names = {"query", "value"}
    replaced = 0
    for name, module in model.named_modules():
        if not _is_target(module, name, target_names):
            continue
        if module.bias is not None and module.bias.requires_grad is False:
            raise RuntimeError(f"LoRA target {name} already frozen (double wrap?)")
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacement = LoRALinear(
            module.in_features,
            module.out_features,
            module.weight.data,
            module.bias.data if module.bias is not None else None,
            r=r,
            alpha=alpha,
            lora_dropout=lora_dropout,
        )
        setattr(parent, child_name, replacement)
        replaced += 1
    if replaced == 0:
        raise RuntimeError("LoRA replaced 0 target modules; scope is empty")
    _freeze_non_lora(model)
    return replaced


def _freeze_non_lora(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("backbone."):
            # Task head stays fully trainable (official run_glue behavior).
            param.requires_grad = True
            continue
        is_adapter = "lora_A" in name or "lora_B" in name
        param.requires_grad = is_adapter


def lora_adapter_parameter_count(model: nn.Module) -> int:
    return sum(
        param.numel()
        for name, param in model.named_parameters()
        if ("lora_A" in name or "lora_B" in name) and param.requires_grad
    )


def assert_lora_eval_noop(model: nn.Module, reference_state: dict) -> None:
    """Gate: with zero-initialized B, eval logits must equal the reloaded model."""
    deltas = []
    for name, param in model.named_parameters():
        if "lora_B" in name:
            deltas.append(float(param.detach().abs().max()))
    if deltas and max(deltas) > 0.0:
        raise RuntimeError("LoRA B must be exactly zero at initialization")
    model_keys = set(model.state_dict())
    reference_keys = set(reference_state)
    missing = reference_keys - model_keys
    added = {key for key in model_keys - reference_keys if "lora_" in key}
    if missing:
        raise RuntimeError(
            f"LoRA dropped reference state keys: {sorted(missing)[:5]}"
        )
    if not added or not all(key.endswith(("lora_A", "lora_B")) for key in added):
        raise RuntimeError(
            f"LoRA must only ADD lora_A/lora_B keys, got: {sorted(added)[:8]}"
        )
