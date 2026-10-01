"""Shared SAM (Sharpness-Aware Minimization) optimizer for EXP-001-R2.

Implements Foret et al. (ICLR 2021) SAM with the Bahri et al.
(arXiv:2110.08529) m=1 / non-adaptive / L2 setting used for GLUE/SuperGLUE.

Algorithm per optimizer step:
1. First forward+backward on the current params (as usual) -> grads.
2. Perturb: theta += rho * g / ||g||_2  (parameter-space ascent, L2 norm).
3. Zero grads, second forward+backward at the perturbed point.
4. Restore original params, then apply the SECOND step's gradients through the
   base optimizer (AdamW + weight decay). Scheduler steps exactly once.

This wrapper keeps the base optimizer untouched, so learning-rate/warmup
scheduling and AMP behavior stay identical to the M20 baseline.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim.optimizer import Optimizer


def gradients_are_finite(params: Iterable[torch.nn.Parameter]) -> bool:
    """True only when at least one gradient exists and every gradient is finite."""
    seen = False
    for param in params:
        if param.grad is None:
            continue
        seen = True
        if not torch.isfinite(param.grad).all().item():
            return False
    return seen


class SAM(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer: Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        max_norm: float | None = None,
    ):
        if not isinstance(rho, (int, float)) or isinstance(rho, bool) or rho <= 0:
            raise ValueError(f"SAM rho must be a positive number, got {rho}")
        if not math.isfinite(float(rho)):
            raise ValueError("SAM rho must be finite")
        defaults = {"rho": float(rho), "adaptive": bool(adaptive)}
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.max_norm = float(max_norm) if max_norm is not None else None
        self.param_groups = self.base_optimizer.param_groups
        # Base optimizer groups carry no rho; inject it like davda54/sam so
        # both wrapper and base see one shared group list.
        for group in self.param_groups:
            group.setdefault("rho", float(rho))
            group.setdefault("adaptive", bool(adaptive))

    def _grad_norm(self) -> torch.Tensor:
        shared_params = self._shared_params()
        if not shared_params:
            raise RuntimeError("SAM found no gradient-bearing parameters")
        return torch.norm(
            torch.stack([
                torch.norm(param.grad.detach().float(), p=2)
                for param in shared_params
            ]),
            p=2,
        )

    def _shared_params(self) -> list[torch.nn.Parameter]:
        return [
            param
            for group in self.param_groups
            for param in group["params"]
            if param.grad is not None
            and (self.max_norm is None or param.grad.norm().item() <= self.max_norm)
        ]

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for param in group["params"]:
                if param.grad is None:
                    continue
                # adaptive=False keeps the standard L2 perturbation.
                e_w = param.grad * scale.to(param.dtype)
                param.add_(e_w)
                self.state[param]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def restore_step(self, zero_grad: bool = False) -> None:
        """Restore pre-perturbation weights without applying the base optimizer."""
        for group in self.param_groups:
            for param in group["params"]:
                if "e_w" in self.state[param]:
                    param.sub_(self.state[param].pop("e_w"))  # back to theta
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        self.restore_step(zero_grad=False)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError(
            "SAM requires the two-step train loop: call first_step() after the "
            "first backward and second_step() after the second backward"
        )

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
