"""RecAdam optimizer for EXP-001-R2 (Chen et al., EMNLP 2020).

RecAdam objective (official ``Sanyuan-Chen/RecAdam`` notation):

    λ(t)·L_target + (1-λ(t))·γ/2·||θ-θ_ref||²

The target Adam update is multiplied by λ(t), while the reference penalty is
decoupled from Adam moments and multiplied by ``1-λ(t)``.  This direction is
important: training starts by recalling the reference and gradually shifts to
the target objective.  ``t0=100`` is an official search-grid value and is used
for M37 after M40 selected WiSE-FT α=1.0, i.e. a deliberately early/weaker
reference phase rather than the originally planned t0=500.
"""
from __future__ import annotations

import math

import torch
from torch.optim.optimizer import Optimizer


def recadam_anneal(t: int, t0: float = 500.0, k: float = 0.1) -> float:
    """Sigmoid annealing of the pretrain penalty (official RecAdam)."""
    if t0 < 0:
        raise ValueError("recadam t0 must be non-negative")
    u = k * (t - t0)
    if u >= 0:
        return float(1.0 / (1.0 + math.exp(-u)))
    exp_u = math.exp(u)
    return float(exp_u / (1.0 + exp_u))


class RecAdam(Optimizer):
    def __init__(
        self,
        params,
        reference_state: dict[str, torch.Tensor],
        named_params: list[tuple[str, torch.nn.Parameter]],
        lr: float = 1e-5,
        betas=(0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.005,
        gamma: float = 5000.0,
        anneal_function: str = "sigmoid",
        anneal_k: float = 0.1,
        anneal_t0: float = 500.0,
    ):
        if not reference_state:
            raise ValueError("RecAdam requires a reference state dict (stage-entry weights)")
        if anneal_function != "sigmoid":
            raise ValueError("RecAdam only supports sigmoid annealing (official)")
        if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or gamma <= 0:
            raise ValueError(f"RecAdam gamma must be positive, got {gamma}")
        defaults = {
            "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
            "gamma": float(gamma), "anneal_function": anneal_function,
            "anneal_k": float(anneal_k), "anneal_t0": float(anneal_t0),
        }
        super().__init__(params, defaults)
        self.reference_state = {
            name: value.detach().clone().cpu() for name, value in reference_state.items()
        }
        # Identity map param -> reference tensor for penalty application.
        self.ref_by_param: dict[int, tuple[torch.Tensor, tuple]] = {}
        for name, param in named_params:
            if name in self.reference_state:
                ref = self.reference_state[name]
                if ref.shape != param.shape:
                    raise RuntimeError(
                        f"RecAdam reference shape mismatch at {name}: "
                        f"{ref.shape} != {param.shape}"
                    )
                self.ref_by_param[id(param)] = (ref, param.shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step_t = state["step"]
                grad = param.grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** step_t
                bias_correction2 = 1 - beta2 ** step_t
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group["eps"])
                update = exp_avg / denom / bias_correction1 * group["lr"]
                anneal = recadam_anneal(
                    step_t, t0=group["anneal_t0"], k=group["anneal_k"]
                )
                # Official objective shifting: target update grows with λ(t).
                param.add_(-anneal * update)
                # Official decoupled pretraining-simulation penalty: strongest
                # early and vanishes as λ(t) approaches one.  No factor two:
                # the paper defines γ/2 * ||theta-theta_ref||^2.
                ref_entry = self.ref_by_param.get(id(param))
                if ref_entry is not None and group["gamma"] > 0:
                    ref, _ = ref_entry
                    penalty_grad = (
                        (1.0 - anneal)
                        * group["gamma"]
                        * (param.detach() - ref.to(param.device, param.dtype))
                    )
                    param.add_(-group["lr"] * penalty_grad)
                # AdamW-style decoupled weight decay, matching official order.
                if group["weight_decay"] > 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"])
        return loss
