"""Config-gated consistency losses for EXP-001-R2 Phase 2 (m21/m22).

Paper-exact, single-variable loss add-ons on top of the M20 gated baseline
(configs/experiments/exp001_r2_m20_gated_small128_bias1_maxlen512_seed42.yaml):

- :func:`rdrop_loss` implements R-Drop (arXiv:2106.14448, NeurIPS'21): two
  forward passes of the SAME batch under independent dropout masks plus a
  symmetric-KL consistency term, applied every optimizer step in BOTH stages:
  ``L_i = 0.5*(CE(y,p^1)+CE(y,p^2)) + (alpha/2)*(KL(p^1||p^2)+KL(p^2||p^1))``.
- :func:`r3f_loss` implements R3F (arXiv:2008.03156, ICLR'21): one Gaussian
  perturbation ``eps ~ N(0, sigma^2)`` per step on the word embeddings (before
  the encoder); the single noisy backward supplies gradients for BOTH terms via
  ``L = CE(x + eps) + lambda * SKL(f(x) || f(x + eps))``.
- :func:`mirror_contradiction_symmetry_loss` implements the mirrored-instance
  (BERT+M) symmetry constraint from Li et al. (EMNLP 2019): the model's
  contradiction log-probability must agree for ``(premise, hypothesis)`` and
  the same pair in reversed order. It applies in M47 Stage 2 only.

Both helpers factor the loss math into standalone functions so they can be
sanity-tested with random logits/embeddings on CPU. Consistency terms are
always computed in FP32 (logits cast to float32, every KL inside
``torch.autocast(device_type=..., enabled=False)``) so FP16 training can never
contaminate them. CE keeps label smoothing via CrossEntropyLoss (0.02).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

R3F_DEFAULT_SIGMA = 1e-5
CONTRADICTION_LABEL_ID = 1  # Project label contract: E=0, C=1, N=2.


def _fp32_log_probs(logits: torch.Tensor) -> torch.Tensor:
    """FP32 log-probabilities, immune to any active autocast context."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return F.log_softmax(logits.float(), dim=-1)


def symmetric_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """``0.5 * (KL(p||q) + KL(q||p))`` from FP32 log-probability tensors.

    Uses the stable ``F.kl_div(..., log_target=True)`` pattern; each direction
    is averaged over batch elements exactly like the papers' per-example KL.
    """
    with torch.autocast(device_type=log_p.device.type, enabled=False):
        if log_p.dtype != torch.float32 or log_q.dtype != torch.float32:
            raise ValueError("symmetric KL must be computed on float32 tensors")
        kl_pq = F.kl_div(
            input=log_q, target=log_p, log_target=True, reduction="batchmean"
        )
        kl_qp = F.kl_div(
            input=log_p, target=log_q, log_target=True, reduction="batchmean"
        )
        return 0.5 * (kl_pq + kl_qp)


def rdrop_loss(logits1: torch.Tensor, logits2: torch.Tensor, labels: torch.Tensor,
               alpha: float = 5.0,
               label_smoothing: float = 0.02) -> dict[str, torch.Tensor]:
    """Paper-exact R-Drop loss for one batch (two independent dropout views).

    The supervised term averages the two cross entropies (label smoothing kept
    via CrossEntropyLoss); ``(alpha/2)`` multiplies the SUM of both symmetric-KL
    directions, matching arXiv:2106.14448 Eq. 4. The KL term is computed in
    FP32 even under autocast.
    """
    ce1 = F.cross_entropy(
        logits1.float(), labels, label_smoothing=label_smoothing
    )
    ce2 = F.cross_entropy(
        logits2.float(), labels, label_smoothing=label_smoothing
    )
    kl_sym = symmetric_kl(_fp32_log_probs(logits1), _fp32_log_probs(logits2))
    total = 0.5 * (ce1 + ce2) + (alpha / 2.0) * kl_sym
    return {
        "loss": total,
        "kl_sym": kl_sym.detach(),
        "ce_mean": (0.5 * (ce1.detach() + ce2.detach())),
    }


def mirror_contradiction_symmetry_loss(
    logits: torch.Tensor,
    mirrored_logits: torch.Tensor,
    labels: torch.Tensor,
    lambda_: float = 1.0,
    label_smoothing: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Li et al. (EMNLP 2019) BERT+M loss on an original and mirrored NLI pair.

    The original ordered pair alone receives supervised CE. The reversed pair is
    deliberately unlabeled: the only constraint is contradiction symmetry,
    ``abs(log p_C(P,H) - log p_C(H,P))``. All log probabilities are FP32 so the
    auxiliary term remains stable under the runner's FP16 autocast.
    """
    if logits.ndim != 2 or mirrored_logits.shape != logits.shape or logits.size(1) != 3:
        raise ValueError("mirror symmetry expects two matching [batch, 3] logit tensors")
    if labels.ndim != 1 or labels.size(0) != logits.size(0):
        raise ValueError("mirror symmetry labels must match the logit batch")
    if not math.isfinite(lambda_) or lambda_ <= 0:
        raise ValueError("mirror symmetry lambda must be finite and positive")
    ce = F.cross_entropy(logits.float(), labels, label_smoothing=label_smoothing)
    log_probs = _fp32_log_probs(logits)
    mirrored_log_probs = _fp32_log_probs(mirrored_logits)
    mirror_l1 = torch.abs(
        log_probs[:, CONTRADICTION_LABEL_ID]
        - mirrored_log_probs[:, CONTRADICTION_LABEL_ID]
    ).mean()
    total = ce + lambda_ * mirror_l1
    return {"loss": total, "mirror_logprob_l1": mirror_l1.detach(), "ce": ce.detach()}


def make_r3f_noise(embedding_layer: torch.nn.Module, input_ids: torch.Tensor,
                   sigma: float = R3F_DEFAULT_SIGMA) -> torch.Tensor:
    """Draw ``eps ~ N(0, sigma^2)`` ONCE per step, shaped like word embeddings."""
    base = embedding_layer(input_ids)
    noise = torch.normal(
        mean=torch.zeros(base.shape, device=base.device),
        std=torch.full(base.shape, sigma, device=base.device),
    )
    return noise.to(dtype=base.dtype)


def r3f_skew_kl(clean_log_probs_p: torch.Tensor,
                noisy_log_probs_q: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between detached clean log-probs p and noisy log-probs q.

    Both inputs are FP32 log-probability tensors (log_softmax output), so the
    identical-distribution case gives exactly 0 without any probability-space
    clamp/renormalization mismatch. ``log q`` is clamped to >= -1e9 to keep
    every term finite; everything runs inside an autocast-disabled FP32 block.
    """
    with torch.autocast(device_type=noisy_log_probs_q.device.type, enabled=False):
        if noisy_log_probs_q.dtype != torch.float32:
            raise ValueError("R3F consistency term must be computed in FP32")
        log_p = clean_log_probs_p.float().clamp(min=-1e9)
        log_q = noisy_log_probs_q.float().clamp(min=-1e9)
        kl_pq = (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()
        kl_qp = (log_q.exp() * (log_q - log_p)).sum(dim=-1).mean()
        return 0.5 * (kl_pq + kl_qp)


def r3f_loss(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
             token_type_ids: torch.Tensor | None, labels: torch.Tensor,
             sigma: float = R3F_DEFAULT_SIGMA,
             lambda_: float = 1.0,
             label_smoothing: float = 0.02) -> dict[str, torch.Tensor]:
    """End-to-end R3F step for FlatCafeBERT (single backward, both gradients).

    Clean pass: standard ``input_ids`` forward without grad, giving the
    detached reference distribution ``p``. Noisy pass: ``inputs_embeds =
    word_embeddings(input_ids) + eps`` with grad enabled, i.e. the perturbation
    hits the WORD EMBEDDINGS ONLY, before the encoder (CafeBERT/XLM-R accepts
    ``inputs_embeds``). Returns ``loss = CE(noisy) + lambda * SKL(p, q)`` whose
    one ``backward()`` supplies gradients for both terms, plus logging scalars.
    """
    embed_layer = model.backbone.get_input_embeddings()

    # ---- Clean forward (no grad): detached reference distribution p ----
    with torch.no_grad():
        clean_out = model(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    with torch.autocast(device_type=input_ids.device.type, enabled=False):
        log_probs_p = F.log_softmax(clean_out["logits"].float(), dim=-1)
    log_probs_p = log_probs_p.detach()

    # ---- Noisy forward: gradients flow for BOTH loss terms ----
    eps = make_r3f_noise(embed_layer, input_ids, sigma=sigma)
    noisy_out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        inputs_embeds=embed_layer(input_ids) + eps,
    )
    with torch.autocast(device_type=input_ids.device.type, enabled=False):
        logits_fp32 = noisy_out["logits"].float()
        ce = F.cross_entropy(
            logits_fp32, labels, label_smoothing=label_smoothing
        )
        log_q = F.log_softmax(logits_fp32, dim=-1)
        sk = r3f_skew_kl(log_probs_p, log_q)
    total = ce + lambda_ * sk
    return {"loss": total, "skl": sk.detach(), "ce": ce.detach()}


def resolve_consistency_mode(model_cfg: dict) -> tuple[str | None, dict]:
    """Config gate: exactly one registered consistency method may be active."""
    alpha = model_cfg.get("rdrop_alpha")
    lam = model_cfg.get("r3f_lambda")
    sigma = model_cfg.get("r3f_sigma", R3F_DEFAULT_SIGMA)
    mirror = model_cfg.get("mirror_contradiction_symmetry")
    if sum(value is not None for value in (alpha, lam, mirror)) > 1:
        raise ValueError("rdrop_alpha and r3f_lambda are mutually exclusive: "
                         "exactly ONE consistency method per run")
    if alpha is not None:
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha <= 0:
            raise ValueError(f"rdrop_alpha must be a positive number, got {alpha}")
        return "rdrop", {"alpha": float(alpha)}
    if lam is not None:
        if not isinstance(lam, (int, float)) or isinstance(lam, bool) or lam <= 0:
            raise ValueError(f"r3f_lambda must be a positive number, got {lam}")
        if not isinstance(sigma, (int, float)) or isinstance(sigma, bool) or sigma <= 0:
            raise ValueError(f"r3f_sigma must be a positive number, got {sigma}")
        return "r3f", {"lambda": float(lam), "sigma": float(sigma)}
    if mirror is not None:
        if not isinstance(mirror, dict) or set(mirror) != {"lambda", "stage"}:
            raise ValueError(
                "mirror_contradiction_symmetry must be exactly {lambda, stage}"
            )
        lambda_ = mirror["lambda"]
        if (
            not isinstance(lambda_, (int, float))
            or isinstance(lambda_, bool)
            or not math.isfinite(float(lambda_))
            or float(lambda_) <= 0
        ):
            raise ValueError("mirror contradiction symmetry lambda must be finite and positive")
        if mirror["stage"] != "stage2":
            raise ValueError("mirror contradiction symmetry is registered for stage2 only")
        return "mirror_symmetry", {"lambda": float(lambda_), "stage": "stage2"}
    if sigma != R3F_DEFAULT_SIGMA:
        raise ValueError("r3f_sigma may only be customized together with r3f_lambda")
    return None, {}


def freelb_params_from_cfg(model_cfg: dict) -> dict | None:
    """FreeLB (arXiv:1909.11764) config gate: exactly one registry per run.

    Reads ``model.freelb`` and returns its validated parameters or None.
    Raises when FreeLB is combined with any other per-step method (R-Drop,
    R3F, Child-Tuning, JTT) or float values are non-finite.
    """
    freelb = model_cfg.get("freelb")
    if freelb is None:
        return None
    if not isinstance(freelb, dict):
        raise ValueError("freelb must be a mapping")  # noqa: TRY004 - config gates are ValueError
    steps = freelb.get("steps")
    step_size = freelb.get("step_size")
    max_norm = freelb.get("max_norm")
    init_mag = freelb.get("init_mag")
    if steps != 2 or not isinstance(steps, int):
        raise ValueError("FreeLB requires exactly steps=2 (paper K=2)")
    for name, value in (("step_size", step_size), ("max_norm", max_norm),
                        ("init_mag", init_mag)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"freelb {name} must be a finite positive number")  # noqa: TRY004
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"freelb {name} must be a finite positive number")
    if freelb.get("padding_perturbation", 0) != 0:
        raise ValueError("FreeLB padding perturbation must be 0")
    if freelb.get("optimizer_updates_per_batch", 1) != 1:
        raise ValueError("FreeLB requires optimizer_updates_per_batch=1")
    if model_cfg.get("rdrop_alpha") is not None or model_cfg.get("r3f_lambda") is not None \
            or model_cfg.get("childtune_p") is not None \
            or model_cfg.get("mirror_contradiction_symmetry") is not None:
        raise ValueError("freelb is mutually exclusive with rdrop/r3f/childtune/mirror")
    return {
        "steps": int(steps),
        "step_size": float(step_size),
        "max_norm": float(max_norm),
        "init_mag": float(init_mag),
        "padding_perturbation": 0,
        "optimizer_updates_per_batch": 1,
    }


def init_freelb_delta(
    embedding: torch.Tensor, attention_mask: torch.Tensor, init_mag: float
) -> torch.Tensor:
    """Official FreeLB init: U(-1,1) * mask, scaled mag/sqrt(nonpad*dim)."""
    mask = attention_mask.to(embedding.dtype).unsqueeze(-1)
    nonpad = attention_mask.sum(dim=1, dtype=torch.float32).clamp(min=1.0)
    dims = nonpad * embedding.size(-1)
    mag = init_mag / torch.sqrt(dims)
    delta = torch.zeros_like(embedding).uniform_(-1.0, 1.0) * mask
    return (delta * mag.view(-1, 1, 1)).detach()


def freelb_delta_step(
    delta: torch.Tensor, delta_grad: torch.Tensor,
    step_size: float, max_norm: float,
) -> torch.Tensor:
    """One FreeLB ascent step: normalized grad, L2 projection to radius."""
    if not torch.isfinite(delta_grad).all():
        raise FloatingPointError("FreeLB delta gradient is non-finite")
    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1)
    denorm = torch.clamp(denorm, min=1e-8).view(-1, 1, 1)
    delta = (delta + step_size * delta_grad / denorm).detach()
    tensor_norm = torch.norm(delta.view(delta.size(0), -1), dim=1).view(-1, 1, 1)
    exceed = (tensor_norm > max_norm).to(delta.dtype)
    delta = (
        delta / tensor_norm * max_norm * exceed + (1 - exceed) * delta
    ).detach()
    return delta


def sam_params_from_cfg(model_cfg: dict) -> dict | None:
    """SAM (Foret et al. ICLR 2021) config gate: rho=0.05, non-adaptive, m=1."""
    sam = model_cfg.get("sam")
    if sam is None:
        return None
    if not isinstance(sam, dict):
        raise ValueError("sam must be a mapping")  # noqa: TRY004 - config gates are ValueError
    rho = sam.get("rho")
    adaptive = sam.get("adaptive", False)
    if isinstance(adaptive, bool) is False or adaptive is not False:
        raise ValueError("SAM requires adaptive=false (official m=1 L2 setting)")
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) \
            or not math.isfinite(float(rho)) or float(rho) <= 0:
        raise ValueError("sam rho must be a finite positive number")
    if sam.get("sharpness_factor_m", 1) != 1:
        raise ValueError("SAM requires sharpness_factor_m=1")
    if model_cfg.get("rdrop_alpha") is not None or model_cfg.get("r3f_lambda") is not None \
            or model_cfg.get("childtune_p") is not None or model_cfg.get("freelb") is not None:
        raise ValueError("sam is mutually exclusive with rdrop/r3f/childtune/freelb")
    return {"rho": float(rho), "adaptive": False, "sharpness_factor_m": 1}
