import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

LABELS = ["E", "C", "N"]


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)."""
    normed = F.normalize(embeddings, p=2, dim=-1)
    sim_matrix = torch.matmul(normed, normed.T) / temperature
    batch_size = embeddings.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=embeddings.device)
    labels_col = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels_col, labels_col.T).float().to(embeddings.device)
    diag_mask = torch.eye(batch_size, device=embeddings.device)
    mask = mask - diag_mask

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_stable = sim_matrix - logits_max.detach()

    exp_logits = torch.exp(sim_stable) * (1.0 - diag_mask)
    log_prob = sim_stable - torch.log(exp_logits.sum(1, keepdim=True).clamp(min=1e-8))

    pos_counts = mask.sum(1)
    has_positives = pos_counts > 0
    if not has_positives.any():
        return torch.tensor(0.0, device=embeddings.device)
    mean_log_prob_pos = (mask * log_prob).sum(1) / pos_counts.clamp(min=1.0)
    return -mean_log_prob_pos[has_positives].mean()


class FlatCafeBERT(nn.Module):
    def __init__(
        self,
        model_name: str,
        fallback_models=None,
        dropout: float = 0.1,
        num_labels: int = 3,
        gradient_checkpointing: bool = False,
        label_smoothing: float = 0.0,
        revision: str | None = None,
        class_weights: list[float] | None = None,
        pool_mode: str = "cls",
        head_hidden: int | None = None,
        sep_token_id: int | None = None,
        alignment_dim: int = 256,
        relation_hidden: int | None = None,
        gate_bias: float = -2.0,
        relation_delta_mode: str = "absolute",
        hierarchical_e_first: bool = False,
        use_layer_mix: bool = False,
        focal_gamma: float = 0.0,
        class_margins: list[float] | None = None,
        use_multi_sample_dropout: bool = False,
        msd_dropouts: list[float] | None = None,
        relation_features_mode: str = "standard",
        use_supcon: bool = False,
        supcon_weight: float = 0.1,
        supcon_temperature: float = 0.07,
        head_architecture: str = "standard",
        classifier_mlp: bool = False,
        classifier_type: str = "linear",
        classifier_hidden: int = 256,
        classifier_dropout: float | None = None,
        classifier_act: str = "gelu",
        segment_pooling: str = "attentive",
        use_gate: bool = True,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.use_layer_mix = use_layer_mix
        self.focal_gamma = focal_gamma
        self.use_multi_sample_dropout = use_multi_sample_dropout
        self.relation_features_mode = relation_features_mode
        self.use_supcon = use_supcon
        self.supcon_weight = supcon_weight
        self.supcon_temperature = supcon_temperature
        self.head_architecture = head_architecture
        self.classifier_mlp = classifier_mlp
        self.classifier_type = classifier_type
        self.classifier_hidden = classifier_hidden
        self.classifier_dropout = classifier_dropout
        self.classifier_act = classifier_act
        self.segment_pooling = str(segment_pooling).lower()
        self.use_gate = bool(use_gate)
        if self.segment_pooling not in ("attentive", "mean"):
            raise ValueError(f"unsupported segment_pooling: {segment_pooling} (must be 'attentive' or 'mean')")
        self.class_margins = (
            torch.tensor(class_margins, dtype=torch.float32) if class_margins else None
        )
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        )
        self.pool_mode = pool_mode
        self.sep_token_id = sep_token_id if sep_token_id is not None else 2
        self.backbone = self._load_backbone(
            model_name, fallback_models or [], revision=revision
        )
        if gradient_checkpointing:
            try:
                self.backbone.gradient_checkpointing_enable()
                print(
                    "[FlatCafeBERT] gradient checkpointing enabled "
                    "(trades compute for VRAM, exact gradients)"
                )
            except (AttributeError, RuntimeError, ValueError) as exc:
                print(f"[FlatCafeBERT] gradient checkpointing not supported: {exc}")

        hidden = self.backbone.config.hidden_size
        self.hidden_size = hidden
        self.dropout = nn.Dropout(dropout)
        self.proj = None

        if use_layer_mix:
            num_layers = getattr(self.backbone.config, "num_hidden_layers", 24)
            self.layer_mix_weights = nn.Parameter(torch.zeros(num_layers + 1))
        else:
            self.layer_mix_weights = None

        if pool_mode == "gated_dual":
            # M13: preserve global pair CLS and add a conservative residual
            # relation signal built from learned premise/hypothesis pools.
            # M15: optional bottleneck via relation_hidden to cut ~6.3M -> ~1.3M params.
            if self.segment_pooling == "attentive":
                self.premise_attention = nn.Linear(hidden, 1, bias=False)
                self.hypothesis_attention = nn.Linear(hidden, 1, bias=False)
            else:
                self.premise_attention = None
                self.hypothesis_attention = None
            if relation_delta_mode not in {"absolute", "signed"}:
                raise ValueError(
                    "relation_delta_mode must be 'absolute' or 'signed' for gated_dual"
                )
            self.relation_delta_mode = relation_delta_mode
            if relation_features_mode == "extended":
                rel_in_dim = hidden * 6
            elif relation_features_mode == "standard":
                rel_in_dim = hidden * 4
            elif relation_features_mode == "concat":
                rel_in_dim = hidden * 2
            else:
                raise ValueError(
                    "relation_features_mode must be one of: "
                    "'standard', 'concat', 'extended'"
                )
            if head_architecture == "pyramid":
                # Graduated Pyramid Compression: 3072 -> 768 -> 256 -> 768 with LayerNorm & GELU
                pyr_mid = relation_hidden or 256
                self.relation_projection = nn.Sequential(
                    nn.Linear(rel_in_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, pyr_mid),
                    nn.LayerNorm(pyr_mid),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(pyr_mid, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            elif head_architecture == "factored":
                # Factored Subspace Interaction: 4x256 -> 1024 -> 512 -> 768
                sub_dim = 256
                self.factored_proj = nn.Linear(hidden, sub_dim)
                cat_dim = sub_dim * (6 if relation_features_mode == "extended" else 4)
                self.relation_projection = nn.Sequential(
                    nn.Linear(cat_dim, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(512, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            else:
                # Standard M20 Bottleneck: 3072 -> 128 -> 768
                if relation_hidden is not None:
                    if relation_hidden <= 0:
                        raise ValueError("relation_hidden must be positive")
                    self.relation_hidden = relation_hidden
                    self.relation_projection = nn.Sequential(
                        nn.Linear(rel_in_dim, relation_hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(relation_hidden, hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                else:
                    self.relation_hidden = None
                    self.relation_projection = nn.Sequential(
                        nn.Linear(rel_in_dim, hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
            if self.use_gate:
                self.fusion_gate = nn.Linear(hidden * 2, hidden)
                nn.init.constant_(self.fusion_gate.bias, gate_bias)
            else:
                self.fusion_gate = None
            self.fusion_norm = nn.LayerNorm(hidden)
            classifier_input = hidden
        elif pool_mode == "token_align":
            # M14: ESIM-style bidirectional local alignment on contextualized
            # CafeBERT tokens, compressed to a relation vector and gated into CLS.
            if alignment_dim <= 0:
                raise ValueError("alignment_dim must be positive")
            self.alignment_dim = alignment_dim
            self.alignment_projection = nn.Sequential(
                nn.Linear(hidden, alignment_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.local_compare = nn.Sequential(
                nn.Linear(alignment_dim * 4, alignment_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            # [premise mean, premise max, hypothesis mean, hypothesis max]
            self.relation_projection = nn.Sequential(
                nn.Linear(alignment_dim * 4, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fusion_gate = nn.Linear(hidden * 2, hidden)
            nn.init.constant_(self.fusion_gate.bias, gate_bias)
            self.fusion_norm = nn.LayerNorm(hidden)
            classifier_input = hidden
        else:
            classifier_input = hidden
            if pool_mode == "dual":
                classifier_input = hidden * 4
            elif pool_mode == "attn":
                self.attn_query = nn.Parameter(torch.empty(hidden))
                nn.init.normal_(self.attn_query, std=hidden**-0.5)
            elif pool_mode != "cls":
                raise ValueError(f"unsupported pool_mode={pool_mode}")
            if head_hidden:
                self.proj = nn.Sequential(
                    nn.Linear(classifier_input, head_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                classifier_input = head_hidden

        self.hierarchical_e_first = hierarchical_e_first
        if hierarchical_e_first:
            if pool_mode != "gated_dual":
                raise ValueError("hierarchical_e_first is only registered for gated_dual")
            if num_labels != 3:
                raise ValueError("hierarchical_e_first requires 3 labels (E,C,N)")
            self.head_e = nn.Linear(classifier_input, 1)
            self.head_cn = nn.Linear(classifier_input, 2)
            self.classifier = nn.Linear(classifier_input, num_labels)
        else:
            cdrop = classifier_dropout if classifier_dropout is not None else dropout
            c_hidden = classifier_hidden if classifier_hidden is not None else 256

            def _get_act(name: str):
                n = str(name).lower()
                if n == "relu":
                    return nn.ReLU()
                elif n == "silu":
                    return nn.SiLU()
                elif n == "tanh":
                    return nn.Tanh()
                elif n == "mish":
                    return nn.Mish()
                return nn.GELU()

            effective_cls_type = str(classifier_type).lower()
            if classifier_mlp and effective_cls_type == "linear":
                effective_cls_type = "mlp_2layer"

            if effective_cls_type == "mlp_2layer":
                self.classifier = nn.Sequential(
                    nn.Linear(classifier_input, c_hidden),
                    nn.LayerNorm(c_hidden),
                    _get_act(classifier_act),
                    nn.Dropout(cdrop),
                    nn.Linear(c_hidden, num_labels),
                )
            elif effective_cls_type == "mlp_3layer":
                mid_dim = max(64, c_hidden // 2)
                self.classifier = nn.Sequential(
                    nn.Linear(classifier_input, c_hidden),
                    nn.LayerNorm(c_hidden),
                    _get_act(classifier_act),
                    nn.Dropout(cdrop),
                    nn.Linear(c_hidden, mid_dim),
                    nn.LayerNorm(mid_dim),
                    _get_act(classifier_act),
                    nn.Dropout(cdrop),
                    nn.Linear(mid_dim, num_labels),
                )
            elif effective_cls_type == "residual_mlp":
                class ResidualBlock(nn.Module):
                    def __init__(self, in_d, h_d, d_p, act_name):
                        super().__init__()
                        self.block = nn.Sequential(
                            nn.Linear(in_d, h_d),
                            nn.LayerNorm(h_d),
                            _get_act(act_name),
                            nn.Dropout(d_p),
                            nn.Linear(h_d, in_d),
                            nn.LayerNorm(in_d),
                            _get_act(act_name),
                            nn.Dropout(d_p),
                        )
                        self.out = nn.Linear(in_d, num_labels)
                    def forward(self, x):
                        return self.out(x + self.block(x))
                self.classifier = ResidualBlock(classifier_input, c_hidden, cdrop, classifier_act)
            else:
                self.classifier = nn.Linear(classifier_input, num_labels)

        if use_multi_sample_dropout:
            drop_rates = msd_dropouts if msd_dropouts is not None else [0.1, 0.2, 0.3, 0.4, 0.5]
            self.msd_layers = nn.ModuleList([nn.Dropout(p) for p in drop_rates])
        else:
            self.msd_layers = None

        print(
            f"[FlatCafeBERT] pool_mode={pool_mode} head_arch={head_architecture} "
            f"classifier_input={classifier_input}"
            f"{' hier_E_first' if hierarchical_e_first else ''}"
            f"{' msd' if use_multi_sample_dropout else ''}"
            f"{' extended_rel' if relation_features_mode == 'extended' else ''}"
            f"{' supcon' if use_supcon else ''}"
            f"{' mlp_classifier' if classifier_mlp else ''}"
        )

    def _load_backbone(self, primary, fallbacks, revision: str | None = None):
        candidates = [primary] + list(fallbacks)
        if len(candidates) != 1 and revision is not None:
            raise ValueError("a pinned revision cannot be combined with fallback models")
        for name in candidates:
            try:
                print(f"[FlatCafeBERT] trying {name} @ {revision or 'default'}...")
                model = AutoModel.from_pretrained(
                    name, revision=revision, trust_remote_code=True
                )
                print(
                    f"[FlatCafeBERT] loaded {name} [OK] "
                    f"hidden={model.config.hidden_size}"
                )
                self.model_name_used = name
                self.model_revision_used = revision
                return model
            except Exception as exc:  # noqa: BLE001 - model hub/library boundary
                print(f"[FlatCafeBERT] {name} failed: {exc}")
        raise RuntimeError("No backbone could be loaded. Check internet / model names.")

    def _segment_masks(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return boolean premise/hypothesis masks, excluding all special separators.

        XLM-R pair encoding is normally ``<s> P </s></s> H </s>``. The first
        separator closes the premise; excluding every separator then leaves the
        hypothesis cleanly even with the doubled middle ``</s>``.
        """
        valid = attention_mask.bool()
        is_sep = input_ids.eq(self.sep_token_id)
        has_sep = is_sep.any(dim=1)
        first_sep = is_sep.int().argmax(dim=1).clamp(min=1)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        non_special = valid & ~is_sep & positions.gt(0)
        premise = non_special & positions.lt(first_sep.unsqueeze(1))
        hypothesis = non_special & positions.gt(first_sep.unsqueeze(1))
        # The tokenizer contract guarantees a separator. This fallback avoids
        # NaNs if a debug tokenizer violates that contract.
        fallback = non_special
        premise = torch.where(has_sep.unsqueeze(1), premise, fallback)
        hypothesis = torch.where(has_sep.unsqueeze(1), hypothesis, fallback)
        return premise, hypothesis

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(hidden.dtype)
        denominator = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (hidden * weights.unsqueeze(-1)).sum(dim=1) / denominator

    @staticmethod
    def _masked_max(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        floor = torch.finfo(hidden.dtype).min
        masked = hidden.masked_fill(~mask.unsqueeze(-1), floor)
        pooled = masked.max(dim=1).values
        has_value = mask.any(dim=1, keepdim=True)
        return torch.where(has_value, pooled, torch.zeros_like(pooled))

    @staticmethod
    def _attentive_pool(
        hidden: torch.Tensor, mask: torch.Tensor, scorer: nn.Module
    ) -> torch.Tensor:
        logits = scorer(hidden).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1) * mask.to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)

    @staticmethod
    def _relation_difference(
        premise: torch.Tensor, hypothesis: torch.Tensor, mode: str
    ) -> torch.Tensor:
        """Return the registered M20/M45 directional relation feature.

        ``absolute`` is the frozen M20 feature.  M45 changes only this term
        to preserve premise-to-hypothesis directionality; it does not add a
        new attention path or change relation capacity.
        """
        delta = premise - hypothesis
        if mode == "absolute":
            return delta.abs()
        if mode == "signed":
            return delta
        raise ValueError(f"unsupported relation delta mode: {mode}")

    def _pool_dual(
        self, hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        premise_mask, hypothesis_mask = self._segment_masks(input_ids, attention_mask)
        premise = self._masked_mean(hidden, premise_mask)
        hypothesis = self._masked_mean(hidden, hypothesis_mask)
        return torch.cat(
            [premise, hypothesis, premise * hypothesis, (premise - hypothesis).abs()],
            dim=1,
        )

    def _residual_fuse(
        self, global_pair: torch.Tensor, relation: torch.Tensor
    ) -> torch.Tensor:
        if not getattr(self, "use_gate", True):
            self.last_gate = None
            return global_pair + self.fusion_norm(relation)
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat([global_pair, relation], dim=-1))
        )
        # Evaluation consumes this detached diagnostic to produce per-example
        # and per-class gate statistics without retaining a training graph.
        self.last_gate = gate.detach()
        return global_pair + gate * self.fusion_norm(relation)

    def _pool_gated_dual(
        self, hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        premise_mask, hypothesis_mask = self._segment_masks(input_ids, attention_mask)
        if getattr(self, "segment_pooling", "attentive") == "mean":
            premise = self._masked_mean(hidden, premise_mask)
            hypothesis = self._masked_mean(hidden, hypothesis_mask)
        else:
            premise = self._attentive_pool(
                hidden, premise_mask, self.premise_attention
            )
            hypothesis = self._attentive_pool(
                hidden, hypothesis_mask, self.hypothesis_attention
            )
        if self.relation_features_mode == "extended":
            p_norm = F.normalize(premise, p=2, dim=-1)
            h_norm = F.normalize(hypothesis, p=2, dim=-1)
            cosine_feat = p_norm * h_norm
            diff_feat = premise - hypothesis
            feats = [
                premise,
                hypothesis,
                premise * hypothesis,
                (premise - hypothesis).abs(),
                diff_feat,
                cosine_feat,
            ]
        elif self.relation_features_mode == "standard":
            feats = [
                premise,
                hypothesis,
                premise * hypothesis,
                self._relation_difference(
                    premise, hypothesis, self.relation_delta_mode
                ),
            ]
        elif self.relation_features_mode == "concat":
            feats = [premise, hypothesis]
        else:  # guarded in __init__; retained as a defensive runtime check
            raise ValueError(
                f"unsupported relation_features_mode={self.relation_features_mode}"
            )

        if self.head_architecture == "factored":
            proj_feats = [self.factored_proj(f) for f in feats]
            rel_cat = torch.cat(proj_feats, dim=-1)
        else:
            rel_cat = torch.cat(feats, dim=-1)

        relation = self.relation_projection(rel_cat)
        self.last_relation = relation
        return self._residual_fuse(hidden[:, 0], relation)

    def _pool_token_alignment(
        self, hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        premise_mask, hypothesis_mask = self._segment_masks(input_ids, attention_mask)
        projected = self.alignment_projection(hidden)
        # Similarity and softmax stay in FP32 even under BF16 training; this is
        # the only O(L^2) operation and avoids overflow on sharp token scores.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            projected_for_alignment = projected.float()
            similarity = torch.bmm(
                projected_for_alignment,
                projected_for_alignment.transpose(1, 2),
            ) / math.sqrt(self.alignment_dim)

            # For each premise token, align over hypothesis tokens; and vice versa.
            p_to_h_logits = similarity.masked_fill(
                ~hypothesis_mask.unsqueeze(1), -1e4
            )
            p_to_h = torch.softmax(p_to_h_logits, dim=-1)
            p_to_h = p_to_h * hypothesis_mask.unsqueeze(1).to(p_to_h.dtype)
            p_to_h = p_to_h / p_to_h.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            aligned_hypothesis = torch.bmm(p_to_h, projected_for_alignment)

            h_to_p_logits = similarity.transpose(1, 2).masked_fill(
                ~premise_mask.unsqueeze(1), -1e4
            )
            h_to_p = torch.softmax(h_to_p_logits, dim=-1)
            h_to_p = h_to_p * premise_mask.unsqueeze(1).to(h_to_p.dtype)
            h_to_p = h_to_p / h_to_p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            aligned_premise = torch.bmm(h_to_p, projected_for_alignment)

        premise_compare = self.local_compare(
            torch.cat(
                [
                    projected_for_alignment,
                    aligned_hypothesis,
                    projected_for_alignment - aligned_hypothesis,
                    projected_for_alignment * aligned_hypothesis,
                ],
                dim=-1,
            )
        )
        hypothesis_compare = self.local_compare(
            torch.cat(
                [
                    projected_for_alignment,
                    aligned_premise,
                    projected_for_alignment - aligned_premise,
                    projected_for_alignment * aligned_premise,
                ],
                dim=-1,
            )
        )
        relation = self.relation_projection(
            torch.cat(
                [
                    self._masked_mean(premise_compare, premise_mask),
                    self._masked_max(premise_compare, premise_mask),
                    self._masked_mean(hypothesis_compare, hypothesis_mask),
                    self._masked_max(hypothesis_compare, hypothesis_mask),
                ],
                dim=-1,
            )
        )
        self.last_relation = relation
        return self._residual_fuse(hidden[:, 0], relation)

    def _classify(self, feat: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            feat = self.proj(feat)
        if self.hierarchical_e_first:
            logit_e = self.head_e(feat).squeeze(-1)  # [B]
            logits_cn = self.head_cn(feat)  # [B,2] -> C,N
            logit_c = -logit_e + logits_cn[:, 0]
            logit_n = -logit_e + logits_cn[:, 1]
            return torch.stack([logit_e, logit_c, logit_n], dim=1)
        return self.classifier(feat)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
        inputs_embeds=None,
    ):
        # ``inputs_embeds`` is a transparent pass-through used ONLY by the R3F
        # consistency loss (noisy word embeddings). No head/architecture change:
        # when it is None this call is identical to the plain path. HF backbones
        # forbid input_ids together with inputs_embeds, so exactly one reaches
        # the encoder; the pooling heads still read the original ``input_ids``
        # for premise/hypothesis segment masks.
        output = self.backbone(
            input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.use_layer_mix,
        )
        if self.use_layer_mix and output.hidden_states is not None:
            all_layers = torch.stack(output.hidden_states, dim=0)
            weights = torch.softmax(self.layer_mix_weights.to(all_layers.device), dim=0).view(-1, 1, 1, 1)
            hidden = (all_layers * weights).sum(dim=0)
        else:
            hidden = output.last_hidden_state

        if self.pool_mode == "dual":
            features = self._pool_dual(hidden, input_ids, attention_mask)
        elif self.pool_mode == "attn":
            scores = torch.einsum("blh,h->bl", hidden, self.attn_query)
            scores = scores.masked_fill(attention_mask == 0, -1e4)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            features = (hidden * weights).sum(dim=1)
        elif self.pool_mode == "gated_dual":
            features = self._pool_gated_dual(hidden, input_ids, attention_mask)
        elif self.pool_mode == "token_align":
            features = self._pool_token_alignment(hidden, input_ids, attention_mask)
        else:
            features = hidden[:, 0]

        msd_outs = None
        if self.use_multi_sample_dropout and self.training and self.msd_layers is not None:
            msd_outs = [self._classify(drop(features)) for drop in self.msd_layers]
            logits = torch.stack(msd_outs, dim=0).mean(dim=0)
        else:
            feat = self.dropout(features)
            logits = self._classify(feat)

        loss = None
        if labels is not None:
            if msd_outs is not None:
                losses = []
                for l_out in msd_outs:
                    if self.class_margins is not None:
                        l_out = l_out + self.class_margins.to(l_out.device)
                    crit = nn.CrossEntropyLoss(
                        label_smoothing=self.label_smoothing,
                        weight=(
                            self.class_weights.to(l_out.device)
                            if self.class_weights is not None
                            else None
                        ),
                    )
                    loss_val = crit(l_out, labels)
                    if self.focal_gamma > 0.0:
                        with torch.no_grad():
                            probs = torch.softmax(l_out, dim=-1)
                            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                            focal_factor = ((1.0 - pt) ** self.focal_gamma).clamp(min=1e-5)
                        loss_val = loss_val * focal_factor.mean()
                    losses.append(loss_val)
                loss = torch.stack(losses).mean()
            else:
                logits_for_loss = logits
                if self.class_margins is not None:
                    logits_for_loss = logits_for_loss + self.class_margins.to(logits.device)
                criterion = nn.CrossEntropyLoss(
                    label_smoothing=self.label_smoothing,
                    weight=(
                        self.class_weights.to(logits.device)
                        if self.class_weights is not None
                        else None
                    ),
                )
                loss = criterion(logits_for_loss, labels)
                if self.focal_gamma > 0.0:
                    with torch.no_grad():
                        probs = torch.softmax(logits_for_loss, dim=-1)
                        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                        focal_factor = ((1.0 - pt) ** self.focal_gamma).clamp(min=1e-5)
                    loss = loss * focal_factor.mean()

            if self.use_supcon and getattr(self, "last_relation", None) is not None:
                sup_loss = supervised_contrastive_loss(
                    self.last_relation, labels, temperature=self.supcon_temperature
                )
                loss = loss + self.supcon_weight * sup_loss

        return {"logits": logits, "loss": loss, "hidden": features}
