"""Dynamic Padding DataCollator adhering to strict per-domain max_length limits."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import torch

from gated_dual_ema_msd.config.contracts import (
    DATASET_MAX_LENGTHS,
    LABEL2ID,
    parse_label,
    require_max_length,
)

# Non-negotiable domain max-lengths per Rule 0b
DOMAIN_MAX_LENGTHS: Dict[str, int] = dict(DATASET_MAX_LENGTHS)


class NLIDataCollator:
    """Dynamic batch-level collator with domain-specific max_length truncation.

    Pads sequences dynamically to the longest sequence within each batch
    (rather than fixed static padding) while enforcing strict upper bounds.
    """

    def __init__(
        self,
        tokenizer: Any,
        default_max_length: int = 512,
        domain_max_lengths: Optional[Dict[str, int]] = None,
    ):
        self.tokenizer = tokenizer
        self.default_max_length = default_max_length
        self.domain_max_lengths = domain_max_lengths or DOMAIN_MAX_LENGTHS

    def __call__(self, rows: Sequence[dict]) -> Dict[str, Any]:
        encs = []
        for r in rows:
            domain = r.get("domain", "").strip().lower()
            if domain:
                ml = require_max_length(domain)
            else:
                ml = self.default_max_length
            e = self.tokenizer(
                str(r["premise"]),
                str(r["hypothesis"]),
                truncation=True,
                max_length=ml,
                return_tensors=None,
            )
            encs.append(e)

        padded = self.tokenizer.pad(encs, padding="longest", return_tensors="pt")

        batch: Dict[str, Any] = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
        }
        if "token_type_ids" in padded:
            batch["token_type_ids"] = padded["token_type_ids"]

        # Parse labels strictly if present
        if "label" in rows[0]:
            labels = [
                parse_label(r["label"], str(r.get("id", f"sample_{i}")))
                for i, r in enumerate(rows)
            ]
            batch["labels"] = torch.tensor(labels, dtype=torch.long)


        # Metadata for debugging / evaluation
        batch["sample_id"] = [str(r.get("id", f"sample_{i}")) for i, r in enumerate(rows)]
        batch["gold_label"] = [r.get("label", "E") for r in rows]
        if "domain" in rows[0]:
            batch["domain"] = [r.get("domain", "unknown") for r in rows]

        return batch
