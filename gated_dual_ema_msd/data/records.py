"""Structured typed records for NLI datasets."""
from __future__ import annotations

from typing import TypedDict


class NLIRecord(TypedDict):
    """Canonical NLI record representation across all domains."""
    id: str
    premise: str
    hypothesis: str
    label: str
    domain: str
