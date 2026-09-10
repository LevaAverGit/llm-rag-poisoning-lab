"""The ``none`` defense -- the undefended baseline.

It keeps every retrieved document and pastes their bodies straight into the context,
quarantining nothing. This is the control cell of the matrix: whatever breach-rate it
produces is what the other defenses are measured against.
"""

from __future__ import annotations

from typing import List

from ..schema import Doc, DefenseName
from .base import Defense, DefenseOutput, render_documents


class NoneDefense(Defense):
    """Baseline: keep all retrieved documents, no framing, nothing quarantined."""

    name = DefenseName.NONE.value

    def apply(self, retrieved_docs: List[Doc], query: str = "") -> DefenseOutput:
        docs = list(retrieved_docs)
        return self._result(render_documents(docs), kept=docs, quarantined=[])


__all__ = ["NoneDefense"]
