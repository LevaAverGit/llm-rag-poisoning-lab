"""The ``provenance_filter`` defense -- filtering by source trust.

It keeps only documents whose provenance is trusted (:attr:`Doc.trusted` is ``True``)
and quarantines the rest before they can reach the generator. In this lab the trust
label tracks where a document came from: vetted internal sources are ``trusted: true``;
documents that arrived from untrusted, externally-influenced channels
(``external_ticket``, ``external_wiki_edit``) are ``trusted: false`` -- which is exactly
how every poisoned document is labelled.

That makes this the strongest defense against the poisoning classes as modelled here,
but it is not free: it is only as good as the provenance labels, and it would also drop
a legitimate answer that happened to live in an untrusted-but-correct source. The
matrix shows the trade-off rather than crowning a winner.
"""

from __future__ import annotations

from typing import List

from ..schema import Doc, DefenseName
from .base import Defense, DefenseOutput, render_documents


class ProvenanceFilterDefense(Defense):
    """Keep only trusted-source documents; quarantine every untrusted one."""

    name = DefenseName.PROVENANCE_FILTER.value

    def apply(self, retrieved_docs: List[Doc], query: str = "") -> DefenseOutput:
        docs = list(retrieved_docs)
        kept = [d for d in docs if d.trusted]
        quarantined = [d for d in docs if not d.trusted]
        return self._result(
            render_documents(kept),
            kept=kept,
            quarantined=quarantined,
            dropped_untrusted=len(quarantined),
        )


__all__ = ["ProvenanceFilterDefense"]
