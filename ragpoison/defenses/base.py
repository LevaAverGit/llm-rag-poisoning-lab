"""Common interface and shared helpers for every defense.

A *defense* is the middle stage of the pipeline: it sits between retrieval and
generation and decides what the generator is actually allowed to see. All defenses
share one interface::

    context, meta = defense.apply(retrieved_docs, query)

``context`` is the final, post-defense string handed to the generator. ``meta`` is a
plain dict that always carries two keys -- ``kept`` and ``quarantined`` (both
``List[Doc]``) -- so the pipeline can record which documents survived and which the
defense held out (into :attr:`GraphState.quarantined`). Individual defenses add their
own diagnostic keys (scores, counts, a graceful-skip flag).

Two orthogonal levers a defense can pull:

* **membership** -- which retrieved documents reach the generator at all
  (``provenance_filter`` drops untrusted ones; ``injection_classifier`` drops
  flagged ones);
* **framing** -- how the surviving documents appear (``context_fencing`` wraps each
  one in a data boundary and prepends a standing "this is data, not instructions"
  rule).

The baseline (``none``) pulls neither lever. This module holds the shared
:func:`render_documents` assembly used by every membership-only defense, plus the
:class:`Defense` base class; it imports only the contracts, so it stays torch-free.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..schema import Doc

# The context string plus a free-form diagnostics dict. `meta` is guaranteed to
# contain "defense" (str), "kept" (List[Doc]) and "quarantined" (List[Doc]).
DefenseMeta = Dict[str, object]
DefenseOutput = Tuple[str, DefenseMeta]

# How individual document bodies are joined in the naive assembly.
DOC_SEPARATOR = "\n\n"


def render_documents(docs: List[Doc]) -> str:
    """Assemble document bodies back-to-back -- the naive, vulnerable rendering.

    This is exactly what an undefended RAG prompt does: paste the retrieved text
    into the context with no boundary between "data" and "instructions". It is the
    baseline (``none``) rendering, and it is reused by the membership-only defenses
    (``provenance_filter``, ``injection_classifier``) on whichever subset of
    documents they chose to keep -- only ``context_fencing`` reshapes the framing.
    """
    return DOC_SEPARATOR.join(d.text.strip() for d in docs)


class Defense:
    """Base class: a named strategy mapping retrieved docs to ``(context, meta)``.

    Subclasses set :attr:`name` (the :class:`~ragpoison.schema.DefenseName` value they
    register under) and implement :meth:`apply`.
    """

    name: str = "defense"

    def apply(self, retrieved_docs: List[Doc], query: str = "") -> DefenseOutput:
        """Turn ``retrieved_docs`` into a post-defense ``(context, meta)`` pair.

        ``query`` is accepted for interface symmetry; membership/framing defenses do
        not need it, but a classifier could condition on it in future.
        """
        raise NotImplementedError

    def _result(
        self,
        context: str,
        kept: List[Doc],
        quarantined: List[Doc],
        **extra: object,
    ) -> DefenseOutput:
        """Build the standard ``meta`` dict (kept/quarantined) plus any extras."""
        meta: DefenseMeta = {
            "defense": self.name,
            "kept": list(kept),
            "quarantined": list(quarantined),
        }
        meta.update(extra)
        return context, meta

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "{cls}(name={name!r})".format(cls=type(self).__name__, name=self.name)


__all__ = [
    "DefenseMeta",
    "DefenseOutput",
    "DOC_SEPARATOR",
    "render_documents",
    "Defense",
]
