"""ragpoison -- defensive RAG-poisoning lab (contracts, corpus, pipeline).

This package holds the shared contracts and the labelled-corpus loader that every
other component builds on. The pipeline modules (index, rag, defenses, judge,
runner, eval) build on top of these.
"""

from __future__ import annotations

from .corpus_loader import (
    load_benign,
    load_by_attack_class,
    load_corpus,
    load_poisoned,
    get_canary_doc,
)
from .schema import (
    CANARY,
    AttackClass,
    DefenseName,
    Doc,
    EmbeddingKind,
    GraphState,
    LLMKind,
    RunConfig,
    Verdict,
)

__all__ = [
    "CANARY",
    "AttackClass",
    "DefenseName",
    "EmbeddingKind",
    "LLMKind",
    "Doc",
    "Verdict",
    "RunConfig",
    "GraphState",
    "load_corpus",
    "load_benign",
    "load_poisoned",
    "load_by_attack_class",
    "get_canary_doc",
]
