"""Tests for the index + retriever (builder #1).

Everything here runs on LlamaIndex's deterministic ``MockEmbedding`` -- no torch,
no model download -- so the suite stays fast and reproducible. A real HuggingFace
smoke check is provided but skipped unless ``RAGPOISON_HF_SMOKE=1`` is set, so the
default ``make test`` never triggers a download.
"""

import os
import sys

import pytest

from ragpoison.index import (
    DEFAULT_MOCK_EMBED_DIM,
    CorpusIndex,
    build_index,
    build_index_from_config,
)
from ragpoison.schema import AttackClass, Doc, EmbeddingKind, RunConfig
from ragpoison.corpus_loader import (
    load_benign,
    load_corpus,
    load_by_attack_class,
)


def test_build_returns_corpus_index():
    idx = build_index(load_corpus(), embedding=EmbeddingKind.MOCK)
    assert isinstance(idx, CorpusIndex)
    # Every corpus document made it into the index.
    assert len(idx.documents) == len(load_corpus())


def test_retrieve_returns_docs_respecting_top_k():
    docs = load_corpus()
    idx = build_index(docs, embedding=EmbeddingKind.MOCK)
    retrieved = idx.retrieve("how do I request vacation?", top_k=3)
    assert 1 <= len(retrieved) <= 3
    assert all(isinstance(d, Doc) for d in retrieved)
    known_ids = {d.id for d in docs}
    assert all(d.id in known_ids for d in retrieved)


def test_retrieved_docs_keep_source_and_trusted():
    idx = build_index(load_benign(), embedding=EmbeddingKind.MOCK)
    retrieved = idx.retrieve("password reset help", top_k=3)
    assert retrieved
    for d in retrieved:
        # Provenance metadata survives the round-trip through the index.
        assert isinstance(d.source, str) and d.source
        assert d.trusted is True  # benign corpus is all trusted


def test_nodes_carry_provenance_metadata():
    idx = build_index(load_benign(), embedding=EmbeddingKind.MOCK)
    nodes = list(idx.index.docstore.docs.values())
    assert nodes
    for node in nodes:
        assert "source" in node.metadata
        assert "trusted" in node.metadata
        assert "id" in node.metadata
    # Only the body is embedded: the labels are excluded from embed content.
    sample = nodes[0]
    assert "source" in sample.excluded_embed_metadata_keys
    assert "trusted" in sample.excluded_embed_metadata_keys


def test_inject_poison_is_retrievable_and_keeps_labels():
    benign = load_benign()
    idx = build_index(benign, embedding=EmbeddingKind.MOCK)

    poison = next(
        d
        for d in load_by_attack_class(AttackClass.RETRIEVAL_HIJACK)
        if d.poisoned
    )
    idx.inject(poison)

    # Ask for everything so the constant-vector mock ranking can't hide it.
    retrieved = idx.retrieve("vacation policy", top_k=len(idx.documents))
    ids = {d.id for d in retrieved}
    assert poison.id in ids

    got = next(d for d in retrieved if d.id == poison.id)
    assert got.poisoned is True
    assert got.trusted is False
    assert got.attack_class == AttackClass.RETRIEVAL_HIJACK.value


def test_inject_adds_exactly_one_document():
    benign = load_benign()
    idx = build_index(benign, embedding=EmbeddingKind.MOCK)
    before = len(idx.documents)
    poison = next(
        d for d in load_by_attack_class(AttackClass.DIRECT_INSTRUCTION_IN_DOC)
        if d.poisoned
    )
    idx.inject(poison)
    assert len(idx.documents) == before + 1
    assert poison in idx.documents


def test_build_from_config_uses_mock_by_default():
    # RunConfig defaults to EmbeddingKind.MOCK, so this must not need torch.
    idx = build_index_from_config(RunConfig())
    retrieved = idx.retrieve("expense reimbursement", top_k=2)
    assert 1 <= len(retrieved) <= 2


def test_reused_embed_model_is_honoured():
    from llama_index.core.embeddings import MockEmbedding

    shared = MockEmbedding(embed_dim=DEFAULT_MOCK_EMBED_DIM)
    idx = build_index(load_benign(), embed_model=shared)
    assert idx._embed_model is shared
    assert idx.retrieve("onboarding first week", top_k=1)


def test_mock_path_does_not_load_torch():
    # Building and querying a mock index must never import torch.
    idx = build_index(load_corpus(), embedding=EmbeddingKind.MOCK)
    idx.retrieve("security policy", top_k=3)
    assert "torch" not in sys.modules


@pytest.mark.skipif(
    os.environ.get("RAGPOISON_HF_SMOKE") != "1",
    reason="real HuggingFace embedding smoke; set RAGPOISON_HF_SMOKE=1 to run",
)
def test_hf_embedding_smoke():
    # Tiny real-embedding smoke: build over a couple of docs and retrieve.
    docs = load_benign()[:3]
    idx = build_index(docs, embedding=EmbeddingKind.HF)
    retrieved = idx.retrieve("how do I reset my password?", top_k=1)
    assert len(retrieved) == 1
    assert isinstance(retrieved[0], Doc)
