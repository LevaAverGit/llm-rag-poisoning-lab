"""In-memory LlamaIndex vector index and retriever over the labelled corpus.

This is the retrieval half of the lab. It turns :class:`~ragpoison.schema.Doc`
objects into a LlamaIndex ``VectorStoreIndex`` (an in-memory ``SimpleVectorStore``,
no external vector database) and exposes a thin retriever that hands back the
original :class:`Doc` objects -- metadata intact -- for the defense and generation
stages downstream.

Two design points matter for the rest of the pipeline:

* **Provenance survives retrieval.** Every node carries the document's
  ``source`` / ``trusted`` / ``poisoned`` labels in its metadata, and
  :meth:`CorpusIndex.retrieve` maps each retrieved node back to the exact
  :class:`Doc` it came from (by id). The ``provenance_filter`` defense relies on
  ``Doc.trusted`` being available here.
* **Only the document text is embedded.** The label metadata is excluded from the
  embedded (and LLM-visible) content, so ranking is driven purely by what an
  attacker actually controls -- the document body -- not by our own trust labels.
  That is what makes the ``retrieval_hijack`` class a meaningful test.

Resource discipline
-------------------
The real embedding backend (HuggingFace ``sentence-transformers/all-MiniLM-L6-v2``)
is imported **lazily**, only when an ``hf`` index is actually requested, so torch
is never pulled in by the deterministic ``MockEmbedding`` path the tests use.
Importing this module (or building a mock index) must not load torch.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from .schema import Doc, EmbeddingKind, RunConfig

# Kept in sync with RunConfig.embed_model_name; the sentence-transformers model
# used for a real ("hf") run. Small (~90MB) and downloaded on first use only.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Vector width for the deterministic test embedding. MockEmbedding returns a
# constant vector, so the exact dimension is immaterial; a small one keeps the
# in-memory store tiny.
DEFAULT_MOCK_EMBED_DIM = 8

# Module-level cache of embedding models keyed by (kind, name, dim), so a real HF
# model loads once per process instead of once per matrix cell (finding 4).
_EMBED_MODEL_CACHE: Dict[tuple, object] = {}

# Tokeniser for the deterministic mock ranking below.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set:
    """Lower-cased alphanumeric token set for lexical overlap scoring."""
    return set(_TOKEN_RE.findall(text.lower()))


def _lexical_overlap(query_tokens: set, doc_text: str) -> int:
    """Number of distinct query tokens that also appear in ``doc_text``."""
    if not query_tokens:
        return 0
    doc_tokens = _tokenize(doc_text)
    return sum(1 for t in query_tokens if t in doc_tokens)


def _is_mock_embed_model(embed_model) -> bool:
    """True iff ``embed_model`` is LlamaIndex's deterministic ``MockEmbedding``.

    Used to pick the query-aware lexical ranking (below) for the offline/test path,
    since ``MockEmbedding`` returns a constant vector and cannot rank by content.
    """
    try:
        from llama_index.core.embeddings import MockEmbedding
    except Exception:  # pragma: no cover - llama-index always ships this
        return False
    return isinstance(embed_model, MockEmbedding)


def _resolve_embed_model(
    embedding: EmbeddingKind,
    embed_model_name: str,
    mock_embed_dim: int,
):
    """Return a LlamaIndex embedding model for the requested backend (memoized).

    ``mock`` uses LlamaIndex's deterministic ``MockEmbedding`` (no torch, no
    download). ``hf`` lazily imports ``HuggingFaceEmbedding`` so the heavy
    sentence-transformers / torch stack is only touched on a real run. Built models
    are cached by (kind, name, dim) so each loads at most once per process.
    """
    kind = EmbeddingKind(embedding)
    cache_key = (kind.value, embed_model_name, mock_embed_dim)
    cached = _EMBED_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if kind == EmbeddingKind.MOCK:
        from llama_index.core.embeddings import MockEmbedding

        model = MockEmbedding(embed_dim=mock_embed_dim)
    elif kind == EmbeddingKind.HF:
        try:
            # Lazy import: this line is what pulls in torch, so it must stay
            # inside the hf branch and never run for mock-based tests.
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise ImportError(
                "The 'hf' embedding backend needs "
                "'llama-index-embeddings-huggingface' and 'sentence-transformers' "
                "(see requirements.txt)."
            ) from exc
        model = HuggingFaceEmbedding(model_name=embed_model_name)
    else:  # pragma: no cover - EmbeddingKind(...) already rejects anything else
        raise ValueError(f"unknown embedding kind: {embedding!r}")
    _EMBED_MODEL_CACHE[cache_key] = model
    return model


def _doc_to_node(doc: Doc) -> TextNode:
    """Build a LlamaIndex ``TextNode`` from a :class:`Doc`.

    The node id equals the document id so retrieval can be mapped back to the
    source document. Label metadata (source / trusted / poisoned / ...) rides
    along on the node but is excluded from both the embedded content and the
    LLM-visible content, so only ``doc.text`` drives ranking and generation.
    """
    metadata = {k: v for k, v in doc.to_metadata().items() if v is not None}
    node = TextNode(text=doc.text, id_=doc.id, metadata=metadata)
    excluded = list(metadata.keys())
    node.excluded_embed_metadata_keys = excluded
    node.excluded_llm_metadata_keys = excluded
    return node


class CorpusIndex:
    """A built vector index plus the mapping back to source :class:`Doc` objects.

    Construct one with :func:`build_index` or :func:`build_index_from_config`.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        docs_by_id: Dict[str, Doc],
        embed_model=None,
        lexical: bool = False,
    ) -> None:
        self._index = index
        self._docs_by_id: Dict[str, Doc] = dict(docs_by_id)
        self._embed_model = embed_model
        # When True, retrieval uses the deterministic query-aware lexical ranking
        # instead of the (constant-vector) MockEmbedding vector store, so the offline
        # mock path actually ranks by content and the poison can win top-k.
        self._lexical = lexical

    @property
    def index(self) -> VectorStoreIndex:
        """The underlying LlamaIndex ``VectorStoreIndex`` (in-memory)."""
        return self._index

    @property
    def documents(self) -> List[Doc]:
        """All documents currently in the index, including any injected ones."""
        return list(self._docs_by_id.values())

    def retrieve(self, query: str, top_k: int = 3) -> List[Doc]:
        """Return the top-``k`` documents for ``query``, highest score first.

        Each retrieved document is the full, labelled :class:`Doc` -- text,
        ``source``, ``trusted`` and the poisoning fields -- not a bare text chunk.
        A lexical (mock) index ranks by query/document keyword overlap; a real (``hf``)
        index ranks via the vector store, resolving each node back to its source
        document by id.
        """
        if self._lexical:
            return self._retrieve_lexical(query, top_k)
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        scored_nodes = retriever.retrieve(query)
        out: List[Doc] = []
        for scored in scored_nodes:
            node = scored.node
            doc_id = node.metadata.get("id", node.node_id)
            doc = self._docs_by_id.get(doc_id)
            if doc is not None:
                out.append(doc)
        return out

    def _retrieve_lexical(self, query: str, top_k: int) -> List[Doc]:
        """Deterministic query-aware ranking for the offline/mock path.

        Scores each document by how many distinct query tokens it contains and
        returns the ``top_k`` highest, breaking ties by insertion order so the result
        is stable. This makes mock retrieval depend on content -- the poison can win
        top-k when it is relevant, and the keyword-stuffed ``retrieval_hijack`` doc
        can ride in on unrelated queries -- instead of the constant-vector no-op.
        """
        query_tokens = _tokenize(query)
        docs = list(self._docs_by_id.values())
        ranked = sorted(
            enumerate(docs),
            key=lambda pair: (-_lexical_overlap(query_tokens, pair[1].text), pair[0]),
        )
        limit = max(0, top_k)
        return [doc for _, doc in ranked[:limit]]

    def inject(self, doc: Doc) -> Doc:
        """Slip a document into an already-built index.

        This models the lab's threat: the attacker does not talk to the chat, they
        get a document into the corpus so the retriever may pull it into context.
        The doc keeps its own labels, so a later provenance check still sees it as
        untrusted / poisoned. Returns the injected document.
        """
        node = _doc_to_node(doc)
        self._index.insert_nodes([node])
        self._docs_by_id[doc.id] = doc
        return doc


def build_index(
    docs: Iterable[Doc],
    embedding: EmbeddingKind = EmbeddingKind.MOCK,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    embed_model=None,
    mock_embed_dim: int = DEFAULT_MOCK_EMBED_DIM,
) -> CorpusIndex:
    """Build an in-memory vector index over ``docs``.

    By default this uses the deterministic ``MockEmbedding`` so it is fast, safe
    and torch-free -- exactly what the tests need. Pass ``embedding='hf'`` for a
    real sentence-transformers run (imported lazily), or hand in a ready-made
    ``embed_model`` to reuse one across cells of the matrix.
    """
    docs = list(docs)
    if embed_model is None:
        embed_model = _resolve_embed_model(embedding, embed_model_name, mock_embed_dim)
    nodes = [_doc_to_node(d) for d in docs]
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    docs_by_id = {d.id: d for d in docs}
    # A MockEmbedding cannot rank by content (constant vector), so the mock path
    # uses the deterministic lexical ranking instead.
    lexical = _is_mock_embed_model(embed_model)
    return CorpusIndex(
        index=index, docs_by_id=docs_by_id, embed_model=embed_model, lexical=lexical
    )


def build_index_from_config(
    config: RunConfig, docs: Optional[Iterable[Doc]] = None
) -> CorpusIndex:
    """Build an index from a :class:`RunConfig` (loading the full corpus if needed).

    Reads the embedding backend and model name from ``config`` so a matrix cell can
    build its index straight from its run configuration.
    """
    if docs is None:
        from .corpus_loader import load_corpus

        docs = load_corpus()
    return build_index(
        docs,
        embedding=config.embedding,
        embed_model_name=config.embed_model_name,
    )


__all__ = [
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_MOCK_EMBED_DIM",
    "CorpusIndex",
    "build_index",
    "build_index_from_config",
]
