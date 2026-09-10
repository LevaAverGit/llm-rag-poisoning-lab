"""End-to-end attack/defense regression on the real RAG pipeline.

One test per poisoning class plus a defense-holds regression, all driven through
the actual ``ragpoison.rag`` pipeline using LlamaIndex ``MockEmbedding`` and the
mock LLM (``EmbeddingKind.MOCK`` / ``LLMKind.MOCK``) -- so nothing is downloaded
and no torch is required.

These are skipped until the pipeline is importable (llama-index installed and
``ragpoison.rag`` present); at that point they run on the mock backends and form
the core breach/hold regression. ``top_k`` is set high enough to include the
poisoned document, so retrieval never silently hides a breach and the tests
isolate defense behaviour.
"""

import pytest

# Skip the whole module until the real pipeline and its dependency are available.
pytest.importorskip("llama_index", reason="real pipeline needs llama-index")
pytest.importorskip("ragpoison.rag", reason="ragpoison.rag not built yet")

from ragpoison import AttackClass, DefenseName, RunConfig  # noqa: E402
from ragpoison import runner  # noqa: E402

ATTACKS = [c.value for c in AttackClass]

# Retrieve widely so the single poisoned doc is always in context; this makes the
# tests about the defense, not about retrieval luck under MockEmbedding.
BASE = RunConfig(embedding="mock", llm="mock", top_k=25)


def _run(attack, defense):
    return runner.run_cell(attack, defense, BASE, cache={}, mode="refresh")


@pytest.mark.parametrize("attack", ATTACKS)
def test_baseline_is_breached(attack):
    """With no defense, every poisoning class hijacks the mock assistant."""
    cell = _run(attack, DefenseName.NONE)
    assert cell["breached"] is True, cell


@pytest.mark.parametrize("attack", ATTACKS)
def test_provenance_filter_holds(attack):
    """Provenance filtering drops the untrusted poison, so nothing breaches.

    LLM-agnostic: the poisoned doc is `trusted: false`, so it is quarantined
    before generation and cannot change behaviour or leak the canary.
    """
    cell = _run(attack, DefenseName.PROVENANCE_FILTER)
    assert cell["breached"] is False, cell
    assert cell["leaked_canary"] is False, cell
    assert cell["quarantined_ids"], cell


def test_exfiltration_baseline_leaks_canary():
    """The data-exfiltration class leaks the canary when undefended."""
    cell = _run(AttackClass.DATA_EXFILTRATION, DefenseName.NONE)
    assert cell["leaked_canary"] is True, cell
    assert cell["breached"] is True, cell


def test_exfiltration_provenance_holds_no_leak():
    """...and provenance filtering stops the leak (the exfil doc is untrusted)."""
    cell = _run(AttackClass.DATA_EXFILTRATION, DefenseName.PROVENANCE_FILTER)
    assert cell["leaked_canary"] is False, cell
