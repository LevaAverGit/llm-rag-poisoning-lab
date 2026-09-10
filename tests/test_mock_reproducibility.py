"""Regression: the offline mock matrix reproduces the committed cache at top_k=3.

This is the guard for the retrieval no-op bug. Before the query-aware
mock ranking, ``MockEmbedding`` returned a constant vector, so at the shipped default
``top_k=3`` nothing content-relevant was retrieved and every cell collapsed to "not
breached" -- while the committed cache claimed otherwise. Running the real mock
pipeline (``MockEmbedding`` + mock LLM, no torch, no downloads) at ``top_k=3`` must
now reproduce the committed cache cell-for-cell, so a future regression to the no-op
is caught here instead of hiding behind a stale cache.
"""

import pytest

pytest.importorskip("llama_index", reason="mock pipeline needs llama-index")
pytest.importorskip("ragpoison.rag", reason="ragpoison.rag not built yet")

from ragpoison import AttackClass, RunConfig  # noqa: E402
from ragpoison import runner  # noqa: E402

# The three torch-free defenses the offline `make run` reproduces (the
# injection_classifier column is only in the real `make run-llm` cache).
MOCK_DEFENSES = ["none", "context_fencing", "provenance_filter"]
ATTACKS = [c.value for c in AttackClass]

# The shipped default retrieval depth. The whole point of the regression is that the
# matrix is meaningful and reproducible at *this exact* top_k -- not only when top_k
# is cranked up high enough to drag the poison in by force.
SHIPPED = RunConfig(embedding="mock", llm="mock", top_k=3)


@pytest.fixture(scope="module")
def committed_cache():
    cache = runner.load_cache(runner.DEFAULT_CACHE)
    assert cache, "committed answer cache is empty/missing"
    return cache


@pytest.mark.parametrize("attack", ATTACKS)
@pytest.mark.parametrize("defense", MOCK_DEFENSES)
def test_mock_cell_reproduces_committed_cache(attack, defense, committed_cache):
    key = runner.cache_key(
        SHIPPED.model_copy(update={"defense": defense}),
        runner.scenario_id(attack, defense),
    )
    assert key in committed_cache, "no committed mock entry for {}".format(key)
    produced = runner.run_cell(attack, defense, SHIPPED, cache={}, mode="refresh")
    assert produced == committed_cache[key], key


def test_mock_baseline_is_non_degenerate():
    # The no-op bug made every baseline cell "not breached"; guard against regressing
    # to that by asserting the undefended baseline still breaches every class, with
    # the class's poison actually riding into the retrieved top-k at top_k=3.
    for attack in ATTACKS:
        cell = runner.run_cell(attack, "none", SHIPPED, cache={}, mode="refresh")
        assert cell["breached"] is True, cell
        assert len(cell["retrieved_ids"]) == 3, cell
        assert any("poison" in rid for rid in cell["retrieved_ids"]), cell
