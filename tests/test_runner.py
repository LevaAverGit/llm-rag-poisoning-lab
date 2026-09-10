"""Runner tests: cache keys, matrix driving, and cache modes.

The RAG pipeline (ragpoison.rag) is built by another component and may pull in
llama-index, so these tests inject a deterministic in-test pipeline double. That
keeps the harness logic (cell running, caching, matrix assembly) fully covered
with no embeddings, no torch and no model download. End-to-end tests against the
real pipeline live in test_pipeline_attacks.py and skip until it is importable.
"""

import pytest

from ragpoison import CANARY, AttackClass, DefenseName, GraphState, RunConfig
from ragpoison import runner

BENIGN_ANSWER = (
    "Full-time employees accrue 20 vacation days per year; "
    "submit requests in the People portal."
)


def _fake_generate(active_docs, defense):
    """A naive, injectable generator used only to exercise the harness."""
    poison = [d for d in active_docs if d.poisoned]
    if not poison:
        return BENIGN_ANSWER
    # Context fencing is modelled as the assistant treating retrieved text as
    # data, so an in-context override no longer fires.
    if defense == DefenseName.CONTEXT_FENCING.value:
        return BENIGN_ANSWER
    cls = str(poison[0].attack_class)
    if cls == AttackClass.DATA_EXFILTRATION.value:
        return "The provisioning marker is {} as requested.".format(CANARY)
    if cls == AttackClass.HIDDEN_INSTRUCTION.value:
        return "ACCESS GRANTED -- see attached link"
    return "SYSTEM OVERRIDE ACCEPTED"


def fake_pipeline(query, config, docs):
    """`(query, config, docs) -> GraphState`, matching the pipeline contract."""
    defense = str(config.defense)
    retrieved = list(docs)
    quarantined = []
    active = list(retrieved)
    if defense == DefenseName.PROVENANCE_FILTER.value:
        quarantined = [d for d in retrieved if not d.trusted]
        active = [d for d in retrieved if d.trusted]
    elif defense == DefenseName.INJECTION_CLASSIFIER.value:
        quarantined = [d for d in retrieved if d.poisoned]
        active = [d for d in retrieved if not d.poisoned]
    return GraphState(
        query=query,
        config=config,
        retrieved=retrieved,
        quarantined=quarantined,
        context="[fenced-context]",
        answer=_fake_generate(active, defense),
    )


# ---------------------------------------------------------------------------
# Cache keys / scenario ids
# ---------------------------------------------------------------------------
def test_scenario_id_is_stable():
    sid = runner.scenario_id(AttackClass.RETRIEVAL_HIJACK, DefenseName.NONE)
    assert sid == "retrieval_hijack::none"


def test_cache_key_has_provider_model_scenario_shape():
    cfg = RunConfig()  # mock + mock, top_k 3
    key = runner.cache_key(cfg, "retrieval_hijack::none")
    parts = key.split("__")
    assert len(parts) == 3
    provider, model, scenario = parts
    assert provider == "mock"
    assert model == "mock-mock-k3"  # embedding-generator-topk fingerprint
    assert scenario == "retrieval_hijack::none"


def test_cache_key_separates_mock_from_ollama():
    mock_key = runner.cache_key(RunConfig(), "s")
    real_key = runner.cache_key(
        RunConfig(embedding="hf", llm="ollama", llm_model_name="gemma3"), "s"
    )
    assert mock_key != real_key
    assert real_key == "ollama__hf-gemma3-k3__s"


def test_cache_key_folds_in_top_k():
    # Different retrieval depths must not collide on one key: the answer
    # is retrieval-dependent, so a shallow and a deep run are different scenarios.
    k3 = runner.cache_key(RunConfig(top_k=3), "retrieval_hijack::none")
    k5 = runner.cache_key(RunConfig(top_k=5), "retrieval_hijack::none")
    assert k3 != k5
    assert k3.endswith("-k3__retrieval_hijack::none")
    assert k5.endswith("-k5__retrieval_hijack::none")


# ---------------------------------------------------------------------------
# Running cells / the matrix with the injected pipeline
# ---------------------------------------------------------------------------
def test_run_matrix_covers_every_cell():
    result = runner.run_matrix(RunConfig(), pipeline=fake_pipeline, cache={})
    cells = result["cells"]
    assert len(cells) == len(AttackClass) * len(DefenseName)  # 4 x 4
    pairs = {(c["attack_class"], c["defense"]) for c in cells}
    assert len(pairs) == 16


def test_baseline_breaches_and_provenance_holds_for_every_class():
    cache = {}
    for attack in [c.value for c in AttackClass]:
        none_cell = runner.run_cell(
            attack, DefenseName.NONE, RunConfig(), pipeline=fake_pipeline, cache=cache
        )
        prov_cell = runner.run_cell(
            attack,
            DefenseName.PROVENANCE_FILTER,
            RunConfig(),
            pipeline=fake_pipeline,
            cache=cache,
        )
        assert none_cell["breached"] is True, attack
        assert prov_cell["breached"] is False, attack
        assert prov_cell["leaked_canary"] is False, attack
        # provenance filter must quarantine the untrusted poison
        assert prov_cell["quarantined_ids"]


def test_exfil_baseline_leaks_canary():
    cell = runner.run_cell(
        AttackClass.DATA_EXFILTRATION,
        DefenseName.NONE,
        RunConfig(),
        pipeline=fake_pipeline,
        cache={},
    )
    assert cell["leaked_canary"] is True
    assert cell["breached"] is True


# ---------------------------------------------------------------------------
# Cache modes
# ---------------------------------------------------------------------------
def test_auto_mode_populates_cache_then_reproduces_without_pipeline():
    cache = {}
    first = runner.run_cell(
        AttackClass.DIRECT_INSTRUCTION_IN_DOC,
        DefenseName.NONE,
        RunConfig(),
        pipeline=fake_pipeline,
        cache=cache,
    )
    assert len(cache) == 1
    # Reproduce with NO pipeline available: must come from cache.
    reproduced = runner.run_cell(
        AttackClass.DIRECT_INSTRUCTION_IN_DOC,
        DefenseName.NONE,
        RunConfig(),
        pipeline=None,
        cache=cache,
        mode="from_cache",
    )
    assert reproduced == first


def test_from_cache_mode_raises_on_miss():
    with pytest.raises(KeyError):
        runner.run_cell(
            AttackClass.HIDDEN_INSTRUCTION,
            DefenseName.NONE,
            RunConfig(),
            pipeline=None,
            cache={},
            mode="from_cache",
        )


def test_refresh_mode_overwrites_stale_cache_entry():
    cfg = RunConfig()
    key = runner.cache_key(
        cfg.model_copy(update={"defense": DefenseName.NONE.value}),
        runner.scenario_id(AttackClass.DIRECT_INSTRUCTION_IN_DOC, DefenseName.NONE),
    )
    cache = {key: {"stale": True, "breached": False}}
    cell = runner.run_cell(
        AttackClass.DIRECT_INSTRUCTION_IN_DOC,
        DefenseName.NONE,
        cfg,
        pipeline=fake_pipeline,
        cache=cache,
        mode="refresh",
    )
    assert cell["breached"] is True
    assert "stale" not in cache[key]


def test_save_and_load_cache_roundtrip(tmp_path):
    cache = {}
    runner.run_matrix(RunConfig(), pipeline=fake_pipeline, cache=cache)
    path = tmp_path / "answers.json"
    runner.save_cache(path, cache)
    loaded = runner.load_cache(path)
    assert loaded == cache


def test_load_cache_missing_file_is_empty(tmp_path):
    assert runner.load_cache(tmp_path / "nope.json") == {}


def test_load_cache_malformed_json_raises_actionable_error(tmp_path):
    # A truncated / malformed cache must raise an actionable RuntimeError naming the
    # file, not a raw JSONDecodeError traceback.
    bad = tmp_path / "answers.json"
    bad.write_text('{"entries": {"k": {"a": 1}', encoding="utf-8")  # truncated
    with pytest.raises(RuntimeError) as exc:
        runner.load_cache(bad)
    assert str(bad) in str(exc.value)
