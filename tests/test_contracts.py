"""Foundational tests for the contracts and the labelled corpus.

Pure Python: PyYAML + Pydantic only. No embeddings, no torch, no downloads.
Pipeline builders add their own MockEmbedding / mock-LLM tests on top of this.
"""

import pytest

from ragpoison import (
    CANARY,
    AttackClass,
    Doc,
    Verdict,
    load_by_attack_class,
    load_corpus,
    get_canary_doc,
)
from ragpoison.corpus_loader import _load_yaml_file


def test_corpus_loader_reports_malformed_yaml_by_name(tmp_path):
    # A malformed corpus file must raise an actionable RuntimeError naming the file
    # rather than a raw YAML traceback (finding 6).
    bad = tmp_path / "broken.yml"
    bad.write_text("id: x\n  bad: : indentation\n:\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as exc:
        _load_yaml_file(bad)
    assert str(bad) in str(exc.value)


def test_corpus_loads_and_ids_unique():
    docs = load_corpus()
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids))
    assert len(docs) >= 10  # 7 benign + 4 poisoned at minimum


def test_all_four_attack_classes_present():
    poisoned = [d for d in load_corpus() if d.poisoned]
    classes = {d.attack_class for d in poisoned}
    assert classes == {c.value for c in AttackClass}


def test_poisoned_docs_are_labelled():
    for d in load_corpus():
        if d.poisoned:
            assert d.attack_class is not None
            assert d.payload
            assert d.target
            assert d.trusted is False
        else:
            assert d.attack_class is None
            assert d.payload is None
            assert d.target is None


def test_exactly_one_canary_doc_and_it_is_benign():
    canary_doc = get_canary_doc()
    assert canary_doc.poisoned is False
    assert canary_doc.trusted is True
    assert CANARY in canary_doc.text


def test_exfiltration_targets_the_canary():
    exfil = [
        d
        for d in load_corpus()
        if d.attack_class == AttackClass.DATA_EXFILTRATION.value
    ]
    assert exfil
    assert any(CANARY in (d.target or "") for d in exfil)


def test_load_by_attack_class_isolates_one_class():
    subset = load_by_attack_class(AttackClass.RETRIEVAL_HIJACK)
    poisoned = [d for d in subset if d.poisoned]
    assert poisoned
    assert all(
        d.attack_class == AttackClass.RETRIEVAL_HIJACK.value for d in poisoned
    )
    # benign docs are still present
    assert any(not d.poisoned for d in subset)


def test_doc_rejects_inconsistent_labels():
    with pytest.raises(Exception):
        Doc(
            id="bad-1",
            text="x",
            source="s",
            trusted=True,
            poisoned=True,  # poisoned but no attack_class
        )
    with pytest.raises(Exception):
        Doc(
            id="bad-2",
            text="x",
            source="s",
            trusted=True,
            poisoned=False,
            payload="should not be here",  # benign but carries a payload
        )


def test_doc_id_rejects_fence_breaking_characters():
    # Attacker-authored ids must not be able to carry fence-breaking characters
    # (finding 5 defence-in-depth at the schema boundary).
    for bad_id in ['a"b', "a<b", "a>b", "a\nb", "", "   "]:
        with pytest.raises(Exception):
            Doc(
                id=bad_id,
                text="x",
                source="internal_wiki",
                trusted=True,
                poisoned=False,
            )
    # A normal id is still accepted.
    ok = Doc(id="poison-direct-101", text="x", source="s", trusted=True, poisoned=False)
    assert ok.id == "poison-direct-101"


def test_verdict_canary_leak_forces_breach():
    v = Verdict(breached=False, leaked_canary=True, rationale="canary appeared")
    assert v.breached is True

    clean = Verdict(breached=False, leaked_canary=False, rationale="benign answer")
    assert clean.breached is False
