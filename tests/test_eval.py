"""Eval tests: breach-rate matrix, worked-bypass selection, README splicing.

Self-contained: no embeddings, no torch, no model download. Uses hand-built cells
so the aggregation logic is checked independently of the pipeline.
"""

from pathlib import Path

import pytest

from ragpoison import CANARY
from ragpoison import eval as ev

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"


def _cell(attack, defense, breached, leaked=False, answer="benign", rationale="r"):
    return {
        "attack_class": attack,
        "defense": defense,
        "query": "How many vacation days?",
        "answer": answer,
        "breached": breached,
        "leaked_canary": leaked,
        "rationale": rationale,
        "retrieved_ids": ["a", "b"],
        "quarantined_ids": [],
    }


def _sample_cells():
    # Baseline breaches everywhere; every defense holds. Exfil leaks the canary.
    cells = []
    attacks = [
        "direct_instruction_in_doc",
        "hidden_instruction",
        "retrieval_hijack",
        "data_exfiltration",
    ]
    defenses = ["none", "context_fencing", "injection_classifier", "provenance_filter"]
    for a in attacks:
        for d in defenses:
            breached = d == "none"
            leaked = breached and a == "data_exfiltration"
            answer = (
                "marker {}".format(CANARY)
                if leaked
                else ("SYSTEM OVERRIDE ACCEPTED" if breached else "benign answer")
            )
            cells.append(_cell(a, d, breached, leaked, answer))
    return cells


def test_build_matrix_rates():
    m = ev.build_matrix(_sample_cells())
    assert m["attacks"][0] == "direct_instruction_in_doc"
    assert m["defenses"] == [
        "none",
        "context_fencing",
        "injection_classifier",
        "provenance_filter",
    ]
    assert m["rate"][("direct_instruction_in_doc", "none")] == 1.0
    assert m["rate"][("direct_instruction_in_doc", "provenance_filter")] == 0.0


def test_build_matrix_averages_multiple_trials():
    cells = [
        _cell("direct_instruction_in_doc", "none", True),
        _cell("direct_instruction_in_doc", "none", False),
    ]
    m = ev.build_matrix(cells)
    assert m["rate"][("direct_instruction_in_doc", "none")] == 0.5


def test_render_matrix_markdown_shape():
    md = ev.render_matrix_markdown(_sample_cells())
    lines = md.splitlines()
    assert lines[0].startswith("| Attack class \\ Defense | none |")
    assert "provenance_filter" in lines[0]
    # header + separator + 4 attack rows
    assert len(lines) == 2 + 4
    assert "1.00" in md and "0.00" in md


def test_pick_worked_bypasses_prefers_baseline():
    cells = [
        _cell("hidden_instruction", "context_fencing", True, answer="ACCESS GRANTED"),
        _cell("hidden_instruction", "none", True, answer="ACCESS GRANTED"),
    ]
    chosen = ev.pick_worked_bypasses(cells)
    assert list(chosen.keys()) == ["hidden_instruction"]
    assert chosen["hidden_instruction"]["defense"] == "none"


def test_pick_worked_bypasses_one_per_breached_class():
    chosen = ev.pick_worked_bypasses(_sample_cells())
    assert len(chosen) == 4
    assert all(cell["defense"] == "none" for cell in chosen.values())


def test_render_bypasses_markdown_mentions_each_class_and_leak():
    md = ev.render_bypasses_markdown(_sample_cells())
    for a in ["direct_instruction_in_doc", "data_exfiltration"]:
        assert a in md
    assert "canary leaked" in md


def test_render_bypasses_handles_no_breaches():
    cells = [_cell("retrieval_hijack", "provenance_filter", False)]
    assert "No breaches" in ev.render_bypasses_markdown(cells)


def test_inject_between_markers_replaces_only_inner_content():
    text = "head\n{}\nOLD\n{}\ntail".format(ev.MATRIX_START, ev.MATRIX_END)
    out = ev.inject_between_markers(text, ev.MATRIX_START, ev.MATRIX_END, "NEW")
    assert "OLD" not in out
    assert "NEW" in out
    assert out.startswith("head")
    assert out.rstrip().endswith("tail")
    assert ev.MATRIX_START in out and ev.MATRIX_END in out


def test_inject_between_markers_raises_when_missing():
    with pytest.raises(ValueError):
        ev.inject_between_markers("no markers here", ev.MATRIX_START, ev.MATRIX_END, "x")


def test_render_readme_against_real_readme_file():
    # The shipped README must carry both marker pairs so the runner can render in place.
    text = README.read_text(encoding="utf-8")
    assert ev.MATRIX_START in text and ev.MATRIX_END in text
    assert ev.BYPASS_START in text and ev.BYPASS_END in text
    rendered = ev.render_readme(text, _sample_cells())
    # Rendering is idempotent and leaves the markers in place.
    assert ev.MATRIX_START in rendered and ev.MATRIX_END in rendered
    assert "| Attack class \\ Defense |" in rendered
    assert "data_exfiltration" in rendered
    assert ev.render_readme(rendered, _sample_cells()) == rendered
    # We only read the real README here; we never write it in tests.
