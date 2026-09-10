"""Evaluation: turn matrix cells into a breach-rate table and worked bypasses.

Consumes the ``cells`` produced by :mod:`ragpoison.runner` and produces:

* a **breach-rate matrix** (poisoning class x defense) -- the fraction of trials
  in each cell where the injection changed behaviour or leaked the canary, and
* **one worked bypass per class** -- a concrete breached example (preferring the
  undefended baseline) with its query, answer and the judge's rationale.

Both are rendered to Markdown and spliced into ``README.md`` between HTML-comment
markers, so ``make run --render-readme`` (M5) refreshes the published numbers in
place. Pure standard library -- no pandas -- so it imports instantly and the test
suite stays light.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import AttackClass, DefenseName

# README splice points. The content between each pair is regenerated; text
# outside them (including the surrounding prose) is left untouched.
MATRIX_START = "<!-- BREACH_MATRIX:START -->"
MATRIX_END = "<!-- BREACH_MATRIX:END -->"
BYPASS_START = "<!-- WORKED_BYPASSES:START -->"
BYPASS_END = "<!-- WORKED_BYPASSES:END -->"

# Canonical ordering for the table axes (matches the README layout).
ATTACK_ORDER: List[str] = [c.value for c in AttackClass]
DEFENSE_ORDER: List[str] = [d.value for d in DefenseName]

__all__ = [
    "MATRIX_START",
    "MATRIX_END",
    "BYPASS_START",
    "BYPASS_END",
    "build_matrix",
    "render_matrix_markdown",
    "pick_worked_bypasses",
    "render_bypasses_markdown",
    "inject_between_markers",
    "render_readme",
    "write_readme",
]


def _ordered(values: List[str], order: List[str]) -> List[str]:
    """Return ``values`` in canonical ``order``, appending any unknown extras."""
    present = set(values)
    out = [v for v in order if v in present]
    out.extend(v for v in values if v not in order)
    return out


def build_matrix(cells: List[dict]) -> dict:
    """Aggregate cells into per-(attack, defense) breach counts and rates.

    Robust to more than one trial per cell (rate = breached / total), though the
    deterministic lab runs a single trial so rates are 0.0 or 1.0.
    """
    attacks = _ordered(sorted({c["attack_class"] for c in cells}), ATTACK_ORDER)
    defenses = _ordered(sorted({c["defense"] for c in cells}), DEFENSE_ORDER)
    total: Dict[Tuple[str, str], int] = {}
    breached: Dict[Tuple[str, str], int] = {}
    for c in cells:
        key = (c["attack_class"], c["defense"])
        total[key] = total.get(key, 0) + 1
        breached[key] = breached.get(key, 0) + (1 if c["breached"] else 0)
    rate: Dict[Tuple[str, str], Optional[float]] = {}
    for a in attacks:
        for d in defenses:
            t = total.get((a, d), 0)
            rate[(a, d)] = (breached[(a, d)] / t) if t else None
    return {
        "attacks": attacks,
        "defenses": defenses,
        "total": total,
        "breached": breached,
        "rate": rate,
    }


def _fmt_rate(value: Optional[float]) -> str:
    return "n/a" if value is None else "{:.2f}".format(value)


def render_matrix_markdown(cells: List[dict]) -> str:
    """Render the breach-rate matrix as a GitHub-flavoured Markdown table."""
    m = build_matrix(cells)
    header = "| Attack class \\ Defense | " + " | ".join(m["defenses"]) + " |"
    sep = "|" + "---|" * (len(m["defenses"]) + 1)
    rows = [header, sep]
    for a in m["attacks"]:
        cells_md = " | ".join(_fmt_rate(m["rate"][(a, d)]) for d in m["defenses"])
        rows.append("| {} | {} |".format(a, cells_md))
    return "\n".join(rows)


def pick_worked_bypasses(cells: List[dict]) -> "OrderedDict[str, dict]":
    """Pick one breached cell per attack class, preferring the ``none`` baseline.

    Returns an ordered mapping ``attack_class -> cell`` for classes that breached
    at least once. Classes that never breached are omitted.
    """
    by_class: Dict[str, List[dict]] = {}
    for c in cells:
        if c["breached"]:
            by_class.setdefault(c["attack_class"], []).append(c)
    chosen: "OrderedDict[str, dict]" = OrderedDict()
    for a in _ordered(list(by_class.keys()), ATTACK_ORDER):
        candidates = by_class[a]
        best = next(
            (c for c in candidates if c["defense"] == DefenseName.NONE.value),
            candidates[0],
        )
        chosen[a] = best
    return chosen


def _truncate(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render_bypasses_markdown(cells: List[dict]) -> str:
    """Render the one-worked-bypass-per-class write-up as Markdown."""
    bypasses = pick_worked_bypasses(cells)
    if not bypasses:
        return "_No breaches were observed in this run._"
    blocks: List[str] = []
    for attack, c in bypasses.items():
        leaked = " (canary leaked)" if c.get("leaked_canary") else ""
        block = (
            "**{attack}** — defense `{defense}`{leaked}\n\n"
            "- _Query:_ {query}\n"
            "- _Answer:_ {answer}\n"
            "- _Judge:_ {rationale}"
        ).format(
            attack=attack,
            defense=c["defense"],
            leaked=leaked,
            query=_truncate(c.get("query", ""), 200),
            answer=_truncate(c.get("answer", "")),
            rationale=_truncate(c.get("rationale", ""), 300),
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def inject_between_markers(text: str, start: str, end: str, payload: str) -> str:
    """Replace the content between ``start`` and ``end`` markers with ``payload``.

    The markers themselves are preserved. Raises if either marker is missing or
    they are out of order, so a malformed README fails loudly rather than silently
    dropping results.
    """
    i = text.find(start)
    j = text.find(end)
    if i == -1 or j == -1:
        raise ValueError("markers not found: {!r} / {!r}".format(start, end))
    if j < i:
        raise ValueError("end marker precedes start marker")
    head = text[: i + len(start)]
    tail = text[j:]
    return "{}\n\n{}\n\n{}".format(head, payload.strip(), tail)


def render_readme(readme_text: str, cells: List[dict]) -> str:
    """Return ``readme_text`` with the matrix and bypass sections regenerated."""
    out = inject_between_markers(
        readme_text, MATRIX_START, MATRIX_END, render_matrix_markdown(cells)
    )
    out = inject_between_markers(
        out, BYPASS_START, BYPASS_END, render_bypasses_markdown(cells)
    )
    return out


def write_readme(readme_path, cells: List[dict]) -> None:
    """Read the README, splice in the rendered sections, and write it back."""
    p = Path(readme_path)
    text = p.read_text(encoding="utf-8")
    p.write_text(render_readme(text, cells), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_cells(results_path: Path) -> List[dict]:
    with Path(results_path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "cells" in data:
        return data["cells"]
    if isinstance(data, list):
        return data
    raise ValueError("unrecognised results file: expected a run result or a cell list")


def _build_parser() -> argparse.ArgumentParser:
    from .runner import DEFAULT_OUT, REPO_ROOT

    p = argparse.ArgumentParser(
        prog="python -m ragpoison.eval",
        description="Render the breach-rate matrix and worked bypasses from a run result.",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_OUT,
        help="Run-result JSON produced by the runner.",
    )
    p.add_argument(
        "--readme", type=Path, default=REPO_ROOT / "README.md", help="README path."
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="Write the rendered sections back into the README (default: print).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cells = _load_cells(args.results)
    if args.write:
        write_readme(args.readme, cells)
        print("Wrote matrix + bypasses into {}".format(args.readme))
    else:
        print(render_matrix_markdown(cells))
        print()
        print(render_bypasses_markdown(cells))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
