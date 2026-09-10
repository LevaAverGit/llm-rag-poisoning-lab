"""Load the labelled corpus (benign + poisoned) from YAML into :class:`Doc`.

Each ``*.yml`` file under ``corpus/`` holds either a single doc mapping or a list
of doc mappings, each matching the :class:`Doc` schema. This loader is pure Python
(PyYAML + Pydantic only): no embeddings, no torch, so tests can validate the whole
corpus without downloading anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml

from .schema import CANARY, AttackClass, Doc

# Repo layout: <root>/ragpoison/corpus_loader.py  ->  <root>/corpus
CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def _load_yaml_file(path: Path) -> List[Doc]:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        return []
    records = raw if isinstance(raw, list) else [raw]
    return [Doc.model_validate(rec) for rec in records]


def load_corpus(corpus_dir: Optional[Path] = None) -> List[Doc]:
    """Load and validate every document in the corpus.

    Raises on duplicate ids so a mislabelled corpus fails loudly.
    """
    base = Path(corpus_dir) if corpus_dir is not None else CORPUS_DIR
    docs: List[Doc] = []
    for path in sorted(base.rglob("*.yml")):
        docs.extend(_load_yaml_file(path))
    seen: dict = {}
    for doc in docs:
        if doc.id in seen:
            raise ValueError(f"duplicate document id: {doc.id!r}")
        seen[doc.id] = doc
    return docs


def load_benign(corpus_dir: Optional[Path] = None) -> List[Doc]:
    return [d for d in load_corpus(corpus_dir) if not d.poisoned]


def load_poisoned(corpus_dir: Optional[Path] = None) -> List[Doc]:
    return [d for d in load_corpus(corpus_dir) if d.poisoned]


def load_by_attack_class(
    attack_class: AttackClass, corpus_dir: Optional[Path] = None
) -> List[Doc]:
    """Benign docs plus only the poisoned docs of one class.

    This is what a single matrix cell indexes: the clean corpus with exactly one
    class of poison mixed in, so a breach is attributable to that class.
    """
    wanted = AttackClass(attack_class).value
    out: List[Doc] = []
    for d in load_corpus(corpus_dir):
        if not d.poisoned or d.attack_class == wanted:
            out.append(d)
    return out


def get_canary_doc(corpus_dir: Optional[Path] = None) -> Doc:
    """Return the single benign document that carries the canary token."""
    matches = [d for d in load_benign(corpus_dir) if CANARY in d.text]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one benign doc carrying the canary, found {len(matches)}"
        )
    return matches[0]


__all__ = [
    "CORPUS_DIR",
    "load_corpus",
    "load_benign",
    "load_poisoned",
    "load_by_attack_class",
    "get_canary_doc",
]
