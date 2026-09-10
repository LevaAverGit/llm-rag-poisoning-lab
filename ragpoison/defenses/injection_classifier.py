"""The ``injection_classifier`` defense -- filtering by a learned detector.

It scores each retrieved document with a HuggingFace prompt-injection classifier
(default ``protectai/deberta-v3-base-prompt-injection-v2``) and quarantines any whose
predicted probability of being an injection reaches a threshold, before the survivors
reach the generator. Unlike ``provenance_filter`` it does not rely on trust labels --
it inspects the content itself -- so it is the only defense here that could, in
principle, catch a poisoned document that arrived through a trusted channel.

Resource discipline and graceful skip
--------------------------------------
The classifier is a heavy, optional extra (``requirements-classifier.txt`` pulls
``transformers`` + ``torch``). It is imported **lazily**: importing this module, or
constructing the defense, never touches ``transformers``. The model is only built the
first time :meth:`InjectionClassifierDefense.apply` needs to score something.

If ``transformers`` (or the model) is unavailable, the defense **graceful-skips**: it
behaves like the ``none`` baseline (keeps everything, quarantines nothing) and records
``skipped=True`` with a human-readable ``reason`` in its ``meta``, instead of failing
the run. This keeps the matrix runnable end-to-end without the optional install; that
cell simply reports "not evaluated" rather than crashing.

Testability
-----------
:meth:`apply` scores through an injectable ``scorer`` callable (``text -> float`` in
``[0, 1]``, the probability of injection). Tests pass a deterministic fake scorer to
exercise the quarantine logic with no torch and no download; production leaves it
``None`` so the real HF pipeline is built lazily.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ..schema import Doc, DefenseName, RunConfig
from .base import Defense, DefenseOutput, render_documents

# A scorer maps a document body to P(injection) in [0, 1].
Scorer = Callable[[str], float]

# Default decision threshold: quarantine a document when P(injection) >= this.
DEFAULT_THRESHOLD = 0.5

# Truncate very long bodies before the classifier; deberta-style models cap at 512
# tokens and would otherwise error on long documents.
_MAX_CHARS = 4000

# Module-level cache of built HF scorers, keyed by model name, so the (heavy) model
# loads once per process instead of once per matrix cell (finding 4). Only the real
# HF path populates this; the mock/test paths never reach _build_hf_scorer.
_HF_SCORER_CACHE: Dict[str, Tuple[Optional[Scorer], Optional[str]]] = {}


def _build_hf_scorer(model_name: str) -> Tuple[Optional[Scorer], Optional[str]]:
    """Lazily build a real HuggingFace injection scorer (memoized by model name).

    Returns ``(scorer, None)`` on success or ``(None, reason)`` if ``transformers``
    or the model is unavailable -- the caller turns a ``None`` scorer into a
    graceful skip. This is the only place ``transformers`` is imported.
    """
    if model_name in _HF_SCORER_CACHE:  # pragma: no cover - real-model path
        return _HF_SCORER_CACHE[model_name]

    result = _load_hf_scorer(model_name)
    _HF_SCORER_CACHE[model_name] = result
    return result


def _load_hf_scorer(model_name: str) -> Tuple[Optional[Scorer], Optional[str]]:
    """Build a fresh HF scorer (uncached); see :func:`_build_hf_scorer`."""
    try:
        # Lazy import: this line is what pulls in transformers/torch, so it must
        # stay inside this function and never run for the mock/test paths.
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - depends on optional deps
        return None, "transformers not installed ({0})".format(exc)

    try:  # pragma: no cover - downloads/loads a model, exercised only on a real run
        clf = pipeline("text-classification", model=model_name)
    except Exception as exc:  # pragma: no cover - network/model dependent
        return None, "could not load classifier model {0!r} ({1})".format(
            model_name, exc
        )

    def scorer(text: str) -> float:  # pragma: no cover - real-model path
        # No fail-open here: a scoring fault propagates so the caller can FAIL CLOSED
        # (quarantine the document and record the error) rather than silently keep it.
        rows = clf(text[:_MAX_CHARS], truncation=True)
        row = rows[0] if isinstance(rows, list) else rows
        label = str(row.get("label", "")).upper()
        score = float(row.get("score", 0.0))
        # protectai returns SAFE / INJECTION; some models emit LABEL_0 / LABEL_1.
        if "INJECT" in label or label in ("LABEL_1", "UNSAFE", "JAILBREAK"):
            return score
        return 1.0 - score

    return scorer, None


class InjectionClassifierDefense(Defense):
    """Score each retrieved document and quarantine likely injections.

    Graceful-skips to a passthrough (with ``skipped=True`` in ``meta``) when the
    classifier is unavailable.
    """

    name = DefenseName.INJECTION_CLASSIFIER.value

    def __init__(
        self,
        model_name: str = RunConfig().injection_classifier_model,
        threshold: float = DEFAULT_THRESHOLD,
        scorer: Optional[Scorer] = None,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        # An explicitly injected scorer (used by tests). When None, a real HF scorer
        # is built lazily on first use and cached in `_scorer`.
        self._scorer: Optional[Scorer] = scorer
        self._scorer_built = scorer is not None
        self._skip_reason: Optional[str] = None

    def _get_scorer(self) -> Optional[Scorer]:
        """Return the scorer, building the HF one lazily on first use.

        Returns ``None`` (and records :attr:`_skip_reason`) when the classifier
        cannot be loaded, which drives the graceful-skip path in :meth:`apply`.
        """
        if self._scorer is not None:
            return self._scorer
        if self._scorer_built:
            # Already tried and failed to build; stay skipped without retrying.
            return None
        scorer, reason = _build_hf_scorer(self._model_name)
        self._scorer = scorer
        self._skip_reason = reason
        self._scorer_built = True
        return scorer

    def apply(self, retrieved_docs: List[Doc], query: str = "") -> DefenseOutput:
        docs = list(retrieved_docs)
        scorer = self._get_scorer()

        if scorer is None:
            # Graceful skip: behave like the baseline, but say so in meta.
            reason = self._skip_reason or "injection classifier unavailable"
            return self._result(
                render_documents(docs),
                kept=docs,
                quarantined=[],
                skipped=True,
                reason=reason,
                threshold=self._threshold,
                model=self._model_name,
            )

        kept: List[Doc] = []
        quarantined: List[Doc] = []
        scores: Dict[str, float] = {}
        score_errors: List[Dict[str, str]] = []
        for doc in docs:
            try:
                prob = float(scorer(doc.text))
            except Exception as exc:
                # Fail CLOSED: a scoring fault must not silently pass an unscored
                # document through. Quarantine it and record why, so the failure is
                # visible in the matrix instead of masquerading as "kept, score 0.0".
                score_errors.append({"id": doc.id, "error": str(exc)})
                quarantined.append(doc)
                continue
            scores[doc.id] = prob
            # High P(injection) -> quarantine; low -> keep. (Gating verified: the
            # scorer returns P(injection), so `>= threshold` correctly drops the
            # documents the classifier flags as injections.)
            if prob >= self._threshold:
                quarantined.append(doc)
            else:
                kept.append(doc)

        return self._result(
            render_documents(kept),
            kept=kept,
            quarantined=quarantined,
            skipped=False,
            scores=scores,
            score_errors=score_errors,
            threshold=self._threshold,
            model=self._model_name,
        )


__all__ = [
    "Scorer",
    "DEFAULT_THRESHOLD",
    "InjectionClassifierDefense",
]
