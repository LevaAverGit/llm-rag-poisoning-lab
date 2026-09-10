"""Defenses: the retrieve -> **defense** -> generate stage of the pipeline.

Every defense implements one interface -- ``apply(retrieved_docs, query) ->
(context, meta)`` -- and is registered here by its
:class:`~ragpoison.schema.DefenseName`:

* ``none`` -- undefended baseline (:class:`~.passthrough.NoneDefense`);
* ``context_fencing`` -- wrap docs in a data boundary + a standing "data, not
  instructions" rule (:class:`~.context_fencing.ContextFencingDefense`);
* ``provenance_filter`` -- keep only trusted-source docs
  (:class:`~.provenance_filter.ProvenanceFilterDefense`);
* ``injection_classifier`` -- score docs with a HuggingFace prompt-injection model and
  quarantine likely injections, lazily loaded with graceful skip
  (:class:`~.injection_classifier.InjectionClassifierDefense`).

Typical use from the pipeline is :func:`run_defense`, which reads the chosen defense
(and the classifier model name) straight from a :class:`~ragpoison.schema.RunConfig`::

    context, meta = run_defense(config, retrieved_docs, query)
    state.context = context
    state.quarantined = meta["quarantined"]

Importing this package is cheap and torch-free: the heavy ``transformers`` dependency
behind ``injection_classifier`` is only touched when that defense actually scores.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Type

from ..schema import Doc, DefenseName, RunConfig
from .base import (
    DOC_SEPARATOR,
    Defense,
    DefenseMeta,
    DefenseOutput,
    render_documents,
)
from .context_fencing import ContextFencingDefense
from .injection_classifier import InjectionClassifierDefense
from .passthrough import NoneDefense
from .provenance_filter import ProvenanceFilterDefense

# Name -> defense class registry. Keys are DefenseName *values* (plain strings), so a
# raw string from a RunConfig (which stores enums as their values) or a DefenseName
# member both resolve.
DEFENSE_CLASSES: Dict[str, Type[Defense]] = {
    DefenseName.NONE.value: NoneDefense,
    DefenseName.CONTEXT_FENCING.value: ContextFencingDefense,
    DefenseName.PROVENANCE_FILTER.value: ProvenanceFilterDefense,
    DefenseName.INJECTION_CLASSIFIER.value: InjectionClassifierDefense,
}


def available_defenses() -> List[str]:
    """The registered defense names, in matrix order."""
    return [
        DefenseName.NONE.value,
        DefenseName.CONTEXT_FENCING.value,
        DefenseName.PROVENANCE_FILTER.value,
        DefenseName.INJECTION_CLASSIFIER.value,
    ]


def get_defense(name, **kwargs) -> Defense:
    """Construct the defense registered under ``name``.

    ``name`` may be a :class:`~ragpoison.schema.DefenseName` or its string value.
    ``kwargs`` are forwarded to the defense constructor (only
    :class:`InjectionClassifierDefense` takes any -- ``model_name`` / ``threshold`` /
    ``scorer``). Raises :class:`ValueError` for an unknown name.
    """
    try:
        key = DefenseName(name).value
    except ValueError as exc:
        raise ValueError(
            "unknown defense {0!r}; available: {1}".format(
                name, ", ".join(available_defenses())
            )
        ) from exc
    return DEFENSE_CLASSES[key](**kwargs)


def build_defense(config: RunConfig) -> Defense:
    """Construct the defense selected by ``config.defense``.

    The classifier is wired up with ``config.injection_classifier_model``; the other
    defenses take no configuration.
    """
    name = DefenseName(config.defense).value
    if name == DefenseName.INJECTION_CLASSIFIER.value:
        return InjectionClassifierDefense(model_name=config.injection_classifier_model)
    return get_defense(name)


def run_defense(
    config: RunConfig, retrieved_docs: List[Doc], query: str = ""
) -> DefenseOutput:
    """Build the config's defense and apply it in one call.

    Convenience for the pipeline: returns the ``(context, meta)`` pair straight from
    :meth:`Defense.apply`.
    """
    return build_defense(config).apply(retrieved_docs, query)


__all__ = [
    "Defense",
    "DefenseMeta",
    "DefenseOutput",
    "DOC_SEPARATOR",
    "render_documents",
    "NoneDefense",
    "ContextFencingDefense",
    "ProvenanceFilterDefense",
    "InjectionClassifierDefense",
    "DEFENSE_CLASSES",
    "available_defenses",
    "get_defense",
    "build_defense",
    "run_defense",
]
