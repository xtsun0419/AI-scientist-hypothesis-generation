"""Deterministic opportunity quality gate; deliberately no LLM scores.

Derived from Research Opportunity Factory ``quality.py`` (Apache-2.0).
Run-bundle publish/validate helpers intentionally omitted.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping

from .models_slim import EvidenceStatus, Opportunity, PaperCard, QualityDecision

QUALITY_GATE_VERSION = "idea_factory.opportunity_quality.v1"
_FIELDS = (
    ("assumption_x", "EMPTY_ASSUMPTION_X"),
    ("observation_y", "EMPTY_OBSERVATION_Y"),
    ("condition_z", "EMPTY_CONDITION_Z"),
    ("failure_f", "EMPTY_FAILURE_F"),
    ("missing_capability_w", "EMPTY_MISSING_CAPABILITY_W"),
    ("alternative_explanation_a", "EMPTY_ALTERNATIVE_A"),
    ("decisive_experiment", "EMPTY_DECISIVE_EXPERIMENT"),
)
_RELATION = re.compile(
    r"\b(assum|because|caus|fail|under|when|if|unless|after|before|during|remain|produ|lead|degrad|evict)\w*\b",
    re.I,
)
_KEYWORD_STACK = re.compile(
    r"\b[a-z][a-z0-9_-]*\s*(?:\+|and)\s*[a-z][a-z0-9_-]*\b",
    re.I,
)
_TEST_WORDS = re.compile(r"\b(compare|versus|vs\.?|against|ablat|control)\b", re.I)
_MEASURE_WORDS = re.compile(
    r"\b(measure|metric|recall|accuracy|latency|loss|rate|error|throughput)\b",
    re.I,
)
_SCOPE_CONTRADICTION = re.compile(
    r"\b(incompatible|contradict(?:s|ory)?|cannot compare|mutually exclusive)\b",
    re.I,
)
QUALITY_GATE_POLICY = {
    "version": QUALITY_GATE_VERSION,
    "policy_kind": "deterministic-transparent-lexical-v1",
    "blank_field_codes": [code for _field, code in _FIELDS],
    "relation_pattern": _RELATION.pattern,
    "keyword_stack_pattern": _KEYWORD_STACK.pattern,
    "test_pattern": _TEST_WORDS.pattern,
    "measure_pattern": _MEASURE_WORDS.pattern,
    "scope_contradiction_pattern": _SCOPE_CONTRADICTION.pattern,
    "alternative_rule": "normalized exact substring in decisive_experiment",
}
QUALITY_GATE_POLICY_SHA256 = hashlib.sha256(
    json.dumps(
        QUALITY_GATE_POLICY,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[\w-]+", value.lower()))


def evaluate_opportunity(
    opportunity: Opportunity,
    cards: Mapping[str, PaperCard] | Iterable[PaperCard],
    *,
    allowed_neighbor_ids: set[str] | None = None,
) -> QualityDecision:
    """Return stable reason codes. ``cards`` must be the validated job universe."""

    card_map = (
        dict(cards)
        if isinstance(cards, Mapping)
        else {card.card_id: card for card in cards}
    )
    reasons: list[str] = []
    for field, code in _FIELDS:
        if not getattr(opportunity, field).strip():
            reasons.append(code)
    if not opportunity.supporting_card_ids:
        reasons.append("NO_SUPPORTING_CARD")
    for card_id in opportunity.supporting_card_ids:
        card = card_map.get(card_id)
        if card is None:
            reasons.append("UNKNOWN_SUPPORTING_CARD")
        elif (
            card.failure_mechanism.status == EvidenceStatus.UNKNOWN
            or not card.failure_mechanism.text.strip()
        ):
            reasons.append("UNKNOWN_FAILURE_MECHANISM")
    if not opportunity.nearest_internal_neighbors:
        reasons.append("NO_INTERNAL_NEIGHBOR")
    else:
        permitted = (
            allowed_neighbor_ids
            if allowed_neighbor_ids is not None
            else set(card_map)
        )
        if any(
            neighbor not in permitted
            for neighbor in opportunity.nearest_internal_neighbors
        ):
            reasons.append("UNKNOWN_INTERNAL_NEIGHBOR")
    if not opportunity.scope_compatibility.strip():
        reasons.append("BLANK_SCOPE_COMPATIBILITY")
    elif _SCOPE_CONTRADICTION.search(opportunity.scope_compatibility):
        reasons.append("SCOPE_CONTRADICTION")
    alternative = _normalized(opportunity.alternative_explanation_a)
    experiment = _normalized(opportunity.decisive_experiment)
    if alternative and alternative not in experiment:
        reasons.append("ALTERNATIVE_NOT_NAMED")
    relation_text = " ".join(
        (
            opportunity.assumption_x,
            opportunity.observation_y,
            opportunity.condition_z,
            opportunity.failure_f,
        )
    )
    if (
        not _RELATION.search(opportunity.assumption_x)
        or len(_normalized(opportunity.assumption_x).split()) < 5
    ):
        reasons.append("VAGUE_ASSUMPTION")
    if _KEYWORD_STACK.search(relation_text) and not _RELATION.search(relation_text):
        reasons.append("KEYWORD_COMBINATION")
    if (
        not experiment
        or not _TEST_WORDS.search(opportunity.decisive_experiment)
        or not _MEASURE_WORDS.search(opportunity.decisive_experiment)
    ):
        reasons.append("NO_DISCRIMINATING_TEST")
    return QualityDecision(
        opportunity_id=opportunity.opportunity_id,
        passed=not reasons,
        reason_codes=sorted(set(reasons)),
    )
