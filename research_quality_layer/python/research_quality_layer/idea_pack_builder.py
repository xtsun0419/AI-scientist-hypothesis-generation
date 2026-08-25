"""Assemble Idea Pack dicts without run-ledger coupling.

Derived from Research Opportunity Factory ``render.py`` field mapping
(Apache-2.0). Host stages supply plain dicts; this helper validates shape
via ``IdeaPack`` when requested.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .models_slim import IdeaPack, IdeaStatus


def _nonempty(value: object) -> bool:
    return type(value) is str and bool(value.strip())


def _as_str(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    if type(value) is str:
        return value
    if isinstance(value, Mapping):
        for key in ("text", "comparison", "evidence_id", "summary"):
            candidate = value.get(key)
            if type(candidate) is str and candidate.strip():
                return candidate
        return default
    return str(value)


def build_idea_pack_dict(
    *,
    idea_id: str,
    opportunity: Mapping[str, Any] | str,
    core_hypothesis: str,
    why_now: str,
    proposed_mechanism: str,
    source_of_gain: str,
    strongest_baseline: str | Mapping[str, Any],
    kill_condition: str,
    main_uncertainty: str,
    expected_reviewer_2_objection: str,
    response_to_objection: str,
    evidence_that_would_make_reviewer_correct: str | Mapping[str, Any],
    nearest_priors: Sequence[Mapping[str, Any]] = (),
    supporting_observations: Sequence[str] = (),
    inference_flags: Sequence[str] = (),
    cheapest_decisive_test: Mapping[str, Any] | None = None,
    human_scores: Mapping[str, Any] | None = None,
    human_reason_codes: Sequence[str] = (),
    status: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Build one Idea Pack dict.

    If ``status`` is omitted, choose ``READY_FOR_CHEAP_TEST`` only when the
    minimal readiness gates are present; otherwise ``HOLD``.
    Cheap decisive experiments are assumed **not** run by this helper.
    """

    if isinstance(opportunity, Mapping):
        opportunity_text = _as_str(
            opportunity.get("opportunity_id"),
            default=_as_str(opportunity.get("assumption_x")),
        )
        if not supporting_observations:
            supporting_observations = [
                item
                for item in (
                    _as_str(opportunity.get("observation_y")),
                    _as_str(opportunity.get("failure_f")),
                )
                if item
            ]
        if not inference_flags:
            flags = opportunity.get("inference_flags") or []
            inference_flags = [str(flag) for flag in flags]
        if not main_uncertainty:
            main_uncertainty = _as_str(opportunity.get("alternative_explanation_a"))
        if not why_now:
            why_now = (
                f"{_as_str(opportunity.get('observation_y'))} "
                f"Under {_as_str(opportunity.get('condition_z'))}."
            ).strip()
    else:
        opportunity_text = opportunity

    baseline_text = _as_str(strongest_baseline)
    evidence_text = _as_str(evidence_that_would_make_reviewer_correct)
    if isinstance(evidence_that_would_make_reviewer_correct, Mapping):
        evidence_text = (
            f"discriminates_against="
            f"{_as_str(evidence_that_would_make_reviewer_correct.get('discriminates_against'))}; "
            f"kill_condition="
            f"{_as_str(evidence_that_would_make_reviewer_correct.get('kill_condition'))}"
        )

    gates = {
        "nearest_priors": bool(nearest_priors)
        and all(_nonempty(prior.get("residual_difference")) for prior in nearest_priors),
        "source_of_gain": _nonempty(source_of_gain),
        "strongest_baseline": _nonempty(baseline_text),
        "decisive_test": (
            isinstance(cheapest_decisive_test, Mapping)
            and all(
                _nonempty(cheapest_decisive_test.get(key))
                for key in ("setup", "discriminates_against", "expected_runtime_or_cost")
            )
        ),
        "kill_condition": _nonempty(kill_condition),
        "human_scores": human_scores is not None,
        "human_reason_codes": bool(human_reason_codes),
    }
    resolved_status = status or (
        IdeaStatus.READY_FOR_CHEAP_TEST.value
        if all(gates.values())
        else IdeaStatus.HOLD.value
    )

    body: dict[str, Any] = {
        "schema_version": "idea_factory.idea_pack.v1",
        "idea_id": idea_id,
        "status": resolved_status,
        "opportunity": opportunity_text,
        "core_hypothesis": core_hypothesis,
        "why_now": why_now,
        "supporting_observations": list(supporting_observations),
        "inference_flags": list(inference_flags),
        "nearest_priors": [dict(prior) for prior in nearest_priors],
        "proposed_mechanism": proposed_mechanism,
        "source_of_gain": source_of_gain,
        "cheapest_decisive_test": (
            dict(cheapest_decisive_test) if cheapest_decisive_test is not None else None
        ),
        "strongest_baseline": baseline_text,
        "kill_condition": kill_condition,
        "main_uncertainty": main_uncertainty,
        "expected_reviewer_2_objection": expected_reviewer_2_objection,
        "response_to_objection": response_to_objection,
        "evidence_that_would_make_reviewer_correct": evidence_text,
        "human_scores": dict(human_scores) if human_scores is not None else None,
        "human_reason_codes": list(human_reason_codes),
        "readiness_gates": gates,
        "notes": (
            "Proposal-level Idea Pack assembled by research_quality_layer. "
            "Cheap decisive experiments have not been run by this helper."
        ),
    }

    if validate:
        # IdeaPack expects tuple fields / nested models; validate then re-dump.
        pack = IdeaPack.model_validate(
            {
                **body,
                "supporting_observations": tuple(body["supporting_observations"]),
                "inference_flags": tuple(body["inference_flags"]),
                "nearest_priors": tuple(body["nearest_priors"]),
                "human_reason_codes": tuple(body["human_reason_codes"]),
            }
        )
        dumped = pack.model_dump(mode="json")
        dumped["readiness_gates"] = gates
        dumped["notes"] = body["notes"]
        return dumped
    return body
