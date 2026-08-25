"""Slim schema contracts for the research-quality sidecar.

Derived from Research Opportunity Factory ``models.py`` (Apache-2.0).
Run-ledger / landscape / corpus-label machinery intentionally omitted.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Score = Annotated[int, Field(ge=0, le=5, strict=True)]
ReconResultCount = Annotated[int, Field(ge=1, le=10, strict=True)]


class StrictModel(BaseModel):
    """Reject unknown fields; validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FrozenStrictModel(StrictModel):
    model_config = ConfigDict(frozen=True, revalidate_instances="always")


class EvidenceStatus(StrEnum):
    REPORTED = "REPORTED"
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


class OpportunityOperator(StrEnum):
    ASSUMPTION_BREAK = "ASSUMPTION_BREAK"
    FAILURE_TRANSFER = "FAILURE_TRANSFER"
    BOUNDARY_CONDITION = "BOUNDARY_CONDITION"
    MEASUREMENT_GAP = "MEASUREMENT_GAP"
    EVALUATION_MISMATCH = "EVALUATION_MISMATCH"
    OBJECTIVE_CONFLICT = "OBJECTIVE_CONFLICT"
    DYNAMIC_MISMATCH = "DYNAMIC_MISMATCH"
    MECHANISM_TRANSPLANT = "MECHANISM_TRANSPLANT"


class ReconDecision(StrEnum):
    COVERED = "COVERED"
    NEAR_PRIOR_WITH_RESIDUAL = "NEAR_PRIOR_WITH_RESIDUAL"
    NO_DIRECT_COVERAGE_FOUND = "NO_DIRECT_COVERAGE_FOUND"


class IdeaStatus(StrEnum):
    READY_FOR_CHEAP_TEST = "READY_FOR_CHEAP_TEST"
    HOLD = "HOLD"
    KILLED_QUALITY = "KILLED_QUALITY"
    KILLED_INTERNAL_DUP = "KILLED_INTERNAL_DUP"
    KILLED_RECON = "KILLED_RECON"
    KILLED_REVIEW = "KILLED_REVIEW"


class ResidualDecision(StrEnum):
    """Residual adjudication vocabulary for Critic / human gate.

    ``KILL`` / ``NARROW`` / ``PASS_TO_HUMAN`` match upstream ReviewDecision.
    ``REFRAME`` is a sidecar extension for "problem worth keeping, current
    claim/method does not hold" — host Critic may adopt it later.
    """

    KILL = "KILL"
    NARROW = "NARROW"
    PASS_TO_HUMAN = "PASS_TO_HUMAN"
    REFRAME = "REFRAME"


class EvidencePointer(StrictModel):
    source: Literal["NOTE", "PDF", "TABLE", "FIGURE", "EXPERIMENT"]
    locator: NonEmptyStr
    supports: NonEmptyStr


class FailureClaim(StrictModel):
    text: str
    status: EvidenceStatus


class PaperMeta(StrictModel):
    title: str
    year: int
    venue: str
    source_path: str


class Scope(StrictModel):
    object: str
    time_horizon: str
    setting: str


class Evaluation(StrictModel):
    measurement: str
    regime: str


class PaperCard(StrictModel):
    """Failure-aware paper card contract (additive overlay for host cards)."""

    schema_version: Literal["idea_factory.paper_card.v1"]
    card_id: str
    paper: PaperMeta
    problem: str
    assumption: FailureClaim
    mechanism: str
    failure_observation: FailureClaim
    failure_mechanism: FailureClaim
    limitation: str
    evaluation: Evaluation
    scope: Scope
    evidence_pointers: list[EvidencePointer]
    extraction_confidence: Literal["HIGH", "MEDIUM", "LOW"]


class Opportunity(StrictModel):
    schema_version: Literal["idea_factory.opportunity.v1"]
    opportunity_id: str
    operator: OpportunityOperator
    assumption_x: str
    observation_y: str
    condition_z: str
    failure_f: str
    missing_capability_w: str
    alternative_explanation_a: str
    decisive_experiment: str
    supporting_card_ids: list[str]
    nearest_internal_neighbors: list[str]
    scope_compatibility: str
    inference_flags: list[str] = Field(default_factory=list)


class QualityDecision(StrictModel):
    opportunity_id: str
    passed: bool
    reason_codes: list[str]


class ReconQuery(StrictModel):
    query_id: str
    opportunity_id: str
    lane: Literal["CONCEPT", "MECHANISM", "FAILURE", "EVALUATION"]
    variant: Literal["CURRENT_TERMS", "GENERIC_SHAPE", "SYNONYM"]
    query: str
    max_results: ReconResultCount


class NearestPrior(FrozenStrictModel):
    paper: NonEmptyStr
    exact_overlap: NonEmptyStr
    residual_difference: NonEmptyStr
    evidence_url_or_id: NonEmptyStr


class ReconReport(StrictModel):
    schema_version: Literal["idea_factory.recon.v1"]
    opportunity_id: str
    searched_query_ids: list[str]
    searched_at: datetime
    nearest_priors: list[NearestPrior]
    decision: ReconDecision
    decision_reason: str


class DecisiveTest(FrozenStrictModel):
    setup: NonEmptyStr
    discriminates_against: NonEmptyStr
    expected_runtime_or_cost: NonEmptyStr


class RouteProposal(StrictModel):
    route_id: str
    opportunity_id: str
    old_assumption_changed: str
    new_assumption: str
    new_mechanism: str
    why_it_addresses_failure: str
    why_nearest_priors_cannot: str
    source_of_gain: str
    cheapest_decisive_test: DecisiveTest
    kill_condition: str


class ReviewVerdict(StrictModel):
    route_id: str
    decision: ResidualDecision
    strongest_baseline: str
    a_plus_b_objection: str
    source_of_gain_verdict: str
    falsifiability_verdict: str
    cost_risk: str
    residual_claim: str


class HumanScores(FrozenStrictModel):
    specific_novelty: Score
    importance: Score
    paper_potential: Score
    feasibility: Score
    excitement: Score
    evidence_clarity: Score


class IdeaPack(FrozenStrictModel):
    schema_version: Literal["idea_factory.idea_pack.v1"] = "idea_factory.idea_pack.v1"
    idea_id: str
    status: IdeaStatus
    opportunity: str
    core_hypothesis: str
    why_now: str
    supporting_observations: tuple[str, ...] = ()
    inference_flags: tuple[str, ...] = ()
    nearest_priors: tuple[NearestPrior, ...] = ()
    proposed_mechanism: str
    source_of_gain: str
    cheapest_decisive_test: DecisiveTest | None = None
    strongest_baseline: str
    kill_condition: str
    main_uncertainty: str
    expected_reviewer_2_objection: str
    response_to_objection: str
    evidence_that_would_make_reviewer_correct: str
    human_scores: HumanScores | None = None
    human_reason_codes: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def require_readiness_evidence(self) -> Self:
        if self.status != IdeaStatus.READY_FOR_CHEAP_TEST:
            return self
        missing: list[str] = []
        for field_name in ("source_of_gain", "strongest_baseline", "kill_condition"):
            if not getattr(self, field_name).strip():
                missing.append(field_name)
        if not self.nearest_priors:
            missing.append("nearest_priors")
        if self.cheapest_decisive_test is None:
            missing.append("cheapest_decisive_test")
        if self.human_scores is None:
            missing.append("human_scores")
        if not self.human_reason_codes:
            missing.append("human_reason_codes")
        if missing:
            raise ValueError("READY_FOR_CHEAP_TEST requires: " + ", ".join(missing))
        return self
