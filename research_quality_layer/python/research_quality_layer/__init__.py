"""Opt-in research-quality sidecar for AutoResearch products.

Extracted/slimmed from Research Opportunity Factory (Apache-2.0).
Does not modify host 01-04 stages.
"""

from .models_slim import (
    DecisiveTest,
    EvidencePointer,
    EvidenceStatus,
    Evaluation,
    FailureClaim,
    HumanScores,
    IdeaPack,
    IdeaStatus,
    NearestPrior,
    Opportunity,
    OpportunityOperator,
    PaperCard,
    PaperMeta,
    QualityDecision,
    ReconDecision,
    ReconQuery,
    ReconReport,
    ResidualDecision,
    ReviewVerdict,
    RouteProposal,
    Scope,
)
from .opportunity_quality_gate import (
    QUALITY_GATE_POLICY,
    QUALITY_GATE_POLICY_SHA256,
    QUALITY_GATE_VERSION,
    evaluate_opportunity,
)
from .recon_protocol import (
    LANES,
    NO_DIRECT_COVERAGE_REASON,
    QUERY_POLICY_VERSION,
    RECON_PROTOCOL_VERSION,
    VARIANTS,
    generate_recon_queries,
    query_text,
)
from .residual_decisions import residual_decision_help
from .idea_pack_builder import build_idea_pack_dict

__all__ = [
    "DecisiveTest",
    "EvidencePointer",
    "EvidenceStatus",
    "Evaluation",
    "FailureClaim",
    "HumanScores",
    "IdeaPack",
    "IdeaStatus",
    "LANES",
    "NO_DIRECT_COVERAGE_REASON",
    "NearestPrior",
    "Opportunity",
    "OpportunityOperator",
    "PaperCard",
    "PaperMeta",
    "QUERY_POLICY_VERSION",
    "QUALITY_GATE_POLICY",
    "QUALITY_GATE_POLICY_SHA256",
    "QUALITY_GATE_VERSION",
    "QualityDecision",
    "RECON_PROTOCOL_VERSION",
    "ReconDecision",
    "ReconQuery",
    "ReconReport",
    "ResidualDecision",
    "ReviewVerdict",
    "RouteProposal",
    "Scope",
    "VARIANTS",
    "build_idea_pack_dict",
    "evaluate_opportunity",
    "generate_recon_queries",
    "query_text",
    "residual_decision_help",
]
