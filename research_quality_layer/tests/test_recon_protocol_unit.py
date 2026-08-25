import pytest

from research_quality_layer.models_slim import Opportunity
from research_quality_layer.recon_protocol import (
    LANES,
    NO_DIRECT_COVERAGE_REASON,
    VARIANTS,
    generate_recon_queries,
)


def _opportunity(**updates: object) -> Opportunity:
    body = {
        "schema_version": "idea_factory.opportunity.v1",
        "opportunity_id": "opp-recon-1",
        "operator": "ASSUMPTION_BREAK",
        "assumption_x": "The cache remains resident during long serving traces.",
        "observation_y": "card-1 reports retrieval degrades after eviction.",
        "condition_z": "under long serving traces",
        "failure_f": "cache eviction causes retrieval loss",
        "missing_capability_w": "eviction-aware retrieval control",
        "alternative_explanation_a": "cache eviction, not routing error",
        "decisive_experiment": (
            "Compare recall after forced cache eviction against cache eviction, "
            "not routing error."
        ),
        "supporting_card_ids": ["card-1"],
        "nearest_internal_neighbors": ["card-2"],
        "scope_compatibility": "Both cards cover KV cache serving over long traces.",
        "inference_flags": [],
    }
    body.update(updates)
    return Opportunity.model_validate(body)


def test_generate_recon_queries_covers_four_lanes_and_two_variants() -> None:
    queries = generate_recon_queries(_opportunity())
    assert len(queries) == len(LANES) * len(VARIANTS)
    assert {query.lane for query in queries} == set(LANES)
    assert {query.variant for query in queries} == set(VARIANTS)
    texts = [query.query.casefold() for query in queries]
    assert len(texts) == len(set(texts))


def test_method_name_firewall_rejects_controlled_terms() -> None:
    with pytest.raises(ValueError, match="firewall rejects"):
        generate_recon_queries(
            _opportunity(missing_capability_w="BERT based retrieval control")
        )


def test_no_direct_coverage_reason_is_bounded() -> None:
    assert "bounded to the supplied queries" in NO_DIRECT_COVERAGE_REASON
    assert "nobody in the world" not in NO_DIRECT_COVERAGE_REASON.lower()
