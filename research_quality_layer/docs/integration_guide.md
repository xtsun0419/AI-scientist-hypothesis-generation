# Integration Guide (opt-in)

This sidecar is designed so **deleting `research_quality_layer/` leaves the host product fully runnable**.

No default hooks are installed into stages 01–04.

## Insertion map

```text
01 文献检索_agent                         (unchanged)
        ↓
02 文献分析_agent / Cards / Wiki / KG     (unchanged)
        ↓  [optional] overlay failure-aware sidecar fields on export
research_quality_layer schemas
        ↓
03 confirmed questions                    (unchanged store)
        ↓  [optional adapter] map confirmed question → Opportunity-shaped dict
Opportunity Quality Gate                  evaluate_opportunity(...)
        ↓ pass only
Prior-art Recon protocol                  generate_recon_queries(...) + host search adapter
        ↓ COVERED → stop; residual / no-direct → continue
04 route generation                       (unchanged)
        ↓
existing CriticAgent / Elo / Evolution    (keep as tournament)
        ↓  [optional] residual decision codes
Idea Pack builder                         build_idea_pack_dict(...)
```

## Minimal Python usage

```python
from research_quality_layer import (
    evaluate_opportunity,
    generate_recon_queries,
    build_idea_pack_dict,
)

decision = evaluate_opportunity(opportunity, cards)
if decision.passed:
    queries = generate_recon_queries(opportunity)
    # run queries through host lit-search; write ReconReport
    pack = build_idea_pack_dict(
        idea_id="idea-1",
        opportunity=opportunity.model_dump(mode="json"),
        core_hypothesis="...",
        why_now="...",
        proposed_mechanism="...",
        source_of_gain="...",
        strongest_baseline="...",
        kill_condition="...",
        main_uncertainty="...",
        expected_reviewer_2_objection="...",
        response_to_objection="...",
        evidence_that_would_make_reviewer_correct="...",
        nearest_priors=[...],
        cheapest_decisive_test={...},
    )
```

## Alignment with host improvement plan

- “no path → high novelty” → rewrite via `graph_novelty_semantics.md`
- novelty pressure between question → route → Quality Gate + Recon, not Critic replacement
- Critic remains route tournament; residual codes are additive

## Dependency note

Runtime needs `pydantic>=2`. Tests need `pytest`. No network required for unit tests.
