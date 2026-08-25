# Prior-art Recon Protocol (non-executing)

This package ships the **protocol**, query templates, and report schema.  
It does **not** ship a live literature-search runner.

## Why this exists

Between confirmed scientific questions and route generation, candidates need an external prior-art pressure test. Host graph gaps are useful recall signals; they are not global novelty certificates.

## Lanes

Every opportunity produces queries on four lanes:

1. `CONCEPT` — assumption / missing capability / condition
2. `MECHANISM` — failure / capability relation
3. `FAILURE` — observation / failure / condition
4. `EVALUATION` — decisive experiment / alternative explanation

Default variants:

- `CURRENT_TERMS`
- `GENERIC_SHAPE`

`SYNONYM` is optional for host adapters.

## Bounded decisions

Only three report decisions are allowed:

| Decision | Meaning |
|---|---|
| `COVERED` | Nearest prior covers the core claim; stop or kill as stated |
| `NEAR_PRIOR_WITH_RESIDUAL` | Overlap exists, but a concrete residual remains |
| `NO_DIRECT_COVERAGE_FOUND` | No direct coverage under **this query pack and receipts** |

`NO_DIRECT_COVERAGE_FOUND` is **not** “nobody in the world has done this.”

Controlled reason text for that decision:

> No direct coverage found under this query pack. This result is bounded to the supplied queries and execution receipts.

## Required report fields

See `schemas/recon_report.schema.json`:

- searched query ids
- searched_at
- nearest priors (`paper`, `exact_overlap`, `residual_difference`, `evidence_url_or_id`)
- decision
- decision_reason

## Method-name firewall

Query generation rejects controlled method/model name tokens so recon searches mechanisms and failures rather than branded method strings. See `recon_protocol.py`.

## Host adapter suggestion

Reuse the host literature-search stack (stage 01) as the executor. Feed `generate_recon_queries(...)` outputs into that stack; write results back as `ReconReport` objects. Keep receipts.
