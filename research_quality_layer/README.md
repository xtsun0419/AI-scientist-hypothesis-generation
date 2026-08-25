# research_quality_layer

Opt-in research-quality sidecar for this AutoResearch product.

Extracted/slimmed from [Research Opportunity Factory](https://github.com/Nathan10969/Research-Opportunity-Factory) (Apache-2.0).  
See `NOTICE`, `ATTRIBUTION.md`, and `LICENSE-APACHE-2.0.txt`.

**Deleting this directory must leave stages 01–04 fully runnable.** No default hooks are installed.

## What this helps with

1. **Failure-aware Paper Card Sidecar** — additive assumption / failure / evaluation / scope / evidence fields; does not replace existing Paper Cards
2. **Opportunity Quality Gate** — deterministic completeness check between confirmed questions (03) and route generation (04)
3. **Graph novelty semantic correction** — graph `no path` = internal gap / coverage uncertainty → queue recon; **not** external high novelty
4. **Prior-art Recon protocol + report schema** — CONCEPT / MECHANISM / FAILURE / EVALUATION; bounded decisions only
5. **Residual decision vocabulary** — `KILL` / `NARROW` / `PASS_TO_HUMAN` / optional `REFRAME`; Elo ranks, residual/recon adjudicate
6. **Idea Pack output contract** — proposal-level experiment-decision package (cheap tests are **not** claimed run)

## What this does *not* do

- Replace SQLite / Web UI / Paper Cards / 4-stage structure
- Port full ROF CLI, ledger, vault, skills, or live recon executor
- Claim Idea Packs are experimentally validated
- Treat graph gaps as global novelty oracles

## Layout

```text
research_quality_layer/
  README.md
  NOTICE / ATTRIBUTION.md / LICENSE-APACHE-2.0.txt
  docs/           integration + protocol notes
  schemas/        JSON Schema contracts
  python/         importable slim package
  tests/          offline unit tests
  samples/        synthetic Idea Pack only
```

## Quick test

```powershell
cd research_quality_layer
# use any Python 3.12+ env with pydantic + pytest
python -m pip install "pydantic>=2,<3" pytest
python -m pytest -q
```

## Insertion map

See `docs/integration_guide.md`.
