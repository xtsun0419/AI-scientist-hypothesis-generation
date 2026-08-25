# Attribution

Portions of `research_quality_layer/` are extracted and slimmed from
[Research Opportunity Factory](https://github.com/Nathan10969/Research-Opportunity-Factory)
(Apache License 2.0).

## What was ported

- Opportunity / PaperCard / Recon / Idea Pack contracts (`models_slim.py`, JSON schemas)
- Deterministic opportunity quality gate (`opportunity_quality_gate.py`)
- Prior-art recon protocol constants, query templates, and bounded decisions (`recon_protocol.py`)
- Residual decision vocabulary (`residual_decisions.py`)
- Idea Pack assembly helper (`idea_pack_builder.py`)

## What was intentionally NOT ported

- Full CLI / pipeline / ledger / run-directory orchestration
- Live literature-search recon executor
- Vault, skills, or private pilot corpora

## License boundary

- Parent repo license: MIT (Copyright (c) 2026 XutaoSun)
- This sidecar package: Apache-2.0 for derived ROF material
- Do not relicense the ported files as MIT; keep NOTICE and LICENSE-APACHE-2.0.txt
