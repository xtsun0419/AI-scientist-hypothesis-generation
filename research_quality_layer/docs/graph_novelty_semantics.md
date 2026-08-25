# Graph Novelty Semantics

## Current risk

On a small or incomplete knowledge graph, it is tempting to treat:

```text
no path in graph  →  high novelty
```

That overclaims. Missing edges often mean **coverage uncertainty**, not an external research blank.

## Recommended semantics

```text
no path in graph  →  internal gap signal
                  →  coverage uncertainty
                  →  queue for external prior-art recon
```

Keep graph gap / analogy operators for **high-recall candidate generation**.  
Let bounded recon adjudicate external prior-art pressure.

## Practical rule

- Graph: propose what is worth checking
- Recon: decide whether it is already covered / residual / no-direct-under-this-pack
- Critic / Elo: rank surviving routes; do not own novelty truth alone
