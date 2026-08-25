# Residual Decisions

Additive vocabulary for Critic / human gate. Does not replace the host CriticAgent or Elo tournament.

| Code | Meaning |
|---|---|
| `KILL` | Nearest prior already covers the core claim |
| `NARROW` | Claim is too wide; a narrower residual remains |
| `PASS_TO_HUMAN` | Evidence-backed, scoped, falsifiable residual exists |
| `REFRAME` | Sidecar extension: problem worth keeping, current claim/method does not hold |

## Questions Critic should still answer

1. What is the strongest baseline?
2. Why would Reviewer #2 call this A+B?
3. Is the claimed source of gain independently evidenced?
4. Which result should kill the route?
5. Is Critic evidence independent of generator evidence?

Elo may rank candidates. Residual / recon decisions own “is this actually new enough to test?”
