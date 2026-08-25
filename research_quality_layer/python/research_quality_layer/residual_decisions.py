"""Residual decision vocabulary for Critic / human gate.

Derived from Research Opportunity Factory review semantics (Apache-2.0).
Does not replace host CriticAgent / Elo; provides additive decision codes.
"""

from __future__ import annotations

from .models_slim import ResidualDecision

_HELP: dict[ResidualDecision, str] = {
    ResidualDecision.KILL: (
        "Nearest prior already covers the core claim; do not continue as stated."
    ),
    ResidualDecision.NARROW: (
        "Original claim is too wide, but a narrower residual remains after "
        "subtracting nearest priors."
    ),
    ResidualDecision.PASS_TO_HUMAN: (
        "Evidence-backed, scoped, falsifiable residual exists; hand to human "
        "for experiment decision."
    ),
    ResidualDecision.REFRAME: (
        "Sidecar extension: the underlying problem is worth keeping, but the "
        "current method or claim does not hold. Host Critic may adopt later."
    ),
}


def residual_decision_help(decision: ResidualDecision | str) -> str:
    """Return the short semantic note for a residual decision code."""

    if isinstance(decision, str):
        decision = ResidualDecision(decision)
    return _HELP[decision]


REQUIRED_CRITIC_QUESTIONS: tuple[str, ...] = (
    "What is the strongest baseline?",
    "Why would Reviewer #2 call this A+B?",
    "Is the claimed source of gain independently evidenced?",
    "Which result should kill the route?",
    "Is Critic evidence independent of generator evidence?",
)
