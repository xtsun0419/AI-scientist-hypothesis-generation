"""Prior-art recon protocol helpers (non-executing).

Derived from Research Opportunity Factory ``recon.py`` (Apache-2.0).
Ships lanes, query templates, method-name firewall, and bounded decisions.
Does NOT include live search / arXiv / OpenAlex execution.
"""

from __future__ import annotations

import hashlib
import json
import re

from .models_slim import Opportunity, ReconQuery

QUERY_POLICY_VERSION = "idea_factory.recon_query_policy.v1"
RECON_PROTOCOL_VERSION = "idea_factory.recon_protocol.v3"
METHOD_FIREWALL_VERSION = "idea_factory.method_model_firewall.v1"
NO_DIRECT_COVERAGE_REASON = (
    "No direct coverage found under this query pack. "
    "This result is bounded to the supplied queries and execution receipts."
)
LANES = ("CONCEPT", "MECHANISM", "FAILURE", "EVALUATION")
VARIANTS = ("CURRENT_TERMS", "GENERIC_SHAPE")

_MEASURE = re.compile(
    r"\b(measure|metric|recall|accuracy|latency|loss|rate|error|throughput|evaluate|test)\w*\b",
    re.I,
)
_NAMED_METHOD = re.compile(
    r"\b(?:called|named|dubbed)\s+[A-Za-z][A-Za-z0-9_-]*",
    re.I,
)
_METHOD_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?![A-Za-z0-9])")
_CONTROLLED_MODEL_STEM = re.compile(
    r"^(?:transformer(?:xl|base|large|small|tiny|\d+)|bert(?:base|large|small|tiny|\d+)|gpt\d+|llama\d+|lora\d+|qlora\d+|dpo\d+|ppo\d+)$",
    re.I,
)
_CONTROLLED_LETTER_NUMBER_MODEL = re.compile(
    r"^t\d+(?:base|small|large|xl)?$",
    re.I,
)
_CONTROLLED_METHOD_TERMS = (
    "adam",
    "bert",
    "claude",
    "deepseek",
    "dpo",
    "flashattention",
    "gemini",
    "gpt",
    "gpt-2",
    "gpt-3",
    "gpt-4",
    "llama",
    "lora",
    "mamba",
    "mistral",
    "phi",
    "ppo",
    "qlora",
    "qwen",
    "roberta",
    "transformer",
)
_DOMAIN_TERM_ALLOWLIST = ("kv", "rag")
_SEPARATOR = "\x1f"
_ID_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

METHOD_FIREWALL_POLICY = {
    "version": METHOD_FIREWALL_VERSION,
    "policy_kind": "deterministic-controlled-vocabulary-and-style-rules",
    "controlled_method_terms": list(_CONTROLLED_METHOD_TERMS),
    "domain_term_allowlist": list(_DOMAIN_TERM_ALLOWLIST),
    "named_method_pattern": _NAMED_METHOD.pattern,
    "token_pattern": _METHOD_TOKEN.pattern,
    "controlled_model_stem_pattern": _CONTROLLED_MODEL_STEM.pattern,
    "controlled_letter_number_model_pattern": _CONTROLLED_LETTER_NUMBER_MODEL.pattern,
    "style_rules": [
        "unallowlisted-uppercase-acronym-length-at-least-3",
        "mixed-case-token",
        "net-or-former-suffix",
    ],
    "boundary": "deterministic lexical firewall; not semantic omniscience",
}
METHOD_FIREWALL_POLICY_SHA256 = hashlib.sha256(
    json.dumps(
        METHOD_FIREWALL_POLICY,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    if not isinstance(prefix, str) or not _ID_PREFIX.fullmatch(prefix):
        raise ValueError(
            "stable ID prefix must contain only letters, digits, or underscores"
        )
    if not parts:
        raise ValueError("stable ID requires at least one part")
    if any(
        not isinstance(part, str) or not part.strip() or _SEPARATOR in part
        for part in parts
    ):
        raise ValueError(
            "stable ID parts must be nonblank strings without unit separators"
        )
    digest = hashlib.sha256(_SEPARATOR.join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def query_text(opportunity: Opportunity, lane: str, variant: str) -> str:
    """Build one lane×variant query string from an Opportunity."""

    x = opportunity.assumption_x
    y = opportunity.observation_y
    z = opportunity.condition_z
    f = opportunity.failure_f
    w = opportunity.missing_capability_w
    a = opportunity.alternative_explanation_a
    test = opportunity.decisive_experiment
    if lane == "CONCEPT":
        current, generic = f"{x} {w} {z}", f"{x} {z}"
    elif lane == "MECHANISM":
        current, generic = f"{f} {w} relation", f"{f} capability relation"
    elif lane == "FAILURE":
        current, generic = f"{y} {f} {z}", f"{y} {f}"
    elif lane == "EVALUATION":
        measure = " ".join(_MEASURE.findall(test)) or "evaluation"
        current, generic = f"{test} {a}", f"{measure} {a} evaluation"
    else:
        raise ValueError(f"unknown recon lane: {lane}")
    return " ".join(
        (current if variant == "CURRENT_TERMS" else generic).split()
    )


def _assert_no_method_names(opportunity: Opportunity) -> None:
    fields = (
        opportunity.assumption_x,
        opportunity.observation_y,
        opportunity.condition_z,
        opportunity.failure_f,
        opportunity.missing_capability_w,
        opportunity.alternative_explanation_a,
        opportunity.decisive_experiment,
        opportunity.scope_compatibility,
        *opportunity.inference_flags,
    )
    denied = set(_CONTROLLED_METHOD_TERMS)
    allowed = set(_DOMAIN_TERM_ALLOWLIST)
    offending: set[str] = set()
    for value in fields:
        named = _NAMED_METHOD.search(value)
        if named:
            offending.add(named.group(0))
        for token in _METHOD_TOKEN.findall(value):
            folded = token.casefold()
            if folded in allowed:
                continue
            uppercase_count = sum(character.isupper() for character in token)
            mixed_case = uppercase_count >= 2 and any(
                character.islower() for character in token
            )
            if (
                folded in denied
                or _CONTROLLED_MODEL_STEM.fullmatch(token)
                or _CONTROLLED_LETTER_NUMBER_MODEL.fullmatch(token)
                or (token.isupper() and len(token) >= 3)
                or mixed_case
                or folded.endswith(("net", "former"))
            ):
                offending.add(token)
    if offending:
        raise ValueError(
            "controlled deterministic firewall rejects method names or model acronyms: "
            + ", ".join(sorted(offending))
        )


def generate_recon_queries(opportunity: Opportunity) -> tuple[ReconQuery, ...]:
    """Pure 4×2 query pack after the method-name firewall.

    SYNONYM variant is documented as optional for host adapters; this helper
    emits CURRENT_TERMS + GENERIC_SHAPE only (upstream default).
    """

    _assert_no_method_names(opportunity)
    queries: list[ReconQuery] = []
    seen: set[str] = set()
    for lane in LANES:
        for variant in VARIANTS:
            query = query_text(opportunity, lane, variant)
            if not query or query.casefold() in seen:
                raise ValueError(
                    "recon query must be nonblank and unique per opportunity"
                )
            seen.add(query.casefold())
            query_id = _stable_id(
                "recon_query",
                opportunity.opportunity_id,
                lane,
                variant,
                QUERY_POLICY_VERSION,
                query,
            )
            queries.append(
                ReconQuery(
                    query_id=query_id,
                    opportunity_id=opportunity.opportunity_id,
                    lane=lane,  # type: ignore[arg-type]
                    variant=variant,  # type: ignore[arg-type]
                    query=query,
                    max_results=10,
                )
            )
    return tuple(queries)
