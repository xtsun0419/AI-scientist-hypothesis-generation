from __future__ import annotations


def classify_failure(exc: Exception) -> str:
    text = str(exc).lower()
    if "429" in text or "too many requests" in text or "rate" in text:
        return "rate_limited"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if " 5" in text or "http 5" in text or "bad gateway" in text or "service unavailable" in text:
        return "server_error"
    if "certificate" in text or "ssl" in text:
        return "tls_error"
    if "network" in text or "urlerror" in text or "nodename" in text:
        return "network_error"
    if "http 4" in text:
        return "client_error"
    return "unknown_error"


def retryable_failure(failure_type: str) -> bool:
    return failure_type in {"rate_limited", "timeout", "server_error", "network_error"}
