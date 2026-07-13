from __future__ import annotations

import ssl
import urllib.request
from typing import Any


def urlopen(request: urllib.request.Request, *, timeout: int | float) -> Any:
    """Open HTTPS URLs with certifi CAs when available."""
    context = _ssl_context()
    return urllib.request.urlopen(request, timeout=timeout, context=context)


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())
