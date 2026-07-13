from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from lit_agent.http import urlopen
from lit_agent.models import QueryPlan, RawSourceRecord


class SourceConnector(ABC):
    name: str

    def __init__(self, *, timeout_seconds: int = 30, polite_delay_seconds: float = 0.2):
        self.timeout_seconds = timeout_seconds
        self.polite_delay_seconds = polite_delay_seconds

    @abstractmethod
    def search(self, query: str, plan: QueryPlan) -> list[RawSourceRecord]:
        """Return raw source records for one query."""

    def fetch_json(self, url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            headers=headers
            or {
                "User-Agent": "lit-agent/0.1 (mailto:research@example.com)",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"{self.name} HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{self.name} network error: {exc.reason}") from exc
        finally:
            if self.polite_delay_seconds:
                time.sleep(self.polite_delay_seconds)
        return json.loads(payload.decode("utf-8"))

    def fetch_text(self, url: str, headers: dict[str, str] | None = None) -> str:
        request = urllib.request.Request(
            url,
            headers=headers
            or {
                "User-Agent": "lit-agent/0.1 (mailto:research@example.com)",
                "Accept": "application/xml,text/xml,application/atom+xml",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"{self.name} HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{self.name} network error: {exc.reason}") from exc
        finally:
            if self.polite_delay_seconds:
                time.sleep(self.polite_delay_seconds)
        return payload.decode("utf-8", errors="replace")

    @staticmethod
    def urlencode(params: dict[str, Any]) -> str:
        return urllib.parse.urlencode(params, doseq=True, safe=":,")
