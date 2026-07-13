from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from lit_agent.constants import (
    ACCESS_CLOSED_HAS_DOI,
    ACCESS_CLOSED_MISSING_DOI,
    ACCESS_DOWNLOADED_OA_PDF,
    ACCESS_OA_NO_PDF_URL,
    ACCESS_OA_PDF_AVAILABLE,
    ACCESS_PREPRINT_PDF,
)
from lit_agent.db import LiteratureDB
from lit_agent.http import urlopen
from lit_agent.models import AccessRecord
from lit_agent.pdf_urls import collect_pdf_url_candidates
from lit_agent.text import doi_url, normalize_doi


class OAResolverAgent:
    """Resolves DOI, OA status, PDF URL, and publisher access hints."""

    def __init__(self, db: LiteratureDB, *, unpaywall_email: str | None = None, timeout_seconds: int = 30):
        self.db = db
        self.unpaywall_email = unpaywall_email or os.environ.get("UNPAYWALL_EMAIL")
        self.timeout_seconds = timeout_seconds

    def run(self) -> int:
        resolved = 0
        for paper in self.db.papers():
            candidates = self.db.rows(
                """
                SELECT pc.*
                FROM paper_candidates pc
                JOIN paper_sources ps ON ps.source_record_id = pc.source_record_id
                WHERE ps.paper_id = ?
                ORDER BY pc.id
                """,
                (paper["id"],),
            )
            doi = normalize_doi(paper["doi"]) or _first_candidate(candidates, "doi")
            pdf_url = _first_candidate(candidates, "pdf_url")
            publisher_url = paper["publisher_url"] or _first_candidate(candidates, "publisher_url")
            source_url = paper["source_url"] or _first_candidate(candidates, "source_url")
            is_oa = _resolve_is_oa(candidates, paper["document_type"])
            unpaywall = None
            if doi:
                unpaywall = self._lookup_unpaywall(doi)
                if unpaywall:
                    is_oa = bool(unpaywall.get("is_oa"))
                    best_location = unpaywall.get("best_oa_location") or {}
                    pdf_url = pdf_url or best_location.get("url_for_pdf")
                    publisher_url = publisher_url or unpaywall.get("doi_url")
            pdf_candidates = collect_pdf_url_candidates(
                paper_id=paper["id"],
                doi=doi,
                publisher_url=publisher_url,
                source_url=source_url,
                candidate_rows=candidates,
                unpaywall_payload=unpaywall,
            )
            for candidate in pdf_candidates:
                self.db.upsert_pdf_candidate(
                    paper_id=paper["id"],
                    pdf_url=candidate.url,
                    source=candidate.source,
                    priority=candidate.priority,
                    reason=candidate.reason,
                )
            if not pdf_url and pdf_candidates:
                pdf_url = pdf_candidates[0].url
            status = _status_for(doi=doi, is_oa=is_oa, pdf_url=pdf_url, document_type=paper["document_type"])
            downloaded = self.db.rows(
                """
                SELECT pdf_url, status
                FROM pdf_assets
                WHERE paper_id = ? AND status IN (?, ?)
                ORDER BY id DESC
                LIMIT 1
                """,
                (paper["id"], ACCESS_DOWNLOADED_OA_PDF, ACCESS_PREPRINT_PDF),
            )
            if downloaded:
                pdf_url = downloaded[0]["pdf_url"] or pdf_url
                status = downloaded[0]["status"]
            record = AccessRecord(
                paper_id=paper["id"],
                doi=doi,
                doi_url=doi_url(doi),
                is_oa=is_oa,
                pdf_url=pdf_url,
                publisher_url=publisher_url or doi_url(doi),
                source_url=source_url,
                access_status=status,
            )
            self.db.upsert_access_record(record)
            resolved += 1
        return resolved

    def _lookup_unpaywall(self, doi: str) -> dict[str, Any] | None:
        if not self.unpaywall_email:
            return None
        url = "https://api.unpaywall.org/v2/" + urllib.parse.quote(doi) + "?" + urllib.parse.urlencode(
            {"email": self.unpaywall_email}
        )
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "lit-agent/0.1",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None


def _first_candidate(rows: list[Any], key: str) -> Any:
    for row in rows:
        value = row[key]
        if value:
            return value
    return None


def _resolve_is_oa(rows: list[Any], document_type: str | None) -> bool | None:
    if document_type == "preprint":
        return True
    saw_false = False
    for row in rows:
        value = row["is_oa"]
        if value == 1:
            return True
        if value == 0:
            saw_false = True
        if row["source"] in {"arxiv", "doaj", "core"} and row["pdf_url"]:
            return True
    if saw_false:
        return False
    return None


def _status_for(
    *,
    doi: str | None,
    is_oa: bool | None,
    pdf_url: str | None,
    document_type: str | None,
) -> str:
    if document_type == "preprint" and pdf_url:
        return ACCESS_PREPRINT_PDF
    if is_oa and pdf_url:
        return ACCESS_OA_PDF_AVAILABLE
    if is_oa:
        return ACCESS_OA_NO_PDF_URL
    if doi:
        return ACCESS_CLOSED_HAS_DOI
    return ACCESS_CLOSED_MISSING_DOI
