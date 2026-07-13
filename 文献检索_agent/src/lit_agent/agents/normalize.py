from __future__ import annotations

import json
from typing import Any

from lit_agent.db import LiteratureDB
from lit_agent.models import PaperCandidate
from lit_agent.text import first_non_empty, normalize_doi, normalize_whitespace


class MetadataNormalizeAgent:
    """Converts source-specific payloads into standard PaperCandidate rows."""

    def __init__(self, db: LiteratureDB):
        self.db = db

    def run(self) -> int:
        inserted = 0
        for row in self.db.source_records():
            payload = json.loads(row["raw_payload_json"])
            candidate = self.normalize_row(row["id"], row["source"], row["source_id"], row["query"], payload)
            if candidate is None:
                continue
            self.db.insert_candidate(candidate)
            inserted += 1
        return inserted

    def normalize_row(
        self,
        source_record_id: int,
        source: str,
        source_id: str,
        query: str,
        payload: dict[str, Any],
    ) -> PaperCandidate | None:
        normalizer = getattr(self, f"_normalize_{source}", self._normalize_generic)
        candidate = normalizer(source_record_id, source, source_id, query, payload)
        if candidate and not candidate.title:
            return None
        return candidate

    def _normalize_openalex(
        self,
        source_record_id: int,
        source: str,
        source_id: str,
        query: str,
        payload: dict[str, Any],
    ) -> PaperCandidate:
        authors = [
            item.get("author", {}).get("display_name")
            for item in payload.get("authorships", [])
            if item.get("author", {}).get("display_name")
        ]
        primary_location = payload.get("primary_location") or {}
        best_oa = payload.get("best_oa_location") or {}
        location = best_oa or primary_location
        return PaperCandidate(
            source_record_id=source_record_id,
            source=source,
            source_id=source_id,
            query=query,
            doi=normalize_doi(payload.get("doi")),
            title=normalize_whitespace(payload.get("title")) or "",
            authors=authors,
            year=payload.get("publication_year"),
            venue=(primary_location.get("source") or {}).get("display_name"),
            abstract=_openalex_abstract(payload.get("abstract_inverted_index")),
            publisher=(primary_location.get("source") or {}).get("host_organization_name"),
            publisher_url=first_non_empty(payload.get("doi"), payload.get("id")),
            source_url=payload.get("id"),
            pdf_url=location.get("pdf_url"),
            is_oa=(payload.get("open_access") or {}).get("is_oa"),
            document_type=payload.get("type"),
            raw_payload=payload,
        )

    def _normalize_crossref(
        self,
        source_record_id: int,
        source: str,
        source_id: str,
        query: str,
        payload: dict[str, Any],
    ) -> PaperCandidate:
        authors = []
        for author in payload.get("author", []):
            name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
            if name:
                authors.append(name)
        year = _first_date_year(payload)
        links = payload.get("link") or []
        pdf_url = None
        for link in links:
            if "pdf" in str(link.get("content-type", "")).lower():
                pdf_url = link.get("URL")
                break
        return PaperCandidate(
            source_record_id=source_record_id,
            source=source,
            source_id=source_id,
            query=query,
            doi=normalize_doi(payload.get("DOI")),
            title=_first(payload.get("title")) or "",
            authors=authors,
            year=year,
            venue=_first(payload.get("container-title")),
            abstract=normalize_whitespace(payload.get("abstract")),
            publisher=payload.get("publisher"),
            publisher_url=payload.get("URL"),
            source_url=payload.get("URL"),
            pdf_url=pdf_url,
            is_oa=None,
            document_type=_first(payload.get("type")) if isinstance(payload.get("type"), list) else payload.get("type"),
            raw_payload=payload,
        )

    def _normalize_semantic_scholar(
        self,
        source_record_id: int,
        source: str,
        source_id: str,
        query: str,
        payload: dict[str, Any],
    ) -> PaperCandidate:
        external = payload.get("externalIds") or {}
        open_pdf = payload.get("openAccessPdf") or {}
        return PaperCandidate(
            source_record_id=source_record_id,
            source=source,
            source_id=source_id,
            query=query,
            doi=normalize_doi(external.get("DOI")),
            title=normalize_whitespace(payload.get("title")) or "",
            authors=[a.get("name") for a in payload.get("authors", []) if a.get("name")],
            year=payload.get("year"),
            venue=payload.get("venue"),
            abstract=normalize_whitespace(payload.get("abstract")),
            publisher=None,
            publisher_url=payload.get("url"),
            source_url=payload.get("url"),
            pdf_url=open_pdf.get("url"),
            is_oa=payload.get("isOpenAccess"),
            document_type=_first(payload.get("publicationTypes")),
            raw_payload=payload,
        )

    def _normalize_arxiv(
        self,
        source_record_id: int,
        source: str,
        source_id: str,
        query: str,
        payload: dict[str, Any],
    ) -> PaperCandidate:
        pdf_url = None
        for link in payload.get("links", []):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
        return PaperCandidate(
            source_record_id=source_record_id,
            source=source,
            source_id=source_id,
            query=query,
            doi=None,
            title=normalize_whitespace(payload.get("title")) or "",
            authors=payload.get("authors", []),
            year=_year_from_date(payload.get("published")),
            venue="arXiv",
            abstract=normalize_whitespace(payload.get("summary")),
            publisher="arXiv",
            publisher_url=payload.get("id"),
            source_url=payload.get("id"),
            pdf_url=pdf_url,
            is_oa=True,
            document_type="preprint",
            raw_payload=payload,
        )

    def _normalize_doaj(
        self,
        source_record_id: int,
        source: str,
        source_id: str,
        query: str,
        payload: dict[str, Any],
    ) -> PaperCandidate:
        bibjson = payload.get("bibjson") or {}
        identifiers = bibjson.get("identifier", [])
        doi = None
        for identifier in identifiers:
            if identifier.get("type", "").lower() == "doi":
                doi = identifier.get("id")
                break
        links = bibjson.get("link", [])
        pdf_url = None
        source_url = None
        for link in links:
            url = link.get("url")
            if link.get("type", "").lower() == "fulltext" and source_url is None:
                source_url = url
            if "pdf" in str(link.get("content_type", "")).lower() or str(url).lower().endswith(".pdf"):
                pdf_url = url
        return PaperCandidate(
            source_record_id=source_record_id,
            source=source,
            source_id=source_id,
            query=query,
            doi=normalize_doi(doi),
            title=normalize_whitespace(bibjson.get("title")) or "",
            authors=[a.get("name") for a in bibjson.get("author", []) if a.get("name")],
            year=_safe_int(bibjson.get("year")),
            venue=(bibjson.get("journal") or {}).get("title"),
            abstract=normalize_whitespace(bibjson.get("abstract")),
            publisher=(bibjson.get("journal") or {}).get("publisher"),
            publisher_url=source_url,
            source_url=source_url,
            pdf_url=pdf_url,
            is_oa=True,
            document_type="journal-article",
            raw_payload=payload,
        )

    def _normalize_core(
        self,
        source_record_id: int,
        source: str,
        source_id: str,
        query: str,
        payload: dict[str, Any],
    ) -> PaperCandidate:
        return PaperCandidate(
            source_record_id=source_record_id,
            source=source,
            source_id=source_id,
            query=query,
            doi=normalize_doi(payload.get("doi")),
            title=normalize_whitespace(payload.get("title")) or "",
            authors=[a.get("name") if isinstance(a, dict) else str(a) for a in payload.get("authors", [])],
            year=_safe_int(payload.get("yearPublished") or payload.get("publishedYear")),
            venue=payload.get("publisher"),
            abstract=normalize_whitespace(payload.get("abstract")),
            publisher=payload.get("publisher"),
            publisher_url=payload.get("downloadUrl") or _first(payload.get("sourceFulltextUrls")),
            source_url=_first(payload.get("sourceFulltextUrls")),
            pdf_url=payload.get("downloadUrl"),
            is_oa=True if payload.get("downloadUrl") else None,
            document_type=payload.get("documentType"),
            raw_payload=payload,
        )

    def _normalize_europe_pmc(
        self,
        source_record_id: int,
        source: str,
        source_id: str,
        query: str,
        payload: dict[str, Any],
    ) -> PaperCandidate:
        pdf_url = None
        full_text_list = payload.get("fullTextUrlList", {}).get("fullTextUrl", [])
        for item in full_text_list:
            if str(item.get("documentStyle", "")).lower() == "pdf":
                pdf_url = item.get("url")
                break
        return PaperCandidate(
            source_record_id=source_record_id,
            source=source,
            source_id=source_id,
            query=query,
            doi=normalize_doi(payload.get("doi")),
            title=normalize_whitespace(payload.get("title")) or "",
            authors=[a.strip() for a in str(payload.get("authorString", "")).split(",") if a.strip()],
            year=_safe_int(payload.get("pubYear")),
            venue=payload.get("journalTitle"),
            abstract=normalize_whitespace(payload.get("abstractText")),
            publisher=None,
            publisher_url=payload.get("doi") or payload.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url"),
            source_url=payload.get("id"),
            pdf_url=pdf_url,
            is_oa=payload.get("isOpenAccess") == "Y",
            document_type=payload.get("pubType"),
            raw_payload=payload,
        )

    def _normalize_generic(
        self,
        source_record_id: int,
        source: str,
        source_id: str,
        query: str,
        payload: dict[str, Any],
    ) -> PaperCandidate:
        return PaperCandidate(
            source_record_id=source_record_id,
            source=source,
            source_id=source_id,
            query=query,
            doi=normalize_doi(payload.get("doi") or payload.get("DOI")),
            title=normalize_whitespace(payload.get("title")) or "",
            authors=[],
            year=_safe_int(payload.get("year")),
            venue=payload.get("venue"),
            abstract=normalize_whitespace(payload.get("abstract")),
            publisher=payload.get("publisher"),
            publisher_url=payload.get("publisher_url") or payload.get("url"),
            source_url=payload.get("url"),
            pdf_url=payload.get("pdf_url"),
            is_oa=payload.get("is_oa"),
            document_type=payload.get("document_type"),
            raw_payload=payload,
        )


def _openalex_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    if not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted.items():
        for index in indexes:
            positions.append((index, word))
    return " ".join(word for _, word in sorted(positions))


def _first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _year_from_date(value: str | None) -> int | None:
    if not value or len(value) < 4:
        return None
    return _safe_int(value[:4])


def _first_date_year(payload: dict[str, Any]) -> int | None:
    for key in ["published-print", "published-online", "published", "issued", "created"]:
        parts = (payload.get(key) or {}).get("date-parts")
        if parts and parts[0]:
            return _safe_int(parts[0][0])
    return None
