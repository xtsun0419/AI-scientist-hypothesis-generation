from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PdfUrlCandidate:
    url: str
    source: str
    priority: int
    reason: str


def collect_pdf_url_candidates(
    *,
    paper_id: int,
    doi: str | None,
    publisher_url: str | None,
    source_url: str | None,
    candidate_rows: list[Any],
    unpaywall_payload: dict[str, Any] | None = None,
) -> list[PdfUrlCandidate]:
    """Collect deterministic legal PDF URL candidates from source metadata."""

    _ = paper_id
    collected: list[PdfUrlCandidate] = []

    for row in candidate_rows:
        source = str(row["source"])
        pdf_url = row["pdf_url"]
        if pdf_url:
            collected.append(PdfUrlCandidate(str(pdf_url), source, 20, "normalized_pdf_url"))
        payload = _jsonish(row["raw_payload_json"])
        collected.extend(_payload_pdf_candidates(source, payload))

    if unpaywall_payload:
        collected.extend(_unpaywall_candidates(unpaywall_payload))

    for url in [publisher_url, source_url]:
        if url:
            collected.extend(_heuristic_candidates(str(url), doi=doi))

    if doi:
        collected.extend(_doi_heuristic_candidates(doi))

    return _dedupe_candidates(collected)


def _payload_pdf_candidates(source: str, payload: dict[str, Any]) -> list[PdfUrlCandidate]:
    candidates: list[PdfUrlCandidate] = []
    if source == "openalex":
        for key, priority in [("best_oa_location", 5), ("primary_location", 10)]:
            location = payload.get(key) or {}
            _add_location_pdf(candidates, source, location, priority, key)
        for index, location in enumerate(payload.get("locations") or []):
            _add_location_pdf(candidates, source, location or {}, 30 + index, "location")
        open_access = payload.get("open_access") or {}
        oa_url = open_access.get("oa_url")
        if oa_url:
            candidates.extend(_heuristic_candidates(str(oa_url), doi=_payload_doi(payload), source=source, priority=35))
    elif source == "crossref":
        for index, link in enumerate(payload.get("link") or []):
            url = link.get("URL")
            content_type = str(link.get("content-type") or "").lower()
            if url and ("pdf" in content_type or _url_looks_like_pdf(str(url))):
                candidates.append(PdfUrlCandidate(str(url), source, 20 + index, "crossref_link"))
    elif source == "semantic_scholar":
        open_pdf = payload.get("openAccessPdf") or {}
        if open_pdf.get("url"):
            candidates.append(PdfUrlCandidate(str(open_pdf["url"]), source, 10, "semantic_scholar_open_access_pdf"))
    elif source == "arxiv":
        for index, link in enumerate(payload.get("links") or []):
            url = link.get("href")
            if url and (link.get("title") == "pdf" or link.get("type") == "application/pdf"):
                candidates.append(PdfUrlCandidate(str(url), source, 5 + index, "arxiv_pdf_link"))
    elif source == "doaj":
        bibjson = payload.get("bibjson") or {}
        for index, link in enumerate(bibjson.get("link") or []):
            url = link.get("url")
            if not url:
                continue
            if "pdf" in str(link.get("content_type") or "").lower() or _url_looks_like_pdf(str(url)):
                candidates.append(PdfUrlCandidate(str(url), source, 10 + index, "doaj_pdf_link"))
            else:
                candidates.extend(_heuristic_candidates(str(url), doi=_doi_from_doaj(bibjson), source=source, priority=35 + index))
    elif source == "core":
        if payload.get("downloadUrl"):
            candidates.append(PdfUrlCandidate(str(payload["downloadUrl"]), source, 10, "core_download_url"))
        for index, url in enumerate(payload.get("sourceFulltextUrls") or []):
            if url:
                candidates.extend(_heuristic_candidates(str(url), doi=_payload_doi(payload), source=source, priority=30 + index))
    elif source == "europe_pmc":
        for index, item in enumerate((payload.get("fullTextUrlList") or {}).get("fullTextUrl") or []):
            url = item.get("url")
            if not url:
                continue
            if str(item.get("documentStyle") or "").lower() == "pdf" or _url_looks_like_pdf(str(url)):
                candidates.append(PdfUrlCandidate(str(url), source, 10 + index, "europe_pmc_pdf_link"))
            else:
                candidates.extend(_heuristic_candidates(str(url), doi=_payload_doi(payload), source=source, priority=35 + index))
    return candidates


def _unpaywall_candidates(payload: dict[str, Any]) -> list[PdfUrlCandidate]:
    candidates: list[PdfUrlCandidate] = []
    locations = []
    if payload.get("best_oa_location"):
        locations.append(payload["best_oa_location"])
    locations.extend(payload.get("oa_locations") or [])
    for index, location in enumerate(locations):
        pdf_url = location.get("url_for_pdf")
        if pdf_url:
            candidates.append(PdfUrlCandidate(str(pdf_url), "unpaywall", 1 + index, "unpaywall_url_for_pdf"))
        landing_url = location.get("url")
        if landing_url:
            candidates.extend(_heuristic_candidates(str(landing_url), doi=payload.get("doi"), source="unpaywall", priority=25 + index))
    return candidates


def _add_location_pdf(
    candidates: list[PdfUrlCandidate],
    source: str,
    location: dict[str, Any],
    priority: int,
    reason: str,
) -> None:
    pdf_url = location.get("pdf_url")
    if pdf_url:
        candidates.append(PdfUrlCandidate(str(pdf_url), source, priority, reason))
    landing = location.get("landing_page_url")
    doi = _payload_doi(location)
    if landing:
        candidates.extend(_heuristic_candidates(str(landing), doi=doi, source=source, priority=priority + 20))


def _heuristic_candidates(
    url: str,
    *,
    doi: str | None,
    source: str = "heuristic",
    priority: int = 40,
) -> list[PdfUrlCandidate]:
    candidates: list[PdfUrlCandidate] = []
    if not url:
        return candidates
    normalized = _normalize_url(url)
    parsed = urllib.parse.urlparse(normalized)
    host = parsed.netloc.lower()
    path = parsed.path

    if _url_looks_like_pdf(normalized):
        candidates.append(PdfUrlCandidate(normalized, source, priority, "direct_pdf_url"))

    if "mdpi.com" in host:
        mdpi_path = path.rstrip("/")
        if not mdpi_path.endswith("/pdf"):
            candidates.append(PdfUrlCandidate(_replace_path_query(parsed, mdpi_path + "/pdf", ""), source, priority + 1, "mdpi_article_pdf"))

    if "tandfonline.com" in host and "/doi/pdf/" in path:
        candidates.append(PdfUrlCandidate(_replace_query(parsed, ""), source, priority + 1, "tandfonline_pdf_without_needaccess"))
        candidates.append(PdfUrlCandidate(_replace_query(parsed, "download=true"), source, priority + 2, "tandfonline_pdf_download"))

    if "doi.org" in host and doi:
        lower_doi = doi.lower()
        if lower_doi.startswith("10.3390/"):
            mdpi = _mdpi_pdf_from_doi(lower_doi)
            if mdpi:
                candidates.append(PdfUrlCandidate(mdpi, source, priority + 3, "mdpi_doi_pdf"))

    return candidates


def _doi_heuristic_candidates(doi: str) -> list[PdfUrlCandidate]:
    candidates: list[PdfUrlCandidate] = []
    mdpi = _mdpi_pdf_from_doi(doi.lower())
    if mdpi:
        candidates.append(PdfUrlCandidate(mdpi, "doi_heuristic", 50, "mdpi_doi_pdf"))
    return candidates


def _mdpi_pdf_from_doi(doi: str) -> str | None:
    match = re.match(r"10\.3390/([a-z]+)(\d{2})(\d{2})(\d+)$", doi)
    if not match:
        return None
    journal, volume, issue, article = match.groups()
    return f"https://www.mdpi.com/2073-4352/{int(volume)}/{int(issue)}/{int(article)}/pdf" if journal == "cryst" else None


def _dedupe_candidates(candidates: list[PdfUrlCandidate]) -> list[PdfUrlCandidate]:
    by_url: dict[str, PdfUrlCandidate] = {}
    for candidate in candidates:
        url = _normalize_url(candidate.url)
        if not url.startswith(("http://", "https://")):
            continue
        current = by_url.get(url)
        normalized = PdfUrlCandidate(url, candidate.source, candidate.priority, candidate.reason)
        if current is None or normalized.priority < current.priority:
            by_url[url] = normalized
    return sorted(by_url.values(), key=lambda item: (item.priority, item.url))


def _normalize_url(url: str) -> str:
    return url.strip().replace(" ", "%20")


def _replace_query(parsed: urllib.parse.ParseResult, query: str) -> str:
    return urllib.parse.urlunparse(parsed._replace(query=query))


def _replace_path_query(parsed: urllib.parse.ParseResult, path: str, query: str) -> str:
    return urllib.parse.urlunparse(parsed._replace(path=path, query=query))


def _url_looks_like_pdf(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.path.lower().endswith(".pdf") or "/pdf" in parsed.path.lower()


def _payload_doi(payload: dict[str, Any]) -> str | None:
    value = payload.get("doi")
    if isinstance(value, str):
        value = value.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        return value.lower()
    ids = payload.get("ids") or {}
    doi = ids.get("doi") if isinstance(ids, dict) else None
    if isinstance(doi, str):
        return doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/").lower()
    return None


def _doi_from_doaj(bibjson: dict[str, Any]) -> str | None:
    for identifier in bibjson.get("identifier") or []:
        if str(identifier.get("type") or "").lower() == "doi":
            return str(identifier.get("id") or "").lower() or None
    return None


def _jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json

        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}
