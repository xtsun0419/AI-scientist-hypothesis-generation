from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from typing import Any

from lit_agent.models import QueryPlan, RawSourceRecord

from .base import SourceConnector


class OpenAlexConnector(SourceConnector):
    name = "openalex"

    def search(self, query: str, plan: QueryPlan) -> list[RawSourceRecord]:
        filters = ",".join(
            [
                f"from_publication_date:{plan.from_year}-01-01",
                f"to_publication_date:{plan.to_year}-12-31",
            ]
        )
        url = "https://api.openalex.org/works?" + self.urlencode(
            {"search": query, "filter": filters, "per-page": plan.max_results_per_query}
        )
        data = self.fetch_json(url)
        return [
            RawSourceRecord(
                source=self.name,
                source_id=str(item.get("id") or item.get("doi") or _stable_id(item)),
                query=query,
                raw_payload=item,
            )
            for item in data.get("results", [])
        ]


class CrossrefConnector(SourceConnector):
    name = "crossref"

    def search(self, query: str, plan: QueryPlan) -> list[RawSourceRecord]:
        filters = ",".join(
            [
                f"from-pub-date:{plan.from_year}-01-01",
                f"until-pub-date:{plan.to_year}-12-31",
            ]
        )
        url = "https://api.crossref.org/works?" + self.urlencode(
            {"query": query, "filter": filters, "rows": plan.max_results_per_query}
        )
        data = self.fetch_json(url)
        items = data.get("message", {}).get("items", [])
        return [
            RawSourceRecord(
                source=self.name,
                source_id=str(item.get("DOI") or item.get("URL") or _stable_id(item)),
                query=query,
                raw_payload=item,
            )
            for item in items
        ]


class SemanticScholarConnector(SourceConnector):
    name = "semantic_scholar"

    def search(self, query: str, plan: QueryPlan) -> list[RawSourceRecord]:
        fields = ",".join(
            [
                "paperId",
                "externalIds",
                "url",
                "title",
                "abstract",
                "venue",
                "year",
                "authors",
                "publicationTypes",
                "openAccessPdf",
                "isOpenAccess",
            ]
        )
        url = "https://api.semanticscholar.org/graph/v1/paper/search?" + self.urlencode(
            {"query": query, "limit": plan.max_results_per_query, "fields": fields}
        )
        data = self.fetch_json(url)
        return [
            RawSourceRecord(
                source=self.name,
                source_id=str(item.get("paperId") or item.get("externalIds", {}).get("DOI") or _stable_id(item)),
                query=query,
                raw_payload=item,
            )
            for item in data.get("data", [])
        ]


class ArxivConnector(SourceConnector):
    name = "arxiv"

    def search(self, query: str, plan: QueryPlan) -> list[RawSourceRecord]:
        url = "https://export.arxiv.org/api/query?" + self.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": plan.max_results_per_query,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        text = self.fetch_text(url)
        return _parse_arxiv_atom(text, query)


class DoajConnector(SourceConnector):
    name = "doaj"

    def search(self, query: str, plan: QueryPlan) -> list[RawSourceRecord]:
        url = "https://doaj.org/api/search/articles/" + self.urlencode_path(query) + "?" + self.urlencode(
            {"pageSize": plan.max_results_per_query}
        )
        data = self.fetch_json(url)
        return [
            RawSourceRecord(
                source=self.name,
                source_id=str(item.get("id") or item.get("bibjson", {}).get("identifier", [{}])[0].get("id") or _stable_id(item)),
                query=query,
                raw_payload=item,
            )
            for item in data.get("results", [])
        ]

    @staticmethod
    def urlencode_path(value: str) -> str:
        from urllib.parse import quote

        return quote(value)


class CoreConnector(SourceConnector):
    name = "core"

    def search(self, query: str, plan: QueryPlan) -> list[RawSourceRecord]:
        url = "https://api.core.ac.uk/v3/search/works?" + self.urlencode(
            {"q": query, "limit": plan.max_results_per_query}
        )
        data = self.fetch_json(url)
        return [
            RawSourceRecord(
                source=self.name,
                source_id=str(item.get("id") or item.get("doi") or _stable_id(item)),
                query=query,
                raw_payload=item,
            )
            for item in data.get("results", [])
        ]


class EuropePmcConnector(SourceConnector):
    name = "europe_pmc"

    def search(self, query: str, plan: QueryPlan) -> list[RawSourceRecord]:
        date_query = f"({query}) AND FIRST_PDATE:[{plan.from_year}-01-01 TO {plan.to_year}-12-31]"
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + self.urlencode(
            {"query": date_query, "format": "json", "pageSize": plan.max_results_per_query}
        )
        data = self.fetch_json(url)
        return [
            RawSourceRecord(
                source=self.name,
                source_id=str(item.get("id") or item.get("doi") or _stable_id(item)),
                query=query,
                raw_payload=item,
            )
            for item in data.get("resultList", {}).get("result", [])
        ]


def _parse_arxiv_atom(text: str, query: str) -> list[RawSourceRecord]:
    namespace = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(text)
    records: list[RawSourceRecord] = []
    for entry in root.findall("atom:entry", namespace):
        entry_id = _xml_text(entry, "atom:id", namespace) or _stable_id({"entry": ET.tostring(entry, encoding="unicode")})
        title = _xml_text(entry, "atom:title", namespace)
        summary = _xml_text(entry, "atom:summary", namespace)
        published = _xml_text(entry, "atom:published", namespace)
        authors = [
            _xml_text(author, "atom:name", namespace)
            for author in entry.findall("atom:author", namespace)
        ]
        links = []
        for link in entry.findall("atom:link", namespace):
            links.append(dict(link.attrib))
        primary_category = entry.find("arxiv:primary_category", namespace)
        payload: dict[str, Any] = {
            "id": entry_id,
            "title": title,
            "summary": summary,
            "published": published,
            "authors": [author for author in authors if author],
            "links": links,
            "primary_category": dict(primary_category.attrib) if primary_category is not None else {},
        }
        records.append(
            RawSourceRecord(
                source="arxiv",
                source_id=entry_id,
                query=query,
                raw_payload=payload,
            )
        )
    return records


def _xml_text(node: ET.Element, path: str, namespace: dict[str, str]) -> str | None:
    child = node.find(path, namespace)
    if child is None or child.text is None:
        return None
    return " ".join(child.text.split())


def _stable_id(payload: Any) -> str:
    text = repr(payload).encode("utf-8", errors="replace")
    return hashlib.sha1(text).hexdigest()
