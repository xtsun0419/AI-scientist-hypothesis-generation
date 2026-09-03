from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import requests


IDCONV_API = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
EFETCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_pubmed_records(ids: list[str], *, email: str, tool: str = "ai_scientist") -> list[dict[str, Any]]:
    cleaned = list(dict.fromkeys(item.strip() for item in ids if item.strip()))
    if not cleaned:
        return []
    session = requests.Session()
    session.headers.update({"User-Agent": f"ai-scientist/1.0 ({email})", "From": email})
    aliases = _idconv(cleaned, email=email, tool=tool, session=session)
    pmids = [aliases.get(item, {}).get("pmid") or item for item in cleaned]
    pmids = [item for item in pmids if item.isdigit()]
    metadata = _efetch(pmids, email=email, tool=tool, session=session)
    records = []
    for requested_id in cleaned:
        info = aliases.get(requested_id, {})
        pmid = info.get("pmid") or (requested_id if requested_id.isdigit() else "")
        item = metadata.get(pmid)
        if not item:
            continue
        records.append({"requested_id": requested_id, "pmid": pmid, "pmcid": info.get("pmcid", ""), **item})
    return records


def _idconv(ids: list[str], *, email: str, tool: str, session: requests.Session) -> dict[str, dict[str, str]]:
    response = session.get(IDCONV_API, params={"ids": ",".join(ids), "tool": tool, "email": email}, timeout=60)
    response.raise_for_status()
    aliases: dict[str, dict[str, str]] = {}
    for record in ET.fromstring(response.text).findall(".//record"):
        info = {
            "pmid": (record.get("pmid") or "").strip(),
            "pmcid": (record.get("pmcid") or "").strip(),
            "doi": (record.get("doi") or "").strip(),
        }
        for key in [record.get("requested-id"), record.get("orig-id"), info["pmid"], info["pmcid"], info["doi"]]:
            if key:
                aliases[str(key).strip()] = info
    return aliases


def _efetch(pmids: list[str], *, email: str, tool: str, session: requests.Session) -> dict[str, dict[str, Any]]:
    if not pmids:
        return {}
    response = session.get(
        EFETCH_API,
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "tool": tool, "email": email},
        timeout=90,
    )
    response.raise_for_status()
    records: dict[str, dict[str, Any]] = {}
    for article in ET.fromstring(response.text).findall(".//PubmedArticle"):
        pmid = (article.findtext(".//PMID") or "").strip()
        if not pmid:
            continue
        abstract = "\n".join((node.text or "").strip() for node in article.findall(".//Abstract/AbstractText") if (node.text or "").strip())
        doi = next(
            ((node.text or "").strip() for node in article.findall(".//ArticleIdList/ArticleId") if node.attrib.get("IdType") == "doi" and (node.text or "").strip()),
            "",
        )
        records[pmid] = {
            "title": (article.findtext(".//ArticleTitle") or "").strip(),
            "abstract": abstract,
            "doi": doi,
            "venue": (article.findtext(".//Journal/Title") or "").strip(),
            "year": (article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate") or "").strip(),
        }
    return records
