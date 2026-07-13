from __future__ import annotations

import re
import unicodedata

DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    text = text.replace("doi:", "").replace("DOI:", "").strip()
    match = DOI_RE.search(text)
    if not match:
        return None
    doi = match.group(1).rstrip(".,;)")
    return doi.lower()


def doi_url(doi: str | None) -> str | None:
    normalized = normalize_doi(doi)
    if not normalized:
        return None
    return f"https://doi.org/{normalized}"


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    text = unicodedata.normalize("NFKD", title)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_whitespace(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def first_non_empty(*values: str | None) -> str | None:
    for value in values:
        cleaned = normalize_whitespace(value)
        if cleaned:
            return cleaned
    return None


def title_similarity_key(title: str | None, year: int | None) -> str:
    normalized = normalize_title(title)
    return f"{normalized}|{year or ''}"
