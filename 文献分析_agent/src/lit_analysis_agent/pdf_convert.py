from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from lit_analysis_agent.config import default_goal_pdf_dir, default_parsed_dir, default_pdf_dir, default_retrieval_agent_dir
from lit_analysis_agent.db import AnalysisDB, utc_now_iso
from lit_analysis_agent.text import DOI_RE, normalize_whitespace


PARSER_VERSION = "pymupdf-rules-v1"
URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
FIGURE_RE = re.compile(r"^(fig\.?|figure)\s*[\dIVXLC]+[\w\-.:)]*\s*(.*)", re.IGNORECASE)
TABLE_RE = re.compile(r"^table\s*[\dIVXLC]+[\w\-.:)]*\s*(.*)", re.IGNORECASE)
REFERENCE_HEADING_RE = re.compile(r"^(references|bibliography)\s*$", re.IGNORECASE)
ABSTRACT_HEADING_RE = re.compile(r"^abstract\s*$", re.IGNORECASE)
ACK_HEADING_RE = re.compile(r"^acknowledg(e)?ments?\s*$", re.IGNORECASE)
SECTION_WORD_RE = re.compile(
    r"^(abstract|introduction|background|methods?|methodology|materials?\s+and\s+methods?|"
    r"experimental|experiments?|results?|discussion|results?\s+and\s+discussion|conclusions?|"
    r"summary|references|bibliography|acknowledg(e)?ments?|supplementary)\b",
    re.IGNORECASE,
)
NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z0-9,;:/()\- ]{2,120}$")
REFERENCE_START_RE = re.compile(r"^(\[\d+\]|\d+[\).]|\([A-Z][A-Za-z\-]+,\s*(19|20)\d{2}\))\s+")
CITATION_RE = re.compile(r"(\[[\d,\-\s;]+\]|\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2}[a-z]?\))")


@dataclass(frozen=True)
class PdfRecord:
    pdf_path: Path
    paper_id: int | None = None
    pdf_asset_id: int | None = None
    title: str | None = None
    doi: str | None = None
    year: int | None = None
    venue: str | None = None
    authors: list[str] | None = None
    abstract: str | None = None
    source: str = "file_scan"


class PdfConversionAgent:
    """Converts collected PDFs into Markdown and structured JSON."""

    def __init__(
        self,
        db: AnalysisDB,
        output_dir: Path | None = None,
        *,
        parser: Callable[[Path, PdfRecord], dict[str, Any]] | None = None,
        scan_roots: list[Path] | None = None,
    ):
        self.db = db
        self.output_dir = output_dir or default_parsed_dir()
        self.parser = parser or parse_pdf_to_document
        self.scan_roots = scan_roots if scan_roots is not None else [default_pdf_dir(), default_goal_pdf_dir()]

    def run(self, *, round_id: int | None = None, limit: int | None = None, force: bool = False) -> dict[str, int]:
        records = self.pdf_records(round_id=round_id)
        if limit is not None:
            records = records[:limit]
        stats = {"total": len(records), "converted": 0, "skipped": 0, "failed": 0}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            if not record.pdf_path.exists() or not record.pdf_path.is_file() or record.pdf_path.stat().st_size <= 0:
                self._record_failure(record, "PDF file is missing or empty.")
                stats["failed"] += 1
                continue
            existing = self.db.paper_conversion_for_path(str(record.pdf_path))
            if existing and not force and _conversion_outputs_exist(existing):
                stats["skipped"] += 1
                continue
            try:
                document = self.parser(record.pdf_path, record)
                paths = self._write_outputs(record, document)
                quality = document.get("quality", {})
                self.db.upsert_paper_conversion(
                    paper_id=record.paper_id,
                    pdf_asset_id=record.pdf_asset_id,
                    pdf_path=str(record.pdf_path),
                    markdown_path=str(paths["markdown"]),
                    json_path=str(paths["json"]),
                    status="converted",
                    parser=str(quality.get("parser") or PARSER_VERSION),
                    error_message=None,
                    page_count=int(document.get("metadata", {}).get("page_count") or 0),
                    text_chars=int(quality.get("text_chars") or 0),
                    section_count=int(quality.get("section_count") or 0),
                    figure_count=int(quality.get("figure_count") or 0),
                    table_count=int(quality.get("table_count") or 0),
                    reference_count=int(quality.get("reference_count") or 0),
                )
                stats["converted"] += 1
            except Exception as exc:
                self._record_failure(record, str(exc))
                stats["failed"] += 1
        return stats

    def pdf_records(self, *, round_id: int | None = None) -> list[PdfRecord]:
        records: list[PdfRecord] = []
        seen: set[str] = set()
        for row in self.db.pdf_assets_for_conversion(round_id):
            record = _record_from_db_row(row)
            key = str(record.pdf_path)
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
        if round_id is not None:
            return records
        for root in self.scan_roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.pdf")):
                resolved = _resolve_pdf_path(path)
                key = str(resolved)
                if key in seen:
                    continue
                seen.add(key)
                records.append(PdfRecord(pdf_path=resolved))
        return records

    def _write_outputs(self, record: PdfRecord, document: dict[str, Any]) -> dict[str, Path]:
        stem = _output_stem(record)
        markdown_path = self.output_dir / f"{stem}.md"
        json_path = self.output_dir / f"{stem}.json"
        markdown_path.write_text(render_markdown(document), encoding="utf-8")
        json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"markdown": markdown_path, "json": json_path}

    def _record_failure(self, record: PdfRecord, error_message: str) -> None:
        self.db.upsert_paper_conversion(
            paper_id=record.paper_id,
            pdf_asset_id=record.pdf_asset_id,
            pdf_path=str(record.pdf_path),
            markdown_path=None,
            json_path=None,
            status="failed",
            parser=PARSER_VERSION,
            error_message=error_message[:1000],
        )


def parse_pdf_to_document(pdf_path: Path, record: PdfRecord) -> dict[str, Any]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed. Install project dependencies before converting PDFs.") from exc

    with fitz.open(pdf_path) as doc:
        pdf_meta = dict(doc.metadata or {})
        blocks: list[dict[str, Any]] = []
        for page_index, page in enumerate(doc, start=1):
            blocks.extend(_page_blocks(page, page_index))
        page_count = doc.page_count

    structured = structure_blocks(blocks, metadata_title=record.title or pdf_meta.get("title"))
    metadata = {
        "title": structured["title"] or record.title or pdf_meta.get("title") or pdf_path.stem,
        "authors": record.authors or _split_authors(pdf_meta.get("author")),
        "year": record.year,
        "doi": record.doi,
        "venue": record.venue,
        "abstract": structured["abstract"] or record.abstract,
        "pdf_path": str(pdf_path),
        "source": record.source,
        "page_count": page_count,
        "parser_version": PARSER_VERSION,
        "converted_at": utc_now_iso(),
    }
    document = {
        "metadata": metadata,
        "sections": structured["sections"],
        "paragraphs": structured["paragraphs"],
        "figures": structured["figures"],
        "tables": structured["tables"],
        "references": structured["references"],
        "quality": {
            "status": "converted",
            "parser": PARSER_VERSION,
            "warnings": structured["warnings"],
            "text_chars": sum(len(str(block.get("text") or "")) for block in blocks),
            "section_count": len(structured["sections"]),
            "figure_count": len(structured["figures"]),
            "table_count": len(structured["tables"]),
            "reference_count": len(structured["references"]),
        },
    }
    if not document["paragraphs"]:
        document["quality"]["status"] = "low_text"
        document["quality"]["warnings"].append("No body paragraphs were detected; PDF may be scanned or extraction quality is low.")
    return document


def structure_blocks(blocks: list[dict[str, Any]], *, metadata_title: str | None = None) -> dict[str, Any]:
    warnings: list[str] = []
    title = normalize_whitespace(metadata_title) or _infer_title(blocks)
    sections: list[dict[str, Any]] = []
    paragraphs: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    reference_lines: list[str] = []
    current = _new_section("Body", 1, _first_page(blocks))
    in_references = False
    abstract_parts: list[str] = []

    for block in blocks:
        text = normalize_whitespace(str(block.get("text") or ""))
        if not text:
            continue
        page = int(block.get("page") or 0)
        if _looks_like_title_duplicate(text, title):
            continue
        if ABSTRACT_HEADING_RE.match(text):
            _finish_section(sections, current, page)
            current = _new_section("Abstract", 1, page)
            in_references = False
            continue
        if _is_section_heading(text):
            _finish_section(sections, current, page)
            current = _new_section(_clean_heading(text), _heading_level(text), page)
            in_references = bool(REFERENCE_HEADING_RE.match(_clean_heading(text)))
            continue
        figure_match = FIGURE_RE.match(text)
        if figure_match:
            figures.append({"label": _caption_label(text), "caption": text, "page": page})
            continue
        table_match = TABLE_RE.match(text)
        if table_match:
            tables.append(
                {
                    "label": _caption_label(text),
                    "caption": text,
                    "page": page,
                    "text": text,
                    "rows": _table_rows(text),
                    "extraction_quality": "text_table",
                }
            )
            continue
        if in_references or current["heading"].lower() in {"references", "bibliography"}:
            reference_lines.append(text)
            continue
        if len(text) < 20 and not current["paragraphs"]:
            continue
        paragraph = {
            "id": f"p{len(paragraphs) + 1:05d}",
            "text": text,
            "page": page,
            "section_path": [current["heading"]],
            "citations": extract_citations(text),
        }
        current["paragraphs"].append(paragraph)
        paragraphs.append(paragraph)
        if current["heading"].lower() == "abstract":
            abstract_parts.append(text)

    _finish_section(sections, current, _last_page(blocks))
    sections = [section for section in sections if section["heading"] != "Body" or section["paragraphs"]]
    references = split_reference_entries(reference_lines)
    if not sections:
        warnings.append("No sections were detected; all text is available through top-level paragraphs.")
    return {
        "title": title,
        "abstract": " ".join(abstract_parts).strip() or None,
        "sections": sections,
        "paragraphs": paragraphs,
        "figures": figures,
        "tables": tables,
        "references": references,
        "warnings": warnings,
    }


def render_markdown(document: dict[str, Any]) -> str:
    metadata = document["metadata"]
    lines = [
        f"# {metadata.get('title') or 'Untitled'}",
        "",
        "## Metadata",
        f"- DOI: {metadata.get('doi') or ''}",
        f"- Year: {metadata.get('year') or ''}",
        f"- Venue: {metadata.get('venue') or ''}",
        f"- PDF: `{metadata.get('pdf_path') or ''}`",
        f"- Parser: {metadata.get('parser_version') or PARSER_VERSION}",
        "",
    ]
    if metadata.get("abstract"):
        lines.extend(["## Abstract", str(metadata["abstract"]), ""])
    for section in document.get("sections", []):
        heading = section.get("heading") or "Section"
        level = min(max(int(section.get("level") or 2), 1), 5)
        lines.extend([f"{'#' * (level + 1)} {heading}", ""])
        for paragraph in section.get("paragraphs", []):
            lines.extend([str(paragraph.get("text") or ""), ""])
    if document.get("figures"):
        lines.extend(["## Figures", ""])
        for figure in document["figures"]:
            lines.append(f"- {figure.get('caption')} (p. {figure.get('page')})")
        lines.append("")
    if document.get("tables"):
        lines.extend(["## Tables", ""])
        for table in document["tables"]:
            lines.append(f"- {table.get('caption')} (p. {table.get('page')})")
        lines.append("")
    if document.get("references"):
        lines.extend(["## References", ""])
        for index, reference in enumerate(document["references"], start=1):
            lines.append(f"{index}. {reference.get('raw_text')}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def split_reference_entries(lines: list[str]) -> list[dict[str, Any]]:
    entries: list[str] = []
    current = ""
    for line in lines:
        cleaned = normalize_whitespace(line)
        if not cleaned:
            continue
        if REFERENCE_START_RE.match(cleaned) and current:
            entries.append(current.strip())
            current = cleaned
        else:
            current = f"{current} {cleaned}".strip()
    if current:
        entries.append(current.strip())
    if not entries and lines:
        entries = [line for line in (normalize_whitespace(item) for item in lines) if line]
    return [_reference_entry(entry) for entry in entries]


def extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(match.group(1) for match in CITATION_RE.finditer(text)))


def _page_blocks(page: Any, page_index: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    try:
        page_dict = page.get_text("dict")
    except Exception:
        page_dict = {}
    for block in page_dict.get("blocks", []):
        lines = []
        sizes = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(str(span.get("text") or "") for span in spans).strip()
            if line_text:
                lines.append(line_text)
            sizes.extend(float(span.get("size") or 0) for span in spans)
        text = normalize_whitespace(" ".join(lines))
        if text:
            blocks.append(
                {
                    "text": text,
                    "page": page_index,
                    "font_size": max(sizes) if sizes else 0,
                    "bbox": block.get("bbox"),
                }
            )
    if blocks:
        return blocks
    text = page.get_text("text")
    return [{"text": item, "page": page_index, "font_size": 0, "bbox": None} for item in _paragraphs_from_text(text)]


def _paragraphs_from_text(text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            if current:
                paragraphs.append(normalize_whitespace(" ".join(current)) or "")
                current = []
            continue
        current.append(cleaned)
    if current:
        paragraphs.append(normalize_whitespace(" ".join(current)) or "")
    return [item for item in paragraphs if item]


def _record_from_db_row(row: Any) -> PdfRecord:
    authors = []
    if row["authors_json"]:
        try:
            authors = [str(item) for item in json.loads(row["authors_json"])]
        except json.JSONDecodeError:
            authors = []
    year = int(row["year"]) if row["year"] is not None else None
    return PdfRecord(
        pdf_path=_resolve_pdf_path(row["pdf_path"]),
        paper_id=int(row["paper_id"]) if row["paper_id"] is not None else None,
        pdf_asset_id=int(row["pdf_asset_id"]) if row["pdf_asset_id"] is not None else None,
        title=row["title"],
        doi=row["doi"],
        year=year,
        venue=row["venue"],
        authors=authors,
        abstract=row["abstract"],
        source="pdf_assets",
    )


def _resolve_pdf_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = default_retrieval_agent_dir() / candidate
    return candidate.resolve()


def _output_stem(record: PdfRecord) -> str:
    digest = hashlib.sha1(str(record.pdf_path).encode("utf-8")).hexdigest()[:10]
    if record.paper_id is not None:
        prefix = f"paper_{record.paper_id:06d}"
    else:
        prefix = "orphan_pdf"
    title = _slug(record.title or record.pdf_path.stem)
    return f"{prefix}_{title}_{digest}"[:160]


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return text[:80] or "untitled"


def _conversion_outputs_exist(row: Any) -> bool:
    markdown = row["markdown_path"]
    json_path = row["json_path"]
    return bool(markdown and json_path and Path(markdown).exists() and Path(json_path).exists())


def _infer_title(blocks: list[dict[str, Any]]) -> str | None:
    candidates = [block for block in blocks[:12] if len(str(block.get("text") or "")) >= 8]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (float(item.get("font_size") or 0), -int(item.get("page") or 0)), reverse=True)
    return normalize_whitespace(str(candidates[0].get("text") or ""))


def _looks_like_title_duplicate(text: str, title: str | None) -> bool:
    return bool(title and text.lower() == title.lower())


def _is_section_heading(text: str) -> bool:
    if len(text) > 140:
        return False
    if FIGURE_RE.match(text) or TABLE_RE.match(text):
        return False
    if SECTION_WORD_RE.match(text):
        return True
    if NUMBERED_HEADING_RE.match(text):
        return True
    words = text.split()
    return 1 <= len(words) <= 8 and text.isupper() and any(len(word) > 3 for word in words)


def _clean_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip(".")


def _heading_level(text: str) -> int:
    match = re.match(r"^(\d+(?:\.\d+)*)", text)
    if not match:
        return 1
    return min(match.group(1).count(".") + 1, 4)


def _new_section(heading: str, level: int, page: int) -> dict[str, Any]:
    return {"heading": heading, "level": level, "page_start": page, "page_end": page, "paragraphs": []}


def _finish_section(sections: list[dict[str, Any]], section: dict[str, Any], page_end: int) -> None:
    section["page_end"] = max(section.get("page_start") or 0, page_end or section.get("page_start") or 0)
    if section["heading"] == "Body" and not section["paragraphs"]:
        return
    sections.append(section)


def _first_page(blocks: list[dict[str, Any]]) -> int:
    return int(blocks[0].get("page") or 1) if blocks else 1


def _last_page(blocks: list[dict[str, Any]]) -> int:
    return int(blocks[-1].get("page") or 1) if blocks else 1


def _caption_label(text: str) -> str:
    return text.split(maxsplit=2)[0:2] and " ".join(text.split(maxsplit=2)[:2])


def _table_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in re.split(r"\s{2,}|\t+", line) if cell.strip()]
        if len(cells) > 1:
            rows.append(cells)
    return rows


def _reference_entry(text: str) -> dict[str, Any]:
    doi_match = DOI_RE.search(text)
    url_match = URL_RE.search(text)
    year_match = YEAR_RE.search(text)
    return {
        "raw_text": text,
        "doi": doi_match.group(1).rstrip(".,;") if doi_match else None,
        "url": url_match.group(0).rstrip(".,;") if url_match else None,
        "year": int(year_match.group(0)) if year_match else None,
    }


def _split_authors(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in re.split(r";|,", value) if item.strip()]
