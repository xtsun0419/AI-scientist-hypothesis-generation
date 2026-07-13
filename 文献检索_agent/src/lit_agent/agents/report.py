from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from lit_agent.db import LiteratureDB


class ReportExportAgent:
    """Writes CSV/JSONL/BibTeX exports and operational reports."""

    def __init__(self, db: LiteratureDB, report_dir: Path):
        self.db = db
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def report(self) -> dict[str, Path]:
        outputs = {
            "all_papers": self.report_dir / "all_papers.csv",
            "open_pdf": self.report_dir / "open_pdf_records.csv",
            "closed_with_doi": self.report_dir / "closed_access_with_doi.csv",
            "missing_doi": self.report_dir / "missing_doi.csv",
            "download_failed": self.report_dir / "download_failed.csv",
            "audit": self.report_dir / "audit_findings.csv",
            "source_failures": self.report_dir / "source_failures.csv",
            "llm_reviews": self.report_dir / "llm_relevance_reviews.csv",
            "pipeline_metrics": self.report_dir / "pipeline_metrics.csv",
            "summary": self.report_dir / "summary.md",
        }
        self._write_csv(outputs["all_papers"], self._paper_rows())
        self._write_csv(outputs["open_pdf"], self._access_rows("downloaded_oa_pdf", "preprint_pdf", "oa_pdf_available", "oa_no_pdf_url"))
        self._write_csv(outputs["closed_with_doi"], self._access_rows("closed_access_has_doi"))
        self._write_csv(outputs["missing_doi"], self._missing_doi_rows())
        self._write_csv(outputs["download_failed"], self._access_rows("download_failed"))
        self._write_csv(outputs["audit"], [dict(row) for row in self.db.audit_findings()])
        self._write_csv(outputs["source_failures"], [dict(row) for row in self.db.source_failures()])
        self._write_csv(outputs["llm_reviews"], [dict(row) for row in self.db.llm_relevance_reviews()])
        self._write_csv(outputs["pipeline_metrics"], [dict(row) for row in self.db.pipeline_metrics()])
        outputs["summary"].write_text(self._summary_markdown(), encoding="utf-8")
        return outputs

    def export(self, fmt: str) -> Path:
        fmt = fmt.lower()
        if fmt == "csv":
            path = self.report_dir / "literature_export.csv"
            self._write_csv(path, self._paper_rows())
            return path
        if fmt == "jsonl":
            path = self.report_dir / "literature_export.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in self._paper_rows():
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            return path
        if fmt == "bibtex":
            path = self.report_dir / "literature_export.bib"
            path.write_text("\n\n".join(self._bibtex_entries()), encoding="utf-8")
            return path
        raise ValueError(f"Unsupported export format: {fmt}")

    def _paper_rows(self) -> list[dict[str, object]]:
        rows = self.db.rows(
            """
            SELECT
                p.id, p.doi, p.title, p.year, p.venue, p.publisher, p.publisher_url,
                p.source_url, p.document_type, p.relevance_score, p.relevance_terms_json,
                p.relevance_reason, ar.doi_url, ar.is_oa, ar.pdf_url, ar.access_status
            FROM papers p
            LEFT JOIN access_records ar ON ar.paper_id = p.id
            ORDER BY p.year DESC, p.title
            """
        )
        return [dict(row) for row in rows]

    def _access_rows(self, *statuses: str) -> list[dict[str, object]]:
        placeholders = ",".join("?" for _ in statuses)
        return [
            dict(row)
            for row in self.db.rows(
                f"""
                SELECT
                    p.id, p.title, p.year, p.venue, ar.doi, ar.doi_url, ar.publisher_url,
                    ar.source_url, ar.pdf_url, ar.access_status,
                    pa.error_message AS download_error
                FROM access_records ar
                JOIN papers p ON p.id = ar.paper_id
                LEFT JOIN pdf_assets pa ON pa.id = (
                    SELECT id
                    FROM pdf_assets
                    WHERE paper_id = p.id
                    ORDER BY id DESC
                    LIMIT 1
                )
                WHERE ar.access_status IN ({placeholders})
                ORDER BY p.year DESC, p.title
                """,
                statuses,
            )
        ]

    def _missing_doi_rows(self) -> list[dict[str, object]]:
        return [
            dict(row)
            for row in self.db.rows(
                """
                SELECT
                    p.id, p.title, p.year, p.venue, p.publisher, p.publisher_url,
                    p.source_url, p.document_type, p.relevance_score, p.relevance_reason,
                    ar.pdf_url, ar.access_status
                FROM papers p
                LEFT JOIN access_records ar ON ar.paper_id = p.id
                WHERE p.doi IS NULL OR p.doi = ''
                ORDER BY p.year DESC, p.title
                """
            )
        ]

    def _summary_markdown(self) -> str:
        counts = {
            "papers": self.db.rows("SELECT COUNT(*) AS n FROM papers")[0]["n"],
            "source_records": self.db.rows("SELECT COUNT(*) AS n FROM source_records")[0]["n"],
            "paper_candidates": self.db.rows("SELECT COUNT(*) AS n FROM paper_candidates")[0]["n"],
            "downloaded": self.db.rows("SELECT COUNT(*) AS n FROM pdf_assets WHERE status IN ('downloaded_oa_pdf', 'preprint_pdf')")[0]["n"],
            "oa_pdf_available": self.db.rows("SELECT COUNT(*) AS n FROM access_records WHERE access_status = 'oa_pdf_available'")[0]["n"],
            "closed_with_doi": self.db.rows("SELECT COUNT(*) AS n FROM access_records WHERE access_status = 'closed_access_has_doi'")[0]["n"],
            "missing_doi": self.db.rows("SELECT COUNT(*) AS n FROM papers WHERE doi IS NULL OR doi = ''")[0]["n"],
            "source_failures": self.db.rows("SELECT COUNT(*) AS n FROM source_failures")[0]["n"],
            "llm_reviews": self.db.rows("SELECT COUNT(*) AS n FROM llm_relevance_reviews")[0]["n"],
            "audit_findings": self.db.rows("SELECT COUNT(*) AS n FROM audit_findings")[0]["n"],
        }
        metrics = {row["metric_name"]: row["metric_value"] for row in self.db.pipeline_metrics()}
        lines = ["# Literature Corpus Construction Summary", ""]
        for key, value in counts.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
        lines.append("## Method Metrics")
        for key in [
            "dedup_compression_ratio",
            "doi_coverage",
            "oa_coverage",
            "pdf_url_coverage",
            "downloaded_pdf_coverage",
        ]:
            if key in metrics:
                lines.append(f"- {key}: {metrics[key]:.3f}")
        lines.append("")
        lines.append("Reports include source coverage, source failures, DOI/OA/PDF availability, LLM relevance reviews, and audit findings.")
        return "\n".join(lines)

    def _bibtex_entries(self) -> list[str]:
        entries = []
        for row in self._paper_rows():
            key = _bibtex_key(row)
            fields = {
                "title": row.get("title"),
                "year": row.get("year"),
                "journal": row.get("venue"),
                "doi": row.get("doi"),
                "url": row.get("doi_url") or row.get("publisher_url") or row.get("source_url"),
            }
            body = []
            for name, value in fields.items():
                if value:
                    body.append(f"  {name} = {{{_bibtex_escape(str(value))}}}")
            entries.append("@article{" + key + ",\n" + ",\n".join(body) + "\n}")
        return entries

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _bibtex_key(row: dict[str, object]) -> str:
    doi = str(row.get("doi") or "")
    if doi:
        return "doi_" + "".join(ch if ch.isalnum() else "_" for ch in doi)[-40:]
    title = str(row.get("title") or "paper")
    year = str(row.get("year") or "na")
    first = "".join(ch for ch in title.split()[0].lower() if ch.isalnum()) if title.split() else "paper"
    return f"{first}_{year}_{row.get('id')}"


def _bibtex_escape(value: str) -> str:
    return value.replace("{", "\\{").replace("}", "\\}")
