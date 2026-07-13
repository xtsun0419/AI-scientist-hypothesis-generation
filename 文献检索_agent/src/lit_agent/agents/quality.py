from __future__ import annotations

from pathlib import Path

from lit_agent.db import LiteratureDB
from lit_agent.models import AuditFinding


class QualityAuditAgent:
    """Creates audit findings for missing DOI, low relevance, broken PDFs, and suspicious metadata."""

    def __init__(self, db: LiteratureDB):
        self.db = db

    def run(self) -> int:
        self.db.clear_audit_findings()
        count = 0
        for paper in self.db.papers():
            if not paper["title"] or len(paper["title"]) < 8:
                self.db.insert_audit_finding(
                    AuditFinding(
                        paper_id=paper["id"],
                        severity="high",
                        issue_type="title_missing_or_short",
                        message="Paper title is missing or suspiciously short.",
                        suggestion="Inspect source payload and fix metadata normalization.",
                    )
                )
                count += 1
            if not paper["doi"]:
                self.db.insert_audit_finding(
                    AuditFinding(
                        paper_id=paper["id"],
                        severity="medium",
                        issue_type="missing_doi",
                        message="Paper has no DOI.",
                        suggestion="Export missing DOI report and resolve manually or through another source.",
                    )
                )
                count += 1
            if paper["relevance_score"] is not None and paper["relevance_score"] < 0.4:
                self.db.insert_audit_finding(
                    AuditFinding(
                        paper_id=paper["id"],
                        severity="medium",
                        issue_type="low_relevance",
                        message="Paper has low permanent-magnet relevance score.",
                        suggestion="Review title and abstract before using this record.",
                    )
                )
                count += 1

        for asset in self.db.rows("SELECT * FROM pdf_assets WHERE file_path IS NOT NULL"):
            path = Path(asset["file_path"])
            if not path.exists() or path.stat().st_size == 0:
                self.db.insert_audit_finding(
                    AuditFinding(
                        paper_id=asset["paper_id"],
                        severity="high",
                        issue_type="pdf_missing_or_empty",
                        message="PDF asset path is missing or empty.",
                        suggestion="Re-run download for this paper.",
                    )
                )
                count += 1
        return count
