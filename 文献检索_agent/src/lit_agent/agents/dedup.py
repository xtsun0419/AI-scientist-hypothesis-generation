from __future__ import annotations

import json

from lit_agent.db import LiteratureDB
from lit_agent.models import PaperRecord
from lit_agent.text import normalize_title


class DeduplicationAgent:
    """Merges candidates into canonical papers while preserving all source links."""

    def __init__(self, db: LiteratureDB):
        self.db = db

    def run(self) -> int:
        paper_ids: set[int] = set()
        for row in self.db.candidates():
            title = row["title"] or ""
            record = PaperRecord(
                id=None,
                doi=row["doi"],
                title=title,
                normalized_title=normalize_title(title),
                authors=json.loads(row["authors_json"]),
                year=row["year"],
                venue=row["venue"],
                abstract=row["abstract"],
                publisher=row["publisher"],
                publisher_url=row["publisher_url"],
                source_url=row["source_url"],
                document_type=row["document_type"],
            )
            paper_id = self.db.upsert_paper(record)
            self.db.link_paper_source(
                paper_id,
                row["source_record_id"],
                row["source"],
                row["source_id"],
                row["query"],
            )
            paper_ids.add(paper_id)
        return len(paper_ids)
