from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path

from lit_agent.constants import ACCESS_DOWNLOAD_FAILED, ACCESS_DOWNLOADED_OA_PDF, ACCESS_PREPRINT_PDF
from lit_agent.db import LiteratureDB
from lit_agent.http import urlopen


class PdfDownloadAgent:
    """Downloads only open or explicitly public PDFs already resolved by OAResolverAgent."""

    def __init__(self, db: LiteratureDB, pdf_dir: Path, *, timeout_seconds: int = 60):
        self.db = db
        self.pdf_dir = pdf_dir
        self.timeout_seconds = timeout_seconds
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *, limit: int | None = None) -> int:
        return self.run_records(self.db.access_records_for_download(), limit=limit)

    def run_records(self, records, *, limit: int | None = None) -> int:
        count = 0
        if limit is not None:
            records = records[:limit]
        for record in records:
            urls = self._candidate_urls(record)
            if not urls:
                continue
            downloaded = self._download_one_record(record, urls)
            if downloaded:
                count += 1
        return count

    def _candidate_urls(self, record) -> list[str]:
        urls = []
        if record["pdf_url"]:
            urls.append(str(record["pdf_url"]))
        for candidate in self.db.pdf_candidates_for_paper(record["paper_id"]):
            urls.append(str(candidate["pdf_url"]))
        return _dedupe_urls(urls)

    def _download_one_record(self, record, urls: list[str]) -> bool:
        last_url = urls[-1]
        last_error = None
        for pdf_url in urls:
            try:
                content, content_type = self._fetch_pdf(pdf_url)
                if not self._looks_like_pdf(content, content_type):
                    raise ValueError(f"URL did not return a PDF-like response: {content_type}")
                sha256 = hashlib.sha256(content).hexdigest()
                suffix = "pdf"
                filename = f"{record['paper_id']:08d}_{sha256[:12]}.{suffix}"
                path = self.pdf_dir / filename
                if not path.exists():
                    path.write_bytes(content)
                status = ACCESS_PREPRINT_PDF if record["access_status"] == ACCESS_PREPRINT_PDF else ACCESS_DOWNLOADED_OA_PDF
                self.db.upsert_pdf_asset(
                    paper_id=record["paper_id"],
                    pdf_url=pdf_url,
                    file_path=str(path),
                    sha256=sha256,
                    file_size=len(content),
                    status=status,
                    error_message=None,
                )
                self.db.update_pdf_candidate_status(
                    paper_id=record["paper_id"],
                    pdf_url=pdf_url,
                    status=status,
                    error_message=None,
                )
                return True
            except Exception as exc:  # pragma: no cover - network failures vary
                last_url = pdf_url
                last_error = str(exc)
                self.db.upsert_pdf_asset(
                    paper_id=record["paper_id"],
                    pdf_url=pdf_url,
                    file_path=None,
                    sha256=None,
                    file_size=None,
                    status=ACCESS_DOWNLOAD_FAILED,
                    error_message=last_error,
                )
                self.db.update_pdf_candidate_status(
                    paper_id=record["paper_id"],
                    pdf_url=pdf_url,
                    status=ACCESS_DOWNLOAD_FAILED,
                    error_message=last_error,
                )
        self.db.upsert_pdf_asset(
            paper_id=record["paper_id"],
            pdf_url=last_url,
            file_path=None,
            sha256=None,
            file_size=None,
            status=ACCESS_DOWNLOAD_FAILED,
            error_message=last_error,
        )
        return False

    def _fetch_pdf(self, url: str) -> tuple[bytes, str | None]:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 lit-agent/0.2"
                ),
                "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": _referer_for(url),
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                content_type = response.headers.get("Content-Type")
                content = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"PDF HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"PDF network error: {exc.reason}") from exc
        return content, content_type

    @staticmethod
    def _looks_like_pdf(content: bytes, content_type: str | None) -> bool:
        if content.startswith(b"%PDF-"):
            return True
        if content_type and "pdf" in content_type.lower() and not _looks_like_html(content):
            return True
        return False


def _looks_like_html(content: bytes) -> bool:
    head = content[:500].decode("utf-8", errors="ignore").lower()
    return bool(re.search(r"<html|<!doctype html", head))


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen = set()
    result = []
    for url in urls:
        normalized = url.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _referer_for(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}/"
    return "https://doi.org/"
