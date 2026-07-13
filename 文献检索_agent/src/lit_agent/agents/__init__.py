from __future__ import annotations

from .dashboard import HtmlDashboardAgent
from .dedup import DeduplicationAgent
from .domain_query import DomainQueryAgent
from .discovery import SourceDiscoveryAgent
from .download import PdfDownloadAgent
from .llm_relevance import LLMRelevanceReviewAgent
from .normalize import MetadataNormalizeAgent
from .oa_resolver import OAResolverAgent
from .orchestrator import OrchestratorAgent
from .quality import QualityAuditAgent
from .relevance import RelevanceJudgeAgent
from .report import ReportExportAgent
from .rounds import (
    EvidenceSynthesisAgent,
    ManualPdfIntakeAgent,
    NextQueryProposalAgent,
    PdfAnalysisAgent,
    RoundAcquisitionAgent,
    RoundApprovalAgent,
    RoundPlanningAgent,
    ScientificGoalAgent,
)

__all__ = [
    "DeduplicationAgent",
    "DomainQueryAgent",
    "EvidenceSynthesisAgent",
    "HtmlDashboardAgent",
    "LLMRelevanceReviewAgent",
    "ManualPdfIntakeAgent",
    "MetadataNormalizeAgent",
    "NextQueryProposalAgent",
    "OAResolverAgent",
    "OrchestratorAgent",
    "PdfAnalysisAgent",
    "PdfDownloadAgent",
    "QualityAuditAgent",
    "RelevanceJudgeAgent",
    "ReportExportAgent",
    "RoundAcquisitionAgent",
    "RoundApprovalAgent",
    "RoundPlanningAgent",
    "ScientificGoalAgent",
    "SourceDiscoveryAgent",
]
