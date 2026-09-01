"""Evaluation harness: score RAG pipeline responses against a golden Q&A set."""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.config import RAG_REFUSAL_MESSAGE
from src.models import RAGResponse

logger = logging.getLogger(__name__)


class EvalCase(BaseModel):
    """A single golden-set test case."""

    id: str
    collection: str
    query: str
    expect_refusal: bool = False
    expected_keywords: list[str] = Field(default_factory=list)
    min_citations: int = 0


class CaseResult(BaseModel):
    """Outcome of scoring one EvalCase against a RAGResponse."""

    case_id: str
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    answer: str


class EvalReport(BaseModel):
    """Aggregate results across a golden-set run."""

    total: int
    passed: int
    results: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        """Fraction of cases that passed (0.0 if there were no cases)."""
        return self.passed / self.total if self.total else 0.0


def load_golden_set(path: Path) -> list[EvalCase]:
    """
    Load and validate golden-set cases from a YAML file.

    Args:
        path: Path to a YAML file containing a list of case dicts.

    Returns:
        List of validated EvalCase objects.
    """
    raw = yaml.safe_load(path.read_text())
    return [EvalCase(**item) for item in raw]


def score_case(case: EvalCase, response: RAGResponse) -> CaseResult:
    """
    Score a single RAGResponse against its EvalCase's expectations.

    Args:
        case: The golden-set case being evaluated.
        response: The RAGResponse produced for case.query.

    Returns:
        CaseResult with pass/fail and, on failure, the reasons.
    """
    reasons: list[str] = []
    is_refusal = response.answer.strip() == RAG_REFUSAL_MESSAGE

    if case.expect_refusal:
        if not is_refusal:
            reasons.append("expected a refusal but got an answer")
        if response.citations:
            reasons.append(f"refusal should have no citations, got {len(response.citations)}")
    else:
        if is_refusal:
            reasons.append("expected an answer but got a refusal")
        if len(response.citations) < case.min_citations:
            reasons.append(
                f"expected >= {case.min_citations} citations, got {len(response.citations)}"
            )
        answer_lower = response.answer.lower()
        missing_keywords = [kw for kw in case.expected_keywords if kw.lower() not in answer_lower]
        if missing_keywords:
            reasons.append(f"missing expected keywords: {missing_keywords}")

    return CaseResult(
        case_id=case.id,
        passed=not reasons,
        reasons=reasons,
        answer=response.answer,
    )


def build_report(results: list[CaseResult]) -> EvalReport:
    """Aggregate individual case results into a report."""
    return EvalReport(
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        results=results,
    )
