"""Unit tests for the eval harness (src/eval.py)."""

from pathlib import Path

import pytest

from src.config import RAG_REFUSAL_MESSAGE
from src.eval import EvalCase, build_report, load_golden_set, score_case
from src.models import Citation, RAGResponse


def make_response(
    answer: str = "DataFrames can be created from a dict.",
    num_citations: int = 1,
) -> RAGResponse:
    """Build a RAGResponse with a given answer and citation count."""
    citations = [
        Citation(
            label=str(i + 1),
            chunk_id=f"chunk_{i}",
            canonical_source_id="doc.md",
            source_path="/docs/doc.md",
        )
        for i in range(num_citations)
    ]
    return RAGResponse(
        answer=answer,
        citations=citations,
        used_chunk_ids=[c.chunk_id for c in citations],
    )


class TestScoreCase:
    """Tests for score_case."""

    def test_answer_case_passes(self):
        case = EvalCase(
            id="c1",
            collection="docs",
            query="q",
            expected_keywords=["DataFrame"],
            min_citations=1,
        )
        result = score_case(case, make_response())

        assert result.passed
        assert result.reasons == []

    def test_answer_case_fails_on_refusal(self):
        case = EvalCase(id="c1", collection="docs", query="q", min_citations=1)
        response = make_response(answer=RAG_REFUSAL_MESSAGE, num_citations=0)

        result = score_case(case, response)

        assert not result.passed
        assert any("expected an answer" in r for r in result.reasons)

    def test_answer_case_fails_on_missing_citations(self):
        case = EvalCase(id="c1", collection="docs", query="q", min_citations=2)
        response = make_response(num_citations=1)

        result = score_case(case, response)

        assert not result.passed
        assert any("citations" in r for r in result.reasons)

    def test_answer_case_fails_on_missing_keyword(self):
        case = EvalCase(id="c1", collection="docs", query="q", expected_keywords=["groupby"])
        response = make_response(answer="Use loc for label-based indexing.")

        result = score_case(case, response)

        assert not result.passed
        assert any("groupby" in r for r in result.reasons)

    def test_keyword_match_is_case_insensitive(self):
        case = EvalCase(id="c1", collection="docs", query="q", expected_keywords=["dataframe"])
        response = make_response(answer="Use a DataFrame for this.")

        result = score_case(case, response)

        assert result.passed

    def test_refusal_case_passes(self):
        case = EvalCase(id="c1", collection="docs", query="q", expect_refusal=True)
        response = make_response(answer=RAG_REFUSAL_MESSAGE, num_citations=0)

        result = score_case(case, response)

        assert result.passed

    def test_refusal_case_fails_when_answer_given(self):
        case = EvalCase(id="c1", collection="docs", query="q", expect_refusal=True)

        result = score_case(case, make_response())

        assert not result.passed
        assert any("expected a refusal" in r for r in result.reasons)

    def test_refusal_case_fails_with_citations(self):
        case = EvalCase(id="c1", collection="docs", query="q", expect_refusal=True)
        response = make_response(answer=RAG_REFUSAL_MESSAGE, num_citations=1)

        result = score_case(case, response)

        assert not result.passed
        assert any("citations" in r for r in result.reasons)


class TestBuildReport:
    """Tests for build_report."""

    def test_aggregates_pass_counts(self):
        case = EvalCase(id="c1", collection="docs", query="q")
        results = [
            score_case(case, make_response()),
            score_case(case, make_response(answer=RAG_REFUSAL_MESSAGE, num_citations=0)),
        ]

        report = build_report(results)

        assert report.total == 2
        assert report.passed == 1
        assert report.pass_rate == 0.5

    def test_pass_rate_with_no_cases_is_zero(self):
        report = build_report([])

        assert report.pass_rate == 0.0


class TestLoadGoldenSet:
    """Tests for load_golden_set."""

    def test_loads_and_validates_cases(self, tmp_path: Path):
        golden_set_file = tmp_path / "golden_set.yaml"
        golden_set_file.write_text(
            """
- id: case_one
  collection: pandas_docs
  query: "How do I create a DataFrame?"
  expected_keywords: ["DataFrame"]
  min_citations: 1
- id: case_two
  collection: pandas_docs
  query: "How do I make pasta?"
  expect_refusal: true
"""
        )

        cases = load_golden_set(golden_set_file)

        assert len(cases) == 2
        assert cases[0].id == "case_one"
        assert cases[0].expected_keywords == ["DataFrame"]
        assert cases[1].expect_refusal is True

    def test_missing_required_field_raises(self, tmp_path: Path):
        golden_set_file = tmp_path / "golden_set.yaml"
        golden_set_file.write_text('- collection: pandas_docs\n  query: "q"\n')

        with pytest.raises(Exception):  # noqa: B017 pydantic ValidationError
            load_golden_set(golden_set_file)
