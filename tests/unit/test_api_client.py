"""Unit tests for the streaming API client."""

import json
from unittest.mock import MagicMock, patch

from src.api_client import APIClient


def _fake_stream(lines: list[dict]) -> MagicMock:
    """Build a mocked httpx.Client whose stream() yields the given NDJSON lines."""
    response = MagicMock()
    response.iter_lines.return_value = [json.dumps(line) for line in lines]
    client = MagicMock()
    client.stream.return_value.__enter__.return_value = response
    client_cls = MagicMock()
    client_cls.return_value.__enter__.return_value = client
    return client_cls


def test_ask_stream_sets_refused_from_done_message():
    lines = [
        {"type": "token", "content": "Uncited claim."},
        {
            "type": "done",
            "citations": [],
            "used_chunk_ids": [],
            "refused": True,
            "refusal_message": "REFUSED",
        },
    ]
    with patch("src.api_client.httpx.Client", _fake_stream(lines)):
        result = APIClient(base_url="http://test").ask_stream(query="q", collection="c")
        tokens = list(result)

    assert tokens == ["Uncited claim."]
    assert result.refused is True
    assert result.refusal_message == "REFUSED"
    assert result.citations == []


def test_ask_stream_accepted_answer_is_not_refused():
    citation = {
        "label": "1",
        "chunk_id": "chunk1",
        "canonical_source_id": "doc.md",
        "source_path": "doc.md",
    }
    lines = [
        {"type": "token", "content": "Claim [1]."},
        {"type": "done", "citations": [citation], "used_chunk_ids": ["chunk1"], "refused": False},
    ]
    with patch("src.api_client.httpx.Client", _fake_stream(lines)):
        result = APIClient(base_url="http://test").ask_stream(query="q", collection="c")
        list(result)

    assert result.refused is False
    assert result.refusal_message is None
    assert [c.chunk_id for c in result.citations] == ["chunk1"]
