"""Unit tests for the LLM client."""

from unittest.mock import MagicMock, patch

import pytest
import requests
from requests.exceptions import Timeout

from src.llm import OllamaClient, OllamaConnectionError


class TestOllamaConnectionError:
    """Tests for OllamaConnectionError exception."""

    def test_ollama_connection_error_is_exception(self):
        """Test that OllamaConnectionError is an Exception."""
        assert issubclass(OllamaConnectionError, Exception)


class TestOllamaClient:
    """Tests for OllamaClient class."""

    def test_ollama_client_initialization_default(self):
        """Test that OllamaClient uses default config values."""
        with patch("src.llm.OLLAMA_BASE_URL", "http://localhost:11434"):
            with patch("src.llm.OLLAMA_MODEL_NAME", "llama3"):
                client = OllamaClient()
                assert client.base_url == "http://localhost:11434"
                assert client.model_name == "llama3"

    def test_ollama_client_initialization_custom_url(self):
        """Test that OllamaClient accepts custom base_url."""
        client = OllamaClient(base_url="http://custom:8080")
        assert client.base_url == "http://custom:8080"

    @patch("src.llm.requests.post")
    @patch("src.llm.OLLAMA_BASE_URL", "http://localhost:11434")
    @patch("src.llm.OLLAMA_MODEL_NAME", "llama3")
    def test_generate_returns_response_text_only(self, mock_post):
        """Test that generate returns only the response text."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "This is the answer",
            "model": "llama3",
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        result = client.generate("test prompt")

        assert result == "This is the answer"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "llama3"
        assert call_kwargs["json"]["prompt"] == "test prompt"
        assert call_kwargs["json"]["stream"] is False

    @patch("src.llm.requests.post")
    @patch("src.llm.OLLAMA_BASE_URL", "http://localhost:11434")
    @patch("src.llm.OLLAMA_MODEL_NAME", "llama3")
    def test_generate_raises_on_connection_error(self, mock_post):
        """Test that generate raises OllamaConnectionError on connection failure."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = OllamaClient()

        with pytest.raises(OllamaConnectionError) as exc_info:
            client.generate("test prompt")

        assert "Failed to connect to Ollama" in str(exc_info.value)
        assert "Is Ollama running?" in str(exc_info.value)
        assert "http://localhost:11434" in str(exc_info.value)

    @patch("src.llm.requests.post")
    @patch("src.llm.OLLAMA_BASE_URL", "http://localhost:11434")
    @patch("src.llm.OLLAMA_MODEL_NAME", "llama3")
    def test_generate_raises_on_timeout(self, mock_post):
        """Test that generate raises OllamaConnectionError on timeout."""
        mock_post.side_effect = Timeout("Request timed out")

        client = OllamaClient()

        with pytest.raises(OllamaConnectionError) as exc_info:
            client.generate("test prompt")

        assert "Failed to connect to Ollama" in str(exc_info.value)
        assert "Is Ollama running?" in str(exc_info.value)

    @patch("src.llm.requests.post")
    @patch("src.llm.OLLAMA_BASE_URL", "http://localhost:11434")
    @patch("src.llm.OLLAMA_MODEL_NAME", "llama3")
    def test_generate_raises_on_http_error(self, mock_post):
        """Test that generate raises OllamaConnectionError on HTTP error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_post.return_value = mock_response

        client = OllamaClient()

        with pytest.raises(OllamaConnectionError) as exc_info:
            client.generate("test prompt")

        assert "Failed to connect to Ollama" in str(exc_info.value)

    @patch("src.llm.requests.post")
    @patch("src.llm.OLLAMA_BASE_URL", "http://localhost:11434")
    @patch("src.llm.OLLAMA_MODEL_NAME", "llama3")
    def test_generate_raises_on_missing_response_field(self, mock_post):
        """Test that generate raises OllamaConnectionError when response field is missing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "llama3",
            "done": True,
            # Missing "response" field
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = OllamaClient()

        with pytest.raises(OllamaConnectionError) as exc_info:
            client.generate("test prompt")

        assert "missing 'response' field" in str(exc_info.value)
        assert "http://localhost:11434" in str(exc_info.value)

    @patch("src.llm.requests.post")
    @patch("src.llm.OLLAMA_BASE_URL", "http://localhost:11434")
    @patch("src.llm.OLLAMA_MODEL_NAME", "llama3")
    def test_generate_uses_custom_timeout(self, mock_post):
        """Test that generate uses timeout parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "answer"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        client.generate("test prompt")

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["timeout"] == 300

    @patch("src.llm.requests.post")
    @patch("src.llm.OLLAMA_BASE_URL", "http://localhost:11434")
    @patch("src.llm.OLLAMA_MODEL_NAME", "llama3")
    def test_generate_preserves_original_exception(self, mock_post):
        """Test that OllamaConnectionError preserves original exception."""
        original_error = requests.exceptions.ConnectionError("Connection refused")
        mock_post.side_effect = original_error

        client = OllamaClient()

        with pytest.raises(OllamaConnectionError) as exc_info:
            client.generate("test prompt")

        # Check that original exception is preserved
        assert exc_info.value.__cause__ == original_error
