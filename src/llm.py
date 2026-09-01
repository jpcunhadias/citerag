"""LLM client for Ollama API."""

import json
import logging
from collections.abc import Iterator

import requests
from requests.exceptions import RequestException

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL_NAME

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Exception raised when connection to Ollama fails."""

    pass


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: str | None = None):
        """
        Initialize OllamaClient.

        Args:
            base_url: Base URL for Ollama API. If None, uses OLLAMA_BASE_URL from config.
        """
        self.base_url = base_url or OLLAMA_BASE_URL
        self.model_name = OLLAMA_MODEL_NAME
        logger.info(
            f"Initialized OllamaClient with base_url={self.base_url}, model={self.model_name}"
        )

    def generate(self, prompt: str) -> str:
        """
        Generate text response from Ollama.

        Args:
            prompt: Prompt text to send to the model.

        Returns:
            Generated response text (only the "response" field from JSON).

        Raises:
            OllamaConnectionError: If connection to Ollama fails.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        try:
            logger.info(f"Sending request to Ollama: {url}")
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            # Extract and return only the "response" text from the JSON
            if "response" not in result:
                error_msg = (
                    f"Invalid response from Ollama at {self.base_url}: "
                    f"missing 'response' field in JSON. Received keys: {list(result.keys())}"
                )
                logger.error(error_msg)
                raise OllamaConnectionError(error_msg)
            return result["response"]
        except RequestException as e:
            error_msg = (
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Original error: {str(e)}"
            )
            logger.error(error_msg)
            raise OllamaConnectionError(error_msg) from e

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        Stream text response from Ollama (NDJSON format).

        Args:
            prompt: Prompt text to send to the model.

        Yields:
            Token chunks from the "response" field of each NDJSON line.

        Raises:
            OllamaConnectionError: If connection to Ollama fails.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
        }

        try:
            logger.info(f"Sending streaming request to Ollama: {url}")
            with requests.post(url, json=payload, stream=True, timeout=300) as response:
                response.raise_for_status()

                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed NDJSON line: {e}")
                        continue

                    if "response" in obj and obj["response"]:
                        yield obj["response"]

                    if obj.get("done"):
                        break

        except RequestException as e:
            error_msg = (
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Original error: {str(e)}"
            )
            logger.error(error_msg)
            raise OllamaConnectionError(error_msg) from e
