"""
llm/client.py — Gemini API client using google-genai SDK.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, LLM_CONFIG
from utils import get_logger, retry

logger = get_logger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Add it to Streamlit Cloud secrets or your .env file."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialised with model '%s'.", GEMINI_MODEL)
    return _client


@retry(max_attempts=3, delay=2.0, exceptions=(Exception,))
def generate(prompt: str) -> str:
    client = _get_client()
    logger.debug("Sending prompt to Gemini (length=%d chars).", len(prompt))

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=LLM_CONFIG["temperature"],
            top_p=LLM_CONFIG["top_p"],
            top_k=LLM_CONFIG["top_k"],
            max_output_tokens=LLM_CONFIG["max_output_tokens"],
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    logger.debug("Received response (length=%d chars).", len(response.text))
    return response.text
