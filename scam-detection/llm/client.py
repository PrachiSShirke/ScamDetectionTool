"""
llm/client.py — Thin wrapper around the Google Generative AI SDK.
"""

from __future__ import annotations

import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL, LLM_CONFIG
from utils import get_logger, retry

logger = get_logger(__name__)

_model: genai.GenerativeModel | None = None


def _get_model() -> genai.GenerativeModel:
    global _model
    if _model is None:
        if not GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Add it to Streamlit Cloud secrets or your .env file."
            )
        genai.configure(api_key=GEMINI_API_KEY)
        _model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=genai.types.GenerationConfig(**LLM_CONFIG),
        )
        logger.info("Gemini model '%s' initialised.", GEMINI_MODEL)
    return _model


@retry(max_attempts=3, delay=2.0, exceptions=(Exception,))
def generate(prompt: str) -> str:
    model = _get_model()
    logger.debug("Sending prompt to Gemini (length=%d chars).", len(prompt))
    response = model.generate_content(prompt)
    if not response.text:
        finish_reason = getattr(response.candidates[0], "finish_reason", "UNKNOWN")
        raise RuntimeError(f"Gemini returned empty response. finish_reason={finish_reason}")
    logger.debug("Received response (length=%d chars).", len(response.text))
    return response.text
