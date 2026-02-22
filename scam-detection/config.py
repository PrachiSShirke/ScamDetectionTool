"""
config.py — Central configuration for the Scam Detection System.
Reads from Streamlit secrets (cloud) or .env file (local).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Streamlit Cloud secrets support ──────────────────────────────────────────
def _get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "llm" / "prompts"

# ── LLM Settings ──────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = _get_secret("GEMINI_API_KEY")
GEMINI_MODEL: str = _get_secret("GEMINI_MODEL", "gemini-1.5-flash")

LLM_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
}

# ── Detection Thresholds ──────────────────────────────────────────────────────
SCAM_THRESHOLD: float = float(_get_secret("SCAM_THRESHOLD", "0.7"))

CONFIDENCE_BANDS = {
    "high":   (0.85, 1.00),
    "medium": (0.60, 0.85),
    "low":    (0.00, 0.60),
}

# ── Prompt Template Names ─────────────────────────────────────────────────────
PROMPT_TEMPLATES = {
    "system":   "system_prompt.txt",
    "analysis": "analysis_template.txt",
    "few_shot": "few_shot_examples.txt",
}

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_OUTPUT_DIR = BASE_DIR / "eval_results"
EVAL_BATCH_SIZE = 10

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
