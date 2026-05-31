"""Load runtime config/secrets into os.environ.

Two sources, so env-based modules (`llm_client`, future `market_source`) work
identically locally and on Streamlit Cloud:
  1. a local `.env` file (KEY=VALUE; gitignored) — for local dev
  2. `st.secrets` — for Streamlit Community Cloud deployment

Precedence: anything already set in the real environment wins, then `.env`,
then Streamlit secrets. Never commit real values; copy `.env.example` → `.env`."""
from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

# Keys bridged from Streamlit secrets into os.environ.
_KEYS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "TUSHARE_TOKEN")


def _load_dotenv(path: Path = _ENV_FILE) -> None:
    """Parse a simple KEY=VALUE `.env` into os.environ.

    Does not override variables already set in the environment. Skips blank
    lines, `#` comments, and empty values (so unfilled placeholders in
    `.env.example` never clobber a module's built-in default)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if " #" in val:                       # drop trailing inline comment
            val = val.split(" #", 1)[0].strip()
        val = val.strip('"').strip("'")
        if not key or not val:                # skip empty placeholders
            continue
        os.environ.setdefault(key, val)


def _load_streamlit_secrets() -> None:
    """Bridge st.secrets into os.environ on Streamlit Cloud. No-op elsewhere."""
    try:
        import streamlit as st
        secrets = st.secrets
    except Exception:
        return
    for k in _KEYS:
        try:
            if k in secrets and not os.environ.get(k):
                os.environ[k] = str(secrets[k])
        except Exception:
            continue


def load_env() -> None:
    """Idempotent. Call once at app startup before any module reads these vars."""
    _load_dotenv()
    _load_streamlit_secrets()
