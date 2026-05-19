"""
Auto-load .env on import.  Import this at the top of app.py before other modules.
Usage: import load_env  # noqa

On local dev   : loads from .env file via python-dotenv.
On Streamlit Cloud : .env doesn't exist; secrets are in st.secrets — we copy
                     them into os.environ so config.py (os.getenv) picks them up.
"""
import os
from pathlib import Path

# ── Local dev: load .env file ─────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent / ".env"
    if _env.exists():
        load_dotenv(_env, override=True)
except ImportError:
    pass  # dotenv optional

# ── Streamlit Cloud: copy st.secrets into os.environ ─────────────────────────
# st.secrets holds the values entered in the Streamlit dashboard.
# config.py uses os.getenv(), so we bridge the two here.
try:
    import streamlit as st
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:
    pass  # Not in Streamlit context, or no secrets configured — safe to ignore
