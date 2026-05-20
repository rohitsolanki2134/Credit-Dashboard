"""
Auto-load .env on import.  Import this at the top of app.py before other modules.
Usage: import load_env  # noqa

On local dev    : loads credentials from .env file via python-dotenv.
On Streamlit Cloud : .env doesn't exist; secrets are injected via st.secrets.
                     We bridge them into os.environ in app.py AFTER set_page_config()
                     so config.py (which uses os.getenv) picks them up correctly.
"""
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent / ".env"
    if _env.exists():
        load_dotenv(_env, override=True)
except ImportError:
    pass  # dotenv optional; env vars may be set externally
