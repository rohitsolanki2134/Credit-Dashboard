"""
Auto-load .env on import.  Import this at the top of app.py before other modules.
Usage: import load_env  # noqa
"""
from pathlib import Path
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent / ".env"
    if _env.exists():
        load_dotenv(_env, override=True)
except ImportError:
    pass  # dotenv optional; env vars may be set externally
