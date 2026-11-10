"""Authorization header and API key state."""

import os

_API_KEY = None


def set_api_key(key):
    """Use `key` for every request's Authorization header (None: OPENAI_API_KEY from the environment, else a dummy)."""
    global _API_KEY
    _API_KEY = key or None


def api_key():
    """The bearer token to send: set_api_key(), else OPENAI_API_KEY, else 'not-needed' (local servers ignore it)."""
    return _API_KEY or os.environ.get("OPENAI_API_KEY") or "not-needed"


def auth_header():
    """Return {} or {'Authorization': f'Bearer {k}'}."""
    k = api_key()
    if not k:
        return {}
    return {"Authorization": f"Bearer {k}"}
