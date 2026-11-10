"""Parsing and error helpers shared across the benchmark."""

import json
import os
import re
import subprocess


class UserError(ValueError):
    """A problem the user can fix (a missing file, a bad combination of options): reported as one line on stderr,
    exit code 2; a traceback only with REPLAY_BENCH_DEBUG=1."""


# A validator returns a bool, a (bool, message) tuple, or (WARN, message) for an answer that is
# acceptable but not the ideal one (counted as a pass, listed in the anomaly report).
WARN = "WARN"


def verdict(result):
    """Normalize a validator result to (status, message), status being PASS, WARN or FAIL."""
    ok, msg = result if isinstance(result, tuple) else (result, "")
    if ok == WARN:
        return WARN, msg
    return ("PASS", "") if ok else ("FAIL", msg or "Logic Failed")


def safe_json(txt):
    try:
        return json.loads(txt)
    except Exception:
        return None


def strip_code_fence(text):
    trimmed = text.strip()
    first_brace = trimmed.find('{')
    first_bracket = trimmed.find('[')

    if first_brace < 0 and first_bracket < 0:
        return text

    if first_brace >= 0 and (first_bracket < 0 or first_brace < first_bracket):
        last_brace = trimmed.rfind('}')
        if last_brace > first_brace:
            return trimmed[first_brace:last_brace + 1]
    elif first_bracket >= 0:
        last_bracket = trimmed.rfind(']')
        if last_bracket > first_bracket:
            return trimmed[first_bracket:last_bracket + 1]

    return text



def version_string():
    """`0.1.0` (the installed package's version, else pyproject.toml's), plus the short git commit in a checkout."""
    from . import paths
    version = None
    try:
        from importlib import metadata
        version = metadata.version("agent-replay-benchmark")
    except Exception:
        pass
    checkout = paths.is_checkout()
    if version is None or checkout:
        try:
            with open(os.path.join(paths.ROOT, "pyproject.toml"), encoding="utf-8") as f:
                m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
            version = (m.group(1) if m else None) or version
        except OSError:
            pass
    version = version or "unknown"
    if checkout:
        try:
            out = subprocess.run(["git", "-C", paths.ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True, timeout=2)
            if out.returncode == 0 and out.stdout.strip():
                version += " (%s)" % out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return version
