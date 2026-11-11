"""Where the shipped data lives, for a source checkout and for an installed package alike.

``fixtures/`` (with ``refs/`` and ``samples/``) is at the repository root in a checkout, and inside the ``bench``
package (``bench/fixtures``) when installed: pyproject.toml maps the root directory to ``bench.fixtures``. Run output goes to the repository's ``states/`` in a checkout and to a per-user data folder when
installed (``user_states_dir``); ``REPLAY_BENCH_STATES_DIR`` overrides both.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(ROOT if os.path.isfile(os.path.join(ROOT, "pyproject.toml")) else _PACKAGE_DIR, "fixtures")
REFS_DIR = os.path.join(FIXTURES_DIR, "refs")
SAMPLES_DIR = os.path.join(FIXTURES_DIR, "samples")


def is_checkout():
    """True when running from a source checkout (pyproject.toml beside the package), not from site-packages."""
    return os.path.isfile(os.path.join(ROOT, "pyproject.toml"))


def _is_windows():
    return os.name == "nt"


def user_states_dir():
    """Where runs are saved. Only computed here, never created: the folder appears with the first state written.

    REPLAY_BENCH_STATES_DIR when set; the repository's states/ in a checkout; else %LOCALAPPDATA%\\replay-bench\\states
    on Windows and $XDG_DATA_HOME/replay-bench/states (default ~/.local/share/replay-bench/states) elsewhere."""
    env = os.environ.get("REPLAY_BENCH_STATES_DIR")
    if env:
        return os.path.expanduser(env)
    if is_checkout():
        return os.path.join(ROOT, "states")
    if _is_windows():
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
        return os.path.join(base, "replay-bench", "states")
    xdg = os.environ.get("XDG_DATA_HOME")
    base = xdg if xdg and os.path.isabs(xdg) else os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "replay-bench", "states")


def default_states_dir():
    return user_states_dir()


def legacy_states_dir():
    """./states under the current directory, where 0.1.0 saved runs when installed; None when it is the states folder."""
    cand = os.path.join(os.getcwd(), "states")
    return None if os.path.abspath(cand) == os.path.abspath(user_states_dir()) else cand


def tilde(path):
    """A path with the home directory shown as ~."""
    path = str(path)
    home = os.path.expanduser("~")
    if home and home != os.sep and (path == home or path.startswith(home.rstrip(os.sep) + os.sep)):
        return "~" + path[len(home.rstrip(os.sep)):]
    return path
