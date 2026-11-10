"""Nerd Font icons for the chat formats, with a plain fallback.

Icons are opt-in (a terminal cannot tell whether its font has them): --icons or REPLAY_BENCH_NERD_FONT=1, and only when
the report is going to a terminal with colors (stdout is a TTY, NO_COLOR unset). --no-nerd-font or
REPLAY_BENCH_NERD_FONT=0 is the explicit off switch. Everywhere else `g(name)` is an empty string (or the given
plain stand-in), so piped output and tests carry no private-use characters.
"""

import contextlib
import os

ICONS = {
    # stats row
    "clock": "\uf017", "delta": "\uf252", "in": "\uf062", "out": "\uf063", "cache": "\uf1c0", "ttft": "\uf2f2", "prefill": "\uf085",
    "speed": "\uf0e4", "ctx": "\uf2db", "eta": "\uf253",
    # verdicts
    "PASS": "\uf00c", "WARN": "\uf071", "FAIL": "\uf00d", "SKIP": "\uf051",
    # tools
    "bash": "\uf120", "read": "\uf15c", "edit": "\uf044", "write": "\uf0c7", "grep": "\uf002", "glob": "\uf07b",
    "find": "\uf07b", "task": "\uf135", "todo": "\uf0ae", "wait": "\uf251", "tool": "\uf0ad",
    # streams
    "main": "\uf086", "subagent": "\uf0e8", "compaction": "\uf066",
}

enabled = False
live_enabled = False  # the live progress line (stderr) in every report format, independent of the stdout report


def configure(on):
    global enabled
    enabled = bool(on)


def configure_live(on):
    global live_enabled
    live_enabled = bool(on)


@contextlib.contextmanager
def live():
    """Icons on for what is built inside when the live line may carry them (or the report format already does)."""
    global enabled
    saved = enabled
    enabled = saved or live_enabled
    try:
        yield
    finally:
        enabled = saved


def env_setting():
    """REPLAY_BENCH_NERD_FONT: True for 1 (any value but 0 or empty), False for 0, None when unset."""
    v = os.environ.get("REPLAY_BENCH_NERD_FONT")
    if v is None or not v.strip():
        return None
    return v.strip() != "0"


def g(name, plain=""):
    """The icon `name` when icons are on, else `plain`."""
    return ICONS.get(name, plain) if enabled else plain


def prefix(name):
    """The icon and a space, or nothing."""
    return g(name) + " " if enabled and name in ICONS else ""
