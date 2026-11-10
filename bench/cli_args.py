"""Small argparse helpers for replay_bench.py (kept separate so they are testable offline)."""

import argparse
import re


def parse_max_context(value):
    """'<N>k' -> N * 1024 tokens (1 <= N <= 1024); plain int >= 1024; 'full' -> None (no limit)."""
    v = str(value).strip().lower()
    if v == "full":
        return None
    if v.endswith("k") and v[:-1].isdigit():
        n = int(v[:-1])
        if 1 <= n <= 1024:
            return n * 1024
    elif v.isdigit():
        n = int(v)
        if n >= 1024:
            return n
    raise argparse.ArgumentTypeError(
        "invalid --max-context %r (expected <N>k with 1..1024, count >= 1024, or full)" % (value,))


def parse_turns_range(value):
    """'A-B' -> (A, B), inclusive, 1-based turn numbers with 1 <= A <= B."""
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(value))
    if not m:
        raise argparse.ArgumentTypeError("invalid --turns %r (expected A-B, e.g. 10-40)" % (value,))
    a, b = int(m.group(1)), int(m.group(2))
    if a < 1 or a > b:
        raise argparse.ArgumentTypeError("invalid --turns %r (need 1 <= A <= B)" % (value,))
    return (a, b)


def parse_run_note(value):
    """--run-note TEXT: whitespace collapsed, 1..60 characters (it goes into the file name and table labels)."""
    note = " ".join(str(value).split())
    if not note or len(note) > 60:
        raise argparse.ArgumentTypeError("invalid --run-note %r (expected 1..60 characters)" % (value,))
    return note


def parse_server_version(value):
    """--server-version VERSION: one token of 1..40 characters without whitespace (it goes into the file name)."""
    v = str(value).strip()
    if not v or len(v) > 40 or any(c.isspace() for c in v):
        raise argparse.ArgumentTypeError("invalid --server-version %r (expected 1..40 characters, no spaces)" % (value,))
    return v
