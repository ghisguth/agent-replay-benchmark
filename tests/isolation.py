"""Makes the unit tests independent of the machine they run on.

It applies to every test of a `unittest discover` run: tests/test_isolation.py imports it, and discovery imports every
test module before the first test runs. A test module run on its own imports it itself (`import isolation  # noqa: F401`).

bench.report.configure() turns Nerd Font icons and colors on when the real stdout/stderr is a terminal. Tests that
expect plain words ("ctx", "TTFT", ...) failed whenever `make test` ran in a terminal. Here the process's own
stdout/stderr never count as terminals; the TTY stubs the tests pass in explicitly still do, so tests that want icons
build them with a stub (or glyphs.configure(True)). REPLAY_BENCH_NERD_FONT and NO_COLOR are dropped from the environment and COLUMNS is pinned to 120.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench import report  # noqa: E402

for _var in ("REPLAY_BENCH_NERD_FONT", "NO_COLOR"):
    os.environ.pop(_var, None)
os.environ["COLUMNS"] = "120"  # a terminal of unknown size (a pty with 0 columns) must not narrow the fake-TTY live line

_real_color_ok = getattr(report.color_ok, "_real", report.color_ok)  # importing twice must not stack the patch


def _color_ok(stream):
    if stream is sys.__stdout__ or stream is sys.__stderr__:
        return False
    try:
        if stream.isatty() and getattr(stream, "fileno", None) and stream.fileno() in (1, 2):
            return False
    except (AttributeError, ValueError, OSError):
        pass
    return _real_color_ok(stream)


_color_ok._real = _real_color_ok
report.color_ok = _color_ok

# The tests' stub servers are stopped with shutdown(), which waits for serve_forever's next poll (0.5 s by default):
# one wait per test dominated the suite's wall time. Poll every 10 ms instead.
import socketserver  # noqa: E402

if not getattr(socketserver.BaseServer.serve_forever, "_fast", False):
    _serve_forever = socketserver.BaseServer.serve_forever

    def _fast_serve_forever(self, poll_interval=0.01):
        return _serve_forever(self, poll_interval)

    _fast_serve_forever._fast = True
    socketserver.BaseServer.serve_forever = _fast_serve_forever
