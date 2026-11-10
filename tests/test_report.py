"""--report-format tui|markdown: format selection, colors, stderr progress, valid markdown, --dry-run, and a full mock
end-to-end run in each format. No real server; the state directory is a temp dir.

    python3 -m unittest discover -s tests
"""

import contextlib
import io
import os
import re
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401
from test_replay import synthetic_fixture, write_fixture  # noqa: E402
from test_state import SplashLike, behaviour  # noqa: E402

import replay_bench  # noqa: E402
from bench import colors, compare, progress, report, state  # noqa: E402

ANSI = re.compile(r"\x1b\[")
SEP = re.compile(r"^\|[-:| ]+\|$")


class TTY(io.StringIO):
    def isatty(self):
        return True


def assert_valid_markdown(test, text):
    """Every table: a header row, a separator row, rows with the same number of cells; no ANSI codes."""
    test.assertFalse(ANSI.search(text), "ANSI code in markdown")
    lines, tables = text.splitlines(), 0
    for i, line in enumerate(lines):
        if SEP.match(line):
            tables += 1
            test.assertTrue(i > 0 and lines[i - 1].startswith("|"), "separator without header")
            n = len(re.findall(r"(?<!\\)\|", lines[i - 1]))
            j = i + 1
            while j < len(lines) and lines[j].startswith("|"):
                test.assertEqual(len(re.findall(r"(?<!\\)\|", lines[j])), n, lines[j])
                j += 1
    return tables


def write_state(path, model="m", speed=50.0, n=6):
    t = {"key": "h:1::" + model, "backend": "h:1", "name": model, "url": "http://h:1/v1/chat/completions",
         "n_ctx": 262144, "backend_info": {"kind": "splash", "version": "1a2b3c4d", "build": "x", "details": {}}}
    w = state.StateWriter(path)
    w.write(state.make_header("coding", t, request={"temperature": None, "reasoning_effort": None}, fixture_path=None,
                              argv=[], selection=None, started_at="2026-11-07T10:00:00+00:00"))
    for i in range(n):
        p = 8000 + i * 30000
        tok = 60 + 10 * i
        w.write(state.result_record({
            "id": "R%03d_main_e0_t%03d" % (i, i), "stream": "main", "kind": "turn", "epoch": 0, "content": "ok", "text": "ok", "ttft_ms": 500 + i * 100,
            "total_ms": 500 + i * 100 + tok / speed * 1000, "tokens": tok, "prompt_tokens": p, "cached_tokens": p - 400,
            "native_ttft_ms": None, "native_prefill_tps": None, "native_decode_tps": speed,
            "eval_result": "FAIL" if i == 3 else "PASS", "fail_reason": "bad | pipe" if i == 3 else "",
            "metrics": {"speculation": {"drafted_tokens": tok, "accepted_draft_tokens": tok // 2, "cycles": tok // 3}}}))
    w.write(state.end_record({}))
    w.close()


def run_main(*argv, stdout=None, stderr=None):
    out, err = stdout or io.StringIO(), stderr or io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            replay_bench.main(list(argv))
        except SystemExit as e:
            if e.code not in (0, None):
                raise
    return (out.getvalue() if hasattr(out, "getvalue") else ""), err.getvalue()


class Setup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.a = os.path.join(self.tmp.name, "a.jsonl")
        self.b = os.path.join(self.tmp.name, "b.jsonl")
        write_state(self.a, "alpha", 40.0)
        write_state(self.b, "beta", 60.0)
        self.both = os.path.join(self.tmp.name, "both.jsonl")  # two model blocks in one file
        with open(self.both, "w") as f:
            for p in (self.a, self.b):
                with open(p) as g:
                    f.write(g.read())
        self.addCleanup(lambda: report.configure("tui", io.StringIO()))


class Formats(Setup):
    def test_default_is_tui_boxes_no_ansi_when_not_a_tty(self):
        out, err = run_main("--state", self.a)
        self.assertIn("┌", out)
        self.assertIn("│ Metric", out)
        self.assertNotIn("| Metric", out)
        self.assertFalse(ANSI.search(out))
        self.assertIn("By context", out)
        self.assertNotIn("###", out)  # terminal headings, not markdown syntax
        self.assertIn("Prefill by append size", out)

    def test_tui_colors_on_a_tty_and_no_color_env(self):
        out, _ = run_main("--state", self.both, stdout=TTY())
        self.assertTrue(ANSI.search(out))
        self.assertIn("\x1b[92m+", out)  # a faster Δ is green
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            out, _ = run_main("--state", self.both, stdout=TTY())
        self.assertFalse(ANSI.search(out))

    def test_markdown_is_valid_and_has_everything(self):
        out, err = run_main("--state", self.a, "--report-format", "markdown", "--no-mdcat", stdout=TTY())
        tables = assert_valid_markdown(self, out)
        self.assertGreaterEqual(tables, 5)  # run, cases, metrics, bands, 2 perf tables
        for needle in ("# Agent replay \u2014 suite coding", "## Run", "## alpha", "| Case", "### Results", "### Metrics",
                       "### Anomalies", "- **R003_main_e0_t003** **FAIL**: bad | pipe", "## Performance", "### By context",
                       "### Prefill by append size"):
            self.assertIn(needle, out)
        self.assertNotIn("┌", out)

    def test_markdown_multi_state_has_compare_and_deltas(self):
        out, _ = run_main("--state", self.both, "--report-format", "markdown", "--no-mdcat")
        assert_valid_markdown(self, out)
        self.assertIn("## Comparison", out)
        self.assertIn("Baseline: alpha", out)
        self.assertRegex(out, r"\| 0-32k\s*\| \*\*beta\*\*\s*\|.*\+\d")

    def test_markdown_pipes_each_section_through_mdcat_on_a_tty_only(self):
        calls = []
        fake = lambda cmd, input, text: calls.append((cmd, input)) or mock.Mock(returncode=0)
        with mock.patch.object(compare.shutil, "which", return_value="/x/mdcat"), \
                mock.patch.object(compare.subprocess, "run", side_effect=fake):
            run_main("--state", self.a, "--report-format", "markdown", stdout=TTY())
            n_tty = len(calls)
            self.assertTrue(all(c == ["mdcat", "--full-width"] for c, _ in calls))
            self.assertTrue(any("| Case" in i for _, i in calls))
            calls.clear()
            out, _ = run_main("--state", self.a, "--report-format", "markdown")  # StringIO: not a tty
            self.assertEqual(calls, [])
            run_main("--state", self.a, "--report-format", "markdown", "--no-mdcat", stdout=TTY())
            self.assertEqual(calls, [])
            run_main("--state", self.a, stdout=TTY())  # tui never uses mdcat
            self.assertEqual(calls, [])
        self.assertGreaterEqual(n_tty, 4)
        assert_valid_markdown(self, out)

    def test_bad_format_rejected(self):
        with self.assertRaises(SystemExit):
            run_main("--state", self.a, "--report-format", "html", stderr=io.StringIO())


class DryRun(Setup):
    def setUp(self):
        super().setUp()
        self.fixture = write_fixture(synthetic_fixture(), self.tmp.name)

    def test_tui(self):
        out, _ = run_main("--suite", "coding", "--dry-run", "--fixture", self.fixture, "--max-context", "64k")
        self.assertIn("┌", out)
        self.assertRegex(out, r"│ id\s+│ stream")
        self.assertIn("R002_S1_t000", out)
        self.assertIn("coding selection:", out)
        self.assertFalse(ANSI.search(out))

    def test_tui_colors_on_a_tty(self):
        out, _ = run_main("--suite", "coding", "--dry-run", "--fixture", self.fixture, stdout=TTY())
        self.assertTrue(ANSI.search(out))

    def test_markdown(self):
        out, _ = run_main("--suite", "coding", "--dry-run", "--fixture", self.fixture, "--report-format", "markdown",
                          "--no-mdcat", stdout=TTY())
        self.assertEqual(assert_valid_markdown(self, out), 1)
        self.assertRegex(out, r"\| id\s+\| stream\s+\| kind")
        self.assertIn("R002_S1_t000", out)
        self.assertIn("coding selection:", out)

    def test_markdown_through_mdcat(self):
        calls = []
        fake = lambda cmd, input, text: calls.append(input) or mock.Mock(returncode=0)
        with mock.patch.object(compare.shutil, "which", return_value="/x/mdcat"), \
                mock.patch.object(compare.subprocess, "run", side_effect=fake):
            run_main("--suite", "coding", "--dry-run", "--fixture", self.fixture, "--report-format", "markdown", stdout=TTY())
        self.assertEqual(len(calls), 1)
        self.assertIn("| id", calls[0])


class Progress(unittest.TestCase):
    def run_spinner(self, stdout, stderr):
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            p = progress.ProgressAnimation()
            p.start("working", state={"start": time.time(), "last": time.time(), "lines": 0})
            time.sleep(0.2)
            p.stop()

    def test_nothing_when_stderr_is_not_a_tty(self):
        out, err = TTY(), io.StringIO()
        self.run_spinner(out, err)
        self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))

    def test_spinner_goes_to_stderr_not_stdout(self):
        out, err = io.StringIO(), TTY()
        self.run_spinner(out, err)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("working", err.getvalue())


class LiveLineColors(unittest.TestCase):
    """The live line (stderr) takes its colors from stderr alone, not from the report format or stdout."""

    def setUp(self):
        self.addCleanup(lambda: report.configure("tui", io.StringIO()))
        self.addCleanup(lambda: os.environ.pop("NO_COLOR", None))

    def spin(self, fmt, stdout, stderr):
        with contextlib.redirect_stderr(stderr):
            report.configure(fmt, stdout)  # stderr's TTY-ness is read here
            before = colors.Clr.BCYN
            with contextlib.redirect_stdout(stdout):
                p = progress.ProgressAnimation()
                p.start("working", state={"start": time.time(), "last": time.time(), "lines": 0})
                time.sleep(0.2)
                p.stop()
        return before

    def test_markdown_report_still_colors_the_live_line(self):
        err = TTY()
        before = self.spin("markdown", io.StringIO(), err)
        self.assertEqual(before, "")  # the report on stdout stays plain
        self.assertIn("\033[96m", err.getvalue())
        self.assertIn("working", err.getvalue())
        self.assertEqual(colors.Clr.BCYN, "")  # restored: tables printed afterwards carry no codes

    def test_tui_to_a_pipe_still_colors_the_live_line(self):
        err = TTY()
        self.spin("tui", io.StringIO(), err)
        self.assertIn("\033[96m", err.getvalue())

    def test_no_color_keeps_the_live_line_plain(self):
        os.environ["NO_COLOR"] = "1"
        err = TTY()
        self.spin("markdown", io.StringIO(), err)
        self.assertNotIn("\033[96m", err.getvalue())

    def test_live_colors_are_off_for_a_stderr_that_is_not_a_terminal(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            report.configure("markdown", io.StringIO())
        with colors.live():
            self.assertEqual(colors.Clr.BCYN, "")


class EndToEnd(unittest.TestCase):
    """A full mock run (Splash-like server, synthetic fixture) in each format."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = write_fixture(synthetic_fixture(), self.tmp.name)
        self.addCleanup(lambda: report.configure("tui", io.StringIO()))

    def run_replay(self, srv, *extra, **kw):
        return run_main("--suite", "coding", "--fixture", self.fixture, "--endpoint", srv.url, "--no-save-state", *extra, **kw)

    def test_tui_run(self):
        with SplashLike(behaviour) as srv:
            out, err = self.run_replay(srv)
        self.assertIn("Splash 1a2b3c4d", out)  # discovery stays on stdout in the tui
        self.assertIn("R003_main_e0_t001", out)  # live case rows
        for needle in ("Metrics", "By context", "Prefill by append size", "┌", "done"):
            self.assertIn(needle, out)
        self.assertRegex(out, r"│ Metric\s+│\s+Value")
        self.assertFalse(ANSI.search(out))
        self.assertEqual(err, "")  # no spinner when stderr is not a terminal

    def test_markdown_run_is_clean_on_stdout(self):
        # the subagent turn with 3 messages answers in text where the recording called a tool: one WARN anomaly
        with SplashLike(lambda p: {"text": "All done here."} if len(p["messages"]) == 3 else behaviour(p)) as srv:
            out, err = self.run_replay(srv, "--report-format", "markdown", "--no-mdcat")
        tables = assert_valid_markdown(self, out)
        self.assertGreaterEqual(tables, 5)
        for needle in ("# Agent replay \u2014 suite coding", "## Run", "**R003_main_e0_t001**", "### Metrics",
                       "### Anomalies", "### By context", "### Prefill by append size", "Splash 1a2b3c4d"):
            self.assertIn(needle, out)
        # the discovery block became the Run section; stderr has one line per finished case, no progress animation
        for chatter in ("Discovery", "selection:"):
            self.assertNotIn(chatter, out)
            self.assertNotIn(chatter, err)
        self.assertRegex(err, r"R003 main/e0 \d+(\.\d)?k? ctx \u00b7 \d\.\d\ds \u00b7 (PASS|WARN|FAIL)")
        self.assertFalse(ANSI.search(err))
        self.assertNotIn("\r", err)

    def test_markdown_run_saved_file_is_valid_markdown(self):
        # the documented use: markdown to a file, progress on the terminal (stderr)
        path = os.path.join(self.tmp.name, "out.md")
        with SplashLike(behaviour) as srv, open(path, "w") as f:
            self.run_replay(srv, "--report-format", "markdown", stdout=f)
        with open(path) as f:
            assert_valid_markdown(self, f.read())

    def test_legend_wrapping_and_ideas(self):
        from bench.outputs import legend
        text_120 = legend(shows_delta=True, spec=True, width=120)
        lines_120 = text_120.splitlines()
        for ln in lines_120:
            self.assertLessEqual(len(ln), 120)
        self.assertIn("Metrics, context and comparison tables use the client clock: decode tok/s = output tokens ÷ (request time − TTFT),", lines_120)
        self.assertIn("end-to-end = output tokens ÷ request time, prefill = new prompt tokens ÷ TTFT (native = the server's own figures).", lines_120)
        self.assertIn("case table with a baseline: ΔTime, ΔDecode and ΔPrefill are the change of that request,", lines_120)

        text_80 = legend(shows_delta=True, spec=True, width=80)
        for ln in text_80.splitlines():
            self.assertLessEqual(len(ln), 80)


if __name__ == "__main__":
    unittest.main()
