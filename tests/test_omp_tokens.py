"""Tests for tools/omp_tokens.py: comparison/report logic and log parsing offline (stdlib); the real case A count
only where Splash's venv packages, the GGUF, the snapshot capture and the case A session exist.

The real check runs under a python that has those packages; with the system python it is skipped. Override the inputs with OMP_GOLDEN_CAPTURE / OMP_CASE_A_SESSION /
OMP_TOKENS_GGUF / OMP_TOKENS_SPLASH / OMP_TOKENS_SPLASH_LOG.
"""
import glob
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.dirname(__file__))
import isolation  # noqa: E402,F401
import omp_tokens as tk  # noqa: E402
import test_omp_requests as tr  # noqa: E402

SPLASH = os.environ.get("OMP_TOKENS_SPLASH", tk.DEFAULT_SPLASH)
GGUF = os.environ.get("OMP_TOKENS_GGUF") or next(iter(sorted(glob.glob(tk.DEFAULT_GGUF_GLOB))), "")
SPLASH_LOG = os.environ.get("OMP_TOKENS_SPLASH_LOG", "")
CASE_D_LOG = os.environ.get("OMP_TOKENS_CASE_D_LOG", "")


def _splash_importable():
    try:
        import jinja2  # noqa: F401
        import tokenizers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return os.path.isdir(os.path.join(SPLASH, "server"))


def row(label, turn, counted, recorded, **kw):
    r = {"label": label, "stream": label.split("/")[0], "turn": turn, "counted": counted, "recorded": recorded,
         "delta": None if counted is None or recorded is None else counted - recorded}
    r.update(kw)
    return r


class ReportTests(unittest.TestCase):
    def test_stream_label(self):
        self.assertEqual(tk.stream_label("main", 2), "main/e2")
        self.assertEqual(tk.stream_label("main", None), "main/e0")
        self.assertEqual(tk.stream_label("Sub", 0), "Sub")

    def test_summarize(self):
        rows = [row("main/e0", 0, 10, 10), row("main/e0", 1, 12, 15), row("main/e0", 2, 20, 18),
                row("Sub", 0, 5, 5), row("Sub", 1, None, 7, error="boom"), row("main/e1", 3, 9, 9)]
        s = {x["stream"]: x for x in tk.summarize(rows)}
        self.assertEqual([x["stream"] for x in tk.summarize(rows)], ["main/e0", "Sub", "main/e1"])
        self.assertEqual((s["main/e0"]["turns"], s["main/e0"]["exact"], s["main/e0"]["min"], s["main/e0"]["max"],
                          s["main/e0"]["median"]), (3, 1, -3, 2, 0))
        self.assertEqual((s["Sub"]["turns"], s["Sub"]["exact"], s["Sub"]["min"]), (2, 1, 0))
        self.assertEqual(len(tk.mismatches(rows)), 3)

    def test_tool_signature_mismatch_is_a_mismatch(self):
        rows = [row("Sub", 0, 5, 5, tools="7·aaaa", tools_ok=False, log_tools=["7·bbbb"])]
        self.assertEqual(len(tk.mismatches(rows)), 1)
        text = tk.format_report(rows)
        self.assertIn("tool signatures equal to the Splash log's: 0/1", text)
        self.assertIn("tools 7·aaaa not in log ['7·bbbb']", text)

    def test_format_report(self):
        rows = [row("main/e0", 0, 10, 10), row("main/e0", 1, 12, 15), row("Sub", 0, None, 7, error="E: x")]
        text = tk.format_report(rows, limit=1)
        lines = text.splitlines()
        self.assertTrue(lines[1].startswith("main/e0") and lines[1].split()[1:] == ["2", "1", "-3", "+0", "-1.5"])
        self.assertIn("total                      3     1", text)
        self.assertIn("mismatching turns (2):", text)
        self.assertIn("counted 12 recorded 15 Δ -3", text)
        self.assertIn("… 1 more", text)

    def test_tools_signature(self):
        tools = [{"type": "function", "function": {"name": "a", "parameters": {}}}]
        sig = tk.tools_signature(tools)
        self.assertRegex(sig, r"^1·[0-9a-f]{8}$")
        self.assertIsNone(tk.tools_signature(None))
        self.assertIsNone(tk.tools_signature([]))


class LogTests(unittest.TestCase):
    def test_parse_splash_log(self):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write("12:23:44 Started · input 20,915 · cached 0 · effort absent → medium\n"
                    "12:24:26 Done · input 20,915 · cached 0 · output 117 · tools 12·ecf4d8da · TTFT 40.1s\n"
                    "12:25:37 Done · input 963 · cached 0 · output 16 · TTFT 1.9s\n"
                    "12:26:00 Cancelled · input 1,000 · tools 7·d7b758fa\n"
                    "12:27:00 Memory: in-service claim\n")
        self.addCleanup(os.unlink, f.name)
        entries = tk.parse_splash_log(f.name)
        self.assertEqual([(e["input"], e["tools"]) for e in entries],
                         [(20915, "12·ecf4d8da"), (963, None), (1000, "7·d7b758fa")])
        self.assertEqual(tk.log_signatures_by_input(entries + [{"input": 963, "tools": "1·x", "time": ""}])[963],
                         {None, "1·x"})


class FakeCounter:
    """Counts a body as its message count (stand-in for SplashCounter)."""

    def count(self, body):
        return 100 + len(body["messages"])

    def tools_signature(self, body):
        return tk.tools_signature(body.get("tools"))


class CheckSessionTests(tr.Base):
    def test_check_session_rows(self):
        b = tr.Builder()
        tr.main_session(b)
        sess = self.load(b)
        rows = tk.check_session(self.path, tr.SNAP, FakeCounter())
        self.assertEqual(len(rows), len(sess.main.turns))
        for r, t in zip(rows, sess.main.turns):
            self.assertEqual((r["label"], r["turn"], r["recorded"]), ("main/e0", t.index, t.prompt_tokens))
            self.assertEqual(r["delta"], r["counted"] - t.prompt_tokens)
        sig = rows[0]["tools"]
        log = [{"input": sess.main.turns[0].prompt_tokens, "tools": sig, "time": ""}]
        rows = tk.check_session(self.path, tr.SNAP, FakeCounter(), log_entries=log)
        self.assertTrue(rows[0]["tools_ok"])

    def test_render_error_is_a_row(self):
        b = tr.Builder()
        tr.main_session(b)
        self.load(b)

        class Broken(FakeCounter):
            def count(self, body):
                raise ValueError("template")

        rows = tk.check_session(self.path, tr.SNAP, Broken())
        self.assertTrue(all(r["counted"] is None and "template" in r["error"] for r in rows))

    def test_check_session_with_compactions(self):
        b = tr.Builder()
        b.reminder(tr.T0 + 5, f"user:{tr.T0}:user")
        b.user(tr.T0, "hello")
        b.assistant(tr.T0 + 10, "thinking", "text", stop="stop")
        b.compaction(tr.T0 + 20, summary="s", method="handoff", tokens_before=105)
        self.load(b)
        rows = tk.check_session(self.path, tr.SNAP, FakeCounter())
        self.assertEqual(len(rows), 2)
        comp_row = rows[1]
        self.assertEqual(comp_row["label"], "main/c0")
        self.assertEqual(comp_row["recorded"], 105)
        self.assertEqual(comp_row["delta"], comp_row["counted"] - 105)

        # With log entries matching counted
        log = [{"input": comp_row["counted"], "tools": comp_row["tools"], "time": ""}]
        rows2 = tk.check_session(self.path, tr.SNAP, FakeCounter(), log_entries=log)
        self.assertEqual(rows2[1]["delta"], 0)
        self.assertTrue(rows2[1]["tools_ok"])

        # With include_compactions=False
        rows3 = tk.check_session(self.path, tr.SNAP, FakeCounter(), include_compactions=False)
        self.assertEqual(len(rows3), 1)


# --------------------------------------------------------------------------------------------- real inputs

@unittest.skipUnless(_splash_importable() and os.path.isfile(GGUF) and os.path.isdir(tr.GOLDEN_CAPTURE)
                     and os.path.isfile(tr.CASE_A_SESSION),
                     "needs Splash's venv python, the UD-IQ1_S GGUF, the snapshot capture and the case A session")
class CaseATokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = tk.SplashCounter(SPLASH, GGUF)

    def test_template_is_patched(self):
        self.assertIn("patched to render later system messages in place", self.counter.describe())

    def test_golden_captures_count_exactly(self):
        """The captured requests themselves count to their response's usage.prompt_tokens."""
        import omp_requests as r
        n = 0
        for _ms, req, resp in r._capture_pairs(tr.GOLDEN_CAPTURE):
            pt = r.resp_usage(resp).get("prompt_tokens")
            if pt is None:
                continue
            with open(req, encoding="utf-8") as f:
                self.assertEqual(self.counter.count(json.load(f)), pt, req)
            n += 1
        self.assertGreaterEqual(n, 13)

    def test_case_a(self):
        log = tk.parse_splash_log(SPLASH_LOG) if os.path.isfile(SPLASH_LOG) else None
        rows = tk.check_session(tr.CASE_A_SESSION, tr.case_a_snapshot(), self.counter, log_entries=log)
        self.assertEqual(len(rows), 475)
        bad = [(r["label"], r["turn"], r["delta"]) for r in tk.mismatches(rows)]
        self.assertEqual(bad, [])
        if log is not None:
            self.assertTrue(all(r["tools_ok"] for r in rows))

    def test_case_a_compactions(self):
        sess = tr.o.load_session(tr.CASE_A_SESSION)
        snap = tr.case_a_snapshot()
        log = tk.parse_splash_log(SPLASH_LOG) if os.path.isfile(SPLASH_LOG) else None
        by_input = tk.log_signatures_by_input(log) if log is not None else {}

        # Compaction 0: input 236,629, tools 12·ecf4d8da
        body0 = tr.r.build_compaction_request(sess.main, 0, snap)
        cnt0 = self.counter.count(body0)
        sig0 = self.counter.tools_signature(body0)
        self.assertEqual(cnt0, 236629)
        self.assertEqual(sig0, "12·ecf4d8da")
        if by_input:
            self.assertIn(cnt0, by_input)
            self.assertIn(sig0, by_input[cnt0])

        # Compaction 1: input 238,162, tools 12·ecf4d8da
        body1 = tr.r.build_compaction_request(sess.main, 1, snap)
        cnt1 = self.counter.count(body1)
        sig1 = self.counter.tools_signature(body1)
        self.assertEqual(cnt1, 238162)
        self.assertEqual(sig1, "12·ecf4d8da")
        if by_input:
            self.assertIn(cnt1, by_input)
            self.assertIn(sig1, by_input[cnt1])



@unittest.skipUnless(_splash_importable() and os.path.isfile(GGUF) and not getattr(tr.CaseDTests, "__unittest_skip__", False),
                     "needs Splash's venv python, the UD-IQ1_S GGUF and the case D inputs")
class CaseDTokenTests(unittest.TestCase):
    def test_case_d(self):
        """Every turn and the compaction count exactly; tool signatures equal the day's Splash log."""
        counter = tk.SplashCounter(SPLASH, GGUF)
        log = tk.parse_splash_log(CASE_D_LOG) if os.path.isfile(CASE_D_LOG) else None
        rows = tk.check_session(tr.CASE_D_SESSION, tr.case_d_snapshot(tr.case_a_snapshot()), counter,
                                log_entries=log)
        self.assertEqual(len(rows), 470)
        self.assertEqual([(r["label"], r["turn"], r["delta"]) for r in tk.mismatches(rows)], [])
        if log is not None:
            self.assertTrue(all(r["tools_ok"] for r in rows))


if __name__ == "__main__":
    unittest.main()
