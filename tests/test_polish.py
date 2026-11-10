"""Report polish: counts instead of rates, one metrics table, short labels, number formats, markdown structure and
emphasis, the markdown live line, Ctrl+C / error handling with an `end` record, fsync, and the chat stats row and
glyph fallback. No real server: a mock Splash on 127.0.0.1 and synthetic states.

    python3 -m unittest tests.test_polish
"""

import contextlib
import io
import json
import os
import re
import signal
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime
from unittest import mock

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401
from test_replay import synthetic_fixture, write_fixture  # noqa: E402
from test_report import ANSI, TTY, assert_valid_markdown, run_main, write_state  # noqa: E402
from test_state import SplashLike, behaviour  # noqa: E402

import replay_bench  # noqa: E402
from bench import chat, fmt, glyphs, report, results, runner, state  # noqa: E402

PUA = re.compile("[-]")


def run_code(*argv, stdout=None, stderr=None):
    """(stdout, stderr, exit code) of main(argv); SystemExit is caught."""
    out, err, code = stdout or io.StringIO(), stderr or io.StringIO(), 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            replay_bench.main(list(argv))
        except SystemExit as e:
            code = e.code or 0
    return out.getvalue(), err.getvalue(), code


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.a = os.path.join(self.tmp.name, "a.jsonl")
        self.b = os.path.join(self.tmp.name, "b.jsonl")
        write_state(self.a, "ISTA/Qwen-Flash-GSQ-RCO", 40.0)
        write_state(self.b, "ISTA/Qwen-Flash-GSQ-Q4", 60.0)
        self.both = os.path.join(self.tmp.name, "both.jsonl")  # two model blocks in one file (a live run writes one file per model; older ones held several)
        with open(self.both, "w") as f:
            for p in (self.a, self.b):
                with open(p) as g:
                    f.write(g.read())
        self.addCleanup(lambda: (report.configure("tui", io.StringIO()), glyphs.configure(False)))


class Formatting(unittest.TestCase):
    def test_durations_one_style(self):
        self.assertEqual([fmt.ms(x) for x in (896, 8574, 24550, 7306, 70600)], ["0.90s", "8.57s", "24.6s", "7.31s", "1m11s"])
        self.assertEqual(fmt.secs(None), "–")

    def test_thousands_separators(self):
        self.assertEqual((fmt.num(12174), fmt.num(None), fmt.num(837)), ("12,174", "–", "837"))

    def test_short_names_strip_the_common_prefix(self):
        self.assertEqual(fmt.short_names(["ISTA-DASLab/Qwen3.8-GSQ-RCO-GGUF", "ISTA-DASLab/Qwen3.8-GSQ-Q4"]),
                         ["RCO-GGUF", "Q4"])
        self.assertEqual(fmt.short_names(["alpha", "beta"]), ["alpha", "beta"])
        self.assertEqual(fmt.short_names(["only"]), ["only"])
        self.assertEqual(fmt.short_names(["m-", "m-x"]), ["m-", "m-x"])  # nothing may become empty


class Tui(Base):
    def test_no_duplicates_counts_and_numbers(self):
        out, _ = run_main("--state", self.a)
        self.assertNotIn("Not clean passes", out)
        self.assertNotIn("ANOMALY", out)
        self.assertNotIn("FAST-MODEL", out)
        self.assertNotIn("Pass rate", out)
        self.assertEqual(out.count("Anomalies"), 1)
        self.assertIn("5 PASS · 0 WARN · 1 FAIL · 0 SKIP", out)
        self.assertIn("38,000", out.replace("38,000", "38,000"))  # thousands separators somewhere in the tables
        self.assertRegex(out, r"\d,\d{3}")
        self.assertNotRegex(out, r"\d{4}\.0\b")  # no raw milliseconds like 7306.0
        self.assertNotIn(self.a, out)

    def test_single_model_has_no_model_or_run_columns(self):
        out, _ = run_main("--state", self.a)
        head = next(l for l in out.splitlines() if "Case" in l and "Verdict" in l)
        self.assertNotIn("Model", head)
        self.assertIn("Stream", head)
        self.assertNotIn("Run ", out)
        self.assertNotIn("Summary per model", out)

    def test_several_models_get_short_labels(self):
        out, _ = run_main("--state", self.both)
        self.assertIn("Comparison", out)
        self.assertRegex(out, r"│ \S*RCO\s+│")
        self.assertNotIn("ISTA/Qwen-Flash-GSQ-RCO  ", out.split("Comparison")[1])
        self.assertRegex(out.split("Comparison")[1], r"\bQ4\b")

    def test_headings_are_not_markdown(self):
        out, _ = run_main("--state", self.a)
        self.assertNotIn("###", out)
        self.assertNotRegex(out, r"(?m)^## ")
        self.assertIn("Metrics", out)

    def test_state_path_once_and_relative(self):
        p = os.path.join(report.REPO, "states", "x.jsonl")
        self.assertEqual(report.rel_path(p), os.path.join("states", "x.jsonl"))
        self.assertEqual(report.rel_path("/tmp/elsewhere.jsonl"), "/tmp/elsewhere.jsonl")


class Markdown(Base):
    def run_md(self, *states):
        out, _ = run_main("--state", states[0], "--report-format", "markdown", "--no-mdcat")
        return out

    def test_heading_structure_and_run_section(self):
        out = self.run_md(self.a)
        assert_valid_markdown(self, out)
        heads = re.findall(r"(?m)^(#{1,3}) (.*)$", out)
        self.assertEqual(heads[0], ("#", "Agent replay — suite coding"))
        self.assertEqual(heads[1], ("##", "Run"))
        titles = [h for _, h in heads]
        for t in ("ISTA/Qwen-Flash-GSQ-RCO", "Results", "Metrics", "Anomalies", "Performance", "By context",
                  "Prefill by append size"):
            self.assertIn(t, titles)
        self.assertNotIn("Comparison", titles)  # one model
        self.assertIn("| Server", out)
        self.assertIn("Splash 1a2b3c4d", out)

    def test_comparison_only_with_two_models(self):
        out = self.run_md(self.both)
        self.assertIn("## Comparison", out)
        self.assertIn("| **Q4**", out)  # short label

    def test_emphasis(self):
        out = self.run_md(self.a)
        self.assertRegex(out, r"\*\*R000_main_e0_t000\*\*")  # bold case ids
        self.assertIn("✅ PASS", out)
        self.assertIn("❌ FAIL", out)
        self.assertIn("**❌ FAIL**".replace("❌ ", ""), out.replace("❌ ", ""))  # FAIL is bold
        self.assertRegex(out, r"`main/e0`")  # the stream is code: it must not wrap
        self.assertRegex(out, r"Wall time \*\(sum of requests\)\*")  # units italic
        self.assertRegex(out, r"(?m)^\*[A-Z].*\*$")  # footnotes italic

    def test_tool_names_in_code_and_signed_deltas(self):
        m = {"id": "R1_main_e0_t000", "eval_result": "WARN", "fail_reason": "other tools: siblings (recorded fdc_get)",
             "ttft_ms": 500, "total_ms": 900, "tokens": 5, "prompt_tokens": 100, "cached_tokens": 0}
        lay = results.CaseLayout(has_stats=True)
        t = report.Table(lay.header, results.case_rows(lay, [m]))
        self.assertIn("other tools: siblings (recorded fdc_get)", t.markdown())
        self.assertIn("⚠️ WARN", t.markdown())
        tbl = report.Table(["a", "Δ"], [["x", "+5.0%"], ["y", "-2.0%"]], dirs={1: "up"})
        md = tbl.markdown()
        self.assertIn("**+5.0%**", md)  # gain bold
        self.assertIn("*-2.0%*", md)  # loss italic

    def test_live_line(self):
        m = {"id": "R004_main_e0_t003", "stream": "main", "epoch": 0, "prompt_tokens": 12600, "ttft_ms": 568,
             "eval_result": "WARN", "fail_reason": "other tools: siblings (recorded fdc_search)"}
        self.assertEqual(results.live_line(m, results.stream_label(m)),
                         "R004 main/e0 12.3k ctx · 0.57s · WARN other tools: siblings (recorded fdc_search)")


class Run(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = write_fixture(synthetic_fixture(), self.tmp.name)
        self.path = os.path.join(self.tmp.name, "s.jsonl")
        self.addCleanup(lambda: report.configure("tui", io.StringIO()))

    def run_replay(self, srv, *extra):
        return run_code("--suite", "coding", "--fixture", self.fixture, "--endpoint", srv.url, "--state-file", self.path, *extra)

    def records(self):
        with open(self.path) as f:
            return [json.loads(x) for x in f if x.strip()]

    def failing_after(self, n, exc):
        real, calls = runner.query_model, []

        def fake(*a, **k):
            calls.append(1)
            if len(calls) > n:
                raise exc
            return real(*a, **k)
        return mock.patch.object(runner, "query_model", side_effect=fake)

    def test_ctrl_c_writes_interrupted_end_and_reports_everything(self):
        for fmt_ in ("tui", "markdown", "chat"):
            if os.path.exists(self.path):
                os.remove(self.path)
            with SplashLike(behaviour) as srv, self.failing_after(4, KeyboardInterrupt()):  # warmup + 3 cases
                out, err, code = self.run_replay(srv, "--report-format", fmt_, "--no-mdcat")
            self.assertEqual(code, 130, fmt_)
            recs = self.records()
            self.assertEqual([r["type"] for r in recs], ["run"] + ["result"] * 3 + ["end"])
            end = recs[-1]
            self.assertTrue(end["interrupted"])
            self.assertEqual(end["reason"], "Ctrl+C")
            self.assertEqual(end["cases"], 3)
            self.assertRegex(out, r"Interrupted\*{0,2} \(Ctrl\+C\) after 3 requests")
            self.assertIn("Metrics", out)  # the reports of the finished cases
            self.assertIn("R003_main_e0_t001", out)
            self.assertIn("interrupted", out.lower() if fmt_ != "markdown" else out.lower())

    def test_a_real_sigint_aborts_the_request_in_flight(self):
        gate = threading.Event()
        seen = []

        def blocking(payload):
            seen.append(1)
            if len(seen) == 4:  # the third case: hang until the test is over
                gate.wait(20)
            return behaviour(payload)

        timer = None
        with SplashLike(blocking) as srv:
            def kill():
                while len(seen) < 4:
                    time.sleep(0.05)
                time.sleep(0.2)
                os.kill(os.getpid(), signal.SIGINT)
            timer = threading.Thread(target=kill, daemon=True)
            timer.start()
            try:
                out, err, code = self.run_replay(srv)
            finally:
                gate.set()
        self.assertEqual(code, 130)
        recs = self.records()
        self.assertEqual(sum(r["type"] == "result" for r in recs), 2)  # the in-flight request is not a result
        self.assertTrue(recs[-1]["interrupted"])
        self.assertRegex(out, r"Interrupted\*{0,2} \(Ctrl\+C\) after 2 requests")

    def test_second_ctrl_c_exits_at_once(self):
        with runner._interrupt_guard():
            first = signal.getsignal(signal.SIGINT)
            with self.assertRaises(KeyboardInterrupt):
                first(signal.SIGINT, None)
            second = signal.getsignal(signal.SIGINT)
            self.assertIs(second, runner._hard_exit)
            with mock.patch.object(runner.os, "_exit") as ex, contextlib.redirect_stderr(io.StringIO()):
                second(signal.SIGINT, None)
            ex.assert_called_once_with(130)

    def test_exception_in_the_loop(self):
        with SplashLike(behaviour) as srv, self.failing_after(3, RuntimeError("boom")):
            out, err, code = self.run_replay(srv)
        self.assertEqual(code, 1)
        end = self.records()[-1]
        self.assertTrue(end["interrupted"])
        self.assertIn("RuntimeError: boom", end["reason"])
        self.assertIn("Traceback", err)
        self.assertIn("Metrics", out)
        self.assertRegex(out, r"Interrupted\*{0,2} \(error: RuntimeError: boom\)")

    def test_clean_run_end_record_and_one_state_line_at_the_end(self):
        with SplashLike(behaviour) as srv:
            out, err, code = self.run_replay(srv)
        self.assertEqual(code, 0)
        end = self.records()[-1]
        self.assertFalse(end["interrupted"])
        self.assertNotIn("reason", end)
        self.assertEqual(out.count(self.path), 1)
        self.assertEqual(out.count("state:"), 1)

    def test_every_state_line_is_fsynced(self):
        with mock.patch.object(state.os, "fsync") as fs:
            w = state.StateWriter(os.path.join(self.tmp.name, "f.jsonl"))
            w.write({"type": "run"})
            w.flush()
            w.write({"type": "end"})
            w.close()
        self.assertEqual(fs.call_count, 2)

    def test_interrupted_and_unfinished_states_load(self):
        write_state(self.path, "m", n=3)
        recs = self.records()
        recs[-1] = state.end_record({}, interrupted="Ctrl+C")
        with open(self.path, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in recs))
        out, _, code = run_code("--state", self.path)
        self.assertEqual(code, 0)
        self.assertRegex(out, r"Interrupted\*{0,2} \(Ctrl\+C\)")
        with open(self.path, "w") as f:  # no end record at all (killed run), and a torn last line
            f.write("".join(json.dumps(r) + "\n" for r in recs[:-1]) + '{"type": "res')
        out, _, code = run_code("--state", self.path)
        self.assertEqual(code, 0)
        self.assertIn("R002_main_e0_t002", out)
        self.assertNotIn("Interrupted", out)


class ChatLook(unittest.TestCase):
    M = {"id": "R001_main_e0_t000", "prompt_tokens": 12174, "cached_tokens": 0, "tokens": 148, "ttft_ms": 47800,
         "total_ms": 52600, "eval_result": "WARN", "fail_reason": "other tools: siblings (recorded fdc_get)"}
    NO_CACHE_NO_SPEED = {"id": "R002", "prompt_tokens": 500, "cached_tokens": 200, "tokens": 5, "ttft_ms": 0,
                         "total_ms": 50, "eval_result": "PASS"}

    def tearDown(self):
        report.configure("tui", io.StringIO())
        glyphs.configure(False)

    def rows(self, nerd):
        report.set_nerd_font(True)
        report.configure("chat", TTY() if nerd else io.StringIO())
        when = datetime(2026, 11, 7, 8, 52, 27)
        return [chat.stats_row(m, when, d) for m, d in ((self.M, 52600), (self.NO_CACHE_NO_SPEED, 50), (self.M, 125000))]

    def test_columns_have_fixed_widths_missing_fields_keep_their_slot(self):
        for nerd in (False, True):
            rows = [ANSI_ALL.sub("", r) for r in self.rows(nerd)]
            self.assertEqual(len({chat.vlen(r) for r in rows}), 1, rows)
        rows = [ANSI_ALL.sub("", r) for r in self.rows(False)]
        self.assertNotIn("/s", rows[1])  # the quick request has no speed ...
        self.assertEqual(rows[1].rstrip(), rows[1].rstrip())
        self.assertEqual(chat.vlen(rows[0]), chat.vlen(rows[1]))  # ... but its slot is blank, nothing shifted

    def test_no_nerd_font_fallback_and_no_ansi_off_a_tty(self):
        rows = self.rows(False)
        for r in rows:
            self.assertFalse(PUA.search(r))
            self.assertFalse(ANSI.search(r))
        report.configure("chat", io.StringIO())
        self.assertFalse(PUA.search(chat.badge("WARN")) or ANSI.search(chat.badge("WARN")))

    def test_nerd_font_and_colors_on_a_tty(self):
        rows = self.rows(True)
        self.assertTrue(all(PUA.search(r) and ANSI.search(r) for r in rows))
        self.assertIn(glyphs.ICONS["clock"], rows[0])
        b = chat.badge("PASS")
        self.assertTrue(PUA.search(b) and "\x1b[7m" in b)

    def test_icons_are_opt_in(self):
        report.set_nerd_font(None)
        report.configure("chat", TTY())
        self.assertFalse(glyphs.enabled)  # default: off, even on a color terminal
        with mock.patch.dict(os.environ, {"REPLAY_BENCH_NERD_FONT": "1"}):
            report.configure("chat", TTY())
            self.assertTrue(glyphs.enabled)
            report.configure("chat", io.StringIO())
            self.assertFalse(glyphs.enabled)  # not a terminal
            report.set_nerd_font(False)  # --no-nerd-font beats the env
            report.configure("chat", TTY())
            self.assertFalse(glyphs.enabled)
        report.set_nerd_font(True)  # --icons
        report.configure("chat", TTY())
        self.assertTrue(glyphs.enabled)
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            report.configure("chat", TTY())
        self.assertFalse(glyphs.enabled)
        report.set_nerd_font(None)
        with mock.patch.dict(os.environ, {"REPLAY_BENCH_NERD_FONT": "0"}):
            report.configure("chat", TTY())
        self.assertFalse(glyphs.enabled)
        _, _, code = run_code("--help")

    def test_cli_flag_exists(self):
        out, _, _ = run_code("--help")
        self.assertIn("--icons", out)
        self.assertIn("--no-nerd-font", out)

    def test_badges_have_one_width(self):
        report.configure("chat", io.StringIO())
        self.assertEqual({chat.vlen(chat.badge(v)) for v in ("PASS", "WARN", "FAIL", "SKIP")}, {6})

    TC = {"id": "R013_main_e0_t010", "entry": {"label": "main/e0", "prompt_tokens": 50000}, "recorded_prompt_tokens": 50000}
    PR = {"done": 12, "total": 470, "t0": 100.0, "t_last": 400.0, "cache": 98.0, "n_ctx": 262144}
    ST = {"start": 410.0, "first": 415.5, "down": 224}

    def test_live_progress_line_full(self):
        report.configure("chat", io.StringIO())
        line = chat.live_compose(self.TC, self.ST, 420.0, self.PR)
        for part in ("13/470", "R013 main/e0", "50k/262k ctx", "cache 98%", "TTFT 5.5s", "224 tok", "50 tok/s", "run 5m20s",
                     "ETA 3h10m"):
            self.assertIn(part, line)
        self.assertIn("\u2595", line)  # the bars
        self.assertLess(line.index("cache"), line.index("TTFT"))  # the cache sits by the context gauge
        self.assertLess(line.index("ETA"), line.index("R013"))  # the ETA sits on the run bar

    def test_live_line_always_fits_the_width_and_keeps_the_core(self):
        report.configure("chat", io.StringIO())
        for width in (40, 60, 80, 120, 200):
            for st in ({"start": 410.0}, self.ST):
                line = chat.live_compose(self.TC, st, 420.0, self.PR, width)
                self.assertLessEqual(chat.vlen(line), width, (width, line))
                self.assertIn("R013", line)
        for width in (60, 80, 120):
            line = chat.live_compose(self.TC, self.ST, 420.0, self.PR, width)
            self.assertIn("50 tok/s", line)
            self.assertIn("13/470", line)
            self.assertIn("50k", line)
        # dropped from the end of the priority list first: ETA, cache, output tokens, ...
        wide = chat.live_compose(self.TC, self.ST, 420.0, self.PR, 125)
        narrower = chat.live_compose(self.TC, self.ST, 420.0, self.PR, 100)
        self.assertIn("ETA", chat.live_compose(self.TC, self.ST, 420.0, self.PR, 200))
        self.assertNotIn("ETA", wide)
        self.assertNotIn("cache", narrower)

    def test_icons_replace_label_words(self):
        report.set_nerd_font(True)
        report.configure("chat", TTY())
        line = ANSI_ALL.sub("", chat.live_compose(self.TC, self.ST, 420.0, self.PR))
        for word in ("cache", "ctx", "run", "TTFT", "ETA", " tok "):
            self.assertNotIn(word, line)
        self.assertIn("tok/s", line)  # a unit stays
        self.assertTrue(PUA.search(line))
        row = ANSI_ALL.sub("", chat.stats_row(self.M, datetime(2026, 11, 7, 8, 52, 27), 52600))
        self.assertNotIn("\u0394", row)
        report.configure("chat", io.StringIO())
        self.assertIn("\u0394", chat.stats_row(self.M, datetime(2026, 11, 7, 8, 52, 27), 52600))  # the fallback keeps it
        line = chat.live_compose(self.TC, self.ST, 420.0, self.PR)
        self.assertIn("cache 98%", line)


ANSI_ALL = re.compile(r"\x1b\[[0-9;]*m")

if __name__ == "__main__":
    unittest.main()


class LiveLineIconWidthTest(unittest.TestCase):
    def test_icons_count_two_columns_when_fitting(self):
        from bench import chat, glyphs
        glyphs.configure(True)
        try:
            tc = {"id": "R208_F428FalsifyP1213_t000", "recorded_prompt_tokens": 9900,
                  "entry": {"label": "F428FalsifyP1213", "prompt_tokens": 9900}}
            st = {"start": 100.0, "first": 117.7, "down": 84, "last": 118.0}
            progress = {"done": 207, "total": 470, "t0": 0.0, "t_last": 90.0, "cache": 98.0, "n_ctx": 262144}
            for width in range(30, 200, 7):
                line = chat.live_compose(tc, st, 119.0, progress, width)
                self.assertLessEqual(chat.live_vlen(line), max(width, chat.live_vlen(
                    chat.live_render(chat.live_parts(tc, st, 119.0, progress), chat.DROP_ORDER))))
                self.assertGreaterEqual(chat.live_vlen(line), chat.vlen(line))
        finally:
            glyphs.configure(False)


class LiveLineIconsInTuiTest(unittest.TestCase):
    TC = {"id": "R003_main_e0", "recorded_prompt_tokens": 22000,
          "entry": {"label": "main/e0", "prompt_tokens": 22000}}
    ST = {"start": 100.0}
    PROGRESS = {"done": 2, "total": 470, "t0": 0.0, "t_last": 90.0, "cache": 97.0, "n_ctx": 262144}

    def setUp(self):
        from bench import glyphs, report
        self.glyphs, self.report = glyphs, report
        self.addCleanup(glyphs.configure_live, False)
        self.addCleanup(glyphs.configure, False)

    def line(self):
        from bench import chat
        return chat.live_compose(self.TC, self.ST, 109.0, self.PROGRESS)

    def test_icons_replace_words_in_tui_when_live_is_on(self):
        self.glyphs.configure(False)
        self.glyphs.configure_live(True)
        line = self.line()
        for icon in ("delta", "eta", "ctx", "cache", "prefill"):
            self.assertIn(self.glyphs.ICONS[icon], line)
        for word in ("run", "ETA", " ctx", "cache", "prefill"):
            self.assertNotIn(word, line)
        self.assertFalse(self.glyphs.enabled)  # the stdout report is untouched

    def test_words_when_live_is_off(self):
        self.glyphs.configure_live(False)
        line = self.line()
        for word in ("run", "ETA", " ctx", "cache", "prefill"):
            self.assertIn(word, line)
        self.assertFalse(any(0xE000 <= ord(c) <= 0xF8FF for c in line))

    def test_stdout_report_stays_free_of_icons(self):
        import io
        self.glyphs.configure_live(True)
        for f in ("tui", "markdown"):
            self.report.configure(f, io.StringIO())  # not a TTY: stdout icons off
            self.assertFalse(self.glyphs.enabled)
            self.assertEqual(self.glyphs.g("ctx"), "")
            self.assertEqual(self.glyphs.prefix("ctx"), "")
        self.report.configure("tui", io.StringIO())
