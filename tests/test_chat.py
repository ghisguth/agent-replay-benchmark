"""The chat report formats (bench/chat.py, ChatOutput in runner.py): renderers on synthetic results, the omp stats
line, compact collapsing, stderr-only live line, colors, tables after the transcript, mock end-to-end runs and
--state re-display. No server except the in-process mock on 127.0.0.1.

    python3 -m unittest discover -s tests
"""

import copy
import io
import json
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import isolation  # noqa: E402,F401
from test_replay import call, entry, synthetic_fixture, write_fixture  # noqa: E402
from test_report import ANSI, TTY, run_main  # noqa: E402
from test_state import SplashLike  # noqa: E402

from bench import chat, colors, compare, report  # noqa: E402
from bench.cases import replay  # noqa: E402


def bash(cmd, **kw):
    return call("bash", json.dumps({"command": cmd, **kw}))


def mini_fixture():
    """main/e0 t0..t2 and subagent S1 t0..t1; the result messages of each call sit in the next entry of its stream."""
    pool = [{"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "<system-reminder>skip me</system-reminder>\nFix the bug in a.py"}]},
            {"role": "assistant", "content": ""},
            {"role": "tool", "tool_call_id": "call_bash", "content": "\n".join("line %d" % i for i in range(15))},
            {"role": "tool", "tool_call_id": "call_edit", "content": "Updated a.py"},
            {"role": "user", "content": "Check the numbers"}]
    edit = call("edit", json.dumps({"path": "src/a.py", "old_string": "x = 1\ny = 2\nz = 3", "new_string": "x = 1\ny = 20\nz = 3\nw = 4"}))
    turns = [
        entry("m0", "main", "main/e0", "turn", 0, 0, 1000, [bash("git status && ls", i="status", timeout=300)], msgs=(0, 1)),
        entry("m1", "main", "main/e0", "turn", 1, 0, 2000, [edit], msgs=(0, 1, 2, 3)),
        entry("s0", "S1", "S1", "turn", 0, 0, 500, [call("read", '{"path":"a.py"}')], msgs=(0, 5)),
        entry("s1", "S1", "S1", "turn", 1, 0, 600, [], msgs=(0, 5, 2)),
        entry("m2", "main", "main/e0", "turn", 2, 0, 2500, [], msgs=(0, 1, 2, 3, 2, 4)),
    ]
    turns[0]["recorded_answer"]["content"] = ""
    return {"suite": "coding", "case": "A", "session_id": "x", "model": "m", "provider": "p", "messages": pool, "turns": turns}


def result(fx, i, calls, content="", **kw):
    e = fx["turns"][i]
    m = {"id": replay.case_id(i + 1, e), "content": content, "tool_calls": calls, "text": content, "ttft_ms": 800,
         "total_ms": 8200, "tokens": 557, "prompt_tokens": 1000, "cached_tokens": 200, "eval_result": "PASS",
         "fail_reason": "", "stream": e["stream"], "kind": e["kind"]}
    m.update(kw)
    return m


class Plain(unittest.TestCase):
    def setUp(self):
        colors.configure(False)
        self.addCleanup(lambda: colors.configure(True))
        self.fx = mini_fixture()
        self.tr = chat.ChatTranscript(width=80)
        self.tr.set_fixture(self.fx)
        self.tr.start_target({"started_at": "2026-11-07T10:00:00+00:00"})

    def rec(self, i):  # a result carrying the recorded answer
        return result(self.fx, i, self.fx["turns"][i]["recorded_answer"]["tool_calls"])


class Numbers(unittest.TestCase):
    def test_format_number_and_duration_like_omp(self):
        self.assertEqual([chat.fmt_number(n) for n in (7, 999, 1000, 1234, 9999, 12345, 123456, 1500000)],
                         ["7", "999", "1K", "1.2K", "10K", "12K", "123K", "1.5M"])
        self.assertEqual([chat.fmt_duration(ms) for ms in (0, 450, 8200, 59950, 65000, 120000, 3 * 3600000 + 120000, 5 * 86400000)],
                         ["0ms", "450ms", "8.2s", "1m00s", "1m05s", "2m00s", "3h02m", "5d00h"])

    def test_stats_line_fields(self):
        m = {"prompt_tokens": 123000, "cached_tokens": 119200, "tokens": 557, "ttft_ms": 1200, "total_ms": 8200}
        line = chat.stats_line(m, datetime(2026, 11, 7, 1, 59, 17), 5 * 3600000 + 3 * 60000)
        self.assertEqual(line, "2026-11-07 01:59:17   Δ 5h03m   3.8K   557   119K   1.2s   67.9/s")

    def test_stats_line_omits_what_omp_omits(self):
        m = {"prompt_tokens": 5000, "cached_tokens": 0, "tokens": 40, "ttft_ms": 0, "total_ms": 50}
        self.assertEqual(chat.stats_line(m, datetime(2026, 1, 2, 3, 4, 5), 50), "2026-01-02 03:04:05   Δ 50ms   5K   40")


class Calls(Plain):
    def test_bash_splits_and_shows_recorded_output(self):
        out = self.tr.case(self.rec(0))
        for needle in ("git status", "  &&", "  ls", "├─── Output", "… (5 earlier lines, showing 10 of 15)", "line 14",
                       "Timeout: 300s"):
            self.assertIn(needle, out)
        self.assertNotIn("line 0\n", out)
        widths = {chat.vlen(ln) for ln in out.splitlines() if ln and ln[0] in "╭│├╰"}
        self.assertEqual(widths, {80})

    def test_single_command_gets_dollar(self):
        out = "\n".join(chat.render_call(bash("make test 2>&1 | tail -40"), "ok", 60))
        self.assertIn("│ $ make test 2>&1 | tail -40", out)

    def test_long_command_wraps_inside_the_box(self):
        out = chat.render_call(bash("echo " + "x" * 200), None, 50)
        self.assertGreater(len(out), 4)
        self.assertEqual({chat.vlen(ln) for ln in out}, {50})

    def test_edit_diff_counts_and_numbers(self):
        out = "\n".join(chat.render_call(self.fx["turns"][1]["recorded_answer"]["tool_calls"][0], "Updated a.py", 80))
        self.assertIn("Edit:  src/a.py ⟨+2/-1⟩", out)
        self.assertRegex(out, r"-2│y = 2")
        self.assertRegex(out, r"\+2│y = 20")
        self.assertRegex(out, r"\+4│w = 4")
        self.assertIn("→ Updated a.py", out)

    def test_write_task_read_todo_grep(self):
        w = "\n".join(chat.render_call(call("write", json.dumps({"path": "n.txt", "content": "a\nb\nc"})), None, 60))
        self.assertIn("Write:  n.txt ⟨+3⟩", w)
        t = "\n".join(chat.render_call(call("task", json.dumps({"i": "spawn", "context": "c", "tasks": [
            {"name": "CodeFalsify", "agent": "reviewer", "task": "Check the census\nmore", "solutionSpace": "s"}]})), None, 80))
        self.assertIn("Task 1 subagent", t)
        self.assertIn("▸ CodeFalsify [reviewer]", t)
        self.assertIn("Check the census", t)
        r = chat.render_call(call("read", json.dumps({"path": "a.py", "offset": 134, "limit": 1})), "x\ny", 60)
        self.assertIn("Read a.py:134-134", r[0])
        self.assertIn("(2 lines)", r[1])
        td = "\n".join(chat.render_call(call("todo", json.dumps({"op": "init", "list": [{"phase": "P1", "items": ["a", "b"]}]})), None, 60))
        self.assertIn("Todo init", td)
        self.assertIn("P1", td)
        g = chat.render_call(call("grep", json.dumps({"pattern": "foo", "path": "src"})), None, 60)
        self.assertIn("Grep ‹foo› src", g[0])

    def test_unparsable_arguments(self):
        out = chat.render_call(call("bash", "{not json"), None, 60)
        self.assertIn("invalid arguments", out[0])

    def test_mismatch_shows_recorded_dimmed(self):
        m = result(self.fx, 0, [bash("git log")])
        out = self.tr.case(m)
        self.assertIn("git log", out)
        self.assertIn("recorded: bash: git status && ls", out)
        self.assertNotIn("Output", out)  # the recorded output belongs to another command

    def test_intent_is_ignored_when_matching(self):
        rec = self.fx["turns"][0]["recorded_answer"]["tool_calls"][0]
        self.assertTrue(chat.same_call(rec, bash("git status && ls", timeout=300)))
        self.assertFalse(chat.same_call(rec, bash("git status && ls", timeout=10)))

    def test_extra_recorded_call_and_no_recorded_call(self):
        out = self.tr.case(result(self.fx, 0, []))
        self.assertIn("recorded, not called by the model: bash: git status && ls", out)
        out = self.tr.case(result(self.fx, 4, [bash("ls")]))  # the recorded answer has no calls
        self.assertIn('recorded: text "ok"', out)  # a one-line preview of the recorded text
        self.fx["turns"][4]["recorded_answer"]["content"] = ""
        out = self.tr.case(result(self.fx, 4, [bash("ls")]))
        self.assertIn("recorded: (no call)", out)


class Stream(Plain):
    def test_first_request_shows_task_and_stream_header(self):
        out = self.tr.case(self.rec(0))
        self.assertIn("── main/e0 · main agent ─", out)
        self.assertIn("❯ Fix the bug in a.py", out)
        self.assertNotIn("skip me", out)
        again = self.tr.case(self.rec(1))
        self.assertNotIn("── main/e0", again)
        self.assertNotIn("Fix the bug", again)

    def test_switch_main_to_subagent_and_back(self):
        self.tr.case(self.rec(0))
        sub = self.tr.case(self.rec(2))
        self.assertIn("── S1 · subagent ─", sub)
        self.assertIn("Check the numbers", sub)
        back = self.tr.case(self.rec(4))
        self.assertIn("── main/e0 · main agent ─", back)
        self.assertNotIn("Fix the bug", back)  # the task is shown once per stream
        self.assertIn("── S1 · subagent", self.tr.case(result(self.fx, 3, [])))

    def test_prose_markdown_and_stats_and_verdict(self):
        text = "# Title\n\nSome **bold** and `code`.\n\n- item one\n- item two\n\n```py\nx = 1\n```"
        out = self.tr.case(result(self.fx, 4, [], content=text, eval_result="WARN", fail_reason="more calls than recorded"))
        self.assertIn(" # Title", out)
        self.assertIn("Some bold and code.", out)
        self.assertIn("- item one", out)
        self.assertIn("```py", out)
        self.assertRegex(out, r"2026-11-07 \d\d:\d\d:\d\d\s+Δ 8.2s\s+800\s+557\s+200\s+0.8s\s+67.9/s\s+WARN\s+R005_main_e0_t002")
        self.assertIn("↳ more calls than recorded", out)

    def test_delta_and_clock_per_stream(self):
        a = self.tr.case(self.rec(0))
        b = self.tr.case(self.rec(2))
        c = self.tr.case(self.rec(1))
        self.assertIn("Δ 8.2s", a)
        self.assertIn("Δ 8.2s", b)  # the subagent's own first request
        self.assertIn("Δ 24.6s", c)  # 2nd main request: three 8.2s requests since the stream began
        self.assertIn("2026-11-07", c)

    def test_skip_and_error_results(self):
        out = self.tr.case(result(self.fx, 0, [], eval_result="SKIP", fail_reason="prompt too big", tokens=0, total_ms=0))
        self.assertIn("SKIP", out)
        self.assertIn("prompt too big", out)

    def test_without_a_fixture_no_outputs_and_no_task(self):
        tr = chat.ChatTranscript(width=80)
        tr.start_target({})
        out = tr.case(self.rec(0))
        self.assertIn("── main_e0 · main agent", out.replace("main/e0", "main_e0"))  # label from the case id
        self.assertIn("git status", out)
        self.assertNotIn("Output", out)
        self.assertNotIn("recorded", out)

    def test_wrong_fixture_is_not_used(self):
        other = mini_fixture()
        other["turns"][0]["label"] = "elsewhere"
        tr = chat.ChatTranscript(width=80)
        tr.set_fixture(other)
        tr.start_target({})
        self.assertNotIn("Output", tr.case(self.rec(0)))


class Compact(Plain):
    def setUp(self):
        super().setUp()
        self.tr = chat.CompactTranscript(width=80)
        self.tr.set_fixture(self.fx)
        self.tr.start_target({})

    def test_consecutive_calls_collapse_into_one_line(self):
        out = [self.tr.case(result(self.fx, 0, [bash("a"), bash("b")])), self.tr.case(result(self.fx, 1, [
            bash("c"), call("read", '{"path":"x"}'), call("read", '{"path":"y"}'), call("read", '{"path":"x"}'),
            call("edit", '{"path":"z","old_string":"a","new_string":"b"}')])), self.tr.case(result(self.fx, 4, [], content="Done."))]
        text = "".join(out)
        self.assertIn("● Ran 3 shell commands, read 2 files, edited 1 file", text)
        self.assertIn("\n● Done.", text)
        self.assertIn("  Fix the bug in a.py", text)
        self.assertLess(text.index("Ran 3"), text.index("Done."))
        self.assertNotIn("Output", text)
        self.assertNotIn("Δ", text)

    def test_prose_is_full_width_and_calls_flush_around_it(self):
        self.tr.case(result(self.fx, 0, [bash("a")]))
        text = self.tr.case(result(self.fx, 1, [bash("b")], content="Looking now."))
        self.assertEqual(text.count("Ran 1 shell command"), 1)
        self.assertRegex(text, r"(?m)^● Looking now\.$")
        self.assertEqual(self.tr.finish().strip(), "⎿ Ran 1 shell command")  # tucked under the message

    def test_subagent_launch_and_stream_switch_flushes(self):
        task = call("task", json.dumps({"tasks": [{"name": "Falsify", "agent": "x", "task": "Check it", "solutionSpace": "s"}]}))
        text = self.tr.case(result(self.fx, 0, [bash("a"), task]))
        self.assertIn("Launched subagent Falsify: Check it", text)
        self.tr.case(result(self.fx, 1, [bash("b")]))
        sw = self.tr.case(result(self.fx, 2, [call("read", '{"path":"p"}')]))
        self.assertLess(sw.index("Ran 1 shell command"), sw.index("── S1 · subagent"))

    def test_verdict_marker_only_for_warn_and_fail(self):
        ok = self.tr.case(result(self.fx, 0, [bash("a")]))
        self.assertNotIn("PASS", ok)
        warn = self.tr.case(result(self.fx, 1, [bash("b")], eval_result="WARN", fail_reason="differs"))
        self.assertIn("⚠ R002_main_e0_t001 WARN: differs", warn)
        self.assertIn("Ran 2 shell commands", warn)  # flushed before the marker
        fail = self.tr.case(result(self.fx, 4, [], eval_result="FAIL", fail_reason="bad"))
        self.assertIn("✗ R005_main_e0_t002 FAIL: bad", fail)

    def test_todo_and_wait_phrases(self):
        self.tr.case(result(self.fx, 0, [call("todo", '{"op":"view"}'), call("wait", "{}")]))
        self.assertIn("Updated todo list, waited for background work", self.tr.finish())


class Live(unittest.TestCase):
    def setUp(self):
        report.configure("chat", io.StringIO())  # no colors, no icons
        self.addCleanup(lambda: report.configure("tui", io.StringIO()))

    def test_label_and_phases(self):
        fx = mini_fixture()
        tc = replay.make_case(fx, 2, fx["turns"][1])
        self.assertEqual(chat.live_label(tc), "R002 main/e0 · 2k ctx")
        self.assertEqual(chat.live_line("L", {"start": 100.0}, 112.4), "L · prefill 12s")
        self.assertEqual(chat.live_line("L", {"start": 100.0, "first": 101.2, "down": 480}, 111.2),
                         "L · TTFT 1.2s · 480 tok · 48 tok/s · 11s")
        self.assertEqual(chat.live_line("L", None, 1.0), "L")


class Outputs(unittest.TestCase):
    """Whole runs through replay_bench.main with the mock server."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        fx = synthetic_fixture()
        for e in fx["turns"]:
            for c in (e["recorded_answer"]["tool_calls"] or []):
                if c["function"]["name"] == "bash":
                    c["function"]["arguments"] = '{"command":"echo hi"}'
        fx["messages"].append({"role": "tool", "tool_call_id": "call_bash", "content": "hi"})
        fx["turns"][2]["messages"] = [0, 1, 2, 4]  # main_turn_1 carries the result of main_turn_0's bash call
        self.fx = fx
        self.fixture = write_fixture(fx, self.tmp.name)
        self.addCleanup(lambda: report.configure("tui", io.StringIO()))
        self.behaviour = lambda payload: {"text": "A summary."} if payload.get("tool_choice") == "none" else \
            {"text": "Thinking.", "calls": [("bash", '{"command":"echo hi"}')]}

    def run_replay(self, fmt, *extra, **kw):
        with SplashLike(self.behaviour) as srv:
            return run_main("--suite", "coding", "--fixture", self.fixture, "--endpoint", srv.url, "--no-save-state",
                            "--report-format", fmt, *extra, **kw)

    def test_chat_run(self):
        out, err = self.run_replay("chat")
        for needle in ("── main/e0 · main agent", "❯ task", "│ $ echo hi", "├─── Output", "│ hi ", "Thinking.",
                       "PASS  R001_main_e0_t000", "── S1 · subagent", "recorded: read:", "── main/c0 · compaction",
                       "A summary.", "Δ ", "Metrics"):
            self.assertIn(needle, out)
        self.assertFalse(ANSI.search(out))
        self.assertNotIn("Discovery", out)
        self.assertIn("Discovery", err)  # discovery chatter is stderr, stdout is the transcript
        self.assertEqual(err.count("\r"), 0)
        self.assertLess(out.index("PASS  R001_main_e0_t000"), out.index("┌"))  # tables after the transcript
        self.assertIn("By context", out)

    def test_chat_compact_run(self):
        out, err = self.run_replay("chat-compact")
        self.assertIn("⎿ Ran 1 shell command", out)
        self.assertIn("A summary.", out)
        self.assertIn("  task", out)
        self.assertNotIn("╭", out)
        self.assertNotIn("Δ", out.split("┌")[0])
        self.assertFalse(ANSI.search(out))
        self.assertLess(out.index("Ran 1 shell command"), out.index("┌"))

    def test_live_line_goes_to_stderr_only_and_is_cleared(self):
        out, err = self.run_replay("chat-compact", stdout=io.StringIO(), stderr=TTY())
        # stderr is a fake TTY: colored (whatever stdout is) and icons stand in for "ctx"
        self.assertRegex(re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", err), r"R001 main/e0 · .*1k/262k")
        self.assertIn("\r\x1b[K", err)
        self.assertNotRegex(out, "[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]")
        self.assertNotIn("\r", out)
        self.assertNotRegex(out, r"prefill \d+s|TTFT \d")

    def test_colors_only_on_a_terminal(self):
        out, _ = self.run_replay("chat", "--no-mdcat", stdout=TTY())
        self.assertTrue(ANSI.search(out))
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            out, _ = self.run_replay("chat", "--no-mdcat", stdout=TTY())
        self.assertFalse(ANSI.search(out))

    def test_tables_are_markdown_through_mdcat_after_the_transcript(self):
        calls = []
        fake = lambda cmd, input, text: calls.append((cmd, input)) or mock.Mock(returncode=0)
        with mock.patch.object(compare.shutil, "which", return_value="/x/mdcat"), \
                mock.patch.object(compare.subprocess, "run", side_effect=fake):
            out, _ = self.run_replay("chat", stdout=TTY())
        self.assertTrue(calls)
        self.assertTrue(all(c == ["mdcat", "--full-width"] for c, _ in calls))
        self.assertTrue(any("| Metric" in i for _, i in calls))
        self.assertIn("R001_main_e0_t000", out)  # the transcript itself is not piped
        self.assertNotIn("┌", out)
        self.assertFalse(any("R001_main_e0_t000 PASS" in i for _, i in calls))

    def test_tables_are_tui_without_mdcat_or_without_a_terminal(self):
        with mock.patch.object(compare.shutil, "which", return_value="/x/mdcat"):
            out, _ = self.run_replay("chat", stdout=io.StringIO())
            self.assertIn("┌", out)
            out, _ = self.run_replay("chat", "--no-mdcat", stdout=TTY())
            self.assertIn("┌", out)
        with mock.patch.object(compare.shutil, "which", return_value=None):
            out, _ = self.run_replay("chat", stdout=TTY())
            self.assertIn("┌", out)

    def test_dry_run_uses_the_tables(self):
        out, _ = run_main("--suite", "coding", "--fixture", self.fixture, "--dry-run", "--report-format", "chat")
        self.assertIn("┌", out)
        self.assertIn("R001_main_e0_t000", out)
        calls = []
        with mock.patch.object(compare.shutil, "which", return_value="/x/mdcat"), \
                mock.patch.object(compare.subprocess, "run", side_effect=lambda c, input, text: calls.append(input) or mock.Mock(returncode=0)):
            run_main("--suite", "coding", "--fixture", self.fixture, "--dry-run", "--report-format", "chat-compact", stdout=TTY())
        self.assertTrue(any("| id" in i for i in calls))

    def test_state_redisplay_with_and_without_the_fixture(self):
        path = os.path.join(self.tmp.name, "s.jsonl")
        with SplashLike(self.behaviour) as srv:
            run_main("--suite", "coding", "--fixture", self.fixture, "--endpoint", srv.url, "--state-file", path)
        for fmt in ("chat", "chat-compact"):
            with mock.patch.object(chat, "FIXTURE_DIRS", [self.tmp.name]):
                out, err = run_main("--state", path, "--report-format", fmt, "--no-mdcat")
            self.assertIn("Thinking.", out)
            self.assertIn("── S1 · subagent", out)
            self.assertIn("Metrics", out)
            if fmt == "chat":
                self.assertIn("├─── Output", out)
                self.assertIn("❯ task", out)
                self.assertRegex(out, r"(PASS|WARN|FAIL)\s+R003_main_e0_t001")
        with mock.patch.object(chat, "FIXTURE_DIRS", [os.path.join(self.tmp.name, "nowhere")]):
            out, _ = run_main("--state", path, "--report-format", "chat", "--no-mdcat")
        self.assertIn("Thinking.", out)
        self.assertIn("│ $ echo hi", out)
        self.assertNotIn("Output", out.split("Metrics")[0])
        self.assertNotIn("❯", out)

    def test_a_fixture_with_another_hash_is_not_used(self):
        path = os.path.join(self.tmp.name, "s.jsonl")
        with SplashLike(self.behaviour) as srv:
            run_main("--suite", "coding", "--fixture", self.fixture, "--endpoint", srv.url, "--state-file", path)
        other = copy.deepcopy(self.fx)
        other["turns"][0]["recorded_answer"]["content"] = "changed"
        d = os.path.join(self.tmp.name, "other")
        os.makedirs(d)
        write_fixture(other, d)
        with mock.patch.object(chat, "FIXTURE_DIRS", [d]):
            out, _ = run_main("--state", path, "--report-format", "chat", "--no-mdcat")
        self.assertNotIn("❯", out)


class Help(unittest.TestCase):
    def test_help_and_choices(self):
        self.assertEqual(report.FORMATS, ("tui", "markdown", "chat", "chat-compact"))
        with self.assertRaises(SystemExit):
            run_main("--report-format", "nope")


if __name__ == "__main__":
    unittest.main()


class TaskListTest(unittest.TestCase):
    def test_shapes(self):
        from bench.chat import task_list
        spec = {"name": "F1", "agent": "a", "task": "x"}
        self.assertEqual(task_list({"tasks": [spec, "junk"]}), [spec])
        self.assertEqual(task_list({"tasks": json.dumps([spec])}), [spec])
        self.assertEqual(task_list({"tasks": json.dumps([spec]) + "]"}), [spec])  # stray bracket after the list
        self.assertEqual(task_list({"context": "c", **spec}), [{"context": "c", **spec}])  # flattened spec
        self.assertEqual(task_list({"tasks": "not json"}), [])
        self.assertEqual(task_list(None), [])
