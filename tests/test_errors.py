"""User-facing errors are one line on stderr and exit 2 (no traceback unless REPLAY_BENCH_DEBUG=1); --compare-state
only between runs of the same suite and fixture; the baseline presentation (totals rows, baseline run in the perf
tables, verdict changes); chat-compact names for Qwen Code tools. No real server.

    python3 -m unittest tests.test_errors
"""

import io
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401
from test_polish import Base, run_code  # noqa: E402
from test_replay import synthetic_fixture, write_fixture  # noqa: E402
from test_report import write_state  # noqa: E402

import replay_bench  # noqa: E402
from bench import chat, report, runner  # noqa: E402


def read_records(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


class OneLine(Base):
    def assert_user_error(self, argv, needle):
        out, err, code = run_code(*argv)
        self.assertEqual(code, 2, (argv, err))
        self.assertNotIn("Traceback", err)
        self.assertIn(needle, err)
        errors = [l for l in err.splitlines() if l.strip() and not l.startswith("warning: ")]
        self.assertEqual(len(errors), 1, err)
        return err

    def test_missing_state_file(self):
        self.assert_user_error(["--state", os.path.join(self.tmp.name, "nope.jsonl")], "state file not found")

    def test_bare_state_name_is_looked_up_in_states(self):
        d = os.path.join(self.tmp.name, "states")
        os.mkdir(d)
        write_state(os.path.join(d, "mine.jsonl"), "m")
        with mock.patch.object(replay_bench, "STATES_DIR", d):
            out, err, code = run_code("--state", "mine.jsonl")
            self.assertEqual((code, err), (0, ""))
            self.assertIn("R002_main_e0_t002", out)
            self.assert_user_error(["--state", "mine.jsn"], "did you mean")

    def test_not_a_state_file(self):
        junk = os.path.join(self.tmp.name, "junk.jsonl")
        with open(junk, "w") as f:
            f.write("this is not json\n")
        self.assert_user_error(["--state", junk], "is not a state file")
        with open(junk, "w") as f:  # torn first line: no header at all
            f.write('{"type": "ru')
        self.assert_user_error(["--state", junk], "is not a state file")
        open(junk, "w").close()
        self.assert_user_error(["--state", junk], "is not a state file")

    def test_missing_fixture(self):
        self.assert_user_error(["--suite", "coding", "--fixture", "/nowhere/fx.json.gz", "--endpoint", "http://127.0.0.1:9/v1"],
                               "cannot read fixture")
        self.assert_user_error(["--suite", "coding", "--fixture", "/nowhere/fx.json.gz", "--dry-run"], "cannot read fixture")

    def test_no_models_reachable(self):
        fx = write_fixture(synthetic_fixture(), self.tmp.name)
        _, err, code = run_code("--suite", "coding", "--fixture", fx, "--endpoint", "http://127.0.0.1:9/v1", "--no-save-state")
        self.assertEqual(code, 2)
        self.assertIn("error: no models to benchmark", err)
        self.assertNotIn("Traceback", err)

    def test_argparse_errors_exit_2(self):
        for argv in (["--suite", "nine"], ["--state", self.a, "--state-file", "x"], ["--test", "X", "--suite", "coding"]):
            self.assertEqual(run_code(*argv)[2], 2)

    def test_compare_state_with_chat_formats(self):
        for fmt in ("chat", "chat-compact"):
            _, err, code = run_code("--state", self.a, "--compare-state", self.a, "--report-format", fmt)
            self.assertEqual(code, 2)
            self.assertIn("--compare-state is not supported with --report-format %s" % fmt, err)

    def test_compare_state_of_another_suite(self):
        other = os.path.join(self.tmp.name, "j.jsonl")
        write_state(other, "m")
        recs = read_records(other)
        recs[0]["suite"] = "judge"
        with open(other, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in recs))
        err = self.assert_user_error(["--state", self.a, "--compare-state", other], "the baseline")
        self.assertIn("is suite judge, this run is coding", err)

    def test_compare_state_of_another_fixture(self):
        other = os.path.join(self.tmp.name, "f.jsonl")
        write_state(other, "m")
        for path, sha in ((self.a, "a" * 64), (other, "b" * 64)):
            recs = read_records(path)
            recs[0]["fixture"] = {"file": "x", "sha256": sha}
            with open(path, "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in recs))
        self.assert_user_error(["--state", self.a, "--compare-state", other], "another fixture")

    def test_debug_env_shows_the_traceback(self):
        with mock.patch.dict(os.environ, {"REPLAY_BENCH_DEBUG": "1"}):
            with self.assertRaises(replay_bench.UserError):
                replay_bench.main(["--state", os.path.join(self.tmp.name, "nope.jsonl")])

    def test_unexpected_exception_is_one_line_too(self):
        with mock.patch.object(replay_bench, "run_benchmark", side_effect=RuntimeError("odd")):
            _, err, code = run_code("--state", self.a)
        self.assertEqual(code, 1)
        self.assertIn("RuntimeError: odd", err)
        self.assertNotIn("Traceback", err)

    def test_http_error_becomes_a_failed_case(self):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                body = b'{"error": {"message": "context overflow: 300000 > 262144"}}'
                self.send_response(400)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    return

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        url = "http://127.0.0.1:%d/v1/chat/completions" % httpd.server_address[1]
        m = runner.query_model(url, "m", {"id": "X", "system": "s", "prompt": "p"})
        self.assertIn("HTTP 400", m["error"])
        self.assertIn("context overflow", m["error"])


class Baseline(Base):
    def setUp(self):
        super().setUp()
        self.slow = os.path.join(self.tmp.name, "slow.jsonl")
        write_state(self.slow, "ISTA/Qwen-Flash-GSQ-RCO", 20.0)
        recs = read_records(self.slow)
        recs[3]["eval_result"], recs[3]["fail_reason"] = "WARN", "tools a vs recorded b"  # one verdict differs
        with open(self.slow, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in recs))

    def md(self, *extra):
        out, _, code = run_code("--state", self.a, "--compare-state", self.slow, "--report-format", "markdown",
                                "--no-mdcat", *extra)
        self.assertEqual(code, 0)
        return out

    def test_totals_rows_instead_of_interleaved_rows(self):
        out = self.md()
        table = [l for l in out.splitlines() if l.startswith("| ")]
        self.assertEqual(sum("R000_main_e0_t000" in l for l in table), 1)  # no baseline row per case
        self.assertNotRegex(out, r"\*\*R\d+_main_e0_t\d+\*[^*]")
        self.assertRegex(out, r"\| \*\*Baseline\*\*")
        self.assertRegex(out, r"\| \*\*This run\*\*")
        self.assertRegex(out, r"\| \*\*Δ\*\*")
        self.assertNotIn("…", "\n".join(l for l in table if "Δ" in l))  # never truncated

    def test_metrics_and_perf_have_the_baseline(self):
        out = self.md()
        self.assertRegex(out, r"\| Metric\s*\|\s*started \d{4}-\d\d-\d\d \d\d:\d\d \|\s*This run \|\s*Δ \|")
        self.assertIn("By context", out)
        self.assertRegex(out, r"\| 0-32k\s*\|\s*1 \|")
        perf = out.split("## Performance")[1]
        self.assertRegex(perf, r"\| < 1k\s*\| \*\*base\*\*\s*\|")
        self.assertRegex(perf, r"\| < 1k\s*\| \*\*this run\*\*\s*\|")

    def test_verdict_changes(self):
        out = self.md()
        self.assertIn("### Verdict changes", out)
        self.assertIn("- **R002_main_e0_t002** WARN → PASS", out)

    def test_tui_totals_and_no_interleaving(self):
        out, _, _ = run_code("--state", self.a, "--compare-state", self.slow)
        self.assertEqual(out.count("R000_main_e0_t000"), 1)
        for word in ("Baseline", "This run", "Δ", "Verdict changes"):
            self.assertIn(word, out)
        self.assertNotIn("+272.…", out)

    def test_legend_mentions_delta_only_with_a_baseline(self):
        with_b = self.md()
        plain, _, _ = run_code("--state", self.a, "--report-format", "markdown", "--no-mdcat")
        self.assertIn("Δ = relative change against the baseline", with_b)
        self.assertNotIn("change against the baseline", plain)
        self.assertIn("case table: Decode / Prefill are the server's own tok/s", plain)
        self.assertIn("client clock", plain)

    def test_state_redisplay_has_no_live_lines_on_stderr(self):
        out, err, _ = run_code("--state", self.a, "--report-format", "markdown", "--no-mdcat")
        self.assertEqual(err, "")


class CompactNames(unittest.TestCase):
    def setUp(self):
        report.configure("chat", io.StringIO())
        self.addCleanup(lambda: report.configure("tui", io.StringIO()))

    def line(self, *calls):
        tr = chat.CompactTranscript(width=100)
        tr.pending = [{"function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]
        return tr.summary_line()

    def test_qwen_code_tools_are_known(self):
        self.assertEqual(self.line(("run_shell_command", {"command": "ls"})), "Ran 1 shell command")
        self.assertEqual(self.line(("read_file", {"file_path": "/a"}), ("read_file", {"file_path": "/b"})), "Read 2 files")
        self.assertEqual(self.line(("todo_write", {"todos": []})), "Updated todo list")
        self.assertEqual(self.line(("skill", {"skill": "ingredient-match"})), "Loaded skill ingredient-match")
        self.assertEqual(self.line(("grep_search", {"pattern": "x"}), ("glob", {"pattern": "*"})), "Searched 2 times")
        self.assertEqual(self.line(("edit", {"file_path": "/a"}), ("write_file", {"file_path": "/b"})), "Edited 1 file, wrote 1 file")

    def test_unknown_tools_are_named(self):
        self.assertEqual(self.line(("fdc_get", {}), ("fdc_get", {}), ("siblings", {})), "Called fdc_get ×2, siblings")


if __name__ == "__main__":
    unittest.main()
