"""Regression tests for the pre-publication review items (secrets in headers, unfinished streams, null usage, ...).

    python3 -m unittest tests.test_review_fixes
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import add_ref  # noqa: E402
import isolation  # noqa: E402,F401

from bench import (  # noqa: E402
    compare,  # noqa: E402
    runner,
    state,
)
from bench import results as R  # noqa: E402
from bench.telemetry import init_acc, normalize  # noqa: E402
from tools import mock_server  # noqa: E402

CASE = {"id": "X", "system": "s", "prompt": "p"}


class SanitizeArgvTest(unittest.TestCase):
    def test_api_key_is_dropped_in_both_forms(self):
        self.assertEqual(state.sanitize_argv(["--api-key", "SECRET", "--suite", "smoke"]), ["--suite", "smoke"])
        self.assertEqual(state.sanitize_argv(["--suite", "smoke", "--api-key=SECRET"]), ["--suite", "smoke"])

    def test_state_file_relative_when_under_cwd_else_basename(self):
        with tempfile.TemporaryDirectory() as d:
            d = os.path.realpath(d)
            inside = os.path.join(d, "states", "a.jsonl")
            self.assertEqual(state.sanitize_argv(["--state-file", inside], cwd=d), ["--state-file", "states/a.jsonl"])
            self.assertEqual(state.sanitize_argv(["--state-file=/Users/someone/x/b.jsonl"], cwd=d),
                             ["--state-file=b.jsonl"])
            self.assertEqual(state.sanitize_argv(["--state-file", "rel.jsonl"], cwd=d), ["--state-file", "rel.jsonl"])

    def test_none_stays_none(self):
        self.assertIsNone(state.sanitize_argv(None))

    def test_header_and_resume_record_use_it(self):
        target = {"backend": "x", "name": "m", "url": "http://127.0.0.1:1/v1"}
        h = state.make_header("smoke", target, argv=["--api-key", "K", "--suite", "smoke"])
        self.assertEqual(h["argv"], ["--suite", "smoke"])
        self.assertEqual(state.resume_record(target, ["--api-key=K"])["argv"], [])


class AddRefApiKeyTest(unittest.TestCase):
    def test_refuses_header_with_api_key(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "x.jsonl")
            with open(src, "w") as f:
                f.write(json.dumps({"type": "run", "endpoint": "http://127.0.0.1:1/v1",
                                    "argv": ["--api-key", "K"]}) + "\n")
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                rc = add_ref.main(["--hosts", os.path.join(d, "h.json"), "--refs-dir", os.path.join(d, "r"),
                                   "--dry-run", src])
            self.assertEqual(rc, 1)
            self.assertIn("--api-key", err.getvalue())


class IndependentCasesAndCacheTest(unittest.TestCase):
    def test_smoke_cases_are_not_one_stream(self):
        rs = [{"id": "a", "prompt_tokens": 500, "ttft_ms": 100}, {"id": "b", "prompt_tokens": 300, "ttft_ms": 100}]
        R.fill_prev_prompts(rs)
        self.assertNotIn("prev_prompt_tokens", rs[1])
        self.assertEqual(R.new_tokens(rs[1]), 300)

    def test_replay_requests_still_chain(self):
        rs = [{"id": "a", "stream": "main", "epoch": 0, "prompt_tokens": 500},
              {"id": "b", "stream": "main", "epoch": 0, "prompt_tokens": 800}]
        R.fill_prev_prompts(rs)
        self.assertEqual(R.new_tokens(rs[1]), 300)

    def test_no_client_prefill_speed_for_a_fully_cached_request(self):
        r = {"prompt_tokens": 5000, "cached_tokens": 4995, "ttft_ms": 200}
        self.assertEqual(R.request_prefill_speed(r), (None, False))
        r = {"prompt_tokens": 5000, "cached_tokens": 4000, "ttft_ms": 1000}
        self.assertEqual(R.request_prefill_speed(r), (1000.0, True))
        from bench import perf
        self.assertIs(perf.MIN_NEW_TOKENS, R.MIN_NEW_TOKENS)


class DisplayOrderTest(unittest.TestCase):
    def test_state_display_keeps_the_run_order(self):
        from test_polish import run_code
        from test_state import TARGET, metrics
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.jsonl")
            w = state.StateWriter(path)
            w.write(state.make_header("sanity", TARGET, argv=["--suite", "sanity"]))
            for i, cid in enumerate(("TC01_X", "SC05_Y", "AA00_Z")):
                w.write(state.result_record(metrics(i, id=cid)))
            w.close()
            out, err, code = run_code("--state", path)
        self.assertEqual(code, 0, err)
        pos = [out.index(c) for c in ("TC01_X", "SC05_Y", "AA00_Z")]
        self.assertEqual(pos, sorted(pos))


class FixturesDirTest(unittest.TestCase):
    def test_default_fixtures_resolve_under_fixtures_dir(self):
        from unittest import mock

        from bench import paths, presets
        from bench.cases import replay
        with tempfile.TemporaryDirectory() as d, mock.patch.object(paths, "FIXTURES_DIR", d):
            for suite in ("coding", "judge"):
                self.assertEqual(os.path.dirname(replay.default_fixture(suite)), d)
            seen = []
            with mock.patch.object(replay, "load_fixture", side_effect=lambda p: seen.append(p) or {}):
                presets._FIXTURE_CACHE.clear()
                presets._get_fixture("coding")
                presets._FIXTURE_CACHE.clear()
            self.assertEqual(seen, [os.path.join(d, "conv-3.json.gz")])


class ModelLabelTest(unittest.TestCase):
    SHA = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"

    def entry(self):
        return compare.entry("localhost:8134", self.SHA, {}, model_info={"label": "mlx-community/Qwen3.8-27B-4bit"})

    def test_markdown_heading_uses_the_label(self):
        blocks = compare.run_blocks(self.entry(), titles=True)
        self.assertIn("## mlx-community/Qwen3.8-27B-4bit", blocks)
        self.assertNotIn("## " + self.SHA, blocks)

    def test_comparison_names_use_the_label(self):
        e = self.entry()
        self.assertEqual(compare.labels([e, dict(e, name="other-model", model_info=None)]),
                         ["mlx-community/Qwen3.8-27B-4bit", "other-model"])


class TotalsConsistencyTest(unittest.TestCase):
    def test_total_row_excludes_a_paused_request_like_metrics(self):
        ok = {"id": "a", "eval_result": "PASS", "ttft_ms": 1000, "total_ms": 1000, "native_ttft_ms": 1000,
              "prompt_tokens": 10, "tokens": 5}
        paused = dict(ok, id="b", ttft_ms=601000, total_ms=601000)
        tot = R.run_totals([ok, paused])
        self.assertEqual(tot["time"], 1.0)
        self.assertEqual(tot["ttft"], 1.0)
        self.assertEqual(compare.stats([ok, paused])["wall_s"], 1.0)

    def test_run_totals_drops_errors(self):
        ok = {"id": "a", "eval_result": "PASS", "ttft_ms": 1000, "total_ms": 2000, "prompt_tokens": 10, "tokens": 5}
        bad = {"id": "b", "eval_result": "FAIL", "error": "boom", "total_ms": 9000, "prompt_tokens": 99, "tokens": 7}
        tot = R.run_totals([ok, bad])
        self.assertEqual((tot["time"], tot["in"], tot["out"]), (2.0, 10, 5))


class BrokenPipeTest(unittest.TestCase):
    def test_piping_into_head_is_quiet(self):
        import subprocess
        cmd = "%s %s --dry-run --preset small | head -n 1" % (
            sys.executable, os.path.join(ROOT, "replay_bench.py"))
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        for bad in ("Traceback", "BrokenPipe", "Broken pipe", "Exception ignored"):
            self.assertNotIn(bad, p.stderr)
        self.assertEqual(len(p.stdout.splitlines()), 1)

    def test_main_exits_zero_on_broken_pipe(self):
        from unittest import mock

        import replay_bench
        with mock.patch.object(replay_bench, "_main", side_effect=BrokenPipeError), \
                mock.patch("os.dup2"), self.assertRaises(SystemExit) as cm:
            replay_bench.main([])
        self.assertEqual(cm.exception.code, 0)


class CliWordingTest(unittest.TestCase):
    def run_cli(self, *argv, env=None):
        import subprocess
        e = dict(os.environ, **(env or {}))
        p = subprocess.run([sys.executable, os.path.join(ROOT, "replay_bench.py"), *argv], capture_output=True,
                           text=True, env=e)
        return p.stdout, p.stderr, p.returncode

    def test_help_names_sanity_and_the_environment(self):
        out, _, code = self.run_cli("--help")
        self.assertEqual(code, 0)
        flat = " ".join(out.split())
        self.assertIn("not with: smoke and sanity suites", flat)
        self.assertNotIn("not with: smoke suite", flat)
        for name in ("REPLAY_BENCH_STALL_SECONDS", "REPLAY_BENCH_SERVER_WAIT", "log every request URL"):
            self.assertIn(name, flat)
        self.assertTrue(out.startswith("usage: replay-bench [options]"), out[:80])

    def test_sanity_with_turns_names_both_suites(self):
        _, err, code = self.run_cli("--suite", "sanity", "--turns", "1-2")
        self.assertEqual(code, 2)
        self.assertIn("smoke or sanity", err)

    def test_version(self):
        from bench.utils import version_string
        out, _, code = self.run_cli("--version")
        self.assertEqual(code, 0)
        self.assertIn(version_string(), out)
        self.assertRegex(out, r"\d+\.\d+\.\d+")

    def test_dry_run_with_an_empty_selection_is_the_same_error(self):
        _, err, code = self.run_cli("--suite", "coding", "--turns", "900-901", "--dry-run")
        self.assertEqual(code, 2)
        self.assertIn("the selection is empty", err)

    def test_misspelled_state_name_suggests_close_files(self):
        from bench.resume import suggest_names
        names = ["mlx-dspark_x_coding-quickstart_t0.6_gguf_20261109.jsonl", "other_smoke_t0.0.jsonl"]
        self.assertEqual(suggest_names("coding-quickstrat_t0.6_gguf", names), names[:1])
        self.assertEqual(suggest_names("zzz", names), [])


class CodeNamesTest(unittest.TestCase):
    def test_no_leading_space_in_markdown_tool_names(self):
        self.assertEqual(R.code_names("bash, read"), "`bash`, `read`")
        self.assertEqual(R.code_names("bash,read"), "`bash`, `read`")
        self.assertEqual(R.code_names("none"), "none")


class WarmupFailureTest(unittest.TestCase):
    def test_failed_warmup_warns_once_and_columns_come_from_the_first_result(self):
        from unittest import mock

        from test_polish import run_code

        from bench import transport
        httpd = mock_server.make_server("generic")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        url = "http://127.0.0.1:%d/v1" % httpd.server_address[1]
        real = runner.query_model

        def fake(u, model, tc, **kw):
            if kw.get("max_tokens") == 256:
                return transport._error("boom", "server")
            return real(u, model, tc, **kw)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(runner, "query_model", side_effect=fake):
            out, err, code = run_code("--suite", "smoke", "--endpoint", url, "--test", "TC01_GIT_COMMIT",
                                      "--state-file", os.path.join(d, "s.jsonl"))
        self.assertEqual(code, 0, err)
        warnings = [x for x in err.splitlines() if x.startswith("warning: warmup")]
        self.assertEqual(len(warnings), 1, err)
        self.assertIn("boom", warnings[0])
        self.assertIn("Prefill", out)  # the In / Out / C% / Prefill columns: the first real result reported counts


class DivergenceToolTest(unittest.TestCase):
    def test_state_names_and_checks_resolve_from_any_directory(self):
        from bench import paths
        from tools import replay_divergence
        refs = sorted(f for f in os.listdir(paths.REFS_DIR) if f.endswith(".jsonl"))
        self.assertTrue(refs)
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                runs = replay_divergence.load_runs([refs[0][:-len(".jsonl")]])  # a bare name, not a path
                self.assertTrue(runs)
                checks = replay_divergence.load_checks(runs)
            finally:
                os.chdir(old)
        self.assertTrue(checks, "conv-3.checks.json was not found from another directory")


class RawServer:
    """A server answering every chat request with the given SSE lines (bytes), then closing the connection."""

    def __init__(self, test, lines):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for line in lines:
                    self.wfile.write(line + b"\n\n")

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        test.addCleanup(httpd.server_close)
        test.addCleanup(httpd.shutdown)
        self.url = "http://127.0.0.1:%d/v1/chat/completions" % httpd.server_address[1]


def sse(obj):
    return b"data: " + json.dumps(obj).encode()


class UnfinishedStreamTest(unittest.TestCase):
    def test_mock_stream_cut_short_is_a_server_error(self):
        httpd = mock_server.make_server("generic", drop_stream_end=True)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        url = "http://127.0.0.1:%d/v1/chat/completions" % httpd.server_address[1]
        m = runner.query_model(url, "m", CASE)
        self.assertIn("stream ended without finish_reason", str(m["error"]))
        self.assertEqual(m["error"].kind, "server")

    def test_complete_streams_still_pass(self):
        httpd = mock_server.make_server("generic")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        m = runner.query_model("http://127.0.0.1:%d/v1/chat/completions" % httpd.server_address[1], "m", CASE)
        self.assertFalse(m.get("error"), m)

    def test_done_without_finish_reason_is_accepted(self):
        srv = RawServer(self, [sse({"choices": [{"index": 0, "delta": {"content": "hi"}}]}), b"data: [DONE]"])
        self.assertFalse(runner.query_model(srv.url, "m", CASE).get("error"))


class NullUsageTest(unittest.TestCase):
    def test_normalize_treats_null_counts_as_absent(self):
        acc = init_acc()
        normalize({"usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": 5}}, acc,
                  current_time=1.0, start_time=0.0, ttft=0.5)
        self.assertEqual(acc["exact_tokens"], 0)
        self.assertEqual(acc["prompt_tokens"], 0)

    def test_a_run_with_null_completion_tokens_completes(self):
        srv = RawServer(self, [
            sse({"choices": [{"index": 0, "delta": {"content": "hello there"}}]}),
            sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            sse({"choices": [], "usage": {"prompt_tokens": None, "completion_tokens": None}}),
            b"data: [DONE]"])
        m = runner.query_model(srv.url, "m", CASE)
        self.assertFalse(m.get("error"), m)
        self.assertGreater(m["tokens"], 0)


if __name__ == "__main__":
    unittest.main()
