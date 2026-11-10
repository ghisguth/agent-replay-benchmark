"""Transport tests: retries, server pauses, stream errors, runaway limits, and api keys.

    python3 -m unittest tests.test_transport
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401
from test_replay import synthetic_fixture, write_fixture  # noqa: E402
from test_state import SplashLike, behaviour  # noqa: E402

import replay_bench  # noqa: E402
from bench import auth, compare, perf, report, runner, state, telemetry, transport  # noqa: E402

CONNECT = "Inference failure: [Errno 61] Connection refused"


def run_code(*argv, stdout=None, stderr=None):
    """(stdout, stderr, exit code) of main(argv); SystemExit is caught."""
    out, err, code = stdout or io.StringIO(), stderr or io.StringIO(), 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            replay_bench.main(list(argv))
        except SystemExit as e:
            code = e.code or 0
    return out.getvalue(), err.getvalue(), code


class Retry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = write_fixture(synthetic_fixture(), self.tmp.name)
        self.path = os.path.join(self.tmp.name, "s.jsonl")
        self.addCleanup(lambda: report.configure("tui", io.StringIO()))
        for name, value in (("SERVER_POLL_S", 0.01), ("SERVER_WAIT_S", 0.5)):
            p = mock.patch.object(transport, name, value)
            p.start()
            self.addCleanup(p.stop)

    def run_with_errors(self, errors_for_case, case="R001_main_e0_t000"):
        """Run the synthetic replay; the first `errors_for_case` real attempts of `case` fail with a connect error."""
        real, left = runner.query_model, [errors_for_case]

        def fake(url, model, tc, *a, **k):
            if tc["id"] == case and k.get("max_tokens") is None and left[0] > 0:
                left[0] -= 1
                return {"error": CONNECT}
            return real(url, model, tc, *a, **k)

        with SplashLike(behaviour) as srv, mock.patch.object(runner, "query_model", side_effect=fake):
            out, err, code = run_code("--suite", "coding", "--fixture", self.fixture, "--endpoint", srv.url,
                                      "--state-file", self.path, "--no-mdcat")
        with open(self.path) as f:
            recs = [json.loads(x) for x in f if x.strip()]
        return {r["id"]: r for r in recs if r["type"] == "result"}, err

    def test_connect_failure_then_success_passes(self):
        res, err = self.run_with_errors(1)
        r = res["R001_main_e0_t000"]
        self.assertEqual(r["eval_result"], "PASS")
        self.assertEqual(r["attempts"], 2)
        self.assertEqual(r["retry_errors"], [CONNECT])
        self.assertIn("server unreachable (%s), waiting for it" % CONNECT, err)
        self.assertIn("attempt 2/3", err)
        self.assertTrue(all("attempts" not in x for i, x in res.items() if i != "R001_main_e0_t000"))

    def test_three_failures_fail_as_server(self):
        res, err = self.run_with_errors(5)
        r = res["R001_main_e0_t000"]
        self.assertEqual(r["eval_result"], "FAIL")
        self.assertEqual(r["fail_reason"], "server: " + CONNECT)
        self.assertEqual(r["attempts"], 3)
        self.assertEqual(r["retry_errors"], [CONNECT, CONNECT])
        self.assertIn("attempt 3/3", err)

    def test_http_400_is_not_retried(self):
        calls = []

        def send():
            calls.append(1)
            return {"error": "HTTP 400 Bad Request: context overflow"}

        m = transport.query_with_retry(send, "http://127.0.0.1:1/v1/chat/completions")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("attempts", m)
        self.assertEqual(m["error"], "HTTP 400 Bad Request: context overflow")

    def test_which_errors_are_transient(self):
        for e in (CONNECT, "timeout", "stalled: no data for 300s after 4 lines", "HTTP 503 Service Unavailable"):
            self.assertTrue(transport.is_transient(e), e)
        for e in ("HTTP 400 Bad Request", "HTTP 500 x", "runaway: over 8192 tokens: abc", "timeout (3600s max duration reached)",
                  "Exception: boom"):
            self.assertFalse(transport.is_transient(e), e)

    def test_wait_gives_up_after_the_limit(self):
        with mock.patch.object(transport, "discover_models", return_value=None):
            self.assertFalse(transport.wait_for_server("http://h:1/v1/chat/completions", limit=0.05))

    def test_ctrl_c_during_the_wait_propagates(self):
        with mock.patch.object(transport, "discover_models", side_effect=KeyboardInterrupt), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                transport.query_with_retry(lambda: {"error": "timeout"}, "http://h:1/v1/chat/completions")


def result(i, ttft, native, total=None, tokens=100):
    p = 8000 + i * 30000
    return state.result_record({
        "id": "R%03d_main_e0_t%03d" % (i, i), "stream": "main", "kind": "turn", "epoch": 0, "text": "ok",
        "ttft_ms": ttft, "total_ms": total or ttft + 3000, "tokens": tokens, "prompt_tokens": p, "cached_tokens": p - 400,
        "native_ttft_ms": native, "native_prefill_tps": 900.0, "native_decode_tps": 30.0, "eval_result": "PASS",
        "fail_reason": ""})


class Pause(unittest.TestCase):
    def test_rule(self):
        self.assertIsNone(perf.server_paused_ms({"ttft_ms": 50000, "native_ttft_ms": 1000}))  # under 60 s
        self.assertEqual(perf.server_paused_ms({"ttft_ms": 257198, "native_ttft_ms": 16586}), 257198 - 16586)
        self.assertIsNone(perf.server_paused_ms({"ttft_ms": 200000, "native_ttft_ms": 120000}))  # under 2 x native
        self.assertIsNone(perf.server_paused_ms({"ttft_ms": 257198, "native_ttft_ms": None}))
        self.assertIsNone(perf.server_paused_ms({"ttft_ms": 257198, "native_ttft_ms": 1, "error": "x"}))

    def make_state(self, d):
        path = os.path.join(d, "s.jsonl")
        t = {"key": "h:1::m", "backend": "h:1", "name": "m", "url": "http://h:1/v1/chat/completions",
             "n_ctx": 262144, "backend_info": {"kind": "splash", "version": "1a2b3c4d", "build": "x", "details": {}}}
        w = state.StateWriter(path)
        w.write(state.make_header("coding", t, request={"temperature": None, "reasoning_effort": None},
                                  fixture_path=None, argv=[], selection=None, started_at="2026-11-07T10:00:00+00:00"))
        for rec in (result(0, 500, 400), result(1, 600, 500), result(2, 257198, 16586, total=433719, tokens=810)):
            w.write(rec)
        w.write(state.end_record({}))
        w.close()
        return path

    def test_paused_case_leaves_client_figures_and_is_noted(self):
        with tempfile.TemporaryDirectory() as d:
            res = [result(0, 500, 400), result(1, 600, 500), result(2, 257198, 16586, total=433719, tokens=810)]
            s = compare.stats(res)
            self.assertEqual(s["requests"], 3)
            self.assertAlmostEqual(s["ttft_p90"], 0.59, places=2)  # the 257 s TTFT is out
            self.assertAlmostEqual(s["wall_s"], 3.5 + 3.6)
            self.assertAlmostEqual(s["client_tps"], 200 / 6.0, places=1)
            self.assertEqual(s["out"], 1010)  # counts keep every request
            self.assertEqual(s["native_dec_p50"], 30.0)  # server-native figures stay
            self.assertEqual(s["paused_note"], "1 request excluded from client-clock figures: the server paused "
                                               "(client TTFT 257s vs server 16.6s)")
            rows = perf.context_rows([("m", res)])
            self.assertEqual(sum(r["requests"] for r in rows), 3)
            self.assertEqual([r["decode_tps"] for r in rows if r["bucket"] == "64-96k"], [None])
            self.assertEqual(sum(r["requests"] for r in perf.append_rows([("m", res)])), 2)
            code_out, _, code = run_code("--state", self.make_state(d), "--no-mdcat")
            self.assertEqual(code, 0)
            self.assertIn("1 request excluded from client-clock figures: the server paused", code_out)
            self.assertIn("PASS", code_out)

    def test_no_note_without_pauses(self):
        s = compare.stats([result(0, 500, 400), result(1, 600, 500)])
        self.assertIsNone(s["paused_note"])

    def test_stall_seconds_env(self):
        from bench.transport import _stall_seconds
        with mock.patch.dict(os.environ, {"REPLAY_BENCH_STALL_SECONDS": "120"}, clear=False):
            self.assertEqual(_stall_seconds(), 120.0)
        env = dict(os.environ)
        env.pop("REPLAY_BENCH_STALL_SECONDS", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(_stall_seconds(), 300.0)


class Stub:
    """Answers every POST with the given SSE chunks (dicts) and [DONE]."""

    def __init__(self, chunks):
        chunks = list(chunks)

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for c in chunks:
                        self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):  # the client stopped reading (a runaway cutoff)
                    pass

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d/v1/chat/completions" % self.srv.server_address[1]

    def __enter__(self):
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()


def send(chunks):
    with Stub(chunks) as s:
        return transport._send(s.url, {"model": "m", "messages": [], "stream": True}, {}, {})


def text_chunk(text):
    return {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": text}}]}


class StreamErrors(unittest.TestCase):
    def test_mtplx_error_chunk(self):
        r = send([text_chunk("par"), {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            "error": {"message": "not enough memory", "type": "server_error", "code": "insufficient_memory"},
            "timings": {"prompt_ms": 0.0, "predicted_ms": 0.0, "prompt_per_second": 0.0,
                        "predicted_per_second": 0.0, "prompt_n": 100, "predicted_n": 5}}])
        self.assertEqual(r, {"error": "server: insufficient_memory: not enough memory"})

    def test_llamacpp_error_only_chunk(self):
        r = send([{"error": {"code": 500, "message": "context overflow", "type": "server_error"}}])
        self.assertEqual(r, {"error": "server: 500: context overflow"})

    def test_error_string_and_missing_fields(self):
        self.assertEqual(send([{"error": "boom"}]), {"error": "server: boom"})
        self.assertEqual(send([{"error": {"type": "oom"}}]), {"error": "server: oom"})
        self.assertEqual(send([{"error": {"foo": 1}}]), {"error": "server: stream error"})

    def test_bare_finish_reason_error(self):
        r = send([{"choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]}])
        self.assertEqual(r, {"error": "server: stream ended with finish_reason error"})

    def test_normal_stream_unaffected(self):
        r = send([text_chunk("hi"), {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}])
        self.assertNotIn("error", r)
        self.assertEqual((r["text"], r["finish_reason"]), ("hi", "stop"))

    def test_not_retried_and_not_double_prefixed(self):
        err = "server: insufficient_memory: x"
        self.assertFalse(transport.is_transient(err))
        calls = []

        def fake():
            calls.append(1)
            return {"error": err}

        m = transport.query_with_retry(fake, "http://x/v1/chat/completions", wait_fn=lambda u: True)
        self.assertEqual((len(calls), m["error"]), (1, err))


class Runaway(unittest.TestCase):
    def test_token_cap_and_time_cap_have_distinct_messages(self):
        chunks = [text_chunk("x" * 100)] * 3
        with Stub(chunks) as s:
            tok = transport._send(s.url, {"stream": True}, {"max_gen_tokens": 10}, {})
            secs = transport._send(s.url, {"stream": True}, {"max_gen_seconds": -1}, {})
        self.assertTrue(tok["error"].startswith("runaway: over 10 tokens"), tok)
        self.assertTrue(secs["error"].startswith("runaway: over -1 s"), secs)
        self.assertEqual((tok["error"].kind, transport.is_transient(tok["error"])), ("runaway", False))

    def test_error_kind_decides_transience(self):
        self.assertTrue(transport.is_transient(transport._error("whatever", "stalled")["error"]))
        self.assertTrue(transport.is_transient(transport._error("HTTP 503 x", "http", 503)["error"]))
        self.assertFalse(transport.is_transient(transport._error("HTTP 400 x", "http", 400)["error"]))


class ApiKey(unittest.TestCase):
    def tearDown(self):
        auth.set_api_key(None)

    def test_configured_key_then_environment_then_dummy(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            self.assertEqual(auth.api_key(), "not-needed")
            os.environ["OPENAI_API_KEY"] = "env-key"
            self.assertEqual(auth.api_key(), "env-key")
            auth.set_api_key("cli-key")
            self.assertEqual(auth.api_key(), "cli-key")


class ZeroTimings(unittest.TestCase):
    def run_chunks(self, *chunks):
        acc = telemetry.init_acc()
        for c in chunks:
            telemetry.normalize(c, acc)
        telemetry.finalize(acc, 1.0, token_count=5)
        return acc

    def test_all_zero_timings_are_missing(self):
        acc = self.run_chunks({"timings": {"prompt_ms": 0.0, "predicted_ms": 0.0, "prompt_per_second": 0.0,
                                           "predicted_per_second": 0.0, "prompt_n": 100, "predicted_n": 5}})
        for k in ("native_ttft_ms", "native_prefill_tps", "native_decode_tps"):
            self.assertIsNone(acc[k], k)
        self.assertEqual((acc["prompt_tokens"], acc["exact_tokens"]), (100, 5))

    def test_zero_mtplx_stats_are_missing(self):
        acc = self.run_chunks({"mtplx_stats": {"ttft_s": 0.0, "prefill_tok_s": 0.0, "decode_tok_s": 0.0}})
        for k in ("native_ttft_ms", "native_prefill_tps", "native_decode_tps"):
            self.assertIsNone(acc[k], k)

    def test_later_real_values_fill_in_and_positive_kept(self):
        acc = self.run_chunks({"timings": {"prompt_ms": 0.0, "predicted_per_second": 0.0}},
                              {"timings": {"prompt_ms": 250.0, "predicted_per_second": 40.0}})
        self.assertEqual((acc["native_ttft_ms"], acc["native_decode_tps"]), (250.0, 40.0))


class TimingsPrefillTest(unittest.TestCase):
    """prompt_per_second means different things per server; the new-token prefill rate must come out right for each."""

    def test_llamacpp_splash_mtplx(self):
        from bench.telemetry import timings_prefill_tps
        # llama.cpp: prompt_n = new tokens, rate over them
        self.assertAlmostEqual(timings_prefill_tps(
            {"cache_n": 17283, "prompt_n": 527, "prompt_ms": 3480.38, "prompt_per_second": 151.42}, 17810, 17283), 151.42)
        # Splash: prompt_n = whole prompt, rate over new tokens
        self.assertAlmostEqual(timings_prefill_tps(
            {"cache_n": 18080, "prompt_n": 18686, "prompt_ms": 3325.27, "prompt_per_second": 182.24}, 18686, 18080), 182.24)
        # MTPLX: rate over the whole prompt -> recomputed from new tokens
        self.assertAlmostEqual(timings_prefill_tps(
            {"prompt_n": 64155, "prompt_ms": 3579.16, "prompt_per_second": 17924.6}, 64155, 63731), 424 / 3.57916, places=2)
        # cold prompt: raw
        self.assertEqual(timings_prefill_tps({"prompt_n": 100, "prompt_ms": 1000, "prompt_per_second": 100.0}, 100, 0), 100.0)

    def test_old_state_recomputed_on_load(self):
        from bench import state
        m = state._metrics_from_record({"type": "result", "id": "R2", "content": "x", "prompt_tokens": 17810,
                                        "cached_tokens": 17283, "native_prefill_tps": 4.48,
                                        "timings": {"cache_n": 17283, "prompt_n": 527, "prompt_ms": 3480.38,
                                                    "prompt_per_second": 151.42}})
        self.assertAlmostEqual(m["native_prefill_tps"], 151.42)


class LoadOldStateTest(unittest.TestCase):
    """States written before the transport/telemetry fixes are cleaned up on load."""

    def test_zero_native_and_finish_reason_error(self):
        from bench import state
        ok = state._metrics_from_record({"type": "result", "id": "R1", "ttft_ms": 90000, "native_ttft_ms": 0.0,
                                         "native_prefill_tps": 0.0, "native_decode_tps": 0.0,
                                         "server_paused_ms": 90000, "content": "x", "eval_result": "PASS"})
        self.assertIsNone(ok["native_ttft_ms"])
        self.assertIsNone(ok["native_decode_tps"])
        self.assertNotIn("server_paused_ms", ok)
        bad = state._metrics_from_record({"type": "result", "id": "R2", "content": "", "tool_calls": [],
                                          "finish_reason": "error", "eval_result": "FAIL",
                                          "fail_reason": "Empty response (no content and no tool calls)"})
        self.assertEqual(bad["error"], "server: stream ended with finish_reason error")
        self.assertEqual(bad["fail_reason"], bad["error"])


if __name__ == "__main__":
    unittest.main()
