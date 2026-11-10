"""Prefill progress: return_progress is sent only to Splash / llama.cpp on streaming requests, prompt_progress chunks
leave TTFT, counters and content alone, a 400 falls back, and the live line shows real or estimated progress and
fits. Mock servers only, on 127.0.0.1.

    python3 -m unittest tests.test_progress
"""

import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401
from test_backend import GENERIC, LLAMACPP, SPLASH  # noqa: E402
from test_polish import ANSI_ALL, run_code  # noqa: E402
from test_replay import synthetic_fixture, write_fixture  # noqa: E402

from bench import chat, live, report, runner, transport  # noqa: E402
from tools import mock_server  # noqa: E402

PP = [{"total": 1000, "cache": 200, "processed": 200, "time_ms": 5},
      {"total": 1000, "cache": 200, "processed": 600, "time_ms": 400},
      {"total": 1000, "cache": 200, "processed": 500, "time_ms": 300},  # a regression: ignored
      {"total": 1000, "cache": 200, "processed": 1000, "time_ms": 800}]


class Server:
    """Streaming chat server; `routes` decide the backend kind. Records payloads. `reject_progress`: HTTP 400 when
    the request has return_progress. `progress`: prompt_progress chunks sent before the output."""

    def __init__(self, routes=SPLASH, progress=PP, reject_progress=False, late_progress=True):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path not in routes:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    self.wfile.write(json.dumps(routes[self.path]).encode())
                except (BrokenPipeError, ConnectionResetError):
                    return

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(payload)
                if reject_progress and "return_progress" in payload:
                    body = b'{"error": {"message": "unknown field return_progress"}}'
                    self.send_response(400)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = []
                if "return_progress" in payload:
                    chunks += [{"choices": [{"delta": {}}], "prompt_progress": p} for p in progress]
                chunks += [{"choices": [{"delta": {"content": "hello"}}]}]
                late = {"choices": [{"delta": {"content": " world"}}]}
                if "return_progress" in payload and late_progress:  # llama.cpp: also on later chunks
                    late["prompt_progress"] = {"total": 1000, "cache": 200, "processed": 1000, "time_ms": 9}
                chunks += [late]
                chunks += [{"choices": [{"delta": {}, "finish_reason": "stop"}]},
                           {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 5,
                                                     "prompt_tokens_details": {"cached_tokens": 200}}}]
                try:
                    for c in chunks:
                        self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.url = "http://127.0.0.1:%d/v1" % self.httpd.server_address[1]

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


CASE = {"id": "X", "system": "s", "prompt": "p"}


class Parsing(unittest.TestCase):
    def query(self, srv, **kw):
        state = {}
        m = runner.query_model(srv.url + "/chat/completions", "m", CASE, state=state, **kw)
        return m, state

    def test_progress_is_parsed_and_the_state_shows_the_last_object(self):
        with Server() as srv:
            m, state = self.query(srv, prompt_progress=True)
        # counters never regress and the object after the first token is ignored
        self.assertEqual(m["prompt_progress"], {"total": 1000, "cache": 200, "processed": 1000, "time_ms": 800})
        self.assertEqual(state["pp"], m["prompt_progress"])

    def test_progress_chunks_are_not_the_first_token_and_change_nothing_else(self):
        with Server() as srv:
            with_pp, st_with = self.query(srv, prompt_progress=True)
            without, st_without = self.query(srv, prompt_progress=False)
        self.assertNotIn("prompt_progress", without)
        for m in (with_pp, without):
            self.assertEqual(m["content"], "hello world")
            self.assertEqual(m["tokens"], 5)
            self.assertEqual(m["tool_calls"], [])
        self.assertEqual(st_with["down"], st_without["down"])
        # the first-token time is that of the content chunk: after every progress chunk was read
        self.assertGreater(st_with["first"], st_with["start"])
        self.assertLessEqual(with_pp["ttft_ms"], with_pp["total_ms"])
        self.assertEqual(set(with_pp) - {"prompt_progress"}, set(without))

    def test_first_is_not_set_by_progress_alone(self):
        # a stream of progress chunks only: no first token, no ttft
        lines = [b'data: {"choices":[{"delta":{}}],"prompt_progress":{"total":9,"cache":0,"processed":3,"time_ms":1}}\n',
                 b'data: {"choices":[{"delta":{"content":null}}],"prompt_progress":{"total":9,"cache":0,"processed":9,"time_ms":2}}\n',
                 b'data: [DONE]\n']

        class R:
            def __iter__(self):
                return iter(lines)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        state = {}
        with mock.patch.object(transport.urllib.request, "urlopen", return_value=R()):
            m = transport._send("http://x/v1/chat/completions", {"stream": True}, CASE, state)
        self.assertNotIn("first", state)
        self.assertEqual(m["ttft_ms"], 0)
        self.assertEqual(m["content"], "")
        self.assertEqual(state["down"], 0)
        self.assertEqual(state["pp"]["processed"], 9)


def fake_stream(lines):
    class R:
        def __iter__(self):
            return iter(lines)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    return R()


KP = 'data: {"kapitsa_progress":{"phase":%d,"prompt_tokens":1000,"cached_tokens":200,"processed_tokens":%d,' \
     '"completion_tokens":0,"final":false}}\n'
KAPITSA_STREAM = [
    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',  # the early empty delta
    b'event: kapitsa_progress\n', (KP % (0, 0)).encode(), b'\n',
    b'event: kapitsa_progress\n', (KP % (1, 600)).encode(), b'\n',
    b'data: {"choices":[{"delta":{}}]}\n',  # keepalive, empty delta
    b'event: kapitsa_progress\n', (KP % (1, 500)).encode(), b'\n',  # a regression: ignored
    b'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
    b'event: kapitsa_progress\n', (KP % (2, 1000)).encode(), b'\n',
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
    b'data: {"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":2}}\n',
    b'data: [DONE]\n']


class KapitsaStream(unittest.TestCase):
    def run_stream(self, lines):
        state = {}
        with mock.patch.object(transport.urllib.request, "urlopen", return_value=fake_stream(lines)) as op:
            m = transport._send("http://x/v1/chat/completions", {"stream": True}, CASE, state,
                             headers=transport.KAPITSA_HEADER)
        return m, state, op

    def test_progress_events_map_to_prompt_progress_and_change_nothing_else(self):
        m, state, op = self.run_stream(KAPITSA_STREAM)
        quiet = [l for l in KAPITSA_STREAM if b"kapitsa" not in l]
        base, bstate, _ = self.run_stream(quiet)
        self.assertEqual(op.call_args[0][0].get_header("X-kapitsa-bench-telemetry"), "1")
        self.assertEqual(m["prompt_progress"]["total"], 1000)
        self.assertEqual(m["prompt_progress"]["cache"], 200)
        self.assertEqual(m["prompt_progress"]["processed"], 600)
        for k in ("content", "tokens", "prompt_tokens", "tool_calls"):
            self.assertEqual(m[k], base[k])
        self.assertEqual(m["content"], "hi")
        self.assertEqual(m["tokens"], 2)
        self.assertEqual(state["down"], bstate["down"])
        self.assertEqual(set(m) - {"prompt_progress"}, set(base))

    def test_empty_deltas_and_phase_0_never_set_the_first_token(self):
        lines = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                 b'data: {"choices":[{"delta":{"content":"","tool_calls":[]}}]}\n',
                 b'event: kapitsa_progress\n', (KP % (0, 0)).encode(),
                 b'data: {"choices":[{"delta":{}}]}\n', b'data: [DONE]\n']
        m, state, _ = self.run_stream(lines)
        self.assertNotIn("first", state)
        self.assertNotIn("pp", state)
        self.assertEqual(m["ttft_ms"], 0)
        self.assertEqual(state["down"], 0)

    def test_header_only_for_kapitsa_and_only_streaming(self):
        sent = []
        with mock.patch.object(transport, "_send", side_effect=lambda *a, **k: sent.append((a[1], k)) or {}):
            runner.query_model("u", "m", CASE, prompt_progress="header")
        self.assertEqual(sent[0][1], {"headers": transport.KAPITSA_HEADER})
        self.assertNotIn("return_progress", sent[0][0])
        self.assertEqual(transport.PromptProgress().sends({"backend_info": {"kind": "kapitsa"}}), "header")
        self.assertFalse(transport.PromptProgress(False).sends({"backend_info": {"kind": "kapitsa"}}))


class Sending(unittest.TestCase):
    def test_field_goes_only_to_splash_and_llamacpp(self):
        p = transport.PromptProgress()
        for kind, want in (("splash", True), ("llamacpp", True), ("ollama", False), ("lmstudio", False),
                           ("vllm", False), ("kapitsa", "header"), ("generic", False)):
            self.assertEqual(p.sends({"backend_info": {"kind": kind}}), want, kind)
        self.assertFalse(p.sends({}))
        self.assertFalse(transport.PromptProgress(False).sends({"backend_info": {"kind": "splash"}}))

    def test_only_streaming_requests_carry_it(self):
        sent = []
        with mock.patch.object(transport, "_send", side_effect=lambda url, payload, tc, st: sent.append(payload) or {}):
            runner.query_model("u", "m", CASE, prompt_progress=True)
            runner.query_model("u", "m", CASE, prompt_progress=False)
            runner.query_model("u", "m", CASE)
        self.assertEqual(["return_progress" in p for p in sent], [True, False, False])
        self.assertIs(sent[0]["return_progress"], True)
        self.assertTrue(sent[0]["stream"])

    def run_cli(self, srv, *extra):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fx = write_fixture(synthetic_fixture(), tmp.name)
        self.addCleanup(lambda: report.configure("tui", io.StringIO()))
        return run_code("--suite", "coding", "--fixture", fx, "--endpoint", srv.url, "--no-save-state", "--no-mdcat",
                        "--turns", "1-2", *extra)

    def real(self, srv):
        return [r for r in srv.requests if r.get("max_tokens") != 256]

    def test_run_sends_to_splash_and_the_opt_out_stops_it(self):
        with Server(SPLASH) as srv:
            self.run_cli(srv)
            self.assertTrue(self.real(srv) and all(r.get("return_progress") is True for r in self.real(srv)))
            self.assertNotIn("return_progress", [r for r in srv.requests if r.get("max_tokens") == 256][0])
        with Server(SPLASH) as srv:
            self.run_cli(srv, "--no-prompt-progress")
            self.assertTrue(all("return_progress" not in r for r in srv.requests))
        with Server(LLAMACPP) as srv:
            self.run_cli(srv)
            self.assertTrue(all(r.get("return_progress") is True for r in self.real(srv)))
        with Server(GENERIC) as srv:
            self.run_cli(srv)
            self.assertTrue(srv.requests and all("return_progress" not in r for r in srv.requests))

    def test_400_retries_once_without_the_field_and_turns_it_off(self):
        with Server(SPLASH, reject_progress=True) as srv:
            out, err, code = self.run_cli(srv)
            reqs = self.real(srv)
        self.assertEqual(code, 0)
        self.assertIn("0 FAIL", out)
        self.assertNotIn("❌ FAIL", out)
        self.assertEqual(err.count("server rejected return_progress"), 1)
        self.assertEqual([("return_progress" in r) for r in reqs], [True, False, False])  # then off for the run

    def test_a_400_that_stays_is_an_error_and_keeps_the_field_on(self):
        policy = transport.PromptProgress()
        target = {"backend_info": {"kind": "splash"}}
        calls = []

        def send(flag):
            calls.append(flag)
            return {"error": "HTTP 400 Bad Request: context overflow"}

        m = transport.send_with_progress_fallback(send, policy, target)
        self.assertEqual(calls, [True, False])
        self.assertIn("HTTP 400", m["error"])
        self.assertTrue(policy.enabled)

    def test_the_result_line_gains_only_the_prompt_progress_key(self):
        with tempfile.TemporaryDirectory() as d:
            fx = write_fixture(synthetic_fixture(), d)
            paths = {}
            for name, extra in (("on", ()), ("off", ("--no-prompt-progress",))):
                paths[name] = os.path.join(d, name + ".jsonl")
                with Server(SPLASH) as srv:
                    run_code("--suite", "coding", "--fixture", fx, "--endpoint", srv.url, "--state-file", paths[name],
                             "--no-mdcat", "--turns", "1-1", *extra)
            recs = {}
            for k, p in paths.items():
                with open(p) as f:
                    recs[k] = [json.loads(l) for l in f if l.strip()]
            on = [r for r in recs["on"] if r["type"] == "result"][0]
            off = [r for r in recs["off"] if r["type"] == "result"][0]
            self.assertEqual(set(on) - set(off), {"prompt_progress"})
            self.assertEqual(set(off) - set(on), set())
            self.assertEqual(on["content"], off["content"])
            self.assertEqual(recs["on"][0]["request"], recs["off"][0]["request"])


TC = {"id": "R013_main_e0_t010", "entry": {"label": "main/e0", "prompt_tokens": 50000}, "recorded_prompt_tokens": 50000,
      "stream": "main", "epoch": 0, "kind": "turn", "fixture": {}}


class Estimate(unittest.TestCase):
    def hist(self, tps=(), prev=None):
        return {"tps": list(tps), "prev": dict(prev or {})}

    def test_no_history_no_estimate(self):
        self.assertEqual(runner.prefill_estimate(TC, self.hist()), {})
        self.assertEqual(runner.prefill_estimate({"id": "smoke"}, self.hist([100.0])), {})

    def test_uncached_over_the_median_speed(self):
        e = runner.prefill_estimate(TC, self.hist([300, 100, 200], {("main", 0): 40000}))
        self.assertEqual(e, {"est_tokens": 10000, "est_tps": 200})
        # no previous request on the stream: nothing cached
        self.assertEqual(runner.prefill_estimate(TC, self.hist([100]))["est_tokens"], 50000)
        # an other stream or epoch does not count; a compaction expects no cache
        self.assertEqual(runner.prefill_estimate(TC, self.hist([100], {("S1", 0): 40000, ("main", 1): 40000}))["est_tokens"], 50000)
        comp = dict(TC, kind="compaction")
        self.assertEqual(runner.prefill_estimate(comp, self.hist([100], {("main", 0): 40000}))["est_tokens"], 50000)
        # a previous prompt larger than this one cannot cache more than it
        self.assertEqual(runner.prefill_estimate(TC, self.hist([100], {("main", 0): 90000}))["est_tokens"], 0)

    def test_only_the_last_five_speeds_count_and_even_counts_average(self):
        h = self.hist([1, 1, 1000, 1000, 1000, 1000, 1000])
        self.assertEqual(runner.prefill_estimate(TC, h)["est_tps"], 1000)
        self.assertEqual(runner.prefill_estimate(TC, self.hist([100, 200]))["est_tps"], 150)

    def test_note_finished_feeds_the_history(self):
        h = self.hist()
        runner.note_finished(TC, {"native_prefill_tps": 250.0, "prompt_tokens": 51000, "eval_result": "PASS"}, h)
        runner.note_finished(TC, {"error": "x"}, h)
        runner.note_finished(TC, {"eval_result": "SKIP", "native_prefill_tps": 9}, h)
        runner.note_finished(dict(TC, kind="compaction"), {"native_prefill_tps": 100.0, "prompt_tokens": 7}, h)
        self.assertEqual(h["tps"], [250.0, 100.0])
        self.assertEqual(h["prev"], {("main", 0): 51000})


    def test_server_without_prefill_speed_feeds_the_client_clock_speed(self):
        # LM Studio: native decode and TTFT only, no cache figure: new tokens = prompt - the previous turn's prompt
        h = self.hist(prev={("main", 0): 40000})
        m = {"prompt_tokens": 50000, "cached_tokens": None, "ttft_ms": 5000, "native_prefill_tps": None, "eval_result": "PASS"}
        runner.note_finished(TC, m, h)
        self.assertEqual(h["tps"], [2000.0])
        # a reported cache count wins; a hit that leaves under 10 new tokens says nothing about speed
        runner.note_finished(TC, dict(m, cached_tokens=49000, ttft_ms=1000), h)
        runner.note_finished(TC, dict(m, cached_tokens=49995), h)
        runner.note_finished(TC, dict(m, ttft_ms=0), h)
        self.assertEqual(h["tps"], [2000.0, 1000.0])
        # so the next live line has an estimate
        live = runner.live_state(TC, h, transport.PromptProgress(), {"backend_info": {"kind": "lmstudio"}})
        self.assertEqual(live["est_tps"], 1500.0)
        h["prev"].clear()
        live = runner.live_state(TC, h, transport.PromptProgress(), {"backend_info": {"kind": "lmstudio"}})
        i = chat.prefill_info(dict(live, start=100.0), 101.0)
        self.assertTrue(i["est"])


class ExpectedProgress(unittest.TestCase):
    EST = {"est_tokens": 40000, "est_tps": 2000}

    def test_pp_expected_shows_real_zero(self):
        i = chat.prefill_info({"start": 100.0, "pp_expected": True}, 105.0)
        self.assertEqual((i["frac"], i["est"], i["remaining"], i["elapsed"]), (0.0, False, None, 5.0))
        s = ANSI_ALL.sub("", chat._prefill_seg(i, set()))
        self.assertIn("0%  5s<?", s)
        self.assertNotIn("~", s)

    def test_pp_expected_ignores_estimate_fields(self):
        i = chat.prefill_info(dict(self.EST, start=100.0, pp_expected=True), 104.0)
        self.assertFalse(i["est"])
        self.assertIsNone(i["remaining"])

    def test_real_pp_wins(self):
        pp = {"total": 1000, "cache": 200, "processed": 600, "time_ms": 400}
        i = chat.prefill_info({"start": 100.0, "pp_expected": True, "pp": pp}, 101.0)
        self.assertEqual(i["frac"], 0.5)

    def test_live_state_by_backend(self):
        hist = {"tps": [100.0], "prev": {}}
        for kind in ("splash", "llamacpp", "kapitsa"):
            st = runner.live_state(TC, hist, transport.PromptProgress(True), {"backend_info": {"kind": kind}})
            self.assertEqual(st, {"pp_expected": True}, kind)
        for target, pol in (({"backend_info": {"kind": "generic"}}, True), ({"backend_info": {"kind": "splash"}}, False),
                            ({}, True)):
            st = runner.live_state(TC, hist, transport.PromptProgress(pol), target)
            self.assertNotIn("pp_expected", st)
            self.assertEqual(st, runner.prefill_estimate(TC, hist))


class LiveLine(unittest.TestCase):
    PR = {"done": 12, "total": 470, "t0": 100.0, "t_last": 400.0, "cache": 98.0, "n_ctx": 262144}

    def setUp(self):
        report.configure("chat", io.StringIO())
        self.addCleanup(lambda: report.configure("tui", io.StringIO()))

    def plain(self, st, width=None, now=420.0):
        return ANSI_ALL.sub("", chat.live_compose(TC, st, now, self.PR, width))

    def test_real_progress_has_no_tilde(self):
        st = {"start": 410.0, "pp": {"total": 1000, "cache": 200, "processed": 600, "time_ms": 4000}}
        line = self.plain(st)
        self.assertIn("50% 10s< 4s", line)
        self.assertNotIn("~50%", line)
        self.assertIn("▕███░░░▏", line)

    def test_estimate_has_a_tilde_and_never_reaches_100(self):
        st = {"start": 410.0, "est_tokens": 2000, "est_tps": 400.0}  # 5s expected
        self.assertIn("~40%  2s<~ 3s", self.plain(st, now=412.0))
        late = self.plain(st, now=430.0)  # far past the estimate
        self.assertIn("~99%", late)
        self.assertNotIn("100%", late)
        self.assertIn("20s<~ 0s", late)  # past the estimate: remaining stays at 0s, never flickers to <?
        self.assertNotIn("20s<?", late)

    def test_real_progress_without_a_rate_yet(self):
        for pp in ({"total": 1000, "cache": 200, "processed": 200, "time_ms": 5},
                   {"total": 1000, "cache": 200, "processed": 300, "time_ms": 400}):
            self.assertIn("10s<?", self.plain({"start": 410.0, "pp": pp}))

    def test_plain_timer_without_progress_or_estimate(self):
        line = self.plain({"start": 410.0})
        self.assertIn("prefill 10s", line)
        self.assertNotIn("%", line.split("prefill")[1])

    def test_prefill_icon_when_live_icons_on_word_when_off(self):
        from bench import glyphs
        self.addCleanup(glyphs.configure_live, glyphs.live_enabled)
        for st in ({"start": 410.0}, {"start": 410.0, "est_tokens": 2000, "est_tps": 400.0}):
            glyphs.configure_live(True)
            on = self.plain(st)
            self.assertIn("\uf085 ", on)
            self.assertNotIn("prefill", on)
            self.assertNotIn("\uf2f2", on)
            glyphs.configure_live(False)
            self.assertIn("prefill", self.plain(st))
            self.assertNotIn("\uf085", self.plain(st))
        self.assertEqual(chat.live_vlen("\uf085"), 2)

    def test_no_estimate_when_everything_is_cached(self):
        self.assertIn("prefill 10s", self.plain({"start": 410.0, "pp": {"total": 5, "cache": 5, "processed": 5, "time_ms": 1}}))

    def test_first_token_ends_the_progress_display(self):
        st = {"start": 410.0, "first": 415.0, "down": 100, "pp": {"total": 10, "cache": 0, "processed": 5, "time_ms": 1}}
        line = self.plain(st)
        self.assertIn("TTFT 5.0s", line)
        self.assertNotIn("prefill", line)

    def test_fits_narrow_and_wide(self):
        for st in ({"start": 410.0, "pp": {"total": 1000, "cache": 0, "processed": 300, "time_ms": 4000}},
                   {"start": 410.0, "est_tokens": 90000, "est_tps": 400.0}):
            for width in (60, 80, 120, 200):
                line = chat.live_compose(TC, st, 420.0, self.PR, width)
                self.assertLessEqual(chat.live_vlen(line), width, (width, ANSI_ALL.sub("", line)))
            self.assertIn("R013", chat.live_compose(TC, st, 420.0, self.PR, 60))
        full = self.plain({"start": 410.0, "pp": {"total": 1000, "cache": 0, "processed": 300, "time_ms": 4000}}, 200)
        self.assertIn("prefill", full)
        self.assertIn("ETA", full)

    def test_plain_chat_live_line(self):
        st = {"start": 100.0, "est_tokens": 1000, "est_tps": 100.0}
        line = ANSI_ALL.sub("", chat.live_line("L", st, 105.0))
        self.assertIn("~50%  5s<~ 5s", line)


RID = "chatcmpl-rb-" + "ab" * 16


def snap(phase="chunk", rid=RID, **ps):
    state = {"phase": phase, "tokens_total": 1000, "tokens_done": 600, "cached_tokens": 200, "elapsed_s": 0.4}
    state.update(ps)
    return {"in_flight": [{"request_id": "chatcmpl-other", "prefill_state": None},
                          {"request_id": rid, "prefill_state": state}]}


class MtplxSnapshot(unittest.TestCase):
    """MTPLX publishes chunked prefill on GET /v1/mtplx/snapshot; the poller maps it to the prompt_progress shape."""

    def test_chunk_state_maps_to_prompt_progress(self):
        pp = transport.mtplx_pp(snap(), RID)
        self.assertEqual({k: v for k, v in pp.items() if k != "at"},
                         {"total": 1000, "cache": 200, "processed": 600, "time_ms": 400, "chunk": 2048})
        self.assertAlmostEqual(pp["at"], time.time(), delta=5)
        self.assertEqual(transport.mtplx_pp(snap(chunk_size=512), RID)["chunk"], 512)
        i = chat.prefill_info({"start": 100.0, "pp": transport.mtplx_pp(snap(), RID)}, 101.0)
        self.assertEqual((i["frac"], i["est"]), (0.5, False))  # (600-200)/(1000-200): a real percentage

    def test_no_reading_before_the_first_chunk_or_for_another_request(self):
        self.assertIsNone(transport.mtplx_pp(snap("started", tokens_done=0), RID))
        self.assertIsNone(transport.mtplx_pp(snap("completed"), RID))
        self.assertIsNone(transport.mtplx_pp(snap(tokens_done=200), RID))  # chunk work started, no new token yet
        self.assertIsNone(transport.mtplx_pp(snap(), "chatcmpl-rb-other"))
        self.assertIsNone(transport.mtplx_pp({"in_flight": []}, RID))
        self.assertIsNone(transport.mtplx_pp({}, RID))
        self.assertIsNone(transport.mtplx_pp(snap(tokens_total=0), RID))
        self.assertIsNone(transport.mtplx_pp(snap(tokens_done="x"), RID))

    def test_a_stale_reading_is_ignored(self):
        last = transport.mtplx_pp(snap(), RID)
        self.assertIsNone(transport.mtplx_pp(snap(tokens_done=500), RID, last))
        self.assertEqual(transport.mtplx_pp(snap(tokens_done=700), RID, last)["processed"], 700)

    def test_a_repeated_reading_keeps_the_interpolation_going(self):
        # MTPLX republishes the same chunk reading on every poll for 11-17 s; the poller must keep the first one
        # (its "at" and "shown"), or the bar snaps back to the chunk boundary every 0.5 s.
        last = transport.mtplx_pp(snap(elapsed_s=2.0), RID)  # 400 new tokens in 2 s: 200 tok/s
        last["at"] = 100.0
        st = {"start": 99.0, "pp": last}
        before = chat.prefill_info(st, 102.0)["frac"]
        self.assertIsNone(transport.mtplx_pp(snap(elapsed_s=2.5), RID, last))
        self.assertGreater(before, 0.5)
        self.assertGreaterEqual(chat.prefill_info(st, 102.5)["frac"], before)

    def test_zero_new_reading_keeps_the_estimate_for_a_small_append(self):
        st = {"start": 100.0, "est_tokens": 1000, "est_tps": 100.0}
        pp = transport.mtplx_pp(snap(tokens_done=200), RID)
        if pp:
            st["pp"] = pp
        i = chat.prefill_info(st, 103.0)
        self.assertEqual((i["est"], round(i["frac"], 2)), (True, 0.3))
        line = ANSI_ALL.sub("", chat.live_line("L", st, 103.0))
        self.assertIn("~30%", line)
        self.assertNotIn("<?", line)

    @staticmethod
    def reading(**kw):
        pp = {"total": 10200, "cache": 200, "processed": 2248, "time_ms": 10000, "chunk": 2048, "at": 110.0}
        pp.update(kw)
        return pp

    def test_multi_chunk_reading_interpolates_up_to_the_next_chunk(self):
        st = {"start": 100.0, "pp": self.reading()}
        i0 = chat.prefill_info(st, 110.0)
        self.assertEqual((i0["est"], round(i0["frac"], 3)), (False, 0.205))  # 2048 of 10000
        i1 = chat.prefill_info(st, 115.0)  # 204.8 tok/s * 5 s
        self.assertEqual(round(i1["frac"], 3), 0.307)
        self.assertAlmostEqual(i1["remaining"], (10200 - 2248 - 1024) / 204.8, places=3)
        i2 = chat.prefill_info(st, 200.0)  # capped at the next chunk boundary
        self.assertEqual(round(i2["frac"], 3), 0.410)

    def test_interpolation_is_capped_at_99_percent_and_never_goes_back(self):
        st = {"start": 100.0, "pp": self.reading(total=2400, processed=2200, time_ms=1000)}
        self.assertEqual(int(chat.prefill_info(st, 150.0)["frac"] * 100), 99)
        st = {"start": 100.0, "pp": self.reading()}
        fracs = [chat.prefill_info(st, t)["frac"] for t in (110.0, 113.0, 112.0, 111.0)]
        self.assertEqual(fracs, sorted(fracs))
        st["pp"] = self.reading(processed=4296, time_ms=20000, at=120.0)  # a new reading may correct
        self.assertEqual(round(chat.prefill_info(st, 120.0)["frac"], 3), 0.410)

    def test_real_reading_without_rate_uses_the_estimate_speed_for_the_remaining_time(self):
        pp = {"total": 1000, "cache": 0, "processed": 100, "time_ms": 100}  # 0.1 s: no usable rate
        i = chat.prefill_info({"start": 100.0, "pp": pp, "est_tps": 300.0}, 101.0)
        self.assertEqual((i["est"], round(i["remaining"], 1)), (False, 3.0))
        line = ANSI_ALL.sub("", chat.live_line("L", {"start": 100.0, "pp": pp, "est_tps": 300.0}, 101.0))
        self.assertIn("10%", line)
        self.assertIn("<~ 3s", line)
        self.assertIsNone(chat.prefill_info({"start": 100.0, "pp": pp}, 101.0)["remaining"])

    def test_llamacpp_progress_is_not_interpolated(self):
        pp = {"total": 1000, "cache": 200, "processed": 600, "time_ms": 500}
        for now in (101.0, 130.0):
            i = chat.prefill_info({"start": 100.0, "pp": dict(pp)}, now)
            self.assertEqual((i["frac"], i["remaining"]), (0.5, 0.5))

    def test_policy_and_live_state(self):
        mt = {"backend_info": {"kind": "mtplx"}}
        self.assertEqual(transport.PromptProgress().sends(mt), "snapshot")
        self.assertFalse(transport.PromptProgress(False).sends(mt))  # --no-prompt-progress turns it off too
        hist = {"tps": [100.0], "prev": {}}
        st = runner.live_state(TC, hist, transport.PromptProgress(True), mt)
        self.assertNotIn("pp_expected", st)  # the estimate stays until the first chunk-phase reading
        self.assertEqual(st, runner.prefill_estimate(TC, hist))

    def test_snapshot_policy_never_resends_on_400(self):
        sent = []
        policy = transport.PromptProgress()
        m = transport.send_with_progress_fallback(lambda flag: sent.append(flag) or {"error": "HTTP 400 Bad"}, policy,
                                               {"backend_info": {"kind": "mtplx"}})
        self.assertEqual((sent, m["error"], policy.enabled), (["snapshot"], "HTTP 400 Bad", True))

    def test_request_id_header_only_on_streaming_mtplx_requests(self):
        sent = []
        with mock.patch.object(transport, "_send", side_effect=lambda url, p, tc, st, headers=None: sent.append(
                (p, headers)) or {}), mock.patch.object(transport, "_poll_snapshot"):
            runner.query_model("http://127.0.0.1:1/v1/chat/completions", "m", CASE, prompt_progress="snapshot")
        (payload, headers), = sent
        self.assertNotIn("return_progress", payload)
        self.assertRegex(headers["x-mtplx-request-id"], r"^rb-[0-9a-f]{32}$")

    def test_url(self):
        self.assertEqual(transport.snapshot_url("http://h:8080/v1/chat/completions"), "http://h:8080/v1/mtplx/snapshot")

    def serve(self, flavor, delay=0.0, **kw):
        httpd = mock_server.make_server(flavor, delay, **kw)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return "http://127.0.0.1:%d/v1/chat/completions" % httpd.server_address[1]

    def test_poller_sets_pp_before_the_first_token_and_stops_after(self):
        url = self.serve("mtplx", delay=0.5, cached_ratio=0.0, chunk=5)
        seen, state = [], {}
        orig = transport._poll_snapshot

        def watch(*a):
            orig(*a)
            seen.append("polling over")
        with mock.patch.object(transport, "MTPLX_POLL_S", 0.05), mock.patch.object(transport, "_poll_snapshot", watch):
            done = threading.Event()

            def sample():
                while not done.is_set():
                    if state.get("pp") and not state.get("first"):
                        seen.append(dict(state["pp"]))
                    time.sleep(0.02)
            t = threading.Thread(target=sample, daemon=True)
            t.start()
            m = runner.query_model(url, "m", CASE, state=state, prompt_progress="snapshot")
            done.set()
        self.assertFalse(m.get("error"), m)
        pps = [x for x in seen if isinstance(x, dict)]
        self.assertTrue(pps, "no snapshot reading before the first token")
        self.assertTrue(all(0 < p["processed"] <= p["total"] and p["cache"] == 0 for p in pps), pps)
        self.assertEqual(m["tokens"], 3)  # mtplx_progress / mtplx_stats during decode
        time.sleep(0.3)
        self.assertIn("polling over", seen)

    def test_a_missing_endpoint_ends_the_polling_and_keeps_the_estimate(self):
        url = self.serve("generic")  # GET /v1/mtplx/snapshot answers 404
        state = {"start": time.time(), "est_tokens": 5000, "est_tps": 100.0}
        with mock.patch.object(transport, "MTPLX_POLL_S", 0.01):
            transport._poll_snapshot(transport.snapshot_url(url), {}, RID, state, threading.Event())  # returns on its own
        self.assertNotIn("pp", state)
        self.assertTrue(chat.prefill_info(state, state["start"] + 10)["est"])


if __name__ == "__main__":
    unittest.main()


class TestFmtLiveTokens(unittest.TestCase):
    def test_under_10k_shows_raw_digits(self):
        cases = (
            (0, "0"),
            (1, "1"),
            (7, "7"),
            (265, "265"),
            (999, "999"),
            (1000, "1000"),
            (1234, "1234"),
            (2800, "2800"),
            (3000, "3000"),
            (9000, "9000"),
            (9999, "9999"),
        )
        for n, want in cases:
            self.assertEqual(live.fmt_live_tokens(n), want, f"n={n}")
            self.assertEqual(chat.fmt_live_tokens(n), want, f"n={n}")

    def test_between_10k_and_100k_shows_one_decimal_digit(self):
        cases = (
            (10000, "10.0K"),
            (10049, "10.0K"),
            (10050, "10.1K"),
            (10100, "10.1K"),
            (11000, "11.0K"),
            (11200, "11.2K"),
            (20000, "20.0K"),
            (99900, "99.9K"),
            (99949, "99.9K"),
        )
        for n, want in cases:
            self.assertEqual(live.fmt_live_tokens(n), want, f"n={n}")

    def test_over_100k_shows_no_float_digit(self):
        cases = (
            (99950, "100K"),
            (100000, "100K"),
            (100499, "100K"),
            (101000, "101K"),
            (101200, "101K"),
            (101900, "102K"),
            (123456, "123K"),
            (250000, "250K"),
            (1000000, "1000K"),
        )
        for n, want in cases:
            self.assertEqual(live.fmt_live_tokens(n), want, f"n={n}")

    def test_edge_cases(self):
        self.assertEqual(live.fmt_live_tokens(None), "0")
        self.assertEqual(live.fmt_live_tokens(-5), "-5")

    def test_live_line_and_compose_use_new_format(self):
        tc = {"id": "R001_main_e0_t000", "entry": {"label": "main/e0", "prompt_tokens": 50000}}
        pr = {"done": 0, "total": 10, "t0": 100.0, "t_last": 100.0, "cache": 50.0}

        # < 10000: 2800 -> 2800 (not 2.8K)
        st_2800 = {"start": 100.0, "first": 101.0, "down": 2800}
        line_2800 = chat.live_line("L", st_2800, 110.0)
        self.assertIn("2800 tok", line_2800)
        compose_2800 = chat.live_compose(tc, st_2800, 110.0, pr)
        self.assertIn("2800", compose_2800)

        # 10000..100k: 11000 -> 11.0K (not 11K)
        st_11k = {"start": 100.0, "first": 101.0, "down": 11000}
        line_11k = chat.live_line("L", st_11k, 110.0)
        self.assertIn("11.0K tok", line_11k)
        compose_11k = chat.live_compose(tc, st_11k, 110.0, pr)
        self.assertIn("11.0K", compose_11k)

        # >= 100k: 100000 -> 100K (no float digit)
        st_100k = {"start": 100.0, "first": 101.0, "down": 100000}
        line_100k = chat.live_line("L", st_100k, 110.0)
        self.assertIn("100K tok", line_100k)
        compose_100k = chat.live_compose(tc, st_100k, 110.0, pr)
        self.assertIn("100K", compose_100k)


class TestLiveFormatters(unittest.TestCase):
    def test_live_label_and_compose_smoke_mode(self):
        tc = {"id": "TC01_GIT_COMMIT"}  # no entry
        progress = {"done": 0, "total": 4, "t0": 100.0, "t_last": 100.0, "cache": 0.0}
        lbl = live.live_label(tc, progress=progress)
        self.assertIn("TC01_GIT_COMMIT", lbl)
        self.assertIn("1/4", lbl)

        from bench.outputs import TuiOutput
        out = TuiOutput()
        l, formatter = out.live(tc, "TC01", progress=progress)
        self.assertIsNotNone(formatter)
        st = {"start": 100.0}
        rendered = formatter(l, st)
        self.assertIn("TC01_GIT_COMMIT", rendered)

    def test_prefill_info_guards_negligible_tokens(self):
        st = {"start": 100.0, "est_tokens": 5, "est_tps": 100.0}
        self.assertIsNone(live.prefill_info(st, 101.0))
        st_zero_tps = {"start": 100.0, "est_tokens": 100, "est_tps": 0.0}
        self.assertIsNone(live.prefill_info(st_zero_tps, 101.0))
        st_valid = {"start": 100.0, "est_tokens": 100, "est_tps": 50.0}
        self.assertIsNotNone(live.prefill_info(st_valid, 101.0))

    def test_prefill_info_clamps_remaining_at_zero_when_elapsed_exceeds_guess(self):
        # 100 tokens at 50 tps = 2.0s expected
        st = {"start": 100.0, "est_tokens": 100, "est_tps": 50.0}
        # Elapsed 1.0s: remaining is 1.0s
        i1 = live.prefill_info(st, 101.0)
        self.assertAlmostEqual(i1["remaining"], 1.0)
        self.assertTrue(i1["est"])
        seg1 = live._prefill_seg(i1, set())
        self.assertIn("1s<~ 1s", seg1)

        # Elapsed 2.5s: elapsed > guess, remaining must be 0.0 (not None / <?), never flickers
        i2 = live.prefill_info(st, 102.5)
        self.assertEqual(i2["remaining"], 0.0)
        self.assertTrue(i2["est"])
        seg2 = live._prefill_seg(i2, set())
        self.assertIn("2s<~ 0s", seg2)
        self.assertNotIn("<?", seg2)


class FmtClock(unittest.TestCase):
    def test_forms(self):
        for sec, want in ((0, " 0s"), (5, " 5s"), (12, "12s"), (59.6, "59s"), (60, "1m00s"), (68, "1m08s"),
                          (600, "10m00s"), (3725, "1h02m")):
            self.assertEqual(chat.fmt_clock(sec), want, sec)
