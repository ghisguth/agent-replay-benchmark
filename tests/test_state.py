"""JSONL state: round trip, crash safety, several models in one file, auto naming, old JSON states, and a full mock
run (header, results with Splash-style telemetry, --state re-display, divergence). Mock servers only, on 127.0.0.1.

    python3 -m unittest tests.test_state
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401
from test_backend import SPLASH  # noqa: E402
from test_replay import synthetic_fixture, write_fixture  # noqa: E402

import replay_bench  # noqa: E402
from bench import baseline, compare, state  # noqa: E402
from bench import results as R  # noqa: E402
from bench.display import display_state  # noqa: E402
from bench.utils import UserError  # noqa: E402

SPLASH_METRICS = {
    "prefill": {"tokens": 50}, "decode": {"tokens": 6},
    "request_latency": {"start_to_first_token_ms": 120.0, "first_token_to_done_ms": 80.0, "wall_ms": 210.0,
                        "ttft_ms": 130.0, "stream_tokens_per_second": 75.0},
    "cache": {"status": "hit", "matched_tokens": 50, "deduplicated_tokens": 0, "lane": 0},
    "speculation": {"drafted_tokens": 12, "accepted_draft_tokens": 9, "cycles": 4},
}
SPLASH_TIMINGS = {"prompt_n": 100, "prompt_ms": 120.0, "prompt_per_second": 416.0, "predicted_n": 7,
                  "predicted_ms": 80.0, "predicted_per_second": 75.0, "cache_n": 50, "draft_n": 12,
                  "draft_n_accepted": 9}


class SplashLike:
    """Mock of Splash: GET routes of the real shapes, streaming chat completions whose final chunk carries usage,
    timings and metrics (server/dialects/openai.py), plus reasoning deltas."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path not in SPLASH:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(SPLASH[self.path]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    return

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(payload)
                spec = outer.behaviour(payload)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [{"choices": [{"delta": {"role": "assistant"}}]}]
                chunks.append({"choices": [{"delta": {"reasoning_content": "hmm, let me think. "}}]})
                chunks.append({"choices": [{"delta": {"content": spec.get("text", "")}}]})
                for idx, (name, args) in enumerate(spec.get("calls", [])):
                    chunks.append({"choices": [{"delta": {"tool_calls": [{
                        "index": idx, "id": "call_%d" % idx, "type": "function",
                        "function": {"name": name, "arguments": args}}]}}]})
                chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls" if spec.get("calls") else "stop"}]})
                chunks.append({"choices": [], "usage": {"prompt_tokens": 100 * len(payload["messages"]),
                                                        "completion_tokens": 7,
                                                        "completion_tokens_details": {"reasoning_tokens": 3},
                                                        "prompt_tokens_details": {"cached_tokens": 50}},
                               "timings": SPLASH_TIMINGS, "metrics": SPLASH_METRICS})
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


SUMMARY = ("A summary of the work so far. " * 40).strip()


def behaviour(payload):
    if payload.get("tool_choice") == "none":
        return {"text": SUMMARY}
    if payload.get("max_tokens") == 256:
        return {"text": "feat: x"}
    return {"text": "Thinking.", "calls": [("bash", '{"x":1}')]}


def run_main(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        replay_bench.main(list(argv))
    return out.getvalue()


def lines(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


TARGET = {"key": "h:1::m", "backend": "h:1", "name": "m", "url": "http://h:1/v1/chat/completions", "n_ctx": 4096,
          "backend_info": {"kind": "splash", "version": "1a2b3c4d", "build": "src-x", "details": {}}}


def metrics(i, **kw):
    m = {"id": "C%02d" % i, "text": "hello", "content": "hello", "ttft_ms": 10, "total_ms": 20, "tokens": 5,
         "prompt_tokens": 100, "cached_tokens": 0, "native_ttft_ms": None, "native_prefill_tps": None,
         "native_decode_tps": None, "eval_result": "PASS", "fail_reason": ""}
    m.update(kw)
    return m


class Format(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "s.jsonl")

    def write(self, target=TARGET, n=3, end=True):
        w = state.StateWriter(self.path)
        w.write(state.make_header("smoke", target, request={"temperature": 0.0}, argv=["--suite", "smoke"]))
        raw = {}
        for i in range(n):
            raw["C%02d" % i] = metrics(i)
            w.write(state.result_record(raw["C%02d" % i]))
        if end:
            w.write(state.end_record(raw))
        w.close()

    def test_round_trip(self):
        calls = [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]
        w = state.StateWriter(self.path)
        w.write(state.make_header("smoke", TARGET))
        m = metrics(0, text="prose\n" + json.dumps(calls), content="prose", tool_calls=calls, finish_reason="tool_calls",
                    usage={"prompt_tokens": 100}, metrics=SPLASH_METRICS, reasoning_chars=18, reasoning_tokens=3)
        w.write(state.result_record(m))
        w.close()
        raw, keys = state.load_state(self.path)
        self.assertEqual(keys[0]["key"], "h:1::m")
        self.assertEqual((keys[0]["backend"], keys[0]["name"], keys[0]["n_ctx"]), ("h:1", "m", 4096))
        self.assertEqual(keys[0]["url"], "http://h:1/v1/chat/completions")
        got = raw["h:1::m"]["C00"]
        self.assertEqual(got, m)  # the runner's shape comes back, "text" rebuilt
        self.assertNotIn("text", lines(self.path)[1])  # one copy of the answer on disk
        self.assertEqual(lines(self.path)[1]["metrics"]["speculation"]["cycles"], 4)

    def test_header_fields(self):
        self.write()
        h = lines(self.path)[0]
        self.assertEqual((h["type"], h["schema"], h["suite"], h["model"], h["n_ctx"]), ("run", 1, "smoke", "m", 4096))
        self.assertEqual(h["endpoint"], "http://h:1/v1")
        self.assertEqual(h["server"]["version"], "1a2b3c4d")
        self.assertEqual(h["request"], {"temperature": 0.0, "reasoning_effort": None, "thinking": None})
        self.assertEqual(h["argv"], ["--suite", "smoke"])
        self.assertRegex(h["started_at"], r"^\d{4}-\d\d-\d\dT")
        self.assertIsNone(h["fixture"])
        self.assertEqual(lines(self.path)[-1]["counts"], {"PASS": 3})

    def test_old_header_without_thinking_loads(self):
        self.write()
        recs = lines(self.path)
        del recs[0]["request"]["reasoning_effort"]  # an older header
        with open(self.path, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in recs))
        raw, keys = state.load_state(self.path)
        self.assertEqual(sorted(raw[keys[0]["key"]]), ["C00", "C01", "C02"])
        self.assertIsNone(lines(self.path)[0]["request"].get("thinking"))
        self.assertEqual(R.sampling_text(lines(self.path)[0]["request"]), "temperature 0.0, reasoning effort none sent")
        self.assertIn("thinking off", R.sampling_text({"thinking": "off"}))

    def test_crash_keeps_everything_so_far(self):
        self.write(end=False)
        with open(self.path, "a") as f:
            f.write('{"type": "result", "id": "C03", "te')  # torn by a crash
        raw, keys = state.load_state(self.path)
        self.assertEqual(sorted(raw["h:1::m"]), ["C00", "C01", "C02"])
        self.assertEqual(baseline.load_baseline(self.path).pick("h:1", "m")["C01"]["eval_result"], "PASS")

    def test_every_line_is_on_disk_before_close(self):
        w = state.StateWriter(self.path)
        w.write(state.make_header("smoke", TARGET))
        w.write(state.result_record(metrics(0)))
        w.flush()
        raw, _ = state.load_state(self.path)  # read while the writer is still open
        self.assertEqual(list(raw["h:1::m"]), ["C00"])
        w.close()

    def test_several_models_in_one_file(self):
        self.write()
        other = dict(TARGET, key="h:1::n", name="n")
        w = state.StateWriter(self.path)  # appends
        w.write(state.make_header("smoke", other))
        w.write(state.result_record(metrics(7, eval_result="FAIL")))
        w.write(state.make_header("smoke", TARGET))  # the same model again in the same file
        w.write(state.result_record(metrics(8)))
        w.close()
        raw, keys = state.load_state(self.path)
        self.assertEqual([k["key"] for k in keys], ["h:1::m", "h:1::n", "h:1::m#2"])
        self.assertEqual(sorted(raw["h:1::m"]), ["C00", "C01", "C02"])
        self.assertEqual(list(raw["h:1::n"]), ["C07"])
        self.assertEqual(list(raw["h:1::m#2"]), ["C08"])
        labels = [e["label"] for e in compare.entries_from_run(keys, raw)]
        self.assertEqual(labels, ["m", "n", "m #2"])

    def test_not_a_state_file(self):
        with open(self.path, "w") as f:
            f.write("hello\n")
        with self.assertRaises(ValueError):
            state.load_state(self.path)

    def test_naming(self):
        when = datetime(2026, 11, 7, 12, 30, 5)
        p = state.default_path("states", TARGET["backend_info"], "unsloth/Qwen3.6 27B (Q4)", "coding", when)
        self.assertEqual(p, os.path.join("states", "splash-1a2b3c4d_unsloth-Qwen3.6-27B-Q4_coding_20261107-123005.jsonl"))
        p = state.default_path(self.tmp.name, {"kind": "generic", "version": None}, "m", "smoke", when)
        self.assertTrue(p.endswith("generic_m_smoke_20261107-123005.jsonl"))
        open(p, "w").close()
        p3 = state.default_path("states", TARGET["backend_info"], "m", "coding", when, {"temperature": 0.6, "reasoning_effort": "low"})
        self.assertRegex(p3, r"splash-1a2b3c4d_m_coding_t0\.6_e-low_\d{8}-\d{6}\.jsonl$")
        p5 = state.default_path("states", TARGET["backend_info"], "m", "coding", when,
                                {"temperature": 0.6, "reasoning_effort": "low", "thinking": "off"})
        self.assertRegex(p5, r"_coding_t0\.6_e-low_think-off_\d{8}-\d{6}\.jsonl$")
        self.assertIn("_think-on_", state.default_path("states", TARGET["backend_info"], "m", "coding", when, {"thinking": "on"}))
        p4 = state.default_path("states", TARGET["backend_info"], "m", "coding", when, {"temperature": 0.0})
        self.assertIn("_coding_t0_", p4)  # greedy is given, so it is named
        p2 = state.default_path(self.tmp.name, {"kind": "generic", "version": None}, "m", "smoke", when)
        self.assertTrue(p2.endswith("_20261107-123005-2.jsonl"))  # never an existing file
        self.assertEqual(state.slug("a/b\\c:d"), "a-b-c-d")

    def test_model_label_and_file_slug(self):
        mi = {"quant": "Q4_K_M"}
        # Appended once
        self.assertEqual(state.model_label("qwen3.8:27b-256k", mi), "qwen3.8:27b-256k \u00b7 Q4_K_M")
        self.assertEqual(state.model_file_slug("qwen3.8:27b-256k", mi), "qwen3.8-27b-256k-Q4_K_M")

        # Not doubled if already in model name (case-insensitive)
        self.assertEqual(state.model_label("qwen3.8:27b-256k-Q4_K_M", mi), "qwen3.8:27b-256k-Q4_K_M")
        self.assertEqual(state.model_label("qwen3.8:27b-256k-q4_k_m", mi), "qwen3.8:27b-256k-q4_k_m")
        self.assertEqual(state.model_file_slug("qwen3.8:27b-256k-q4_k_m", mi), "qwen3.8-27b-256k-q4_k_m")

        # Without model_info or without quant
        self.assertEqual(state.model_label("qwen3.8:27b-256k", None), "qwen3.8:27b-256k")
        self.assertEqual(state.model_file_slug("qwen3.8:27b-256k", {}), "qwen3.8-27b-256k")

        # default_path incorporates quant
        when = datetime(2026, 11, 7, 12, 30, 5)
        p = state.default_path("states", {"kind": "ollama", "version": "0.40.15"}, "qwen3.8:27b-256k", "coding",
                               when=when, model_info=mi)
        self.assertEqual(p, os.path.join("states", "ollama-0.40.15_qwen3.8-27b-256k-Q4_K_M_coding_20261107-123005.jsonl"))

    def test_make_header_stores_model_info(self):
        target = {"backend": "localhost:11434", "name": "qwen3.8:27b-256k", "model_info": {"quant": "Q4_K_M"}}
        h = state.make_header("coding", target)
        self.assertEqual(h.get("model_info"), {"quant": "Q4_K_M"})


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = write_fixture(synthetic_fixture(), self.tmp.name)
        self.states = os.path.join(self.tmp.name, "states")
        patch = mock.patch.object(replay_bench, "STATES_DIR", self.states)
        patch.start()
        self.addCleanup(patch.stop)

    def run_replay(self, srv, *extra):
        return run_main("--suite", "coding", "--fixture", self.fixture, "--endpoint", srv.url, "--no-mdcat", *extra)

    def test_full_run_auto_saves_jsonl(self):
        with SplashLike(behaviour) as srv:
            out = self.run_replay(srv, "--temperature", "0.5")
        self.assertIn("Splash 1a2b3c4d", out)  # discovery shows the backend
        files = os.listdir(self.states)
        self.assertEqual(len(files), 1)
        self.assertRegex(files[0], r"^splash-1a2b3c4d_([A-Za-z0-9.+-]+_)?m_coding_t0\.5_\d{8}-\d{6}\.jsonl$")
        recs = lines(os.path.join(self.states, files[0]))
        h, results, end = recs[0], recs[1:-1], recs[-1]
        self.assertEqual((h["type"], h["suite"], h["model"], h["endpoint"]), ("run", "coding", "m", srv.url))
        self.assertEqual(h["server"]["kind"], "splash")
        self.assertEqual(h["request"], {"temperature": 0.5, "reasoning_effort": None, "thinking": None})
        self.assertEqual(h["fixture"]["file"], "fx.json.gz")
        self.assertRegex(h["fixture"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(h["selection"]["turns"], None)
        self.assertIn("--temperature", h["argv"])
        self.assertEqual(len(results), 10)
        self.assertEqual(end["type"], "end")
        r = next(x for x in results if x["id"] == "R003_main_e0_t001")
        self.assertEqual(r["type"], "result")
        self.assertEqual(r["content"], "Thinking.")
        self.assertEqual(r["tool_calls"][0]["function"], {"name": "bash", "arguments": '{"x":1}'})
        self.assertEqual(r["finish_reason"], "tool_calls")
        self.assertEqual((r["reasoning_tokens"], r["reasoning_chars"]), (3, len("hmm, let me think. ")))
        self.assertNotIn("hmm", json.dumps(r))  # counts only, no thinking text
        self.assertEqual(r["metrics"]["speculation"], SPLASH_METRICS["speculation"])
        self.assertEqual(r["metrics"]["request_latency"]["wall_ms"], 210.0)
        self.assertEqual(r["timings"]["draft_n_accepted"], 9)
        self.assertEqual(r["usage"]["completion_tokens"], 7)
        self.assertEqual((r["stream"], r["kind"], r["epoch"], r["recorded_prompt_tokens"]), ("main", "turn", 0, 2000))
        self.assertEqual((r["recorded_tools"], r["tool_names"], r["eval_result"]), (["bash", "read"], ["bash"], "PASS"))
        self.assertEqual((r["prompt_tokens"], r["cached_tokens"], r["tokens"]), (200, 50, 7))
        c = next(x for x in results if x["id"] == "R007_main_c0")
        self.assertEqual((c["content"], c["tool_calls"], c["eval_result"]), (SUMMARY, [], "PASS"))

        # --state re-displays it (no server, no fixture), alone and against itself
        path = os.path.join(self.states, files[0])
        os.remove(self.fixture)
        out = run_main("--state", path, "--no-mdcat")
        self.assertIn("R003_main_e0_t001", out)
        self.assertIn("Agent replay \u00b7 suite coding", out)
        out = run_main("--state", path, "--compare-state", path, "--no-mdcat")
        self.assertIn("R001_main_e0_t000", out)

    def test_state_file_and_no_save(self):
        forced = os.path.join(self.tmp.name, "forced.jsonl")
        with SplashLike(behaviour) as srv:
            self.run_replay(srv, "--turns", "1-2", "--state-file", forced)
            self.run_replay(srv, "--turns", "1-2", "--state-file", forced)  # appended: a second block
            self.run_replay(srv, "--turns", "1-2", "--no-save-state")
        self.assertFalse(os.path.exists(self.states))
        recs = lines(forced)
        self.assertEqual([r["type"] for r in recs], ["run", "result", "result", "end"] * 2)
        raw, keys = state.load_state(forced)
        self.assertEqual([k["key"] for k in keys], ["%s::m" % keys[0]["backend"],
                                                    "%s::m#2" % keys[0]["backend"]])


class StateComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write_state(self, path, suite="smoke", fixture=None, target=TARGET, case_ids=("C00", "C01", "C02"), **metric_kw):
        w = state.StateWriter(path)
        w.write(state.make_header(suite, target, request={"temperature": 0.0}, argv=["--suite", suite], fixture_path=fixture))
        raw = {}
        for i, cid in enumerate(case_ids):
            raw[cid] = metrics(i, id=cid, **metric_kw)
            w.write(state.result_record(raw[cid]))
        w.write(state.end_record(raw))
        w.close()

    def run_main_with_stderr(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            replay_bench.main(list(argv))
        return out.getvalue(), err.getvalue()

    def test_different_suite_is_refused(self):
        s1 = os.path.join(self.tmp.name, "s1.jsonl")
        s2 = os.path.join(self.tmp.name, "s2.jsonl")
        self.write_state(s1, suite="coding")
        self.write_state(s2, suite="judge")
        with self.assertRaisesRegex(UserError, "is suite coding, this run is judge"):
            display_state(s2, compare_state_path=s1)

    def test_different_fixture_is_refused(self):
        fx1 = os.path.join(self.tmp.name, "fx1.json.gz")
        fx2 = os.path.join(self.tmp.name, "fx2.json.gz")
        with open(fx1, "wb") as f:
            f.write(b"data1")
        with open(fx2, "wb") as f:
            f.write(b"data2")
        s1 = os.path.join(self.tmp.name, "s1.jsonl")
        s2 = os.path.join(self.tmp.name, "s2.jsonl")
        self.write_state(s1, suite="coding", fixture=fx1)
        self.write_state(s2, suite="coding", fixture=fx2)
        with self.assertRaisesRegex(UserError, "another fixture"):
            display_state(s2, compare_state_path=s1)

    def test_identical_runs_delta_zero(self):
        s1 = os.path.join(self.tmp.name, "s1.jsonl")
        self.write_state(s1, case_ids=["C00", "C01", "C02"])
        out, err = self.run_main_with_stderr("--state", s1, "--compare-state", s1, "--no-mdcat")
        self.assertNotIn("compared on", out)
        self.assertEqual(err, "")
        self.assertIn("+0.0%", out)

    def test_compare_state_different_lengths(self):
        s1 = os.path.join(self.tmp.name, "s1.jsonl")
        s2 = os.path.join(self.tmp.name, "s2.jsonl")
        self.write_state(s1, case_ids=["C00", "C01", "C02"])
        self.write_state(s2, case_ids=["C00", "C01"])
        out, err = self.run_main_with_stderr("--state", s2, "--compare-state", s1, "--no-mdcat")
        self.assertIn("compared on 2 common requests (baseline 3, this run 2)", out)
        self.assertIn("Baseline", out)
        self.assertIn("This run", out)

    def test_compare_state_empty_intersection(self):
        s1 = os.path.join(self.tmp.name, "s1.jsonl")
        s2 = os.path.join(self.tmp.name, "s2.jsonl")
        self.write_state(s1, case_ids=["C00", "C01"])
        self.write_state(s2, case_ids=["C02", "C03"])
        out, err = self.run_main_with_stderr("--state", s2, "--compare-state", s1, "--no-mdcat")
        self.assertIn("no common requests (baseline 2, this run 2), nothing to compare", out)  # the compared-with line
        self.assertIn("no common requests with the baseline", out)
        self.assertNotIn("Baseline", out)


class RegradeWarningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.fx = synthetic_fixture()
        self.fx_path = write_fixture(self.fx, self.dir, "fx.json.gz")
        old_fixtures = state.FIXTURES_DIR
        state.FIXTURES_DIR = self.dir
        self.addCleanup(setattr, state, "FIXTURES_DIR", old_fixtures)
        state._fixture_cache.clear()
        self.addCleanup(state._fixture_cache.clear)
        self.cid = "R001_main_e0_t000"

    def write_jsonl(self, recs, filename="a.jsonl"):
        path = os.path.join(self.dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        return path

    def test_fixture_sha_differs_warns_and_keeps_stored_verdicts(self):
        h = {"type": "run", "schema": 1, "suite": "coding",
             "fixture": {"file": "fx.json.gz", "sha256": "0" * 64},
             "backend": "mock:8080", "model": "m"}
        r = {"type": "result", "id": self.cid, "eval_result": "PASS", "fail_reason": None}
        p = self.write_jsonl([h, r])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            blocks = state.read_blocks(p)
        self.assertIn(f"warning: {p}: fixture fx.json.gz changed since this run; showing stored verdicts\n", err.getvalue())
        self.assertEqual(blocks[0]["results"][self.cid]["eval_result"], "PASS")

    def test_fixture_missing_warns(self):
        h = {"type": "run", "schema": 1, "suite": "coding",
             "fixture": {"file": "missing.json.gz", "sha256": "abc"},
             "backend": "mock:8080", "model": "m"}
        r = {"type": "result", "id": self.cid, "eval_result": "PASS", "fail_reason": None}
        p = self.write_jsonl([h, r])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            blocks = state.read_blocks(p)
        self.assertIn(f"warning: {p}: fixture missing.json.gz changed since this run; showing stored verdicts\n", err.getvalue())
        self.assertEqual(blocks[0]["results"][self.cid]["eval_result"], "PASS")

    def test_grade_raises_warns_with_count_and_keeps_stored_verdicts(self):
        h = {"type": "run", "schema": 1, "suite": "coding",
             "fixture": {"file": "fx.json.gz", "sha256": state.file_sha256(self.fx_path)},
             "backend": "mock:8080", "model": "m"}
        r = {"type": "result", "id": self.cid, "eval_result": "PASS", "fail_reason": None}
        p = self.write_jsonl([h, r])
        from bench.cases import grading
        err = io.StringIO()
        with mock.patch.object(grading, "grade", side_effect=ValueError("bad syntax")):
            with contextlib.redirect_stderr(err):
                blocks = state.read_blocks(p)
        self.assertIn(f"warning: {p}: 1 results could not be re-graded (bad syntax)\n", err.getvalue())
        self.assertEqual(blocks[0]["results"][self.cid]["eval_result"], "PASS")

    def test_skip_and_error_results_never_regraded_and_no_warnings(self):
        h = {"type": "run", "schema": 1, "suite": "coding",
             "fixture": {"file": "fx.json.gz", "sha256": "0" * 64},
             "backend": "mock:8080", "model": "m"}
        r1 = {"type": "result", "id": self.cid, "eval_result": "SKIP", "fail_reason": "turn skipped"}
        r2 = {"type": "result", "id": "R002_S1_t000", "error": "timeout", "eval_result": "FAIL"}
        p = self.write_jsonl([h, r1, r2])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            blocks = state.read_blocks(p)
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(blocks[0]["results"][self.cid]["eval_result"], "SKIP")

        h2 = {"type": "run", "schema": 1, "suite": "coding",
              "fixture": {"file": "fx.json.gz", "sha256": state.file_sha256(self.fx_path)},
              "backend": "mock:8080", "model": "m"}
        p2 = self.write_jsonl([h2, r1, r2], filename="b.jsonl")
        from bench.cases import grading
        err2 = io.StringIO()
        with mock.patch.object(grading, "grade", side_effect=ValueError("should not be called")):
            with contextlib.redirect_stderr(err2):
                state.read_blocks(p2)
        self.assertEqual(err2.getvalue(), "")

    def test_multiple_grade_errors_aggregated_once(self):
        h = {"type": "run", "schema": 1, "suite": "coding",
             "fixture": {"file": "fx.json.gz", "sha256": state.file_sha256(self.fx_path)},
             "backend": "mock:8080", "model": "m"}
        r1 = {"type": "result", "id": "R001_main_e0_t000", "eval_result": "PASS", "fail_reason": None}
        r2 = {"type": "result", "id": "R002_S1_t000", "eval_result": "PASS", "fail_reason": None}
        p = self.write_jsonl([h, r1, r2])
        from bench.cases import grading
        err = io.StringIO()
        with mock.patch.object(grading, "grade", side_effect=RuntimeError("eval crash")):
            with contextlib.redirect_stderr(err):
                blocks = state.read_blocks(p)
        self.assertEqual(err.getvalue(), f"warning: {p}: 2 results could not be re-graded (eval crash)\n")
        self.assertEqual(blocks[0]["results"]["R001_main_e0_t000"]["eval_result"], "PASS")
        self.assertEqual(blocks[0]["results"]["R002_S1_t000"]["eval_result"], "PASS")


class StateWriterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "s.jsonl")

    def _read_records(self, path=None):
        target = path or self.path
        with open(target, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_write_header_and_three_results_in_order(self):
        w = state.StateWriter(self.path)
        w.write(state.make_header("smoke", TARGET))
        for i in range(3):
            w.write(state.result_record(metrics(i)))
        w.close()
        recs = self._read_records()
        self.assertEqual(len(recs), 4)
        self.assertEqual([r["type"] for r in recs], ["run", "result", "result", "result"])
        self.assertEqual([r["id"] for r in recs[1:]], ["C00", "C01", "C02"])

    def test_keyboard_interrupt_has_every_finished_case(self):
        w = state.StateWriter(self.path)
        w.write(state.make_header("smoke", TARGET))
        raw = {}
        for i in range(4):
            raw[f"C{i:02d}"] = metrics(i)
            w.write(state.result_record(raw[f"C{i:02d}"]))
        w.write(state.end_record(raw, interrupted="Ctrl+C"))
        w.close()
        recs = self._read_records()
        self.assertEqual(len(recs), 6)
        self.assertEqual(recs[-1]["type"], "end")
        self.assertTrue(recs[-1]["interrupted"])
        self.assertEqual(recs[-1]["reason"], "Ctrl+C")
        self.assertEqual([r["id"] for r in recs[1:5]], ["C00", "C01", "C02", "C03"])

    def test_no_temp_file_left_behind_after_normal_finish(self):
        w = state.StateWriter(self.path)
        w.write(state.make_header("smoke", TARGET))
        for i in range(5):
            w.write(state.result_record(metrics(i)))
        w.close()
        dir_files = os.listdir(os.path.dirname(self.path))
        tmp_files = [f for f in dir_files if f.endswith(".tmp") or f.startswith(".")]
        self.assertEqual(tmp_files, [])
        self.assertIn(os.path.basename(self.path), dir_files)

    def test_appends_to_existing_file(self):
        w = state.StateWriter(self.path)
        w.write(state.make_header("smoke", TARGET))
        w.write(state.result_record(metrics(0)))
        w.close()

        w2 = state.StateWriter(self.path)
        w2.write(state.result_record(metrics(1)))
        w2.close()

        recs = self._read_records()
        self.assertEqual(len(recs), 3)
        self.assertEqual([r["type"] for r in recs], ["run", "result", "result"])
        self.assertEqual([r["id"] for r in recs[1:]], ["C00", "C01"])

    def test_partial_last_line_skipped_on_read(self):
        w = state.StateWriter(self.path)
        w.write(state.make_header("smoke", TARGET))
        w.write(state.result_record(metrics(0)))
        w.write(state.result_record(metrics(1)))
        w.close()

        # Simulate a crash leaving an incomplete last line
        with open(self.path, "a", encoding="utf-8") as f:
            f.write('{"type": "result", "id": "C02", "tokens": 5')

        blocks = state.read_blocks(self.path)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(list(blocks[0]["results"].keys()), ["C00", "C01"])

    def test_file_sha256_caching(self):
        fpath = os.path.join(self.tmp.name, "hash_test.txt")
        with open(fpath, "w") as f:
            f.write("hello")
        h1 = state.file_sha256(fpath)
        h2 = state.file_sha256(fpath)
        self.assertEqual(h1, h2)
        self.assertIn((os.path.abspath(fpath), os.stat(fpath).st_mtime_ns, os.stat(fpath).st_size), state._SHA_CACHE)


class SanityRegrade(unittest.TestCase):
    def test_saved_sanity_and_smoke_states_are_regraded(self):
        for suite in ("sanity", "smoke"):
            results = {
                "SC05_COUNT_AND_STOP": {"text": "1, 2, 3, 4, 5", "eval_result": "FAIL", "fail_reason": "old"},
                "SC08_ARITHMETIC": {"text": "7", "eval_result": "PASS", "fail_reason": ""},
                "SC10_UNICODE_ECHO": {"text": "x", "error": "timeout", "eval_result": "FAIL"},
            }
            self.assertEqual(state.regrade({"header": {"suite": suite}, "results": results}), (0, None, None))
            self.assertEqual(results["SC05_COUNT_AND_STOP"]["eval_result"], "PASS", suite)
            self.assertEqual(results["SC08_ARITHMETIC"]["eval_result"], "FAIL", suite)
            self.assertEqual(results["SC10_UNICODE_ECHO"]["eval_result"], "FAIL")  # error: verdict kept
            self.assertNotIn("fail_reason", results["SC10_UNICODE_ECHO"])


class RouteChange(unittest.TestCase):
    def test_fallback_after_the_header_is_recorded_and_loaded(self):
        from bench import transport
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.jsonl")
            target = {**TARGET, "url": "http://h:1/api/v0/chat/completions",
                      "fallback_url": "http://h:1/v1/chat/completions"}
            w = state.StateWriter(path)
            target["header"] = state.make_header("smoke", target)
            w.write(target["header"])
            target["state_writer"] = w
            sent = []
            def send(url):
                sent.append(url)
                return {"error": "HTTP 404 Not Found"} if "/api/v0" in url else {"text": "ok", "prompt_tokens": 3}
            with contextlib.redirect_stderr(io.StringIO()):
                transport.post_with_fallback(target, send)
            w.write(state.result_record(metrics(0)))
            w.close()
            self.assertEqual(sent[-1], "http://h:1/v1/chat/completions")
            with open(path) as f:
                recs = [json.loads(line) for line in f]
            self.assertEqual([r["type"] for r in recs], ["run", "route", "result"])
            self.assertEqual(recs[0]["endpoint"], "http://h:1/api/v0")  # the line on disk keeps the old route
            self.assertEqual(recs[1]["endpoint"], "http://h:1/v1")
            block = state.read_blocks(path)[0]
            self.assertEqual(block["header"]["endpoint"], "http://h:1/v1")


if __name__ == "__main__":
    unittest.main()
