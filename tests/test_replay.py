"""Offline tests for the replay suite (--suite coding): selection, lazy payloads, grading, SKIP and an end-to-end run
against an in-process mock server on 127.0.0.1 (no real model server is ever contacted).

    python3 -m unittest discover -s tests
"""

import contextlib
import copy
import gzip
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

import replay_bench  # noqa: E402
from bench.cases import grading, replay  # noqa: E402
from bench.state import load_state  # noqa: E402

REAL_FIXTURE_3 = os.path.join(ROOT, "fixtures", "conv-3.json.gz")

TOOLS = [{"type": "function", "function": {"name": n, "description": n, "parameters": {"type": "object"}}}
         for n in ("bash", "read")]


def call(name, args='{"x":1}'):
    return {"id": "call_" + name, "type": "function", "function": {"name": name, "arguments": args}}


def entry(id_, stream, label, kind, turn, epoch, prompt, calls=(), msgs=(0, 1), compaction=None, **params):
    p = {"model": "recorded-model", "stream": True, "stream_options": {"include_usage": True},
         "max_completion_tokens": 32768, "chat_template_kwargs": {"enable_thinking": True}}
    p.update(params)
    return {"id": id_, "stream": stream, "label": label, "kind": kind, "turn": turn, "compaction": compaction,
            "epoch": epoch, "messages": list(msgs), "tools": copy.deepcopy(TOOLS), "params": p,
            "prompt_tokens": prompt,
            "recorded_answer": {"role": "assistant", "content": "ok", "tool_calls": list(calls) or None,
                                "reasoning_content": None, "stopReason": "toolUse" if calls else "stop"},
            "time": "2026-11-06T20:00:00.000Z"}


def synthetic_fixture():
    """Main epoch 0, a compaction, epoch 1, and two subagent streams, interleaved."""
    pool = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
            {"role": "assistant", "content": "a"}, {"role": "tool", "tool_call_id": "c", "content": "r"}]
    turns = [
        entry("main_turn_0", "main", "main/e0", "turn", 0, 0, 1000, [call("bash")]),               # R001
        entry("S1_turn_0", "S1", "S1", "turn", 0, 0, 500, [call("read")], msgs=(0, 1, 2)),         # R002
        entry("main_turn_1", "main", "main/e0", "turn", 1, 0, 2000, [call("bash"), call("read")]),  # R003
        entry("S1_turn_1", "S1", "S1", "turn", 1, 0, 3000, [], msgs=(0, 1, 2, 3)),                 # R004
        entry("main_turn_2", "main", "main/e0", "turn", 2, 0, 4000, [call("read")]),                # R005
        entry("main_turn_3", "main", "main/e0", "turn", 3, 0, 1500, [call("read")]),                # R006
        entry("main_compaction_0", "main", "main/c0", "compaction", 0, 0, 4500, [], compaction=0,
              tool_choice="none"),                                                                  # R007
        entry("main_turn_4", "main", "main/e1", "turn", 0, 1, 800, [call("bash")]),                 # R008
        entry("S2_turn_0", "S2", "S2", "turn", 0, 0, 600, [call("bash")]),                          # R009
        entry("main_turn_5", "main", "main/e1", "turn", 1, 1, 900, []),                             # R010
    ]
    turns[6]["recorded_answer"]["content"] = "summary " * 150
    return {"suite": "coding", "case": "A", "session_id": "x", "model": "recorded-model", "provider": "p",
            "messages": pool, "turns": turns}


def write_fixture(fx, d, name="fx.json.gz"):
    path = os.path.join(d, name)
    raw = json.dumps(fx).encode()
    with (gzip.open(path, "wb") if name.endswith(".gz") else open(path, "wb")) as f:
        f.write(raw)
    return path


def ids(selected):
    return [replay.case_id(p, e) for p, e in selected]


class Selection(unittest.TestCase):
    def setUp(self):
        self.fx = synthetic_fixture()

    def positions(self, **kw):
        # max_context here is the prompt limit; the real option also reserves SKIP_RESERVE for the answer
        if kw.get("max_context") is not None:
            kw["max_context"] += replay.SKIP_RESERVE
        return [p for p, _ in replay.select_entries(self.fx, **kw)]

    def test_full_keeps_everything_in_order(self):
        self.assertEqual(self.positions(), list(range(1, 11)))
        self.assertEqual(self.positions(max_context=None), list(range(1, 11)))

    def test_ids(self):
        self.assertEqual(ids(replay.select_entries(self.fx)), [
            "R001_main_e0_t000", "R002_S1_t000", "R003_main_e0_t001", "R004_S1_t001", "R005_main_e0_t002",
            "R006_main_e0_t003", "R007_main_c0", "R008_main_e1_t000", "R009_S2_t000", "R010_main_e1_t001"])
        listed = ids(replay.select_entries(self.fx))
        self.assertEqual(listed, sorted(listed))

    def test_no_subagents(self):
        self.assertEqual(self.positions(no_subagents=True), [1, 3, 5, 6, 7, 8, 10])

    def test_max_context(self):
        # compactions and main epoch >= 1 dropped; main/e0 ends at its first turn above 3500 (R005, so R006 too);
        # S1 ends at its first turn above the limit (R004 at 3000 is within)
        self.assertEqual(self.positions(max_context=3500), [1, 2, 3, 4, 9])
        # S1's second turn is above 2500: S1 ends there; main/e0 ends at R005 (4000)
        self.assertEqual(self.positions(max_context=2500), [1, 2, 3, 9])
        # a stream ends at its first turn above the limit even if a later turn is below it again
        self.assertEqual(self.positions(max_context=1000, no_subagents=True), [1])
        self.assertEqual(self.positions(max_context=10000), [1, 2, 3, 4, 5, 6, 9])

    def test_max_context_leaves_the_skip_reserve(self):
        # R001 has 1000 prompt tokens: it fits a server of exactly 1000 + SKIP_RESERVE, not one token less
        n = 1000 + replay.SKIP_RESERVE
        self.assertEqual([p for p, _ in replay.select_entries(self.fx, max_context=n, no_subagents=True)], [1])
        self.assertEqual([p for p, _ in replay.select_entries(self.fx, max_context=n - 1, no_subagents=True)], [])

    def test_max_context_uses_recorded_prompt_tokens(self):
        self.fx["turns"][0]["recorded_prompt_tokens"] = 9000  # original count above the re-counted 1000
        self.assertEqual(self.positions(max_context=3500, no_subagents=True), [])
        self.fx["turns"][0]["recorded_prompt_tokens"] = 100
        self.assertEqual(self.positions(max_context=3500, no_subagents=True), [1, 3])

    def test_turns_slices_what_is_left(self):
        self.assertEqual(self.positions(turns=(2, 4)), [2, 3, 4])
        self.assertEqual(self.positions(turns=(8, 99)), [8, 9, 10])
        self.assertEqual(self.positions(turns=(3, 3)), [3])
        self.assertEqual(self.positions(turns=(1, 3), no_subagents=True), [1, 3, 5])
        self.assertEqual(self.positions(turns=(2, 3), max_context=3500), [2, 3])


    def test_all_combined(self):
        # no subagents -> [1,3,5,6,7,8,10]; limit 3500 -> [1,3]; turns 1-2 -> [1,3]
        self.assertEqual(self.positions(no_subagents=True, max_context=3500, turns=(1, 2)), [1, 3])
        self.assertEqual(self.positions(max_context=10000, turns=(2, 6)), [2, 3, 4, 5, 6])

    def test_summary_line(self):
        line = replay.summary_line(replay.select_entries(self.fx))
        self.assertIn("10 entries (main 6, subagent 3, compaction 1)", line)
        self.assertIn("max prompt 4500 tokens", line)
        self.assertIn("0 entries", replay.summary_line([]))

    def test_dry_run_table(self):
        tbl = replay.dry_run_table(replay.select_entries(self.fx, turns=(3, 3)))
        self.assertEqual(len(tbl.rows), 1)
        self.assertEqual(tbl.rows[0][0], "R003_main_e0_t001")
        self.assertEqual(tbl.rows[0][-1], "bash,read")


class CasesAndPayload(unittest.TestCase):
    def setUp(self):
        self.fx = synthetic_fixture()

    def cases(self, **kw):
        return [replay.make_case(self.fx, p, e) for p, e in replay.select_entries(self.fx, **kw)]

    def test_case_fields(self):
        cs = {c["id"]: c for c in self.cases()}
        self.assertEqual(cs["R001_main_e0_t000"]["type"], "Replay main")
        self.assertEqual(cs["R002_S1_t000"]["type"], "Replay subagent")
        self.assertEqual(cs["R007_main_c0"]["type"], "Replay compaction")
        c = cs["R003_main_e0_t001"]
        self.assertEqual((c["stream"], c["kind"], c["epoch"]), ("main", "turn", 0))
        self.assertEqual(c["recorded_tools"], ["bash", "read"])
        self.assertEqual(c["recorded_prompt_tokens"], 2000)
        self.assertEqual((c["max_gen_tokens"], c["max_gen_seconds"]), (32768, 1800))
        self.assertEqual(cs["R004_S1_t001"]["recorded_tools"], [])

    def test_case_is_lazy(self):
        for c in self.cases():
            self.assertNotIn("messages", c)
            self.assertNotIn("tools", c)
            self.assertNotIn("payload", c)
            self.assertIs(c["fixture"], self.fx)  # a reference, not a copy
            for v in c.values():
                if isinstance(v, list):
                    self.assertFalse(any(isinstance(x, dict) for x in v))

    def test_payload(self):
        cs = {c["id"]: c for c in self.cases()}
        p = replay.build_payload(cs["R003_main_e0_t001"], "target-model")
        self.assertEqual(p["model"], "target-model")
        self.assertEqual([m["role"] for m in p["messages"]], ["system", "user"])
        self.assertEqual(p["tools"], TOOLS)
        self.assertEqual(p["max_completion_tokens"], 32768)
        self.assertEqual(p["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(p["stream_options"], {"include_usage": True})
        self.assertTrue(p["stream"])
        for absent in ("max_tokens", "temperature", "reasoning_effort", "tool_choice"):
            self.assertNotIn(absent, p)
        self.assertEqual(self.fx["turns"][2]["params"]["model"], "recorded-model")  # the fixture is untouched
        pc = replay.build_payload(cs["R007_main_c0"], "m")
        self.assertEqual(pc["tool_choice"], "none")
        self.assertEqual(len(pc["messages"]), 2)
        p4 = replay.build_payload(cs["R004_S1_t001"], "m", reasoning_effort="high", temperature=0.2)
        self.assertEqual((p4["reasoning_effort"], p4["temperature"]), ("high", 0.2))
        self.assertEqual(len(p4["messages"]), 4)

    def test_payload_thinking(self):
        c = {c["id"]: c for c in self.cases()}["R003_main_e0_t001"]
        rec = {"enable_thinking": True, "preserve_thinking": True}
        c["entry"]["params"]["chat_template_kwargs"] = dict(rec)
        off = replay.build_payload(c, "m", thinking=False)
        self.assertEqual(off["chat_template_kwargs"], {"enable_thinking": False, "preserve_thinking": True})
        self.assertIsNot(off["chat_template_kwargs"], c["entry"]["params"]["chat_template_kwargs"])
        self.assertEqual(c["entry"]["params"]["chat_template_kwargs"], rec)  # the shared params are untouched
        c["entry"]["params"]["chat_template_kwargs"] = {"enable_thinking": False}
        self.assertTrue(replay.build_payload(c, "m", thinking=True)["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(c["entry"]["params"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(replay.build_payload(c, "m")["chat_template_kwargs"], {"enable_thinking": False})  # as recorded
        self.assertEqual(replay.build_payload(c, "m", reasoning_effort="none")["chat_template_kwargs"],
                         {"enable_thinking": False})
        del c["entry"]["params"]["chat_template_kwargs"]
        self.assertEqual(replay.build_payload(c, "m", thinking=False)["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("chat_template_kwargs", replay.build_payload(c, "m"))
        self.assertNotIn("chat_template_kwargs", c["entry"]["params"])

    def test_skip_reason(self):
        c = {c["id"]: c for c in self.cases()}["R007_main_c0"]  # 4500 tokens
        self.assertIsNone(replay.skip_reason(c, {"n_ctx": None}))
        self.assertIsNone(replay.skip_reason(c, {}))
        self.assertIsNone(replay.skip_reason(c, {"n_ctx": 4500 + replay.SKIP_RESERVE}))
        self.assertIn("ctx 8000", replay.skip_reason(c, {"n_ctx": 8000}))

    def test_fixture_loading(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.json.gz", "a.json"):
                self.assertEqual(replay.load_fixture(write_fixture(self.fx, d, name))["case"], "A")


class Grading(unittest.TestCase):
    def setUp(self):
        self.cases = {c["id"]: c for c in (replay.make_case(fx, p, e) for fx in [synthetic_fixture()] for p, e in replay.select_entries(fx))}

    def grade(self, cid, text="", calls=None, **extra):
        m = {"text": text}
        if calls is not None:
            m["tool_calls"] = calls
        m.update(extra)
        return grading.grade(self.cases[cid], m), m

    def test_tool_names_and_agreement_metric(self):
        (v, r), m = self.grade("R003_main_e0_t001", calls=[call("bash"), call("read")])
        self.assertEqual((v, r), ("PASS", "same as recorded: bash,read"))
        self.assertEqual((m["tool_names"], m["matches_recorded"]), (["bash", "read"], True))
        (v, r), m = self.grade("R003_main_e0_t001", calls=[call("read"), call("bash")])  # any order
        self.assertEqual((v, r, m["matches_recorded"]), ("PASS", "same as recorded: bash,read", True))
        (v, r), m = self.grade("R001_main_e0_t000", calls=[call("bash"), call("read")])
        self.assertEqual((v, r, m["matches_recorded"]), ("PASS", "other tools: bash,read (recorded bash)", False))
        (v, r), _ = self.grade("R004_S1_t001", calls=[call("read")])
        self.assertEqual((v, r), ("PASS", "other tools: read (recorded text)"))

    def test_final_answer_pass(self):
        (v, r), m = self.grade("R004_S1_t001", text="all done")
        self.assertEqual((v, r, m["tool_names"], m["matches_recorded"]), ("PASS", "same as recorded: text", [], True))


# ---------------------------------------------------------------------------------------- mock server

class MockServer:
    """OpenAI-compatible mock on 127.0.0.1:<ephemeral>: /v1/models and streaming /v1/chat/completions."""

    def __init__(self, n_ctx=262144, behaviour=None):
        self.requests = []
        self.n_ctx = n_ctx
        self.behaviour = behaviour or (lambda payload: {"text": "Thinking.", "calls": [("bash", '{"x":1}')]})
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"data": [{"id": "mock-model", "meta": {"n_ctx": outer.n_ctx}}]}).encode()
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
                for i in range(0, len(spec.get("text", "")), 4):
                    chunks.append({"choices": [{"delta": {"content": spec["text"][i:i + 4]}}]})
                for idx, (name, args) in enumerate(spec.get("calls", [])):
                    chunks.append({"choices": [{"delta": {"tool_calls": [{
                        "index": idx, "id": "call_%d" % idx, "type": "function",
                        "function": {"name": name, "arguments": ""}}]}}]})
                    half = len(args) // 2
                    for part in (args[:half], args[half:]):
                        chunks.append({"choices": [{"delta": {"tool_calls": [{
                            "index": idx, "function": {"arguments": part}}]}}]})
                n_msgs = len(payload.get("messages", []))
                chunks.append({"choices": [], "usage": {"prompt_tokens": 100 * n_msgs, "completion_tokens": 7,
                                                        "prompt_tokens_details": {"cached_tokens": 50}}})
                try:
                    for c in chunks:
                        self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = "http://127.0.0.1:%d/v1" % self.httpd.server_address[1]

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def plain_dry_lines(text):
    """A tui --dry-run (box table, then the summary note) as the old plain lines: header, rows, summary."""
    return [l.replace("\u2502", " ") if l.startswith("\u2502") else l
            for l in text.splitlines() if l.strip() and l[0] not in "\u250c\u251c\u2514"]


def run_main(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        replay_bench.main(list(argv))
    return out.getvalue()


SUMMARY = ("A summary of the work so far. " * 40).strip()


def compaction_aware(payload):
    if payload.get("tool_choice") == "none":
        return {"text": SUMMARY}
    if payload.get("max_tokens") == 256:  # warmup
        return {"text": "feat: x"}
    return {"text": "Thinking.", "calls": [("bash", '{"x":1}')]}


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = write_fixture(synthetic_fixture(), self.tmp.name)
        self.state = os.path.join(self.tmp.name, "state.jsonl")

    def run_replay(self, srv, *extra):
        return run_main("--suite", "coding", "--fixture", self.fixture, "--endpoint", srv.url, "--no-mdcat",
                        "--state-file", self.state, *extra)

    def results(self):
        raw, targets = load_state(self.state)
        return raw[targets[0]["key"]], targets[0]

    def test_full_run(self):
        with MockServer(behaviour=compaction_aware) as srv:
            out = self.run_replay(srv)
        self.assertIn("selection: 10 entries", out)
        warm, replayed = srv.requests[0], srv.requests[1:]
        self.assertEqual(warm["max_tokens"], 256)  # the generic warmup case, not a replay request
        self.assertEqual(len(replayed), 10)
        for p in replayed:
            self.assertEqual(p["model"], "mock-model")
            self.assertNotIn("max_tokens", p)
            self.assertNotIn("temperature", p)  # omp sent none and the replay sends none unless asked
            self.assertEqual(p["max_completion_tokens"], 32768)
            self.assertEqual(p["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual([p.get("tool_choice") for p in replayed], [None] * 6 + ["none"] + [None] * 3)
        self.assertEqual([len(p["messages"]) for p in replayed], [2, 3, 2, 4, 2, 2, 2, 2, 2, 2])

        res, target = self.results()
        self.assertEqual(target["n_ctx"], 262144)
        r = res["R003_main_e0_t001"]
        self.assertEqual((r["stream"], r["kind"], r["epoch"]), ("main", "turn", 0))
        self.assertEqual(r["recorded_prompt_tokens"], 2000)
        self.assertEqual(r["recorded_tools"], ["bash", "read"])
        self.assertEqual(r["tool_names"], ["bash"])
        self.assertEqual(r["eval_result"], "PASS")
        self.assertEqual(r["fail_reason"], "other tools: bash (recorded bash,read)")
        self.assertFalse(r["matches_recorded"])
        self.assertEqual((r["prompt_tokens"], r["cached_tokens"], r["tokens"]), (200, 50, 7))
        self.assertTrue(r["text"].startswith("Thinking.\n[{"))
        self.assertEqual(res["R001_main_e0_t000"]["eval_result"], "PASS")
        c = res["R007_main_c0"]
        self.assertEqual((c["kind"], c["eval_result"], c["text"]), ("compaction", "PASS", SUMMARY))
        self.assertEqual(res["R010_main_e1_t001"]["eval_result"], "PASS")  # recorded text, got bash: other tools

    def test_state_replay(self):
        with MockServer(behaviour=compaction_aware) as srv:
            self.run_replay(srv, "--no-subagents")
        res, _ = self.results()
        self.assertEqual(sorted(res), ["R001_main_e0_t000", "R003_main_e0_t001", "R005_main_e0_t002",
                                       "R006_main_e0_t003", "R007_main_c0", "R008_main_e1_t000", "R010_main_e1_t001"])
        # --state: no server, no fixture needed
        os.remove(self.fixture)
        out = run_main("--state", self.state, "--no-mdcat")
        self.assertIn("R003_main_e0_t001", out)
        self.assertIn("Agent replay \u00b7 suite coding", out)
        out = run_main("--state", self.state, "--compare-state", self.state, "--no-mdcat")
        self.assertIn("R001_main_e0_t000", out)

    def test_compare_state_live(self):
        with MockServer(behaviour=compaction_aware) as srv:
            self.run_replay(srv, "--turns", "1-3")
            base = os.path.join(self.tmp.name, "base.jsonl")
            os.replace(self.state, base)
            out = self.run_replay(srv, "--turns", "1-3", "--compare-state", base)
        self.assertIn("R001_main_e0_t000", out)

    def test_skip_by_n_ctx(self):
        # n_ctx 6000, reserve 4096: only prompts <= 1904 tokens are sent
        with MockServer(n_ctx=6000, behaviour=compaction_aware) as srv:
            out = self.run_replay(srv)
        self.assertEqual(len(srv.requests) - 1, 6)  # 1000, 500, 1500, 800, 600, 900 tokens are sent
        res, _ = self.results()
        skipped = sorted(k for k, v in res.items() if v["eval_result"] == "SKIP")
        self.assertEqual(skipped, ["R003_main_e0_t001", "R004_S1_t001", "R005_main_e0_t002", "R007_main_c0"])
        s = res["R005_main_e0_t002"]
        self.assertEqual(s["eval_result"], "SKIP")
        self.assertIn("ctx 6000", s["fail_reason"])
        self.assertEqual(s["stream"], "main")
        self.assertIn("SKIP", out)

    def test_selection_flags_reach_the_run(self):
        with MockServer(behaviour=compaction_aware) as srv:
            self.run_replay(srv, "--max-context", "64k", "--turns", "1-5")
        res, _ = self.results()
        # 64k keeps all but compaction and epoch 1 (R001-R006, R009); --turns 1-5 takes the first five
        self.assertEqual(sorted(res), ["R001_main_e0_t000", "R002_S1_t000", "R003_main_e0_t001", "R004_S1_t001",
                                       "R005_main_e0_t002"])

    def test_reasoning_effort_and_temperature_on_request(self):
        with MockServer(behaviour=compaction_aware) as srv:
            self.run_replay(srv, "--turns", "1-1", "--reasoning-effort", "high", "--temperature", "0.3")
        p = srv.requests[1]
        self.assertEqual((p["reasoning_effort"], p["temperature"]), ("high", 0.3))

    def test_thinking_off_on_request(self):
        with MockServer(behaviour=compaction_aware) as srv:
            self.run_replay(srv, "--turns", "1-1", "--thinking", "off")
        self.assertIs(srv.requests[1]["chat_template_kwargs"]["enable_thinking"], False)

    def test_errors_fail_the_request(self):
        def broken(payload):
            return {"text": "x", "calls": [("bash", '{"x": ')]}
        with MockServer(behaviour=broken) as srv:
            self.run_replay(srv, "--turns", "1-1")
        res, _ = self.results()
        r = res["R001_main_e0_t000"]
        self.assertEqual(r["eval_result"], "FAIL")
        self.assertIn("not valid JSON", r["fail_reason"])


class CliSelectionFlags(unittest.TestCase):
    def run_err(self, *argv):
        err = io.StringIO()
        code = None
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            try:
                replay_bench.main(list(argv))
            except SystemExit as e:
                code = e.code
        return code, err.getvalue()

    def test_selection_flags_rejected_with_smoke(self):
        for argv in (["--suite", "smoke", "--max-context", "64k"], ["--suite", "smoke", "--no-subagents"],
                     ["--suite", "smoke", "--dry-run"], ["--suite", "smoke", "--fixture", "x"],
                     ["--suite", "smoke", "--turns", "1-3"], ["--preset", "smoke", "--turns", "1-3"],
                     ["--test", "TC01_GIT_COMMIT", "--no-subagents"], ["--test", "TC01_GIT_COMMIT", "--dry-run"]):
            code, err = self.run_err(*argv)
            self.assertEqual(code, 2, argv)
            self.assertTrue("--suite" in err or "smoke takes no selection options" in err, (argv, err))

    def test_test_flag_rejected_with_conv2(self):
        code, err = self.run_err("--suite", "coding", "--test", "TC01_GIT_COMMIT")
        self.assertEqual(code, 2)

    def test_missing_fixture(self):
        code, err = self.run_err("--suite", "coding", "--dry-run", "--fixture", "/nonexistent/fx.json.gz")
        self.assertEqual(code, 2)
        self.assertIn("cannot read fixture", err)

    def test_dry_run_synthetic_makes_no_requests(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_fixture(synthetic_fixture(), d)
            with MockServer() as srv:
                with self.assertRaises(SystemExit) as cm:
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        replay_bench.main(["--suite", "coding", "--dry-run", "--fixture", path, "--endpoint", srv.url,
                                           "--max-context", "64k", "--turns", "2-4"])
                self.assertEqual(cm.exception.code, 0)
                self.assertEqual(srv.requests, [])
            text = out.getvalue()
            self.assertIn("R002_S1_t000", text)
            self.assertIn("R004_S1_t001", text)
            self.assertNotIn("R005", text)
            self.assertIn("coding selection: 3 entries", text)


class RealFixture3DryRun(unittest.TestCase):
    def dry(self, *flags):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            replay_bench.main(["--suite", "coding", "--dry-run", *flags])
        self.assertEqual(cm.exception.code, 0)
        lines = plain_dry_lines(out.getvalue())
        return lines[1:-1], lines[-1]

    def test_full(self):
        rows, summary = self.dry()
        self.assertEqual(len(rows), 470)
        self.assertIn("coding selection: 470 entries (main 297, subagent 172, compaction 1)", summary)
        self.assertEqual(len({r.split()[0] for r in rows}), 470)
        self.assertEqual([r.split()[1] for r in rows].count("main/c0"), 1)

    def test_no_subagents_and_limit(self):
        rows, summary = self.dry("--no-subagents")
        self.assertEqual(len(rows), 298)
        self.assertIn("subagent 0", summary)
        rows, _ = self.dry("--max-context", "64k")
        self.assertEqual(len(rows), 76)  # all streams that fit 64k with the answer reserve
        for r in rows:
            f = r.split()
            self.assertNotEqual(f[2], "compaction")
            self.assertLessEqual(int(f[4]) + replay.SKIP_RESERVE, 64 * 1024)


if __name__ == "__main__":
    unittest.main()
