"""End to end: the real CLI (subprocess) against tools/mock_server.py (subprocess) on 127.0.0.1. No real server is
contacted. Not part of `make test`; run with `make integration`.

    python3 -m unittest discover -s tests/integration
"""

import fcntl
import json
import os
import pty
import re
import select
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CLI = os.path.join(ROOT, "replay_bench.py")
MOCK = os.path.join(ROOT, "tools", "mock_server.py")
ENV = dict(os.environ, NO_COLOR="1", PYTHONUNBUFFERED="1")
ENV.pop("REPLAY_BENCH_NERD_FONT", None)
REPLAY = ["--suite", "coding", "--turns", "1-6"]
TIMEOUT = 45


class MockServer:
    """tools/mock_server.py as a subprocess on a free port; `log` collects the request bodies."""

    def __init__(self, tmp, flavor="llamacpp", delay=0.0, extra=()):
        self.log = os.path.join(tmp, "requests-%s.jsonl" % flavor)
        self.paths_log = os.path.join(tmp, "paths-%s.txt" % flavor)
        cmd = [sys.executable, MOCK, "--flavor", flavor, "--delay", str(delay), "--log", self.log,
               "--paths-log", self.paths_log, *extra]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=ENV)
        line = self.proc.stdout.readline().strip()
        if not line.startswith("listening on "):
            self.stop()
            raise RuntimeError("mock server did not start: %r" % line)
        self.url = line[len("listening on "):]

    def requests(self):
        """Replay requests (they carry tools); the small warm-up request before the first one is left out."""
        return [r for r in self.all_requests() if r.get("tools")]

    def paths(self):
        """`METHOD /path` of every request, in order (needs --paths-log)."""
        if not os.path.exists(self.paths_log):
            return []
        with open(self.paths_log) as f:
            return [line.strip() for line in f if line.strip()]

    def all_requests(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log) as f:
            return [json.loads(line) for line in f if line.strip()]

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc.stdout.close()


def cli(*args, **kw):
    return subprocess.run([sys.executable, CLI, *args], capture_output=True, text=True, env=ENV,
                          timeout=TIMEOUT, cwd=kw.get("cwd"))


def read_state(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def of_type(records, t):
    return [r for r in records if r.get("type") == t]


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="replay-bench-it-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.servers = []
        self.addCleanup(lambda: [s.stop() for s in self.servers])

    def server(self, **kw):
        s = MockServer(self.tmp, **kw)
        self.servers.append(s)
        return s

    def path(self, name):
        return os.path.join(self.tmp, name)

    def run_ok(self, srv, *args):
        r = cli("--endpoint", srv.url, *args)
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
        return r

    def test_smoke_writes_a_state_file(self):
        srv = self.server()
        state = self.path("smoke.jsonl")
        self.run_ok(srv, "--suite", "smoke", "--state-file", state)
        recs = read_state(state)
        self.assertEqual(recs[0]["type"], "run")
        self.assertEqual(recs[0]["suite"], "smoke")
        self.assertEqual(recs[-1]["type"], "end")
        results = of_type(recs, "result")
        self.assertEqual(len(results), 4)
        self.assertEqual(len({r["id"] for r in results}), 4)
        self.assertEqual(len(srv.all_requests()), 5)  # four cases and the warm-up
        self.assertTrue(all(r.get("eval_result") for r in results))

    def test_sanity_runs_to_verdicts(self):
        # The mock recognises the sanity prompts (tools/mock_server.py _sanity_answer) and answers them correctly;
        # the four smoke cases too: all ten PASS.
        srv = self.server()
        state = self.path("sanity.jsonl")
        r = self.run_ok(srv, "--preset", "sanity", "--state-file", state)
        recs = read_state(state)
        self.assertEqual(recs[0]["suite"], "sanity")
        self.assertEqual(recs[-1]["type"], "end")
        results = of_type(recs, "result")
        self.assertEqual(len(results), 10)
        self.assertTrue(all(x["eval_result"] in ("PASS", "WARN", "FAIL") for x in results))
        by_id = {x["id"]: x["eval_result"] for x in results}
        self.assertEqual(list(by_id.values()), ["PASS"] * 10, by_id)
        self.assertIn("SC09_NEEDLE_8K", r.stdout)
        shown = self.run_ok(srv, "--state", state)  # a saved sanity state shows again
        self.assertIn("SC10_UNICODE_ECHO", shown.stdout)

    def test_judge_and_replay_in_report_formats(self):
        srv = self.server()
        judge = self.run_ok(srv, "--suite", "judge", "--report-format", "markdown", "--state-file", self.path("j.jsonl"))
        self.assertIn("|", judge.stdout)
        for fmt in ("markdown", "chat-compact"):
            state = self.path("r-%s.jsonl" % fmt)
            r = self.run_ok(srv, *REPLAY, "--report-format", fmt, "--state-file", state)
            self.assertIn("R001_main_e0_t000", r.stdout)
            if fmt == "markdown":
                self.assertIn("|", r.stdout)
            self.assertEqual(len(of_type(read_state(state), "result")), 6)

    def test_state_replays_without_a_server(self):
        srv = self.server()
        state = self.path("s.jsonl")
        live = self.run_ok(srv, *REPLAY, "--state-file", state)
        srv.stop()
        replayed = cli("--state", state)
        self.assertEqual(replayed.returncode, 0, replayed.stdout[-1500:] + replayed.stderr[-1500:])
        self.assertIn("R006_main_e0_t005", replayed.stdout)
        self.assertIn("R006_main_e0_t005", live.stdout)

    def test_resume_sends_only_the_missing_requests(self):
        srv = self.server(delay=0.4)
        state = self.path("resume.jsonl")
        proc = subprocess.Popen([sys.executable, CLI, "--endpoint", srv.url, *REPLAY, "--state-file", state],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=ENV)
        try:
            deadline = time.time() + 30
            while time.time() < deadline and proc.poll() is None:
                if os.path.exists(state) and len(of_type(read_state(state), "result")) >= 2:
                    break
                time.sleep(0.05)
            self.assertIsNone(proc.poll(), "the run ended before it could be interrupted")
            proc.send_signal(signal.SIGINT)
            stderr = proc.communicate(timeout=20)[1]
        finally:
            if proc.poll() is None:
                proc.kill()
        graded = {r["id"] for r in of_type(read_state(state), "result") if not r.get("error")}
        self.assertTrue(2 <= len(graded) < 6, graded)
        before = len(srv.requests())
        tail = stderr.rstrip().splitlines()[-2:]  # the last lines: how to continue
        self.assertTrue(tail[0].startswith("interrupted: "), stderr[-500:])
        parts = shlex.split(tail[1])
        self.assertEqual(parts[0], "python3")
        self.assertEqual(parts[2], "--resume")
        r = subprocess.run([sys.executable, *parts[1:]],  # exactly the printed command
                           capture_output=True, text=True, env=ENV, timeout=TIMEOUT)
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
        recs = read_state(state)
        self.assertTrue(of_type(recs, "resume"), "no resume record")
        final = {}
        for r in of_type(recs, "result"):
            final[r["id"]] = r
        self.assertEqual(len(final), 6)
        for r in final.values():
            self.assertFalse(r.get("error"), r)
            self.assertTrue(r.get("eval_result"), r)
        self.assertEqual(len(srv.requests()) - before, 6 - len(graded))
        self.assertEqual(of_type(recs, "end")[-1]["cases"], 6)

    def test_return_progress_only_to_llamacpp(self):
        llama = self.server(flavor="llamacpp")
        self.run_ok(llama, "--suite", "coding", "--turns", "1-2", "--no-save-state")
        reqs = llama.requests()
        self.assertTrue(reqs)
        self.assertTrue(all(r.get("return_progress") is True for r in reqs))
        generic = self.server(flavor="generic")
        self.run_ok(generic, "--suite", "coding", "--turns", "1-2", "--no-save-state")
        reqs = generic.requests()
        self.assertTrue(reqs)
        self.assertFalse(any("return_progress" in r for r in reqs))

    def live_prefill_segments(self, srv, *args):
        """Run the CLI with stderr on a pty (the live line only draws on a terminal) and return the `prefill …` pieces
        of the live line it drew, as text."""
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 160, 0, 0))
        proc = subprocess.Popen([sys.executable, CLI, "--endpoint", srv.url, "--suite", "coding", "--turns", "1-1",
                                 "--no-save-state", *args], stdout=subprocess.DEVNULL, stderr=slave, env=ENV)
        os.close(slave)
        buf, deadline = b"", time.time() + TIMEOUT
        try:
            while time.time() < deadline:
                ready, _, _ = select.select([master], [], [], 0.5)
                if ready:
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    buf += data
                elif proc.poll() is not None:
                    break
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
            os.close(master)
        self.assertEqual(proc.returncode, 0, buf[-1500:])
        return re.findall(r"prefill [^\x1b]*", buf.decode("utf-8", "replace"))

    def test_mtplx_small_append_shows_the_estimate_not_zero_percent(self):
        srv = self.server(flavor="mtplx", delay=3.0)  # the suffix fits one 2048-token chunk: no reading until the end
        segs = self.live_prefill_segments(srv)
        pcts = [m.group(1) for seg in segs for m in [re.search(r"(~?\d+)%", seg)] if m]
        # no history for an estimate here, so only the timer shows: never "0% <?" from the zero-new-token reading
        self.assertFalse(pcts, segs)

    def test_mtplx_multi_chunk_prefill_gives_a_real_percentage(self):
        srv = self.server(flavor="mtplx", delay=4.0, extra=("--chunk", "4", "--cached-ratio", "0.2"))
        segs = self.live_prefill_segments(srv)
        pcts = [m.group(1) for seg in segs for m in [re.search(r"(~?\d+)%", seg)] if m]
        self.assertTrue(pcts, segs)
        self.assertTrue(any(not p.startswith("~") and int(p) > 0 for p in pcts), segs)  # real: from the snapshot
        real = [int(p) for p in pcts if not p.startswith("~")]
        self.assertEqual(real, sorted(real), segs)  # never backwards

    def test_no_prompt_progress_turns_the_snapshot_off(self):
        srv = self.server(flavor="mtplx", delay=2.0)
        segs = self.live_prefill_segments(srv, "--no-prompt-progress")
        self.assertFalse(any(re.search(r"(?<!~)\b\d+%", seg) for seg in segs), segs)

    def test_ollama_mock_server(self):
        srv = self.server(flavor="ollama")
        state = self.path("ollama-smoke.jsonl")
        r = self.run_ok(srv, "--suite", "smoke", "--state-file", state)
        self.assertIn("ollama 0.40.15", r.stdout)
        self.assertIn("Q4_K_M", r.stdout)
        recs = read_state(state)
        self.assertEqual(recs[0]["type"], "run")
        self.assertEqual(recs[0]["server"]["kind"], "ollama")
        self.assertEqual(recs[0]["n_ctx"], 262144)
        self.assertEqual(recs[0]["model_info"], {
            "quant": "Q4_K_M",
            "format": "gguf",
            "family": "qwen35",
            "parameter_size": "27.3B",
            "context_length": 262144,
            "runner": "llamacpp",
            "digest": "4010bd202712ea6bee89a163733011c461085c2bd1154caa2a49bcabb7c5eaa6",
        })

    def test_lmstudio_mock_server(self):
        srv = self.server(flavor="lmstudio")
        state = self.path("lmstudio-smoke.jsonl")
        r = self.run_ok(srv, "--suite", "smoke", "--state-file", state)
        self.assertIn("LM Studio", r.stdout)
        self.assertIn("4bit", r.stdout)
        recs = read_state(state)
        self.assertEqual(recs[0]["type"], "run")
        self.assertEqual(recs[0]["server"]["kind"], "lmstudio")
        self.assertEqual(recs[0]["n_ctx"], 65536)
        self.assertEqual(recs[0]["model_info"], {
            "quant": "4bit",
            "format": "mlx",
            "family": "mock_arch",
            "context_length": 65536,
        })
        res = [rec for rec in recs if rec.get("type") == "result"]
        self.assertTrue(res)
        self.assertIsNone(res[0]["native_decode_tps"])  # a stream carries no speeds
        self.assertIsNone(res[0]["native_ttft_ms"])
        self.assertIsNone(res[0].get("cached_tokens"))
        self.assertIsNone(res[0]["native_prefill_tps"])  # prompt_tokens / TTFT is no native figure
        self.assertEqual(res[0]["metrics"]["speculation"], {"drafted_tokens": 49, "accepted_draft_tokens": 21})
        posts = [p for p in srv.paths() if p.startswith("POST")]
        self.assertTrue(posts)
        self.assertEqual(set(posts), {"POST /api/v0/chat/completions"})
        self.assertTrue(recs[0]["endpoint"].endswith("/api/v0"), recs[0]["endpoint"])

    def test_mlx_dspark_mock_server(self):
        srv = self.server(flavor="mlx-dspark")
        state = self.path("dspark-smoke.jsonl")
        r = self.run_ok(srv, "--suite", "smoke", "--state-file", state)
        self.assertIn("mlx-dspark 0.20.4", r.stdout)
        self.assertIn("mlx-community/Qwen3.8-27B-4bit", r.stdout)
        self.assertIn("incoai/Qwen3.8-27B-DFlash2", r.stdout)
        self.assertNotIn("/Users/", r.stdout)
        recs = read_state(state)
        self.assertEqual(recs[0]["server"]["kind"], "mlx-dspark")
        self.assertEqual(recs[0]["server"]["version"], "0.20.4")
        self.assertEqual(recs[0]["model_info"], {"label": "mlx-community/Qwen3.8-27B-4bit", "mode": "dflash",
                                                 "drafter": "incoai/Qwen3.8-27B-DFlash2"})
        self.assertNotIn("/Users/", json.dumps(recs[0]))
        res = [rec for rec in recs if rec.get("type") == "result"]
        self.assertEqual(res[0]["native_decode_tps"], 50.9)  # decode_tokens_per_sec, not tokens_per_sec
        self.assertEqual(res[0]["metrics"]["speculation"]["cycles"], 15)

    def test_mlx_lm_mock_server(self):
        srv = self.server(flavor="mlx_lm")
        state = self.path("mlxlm-smoke.jsonl")
        r = self.run_ok(srv, "--suite", "smoke", "--state-file", state)
        self.assertIn("mlx_lm.server", r.stdout)
        recs = read_state(state)
        self.assertEqual(recs[0]["server"]["kind"], "mlx_lm")
        self.assertEqual(recs[0]["model"], "mlx-community/Mock-4bit")

    def test_generic_mock_is_not_taken_for_mlx_lm(self):
        srv = self.server(flavor="generic")
        state = self.path("generic-smoke.jsonl")
        self.run_ok(srv, "--suite", "smoke", "--state-file", state)
        self.assertEqual(read_state(state)[0]["server"]["kind"], "generic")

    def test_lmstudio_debug_names_the_route_and_the_last_chunk(self):
        srv = self.server(flavor="lmstudio")
        env = dict(ENV, REPLAY_BENCH_DEBUG="1")
        r = subprocess.run([sys.executable, CLI, "--suite", "smoke", "--endpoint", srv.url, "--no-save-state"],
                           capture_output=True, text=True, env=env, timeout=TIMEOUT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("[debug] POST ", r.stderr)
        self.assertIn("/api/v0/chat/completions", r.stderr)
        self.assertIn("last chunk keys", r.stderr)
        self.assertIn("'stats'", r.stderr)  # the last chunk carries `stats`
        self.assertIn("'total_draft_tokens_count': 49", r.stderr)
        quiet = cli("--suite", "smoke", "--endpoint", srv.url, "--no-save-state")
        self.assertNotIn("[debug]", quiet.stderr)

    def test_lmstudio_falls_back_to_v1_when_v0_chat_is_404(self):
        srv = self.server(flavor="lmstudio", extra=("--v0-chat-status", "404"))
        state = self.path("lmstudio-fallback.jsonl")
        r = self.run_ok(srv, "--suite", "smoke", "--state-file", state)
        self.assertEqual(r.stderr.count("using http://"), 1, r.stderr)
        posts = [p for p in srv.paths() if p.startswith("POST")]
        self.assertEqual(posts[0], "POST /api/v0/chat/completions")  # tried once, then /v1 for the rest
        self.assertEqual(posts.count("POST /api/v0/chat/completions"), 1, posts)
        self.assertTrue(len(posts) > 1)
        self.assertEqual(set(posts[1:]), {"POST /v1/chat/completions"})
        recs = read_state(state)
        self.assertTrue(recs[0]["endpoint"].endswith("/v1"), recs[0]["endpoint"])
        res = of_type(recs, "result")
        self.assertEqual(len(res), 4)
        self.assertFalse([x for x in res if x.get("error")])


if __name__ == "__main__":
    unittest.main()
