"""Offline tests for the replay_bench.py CLI flags. No server requests.

    python3 -m unittest discover -s tests
"""

import argparse
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

import replay_bench  # noqa: E402
from bench import (
    discovery,  # noqa: E402
    runner,  # noqa: E402
)
from bench.cli_args import parse_max_context, parse_turns_range  # noqa: E402
from bench.endpoints import DEFAULT_ENDPOINT, parse_endpoint, resolve_endpoints  # noqa: E402
from bench.state import load_state  # noqa: E402


def run_main(*argv):
    """Run main(argv) with run_benchmark mocked; return (exit code or None, stderr, run_benchmark mock)."""
    err = io.StringIO()
    code = None
    with mock.patch.object(replay_bench, "run_benchmark") as rb, contextlib.redirect_stderr(err):
        try:
            replay_bench.main(list(argv))
        except SystemExit as e:
            code = e.code
    return code, err.getvalue(), rb


class ParseHelpers(unittest.TestCase):
    def test_max_context(self):
        self.assertIsNone(parse_max_context("full"))
        self.assertIsNone(parse_max_context("FULL"))
        self.assertEqual(parse_max_context("64k"), 64 * 1024)
        self.assertEqual(parse_max_context("236K"), 236 * 1024)
        self.assertEqual(parse_max_context("16k"), 16 * 1024)
        self.assertEqual(parse_max_context("48k"), 48 * 1024)
        self.assertEqual(parse_max_context("256k"), 256 * 1024)
        self.assertEqual(parse_max_context("1k"), 1024)
        self.assertEqual(parse_max_context("1024k"), 1024 * 1024)
        self.assertEqual(parse_max_context("1024"), 1024)
        self.assertEqual(parse_max_context("65536"), 65536)
        for bad in ("0k", "1025k", "", "64", "1023", "-64k", "fullish", "bad"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_max_context(bad)

    def test_turns(self):
        self.assertEqual(parse_turns_range("10-40"), (10, 40))
        self.assertEqual(parse_turns_range("5-5"), (5, 5))
        for bad in ("10", "a-b", "40-10", "0-3", "1-", "-3", "1-2-3"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_turns_range(bad)


class Endpoints(unittest.TestCase):
    def test_default(self):
        self.assertEqual(DEFAULT_ENDPOINT, "http://localhost:8080/v1")
        self.assertEqual(resolve_endpoints(None), [{"backend": "localhost:8080", "base_url": "http://localhost:8080/v1"}])

    def test_parse_valid(self):
        r = parse_endpoint("https://mech.example:8132/v1")
        self.assertEqual(r, {"backend": "mech.example:8132", "base_url": "https://mech.example:8132/v1"})
        self.assertEqual(parse_endpoint("http://box.local.:8133/v1/")["backend"], "box.local:8133")
        self.assertEqual(parse_endpoint("http://box.local.:8133/v1/")["base_url"], "http://box.local.:8133/v1")

    def test_missing_v1_is_appended(self):
        for url in ("http://h:9000", "http://h:9000/"):
            self.assertEqual(parse_endpoint(url)["base_url"], "http://h:9000/v1")

    def test_default_ports_in_label(self):
        self.assertEqual(parse_endpoint("https://h/v1")["backend"], "h:443")
        self.assertEqual(parse_endpoint("http://h/v1")["backend"], "h:80")

    def test_invalid(self):
        for bad in ("localhost:8080/v1", "ftp://h/v1", "h", "", "http:///v1", "http://h:99999/v1",
                    "http://h:1/v1/chat/completions", "http://h:1/api", "http://u:p@h:1/v1", "http://h:1/v1?x=1"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                parse_endpoint(bad)

    def test_multiple_and_repeats(self):
        r = resolve_endpoints(["http://a:1/v1", "http://b:2", "http://a:1"])
        self.assertEqual([x["backend"] for x in r], ["a:1", "b:2"])

    def test_label_clash(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            resolve_endpoints(["http://a:1/v1", "https://a:1/v1"])


class CliFlags(unittest.TestCase):
    def test_default_endpoint_passed(self):
        _, _, rb = run_main()
        self.assertEqual([e["base_url"] for e in rb.call_args[1]["endpoints"]], ["http://localhost:8080/v1"])

    def test_multiple_endpoints(self):
        code, _, rb = run_main("--endpoint", "http://a:1/v1", "--endpoint", "https://b:2")
        self.assertIsNone(code)
        self.assertEqual([e["base_url"] for e in rb.call_args[1]["endpoints"]], ["http://a:1/v1", "https://b:2/v1"])

    def test_bad_endpoint_rejected(self):
        code, err, rb = run_main("--endpoint", "ftp://a/v1")
        self.assertEqual(code, 2)
        self.assertIn("scheme", err)
        rb.assert_not_called()

    def test_removed_flags_rejected(self):
        for flag in ("--fast", "--slow", "--tool", "--long", "--all-with-long", "--match-judge", "--match-judge-live",
                     "--luc-root=x", "--fdc-db=x", "--luc-app-url=x", "--target=Mech", "--ca-cert=x",
                     "--no-report", "--name=x"):
            code, _, rb = run_main(flag)
            self.assertEqual(code, 2, flag)
            rb.assert_not_called()


    def test_default_mode_is_the_large_preset(self):
        code, _, rb = run_main()
        self.assertIsNone(code)
        self.assertEqual(rb.call_args[0][0], "coding")
        self.assertEqual(rb.call_args[1]["preset"], "large")
        code, _, rb = run_main("--test", "TC01_GIT_COMMIT")  # --test needs the smoke cases, not a preset
        self.assertEqual(rb.call_args[0][0], "smoke")

    def test_smoke(self):
        for argv in (("--suite", "smoke"), ("--suite", "SMOKE")):
            code, _, rb = run_main(*argv)
            self.assertIsNone(code)
            self.assertEqual(rb.call_args[0][0], "smoke")
            self.assertIsNone(rb.call_args[1]["replay"])

    def test_old_suite_flags_rejected(self):
        for flag in ("--smoke", "--conv", "--conv-2", "--conv-3", "--every=2", "-n", "--iterations=2"):
            code, _, rb = run_main(flag)
            self.assertEqual(code, 2, flag)
            rb.assert_not_called()

    def test_unknown_suite_rejected(self):
        code, _, rb = run_main("--suite", "5")
        self.assertEqual(code, 2)
        rb.assert_not_called()

    def test_suite_names_and_fixtures(self):
        for name, mode, fx in (("coding", "coding", "conv-3.json.gz"),
                               ("judge", "judge", "judge.json.gz")):
            code, _, rb = run_main("--suite", name)
            self.assertIsNone(code, name)
            self.assertEqual(rb.call_args[0][0], mode)
            self.assertTrue(rb.call_args[1]["replay"]["fixture"].endswith(os.path.join("fixtures", fx)), name)
            self.assertIsNone(rb.call_args[1]["temperature"])
        _, _, rb = run_main("--suite", "judge", "--fixture", "x.json.gz")
        self.assertEqual(rb.call_args[1]["replay"]["fixture"], "x.json.gz")

    def test_replay_flags_error_with_smoke(self):
        for extra in (["--dry-run"], ["--no-subagents"], ["--max-context", "64k"], ["--turns", "1-3"],
                      ["--fixture", "x"]):
            code, _, rb = run_main("--suite", "smoke", *extra)
            self.assertEqual(code, 2, extra)
            if extra != ["--dry-run"]:  # alone, selection options mean the coding suite (--dry-run: the default preset)
                continue
            code, _, rb = run_main("--test", "TC01_GIT_COMMIT", *extra)
            self.assertEqual(code, 2, extra)
            rb.assert_not_called()

    def test_ultrafast_rejected(self):
        code, err, rb = run_main("--ultrafast")
        self.assertEqual(code, 2)
        rb.assert_not_called()

    def test_suite_coding_turns(self):
        code, _, rb = run_main("--suite", "coding", "--turns", "2-5")
        self.assertIsNone(code)
        self.assertEqual(rb.call_args[0][0], "coding")
        self.assertEqual(rb.call_args[1]["replay"]["turns"], (2, 5))
        self.assertIsNone(rb.call_args[1]["preset"])

    def test_suite_numeric_aliases_rejected(self):
        for bad in ("3", "4"):
            code, err, rb = run_main("--suite", bad)
            self.assertEqual(code, 2)
            rb.assert_not_called()
            self.assertIn("invalid suite '%s'" % bad, err)

    def test_suite_coding_options(self):
        code, _, rb = run_main("--suite", "coding", "--no-subagents")
        self.assertIsNone(code)
        self.assertEqual(rb.call_args[0][0], "coding")
        self.assertTrue(rb.call_args[1]["replay"]["no_subagents"])
        code, err, rb = run_main("--suite", "coding", "--test", "TC01_GIT_COMMIT")
        self.assertEqual(code, 2)
        self.assertIn("--test selects smoke or sanity cases", err)

    def test_new_options_parse(self):
        code, _, rb = run_main("--suite", "coding", "--max-context", "96k", "--no-subagents", "--turns", "3-9")
        self.assertIsNone(code)
        self.assertEqual(rb.call_args[0][0], "coding")
        self.assertNotIn("iterations", rb.call_args[1])
        r = rb.call_args[1]["replay"]
        self.assertEqual((r["max_context"], r["no_subagents"], r["turns"]), (96 * 1024, True, (3, 9)))
        self.assertNotIn("every", r)
        self.assertTrue(r["fixture"].endswith(os.path.join("fixtures", "conv-3.json.gz")))
        _, _, rb = run_main("--suite", "coding")
        self.assertIsNone(rb.call_args[1]["replay"]["max_context"])
        self.assertIsNone(rb.call_args[1]["temperature"])  # replay sends no temperature unless asked
        _, _, rb = run_main("--suite", "coding", "--temperature", "0.5")
        self.assertEqual(rb.call_args[1]["temperature"], 0.5)
        _, _, rb = run_main("--suite", "smoke")
        self.assertEqual(rb.call_args[1]["temperature"], 0.0)
        self.assertIsNone(rb.call_args[1]["replay"])

    def test_bad_values(self):
        for argv in (("--suite", "coding", "--max-context", "0k"), ("--suite", "coding", "--turns", "9-3")):
            code, _, rb = run_main(*argv)
            self.assertEqual(code, 2)
            rb.assert_not_called()

    def test_flexible_max_context_cli(self):
        code, _, rb = run_main("--suite", "coding", "--max-context", "48k")
        self.assertIsNone(code)
        self.assertEqual(rb.call_args[1]["replay"]["max_context"], 48 * 1024)
        code, _, rb = run_main("--suite", "coding", "--max-context", "65536")
        self.assertIsNone(code)
        self.assertEqual(rb.call_args[1]["replay"]["max_context"], 65536)


MODELS = [{"id": "org/Alpha-8B", "n_ctx": 1}, {"id": "org/Beta-27B", "n_ctx": 1}, {"id": "org/Alpha-8B-Q4", "n_ctx": 1}]


def discover(*, filters=None, first=False, models=MODELS):
    with mock.patch.object(discovery, "discover_models", return_value=models), \
            mock.patch("bench.backend.detect", return_value={"kind": "generic", "version": None, "build": None,
                                                              "details": {}}), \
            contextlib.redirect_stdout(io.StringIO()) as out:
        targets = discovery.discover_and_report(
            [{"backend": "h:1", "base_url": "http://h:1/v1"}], filters, first_model=first)
    return [t["name"] for t in targets], out.getvalue()


class ModelSelection(unittest.TestCase):
    def test_all_models_by_default(self):
        names, _ = discover()
        self.assertEqual(names, ["org/Alpha-8B", "org/Beta-27B", "org/Alpha-8B-Q4"])

    def test_first_model(self):
        self.assertEqual(discover(first=True)[0], ["org/Alpha-8B"])

    def test_filter_then_first(self):
        self.assertEqual(discover(filters=["beta", "q4"], first=True)[0], ["org/Beta-27B"])
        self.assertEqual(discover(filters=["q4"], first=True)[0], ["org/Alpha-8B-Q4"])
        self.assertEqual(discover(filters=["alpha"])[0], ["org/Alpha-8B", "org/Alpha-8B-Q4"])

    def test_no_match(self):
        names, out = discover(filters=["nope"], first=True)
        self.assertEqual(names, [])
        self.assertIn("no model matches --model nope", out)
        self.assertIn("org/Beta-27B", out)

    def test_flag_reaches_run_benchmark(self):
        _, _, rb = run_main("--first-model", "--model", "x")
        self.assertTrue(rb.call_args[1]["first_model"])
        self.assertEqual(rb.call_args[1]["model_filters"], ["x"])
        _, _, rb = run_main()
        self.assertFalse(rb.call_args[1]["first_model"])


FAKE_METRICS = {"text": '{"route": "x", "confidence": 1, "args": {}}', "ttft_ms": 10, "total_ms": 20, "tokens": 5,
                "prompt_tokens": 10, "cached_tokens": 0, "native_ttft_ms": None, "native_prefill_tps": None,
                "native_decode_tps": None}


class NoReportFiles(unittest.TestCase):
    """A smoke run against a mocked server saves its JSONL state under states/ (and writes nothing else)."""

    def run_smoke(self, *extra, models=("m1",)):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                with mock.patch("bench.discovery.discover_models", return_value=[{"id": m, "n_ctx": 262144} for m in models]), \
                        mock.patch("bench.backend.detect", return_value={"kind": "splash", "version": "ab12cd34",
                                                                         "build": None, "details": {}}), \
                        mock.patch.object(replay_bench, "STATES_DIR", os.path.join(tmp, "states")), \
                        mock.patch.object(runner, "query_model", side_effect=lambda *a, **k: dict(FAKE_METRICS)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    replay_bench.main(["--suite", "smoke", "--endpoint", "http://h:1/v1", "--no-mdcat"] + list(extra))
            finally:
                os.chdir(cwd)
            self.saved = sorted(os.listdir(os.path.join(tmp, "states"))) if os.path.isdir(os.path.join(tmp, "states")) else []
            return sorted(os.listdir(tmp)), tmp

    def test_auto_saves_one_file_under_states(self):
        repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        before = os.path.isdir(os.path.join(repo, "states"))
        files, tmp = self.run_smoke()
        self.assertEqual(files, ["states"])  # only the (patched) states dir
        self.assertEqual(os.path.isdir(os.path.join(repo, "states")), before)  # the repo's own is untouched
        self.assertEqual(len(self.saved), 1)
        self.assertRegex(self.saved[0], r"^splash-ab12cd34_m1_smoke_t0_\d{8}-\d{6}\.jsonl$")

    def test_no_save_state_writes_nothing(self):
        files, _ = self.run_smoke("--no-save-state")
        self.assertEqual(files, [])

    def test_state_file_writes_only_that_file(self):
        with tempfile.TemporaryDirectory() as out:
            path = os.path.join(out, "run.jsonl")
            files, _ = self.run_smoke("--state-file", path)
            self.assertEqual(files, [])
            raw, targets = load_state(path)
            self.assertEqual(targets[0]["backend"], "h:1")
            self.assertEqual(targets[0]["url"], "http://h:1/v1/chat/completions")
            self.assertIn("TC01_GIT_COMMIT", raw[targets[0]["key"]])
            self.assertEqual(sorted(os.listdir(out)), ["run.jsonl"])
            self.assertEqual(raw[targets[0]["key"]]["TC01_GIT_COMMIT"]["tokens"], 5)

    def test_one_state_file_per_model(self):
        with tempfile.TemporaryDirectory() as out:
            path = os.path.join(out, "run.jsonl")
            self.run_smoke("--state-file", path, models=("m1", "org/m2"))
            self.assertEqual(sorted(os.listdir(out)), ["run_m1.jsonl", "run_org-m2.jsonl"])
            for name, model in (("run_m1.jsonl", "m1"), ("run_org-m2.jsonl", "org/m2")):
                _, targets = load_state(os.path.join(out, name))
                self.assertEqual([t["name"] for t in targets], [model])  # one block, one model
        files, _ = self.run_smoke(models=("m1", "org/m2"))
        self.assertEqual(len(self.saved), 2)  # the automatic names differ by model already

    def test_resume_hint_only_for_the_interrupted_model(self):
        calls = []

        def fake(url, model, tc, *a, **k):
            calls.append(model)
            if model == "org/m2" and tc["id"] != "TC01_GIT_COMMIT":  # past the warmup, in the second model
                raise KeyboardInterrupt
            return dict(FAKE_METRICS)

        err = io.StringIO()
        with mock.patch("bench.discovery.discover_models", return_value=[{"id": m, "n_ctx": 1} for m in ("m1", "org/m2")]), \
                mock.patch("bench.backend.detect", return_value={"kind": "splash", "version": "v", "build": None, "details": {}}), \
                mock.patch.object(runner, "query_model", side_effect=fake), \
                tempfile.TemporaryDirectory() as tmp, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as cm:
                replay_bench.main(["--suite", "smoke", "--endpoint", "http://h:1/v1", "--no-mdcat",
                                   "--state-file", os.path.join(tmp, "run.jsonl")])
        self.assertEqual(cm.exception.code, 130)
        text = err.getvalue()
        self.assertEqual(text.count("--resume"), 1)
        self.assertIn("run_org-m2.jsonl", text.split("--resume")[1])

    def test_state_flags_are_exclusive(self):
        for argv in (("--state", "a", "--state-file", "b"), ("--state-file", "b", "--no-save-state")):
            code, _, rb = run_main(*argv)
            self.assertEqual(code, 2)
            rb.assert_not_called()

    def test_short_argparse_errors(self):
        for argv in (["--turns", "5-3"], ["--unknown-option"], ["--max-context", "bad"]):
            code, err, rb = run_main(*argv)
            self.assertEqual(code, 2, argv)
            rb.assert_not_called()
            lines = err.strip().splitlines()
            self.assertTrue(err.startswith("error:"), f"{argv}: expected stderr to start with 'error:', got {err!r}")
            self.assertLessEqual(len(lines), 3, f"{argv}: expected at most 3 lines, got {len(lines)}: {err!r}")
            self.assertIn("(see --help)", err, f"{argv}: expected '(see --help)' in stderr")
            self.assertNotIn("usage: replay_bench.py", err, f"{argv}: usage block should not be printed")


class HelpFormatting(unittest.TestCase):
    def test_help_width_and_structure(self):
        env = dict(os.environ, COLUMNS="100")
        out = io.StringIO()
        with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            replay_bench.main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        text = out.getvalue()

        # Epilog order: Presets before Notes before Examples
        idx_presets = text.index("Presets (")
        idx_notes = text.index("Notes:")
        idx_examples = text.index("Examples:")
        self.assertLess(idx_presets, idx_notes, "Presets should precede Notes in epilog")
        self.assertLess(idx_notes, idx_examples, "Notes should precede Examples in epilog")

        # Every help line <= 100 columns at COLUMNS=100 except presets table
        lines = text.splitlines()
        in_presets = False
        for i, line in enumerate(lines, 1):
            if line.startswith("Presets ("):
                in_presets = True
            elif line.startswith("Notes:"):
                in_presets = False
            if not in_presets:
                self.assertLessEqual(
                    len(line), 100, f"Line {i} exceeds 100 columns ({len(line)}): {line!r}"
                )

        self.assertIn("default for replay suites", text)
        self.assertIn("default: on (Splash, llama.cpp, Kapitsa, MTPLX)", text)
        self.assertIn("not with: --resume, --report-format chat, chat-compact (tabular)", text)


class StateSelectionTests(unittest.TestCase):
    def test_selection_options_rejected_with_state(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as f:
            f.write(json.dumps({"type": "run", "schema": 1, "suite": "smoke", "model": "m"}) + "\n")
            f.flush()
            for opt in (["--preset", "large"], ["--turns", "1-5"], ["--max-context", "64k"], ["--no-subagents"], ["--suite", "coding"]):
                code, err, rb = run_main("--state", f.name, *opt)
                self.assertEqual(code, 2, opt)
                rb.assert_not_called()
                self.assertIn("selection options don't apply to --state; the saved run is shown as recorded", err)

    def test_smoke_state_with_test_allowed(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as f:
            f.write(json.dumps({"type": "run", "schema": 1, "suite": "smoke", "model": "m"}) + "\n")
            f.flush()
            code, err, rb = run_main("--state", f.name, "--test", "TC01_GIT_COMMIT")
            self.assertNotIn("selection options don't apply to --state", err)


class AuthHeaderTests(unittest.TestCase):
    def setUp(self):
        self.recorded = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_GET(self):
                auth_hdr = self.headers.get("Authorization")
                outer.recorded.append((self.command, self.path, auth_hdr))
                if self.path == "/v1/models":
                    body = json.dumps({"object": "list", "data": [{"id": "m1", "n_ctx": 4096}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/status":
                    body = json.dumps({"schema_version": 8, "identity": {"cache": {"build_id": "src-12345678"}}}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                auth_hdr = self.headers.get("Authorization")
                outer.recorded.append((self.command, self.path, auth_hdr))
                if self.path == "/v1/chat/completions":
                    chunk = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(chunk)
                else:
                    self.send_response(404)
                    self.end_headers()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.base_url = "http://127.0.0.1:%d/v1" % self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        from bench import auth
        auth.set_api_key(None)
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_auth_header_recorded_for_discovery_backend_and_chat(self):
        from bench import auth, backend
        from bench.cases import select_cases
        auth.set_api_key("test-token-123")

        # 1. Backend detection
        backend.detect(self.base_url, timeout=2)
        # 2. Model discovery
        discovery.discover_models(self.base_url)
        # 3. Chat query
        tc = select_cases(["TC01_GIT_COMMIT"])[0]
        runner.query_model(f"{self.base_url}/chat/completions", "m1", tc)

        # Check recorded authorization headers
        paths_seen = [p for _, p, _ in self.recorded]
        self.assertIn("/v1/models", paths_seen)
        self.assertIn("/v1/chat/completions", paths_seen)
        for cmd, path, auth_val in self.recorded:
            self.assertEqual(auth_val, "Bearer test-token-123", f"{cmd} {path} got auth: {auth_val}")

    def test_api_key_flag_reaches_auth(self):
        from bench import auth
        run_main("--api-key", "cli-key-999")
        self.assertEqual(auth.api_key(), "cli-key-999")
        auth.set_api_key(None)
