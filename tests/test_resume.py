"""--resume: continue an interrupted run from its state file. Mock Splash on 127.0.0.1 only; no real server.

    python3 -m unittest tests.test_resume
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401
from test_replay import synthetic_fixture, write_fixture  # noqa: E402
from test_state import SplashLike, behaviour  # noqa: E402

import replay_bench  # noqa: E402
from bench import display, report, runner, state  # noqa: E402
from bench import resume as resume_mod  # noqa: E402

N_CASES = 10  # the synthetic fixture


def run_code(*argv):
    """(stdout, stderr, exit code) of main(argv); SystemExit is caught."""
    out, err, code = io.StringIO(), io.StringIO(), 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            replay_bench.main(list(argv))
        except SystemExit as e:
            code = e.code or 0
    return out.getvalue(), err.getvalue(), code


class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = write_fixture(synthetic_fixture(), self.tmp.name)
        self.path = os.path.join(self.tmp.name, "s.jsonl")
        self.addCleanup(lambda: report.configure("tui", io.StringIO()))

    def first_run(self, srv, stop_after=None, *extra):
        """The recorded run: --state-file, stopped by Ctrl+C after `stop_after` cases (warmup counts as a query)."""
        argv = ("--suite", "coding", "--fixture", self.fixture, "--endpoint", srv.url, "--state-file", self.path, *extra)
        if stop_after is None:
            return run_code(*argv)
        real, calls = runner.query_model, []

        def fake(*a, **k):
            calls.append(1)
            if len(calls) > stop_after + 1:
                raise KeyboardInterrupt()
            return real(*a, **k)
        with mock.patch.object(runner, "query_model", side_effect=fake):
            return run_code(*argv)

    def read(self):
        with open(self.path) as f:
            return f.read()

    def write(self, text):
        with open(self.path, "w") as f:
            f.write(text)

    def records(self):
        with open(self.path) as f:
            return [json.loads(x) for x in f if x.strip()]

    def types(self):
        return [r["type"] for r in self.records()]

    def case_requests(self, srv):
        return [r for r in srv.requests if r.get("max_tokens") != 256]  # not the warmup

    def test_interrupted_run_resumes_and_sends_only_the_missing_cases(self):
        with SplashLike(behaviour) as srv:
            _, _, code = self.first_run(srv, 3)
            self.assertEqual(code, 130)
            self.assertEqual(self.types(), ["run"] + ["result"] * 3 + ["end"])
            srv.requests.clear()
            out, err, code = run_code("--resume", self.path)
            self.assertEqual(code, 0, err)
            self.assertEqual(len(self.case_requests(srv)), N_CASES - 3)
        prog = os.path.basename(sys.argv[0])
        self.assertIn("resuming with: %s --suite coding" % prog, err)
        self.assertIn("3 of 10 cases done, 7 to run", err)
        self.assertEqual(self.types(), ["run"] + ["result"] * 3 + ["end", "resume"] + ["result"] * 7 + ["end"])
        recs = self.records()
        resume = recs[5]
        self.assertEqual(resume["argv"], ["--resume", self.path])
        self.assertIn("server", resume)
        self.assertFalse(recs[-1]["interrupted"])
        self.assertEqual(recs[-1]["cases"], N_CASES)
        for needle in ("R001_main_e0_t000", "R010_main_e1_t001", "Metrics"):  # loaded and new, one report
            self.assertIn(needle, out)
        raw, keys = state.load_state(self.path)
        self.assertEqual(len(keys), 1)
        order = [r["id"] for r in recs if r["type"] == "result"]
        self.assertEqual(list(raw[keys[0]["key"]]), order)
        self.assertIsNone(keys[0]["end"].get("reason"))

    def test_resuming_a_resumed_file_again(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv, 2)
            real, calls = runner.query_model, []

            def fake(*a, **k):
                calls.append(1)
                if len(calls) > 3:
                    raise KeyboardInterrupt()
                return real(*a, **k)
            with mock.patch.object(runner, "query_model", side_effect=fake):
                _, _, code = run_code("--resume", self.path)
            self.assertEqual(code, 130)
            self.assertEqual(len(state.load_state(self.path)[0][next(iter(state.load_state(self.path)[0]))]), 4)
            srv.requests.clear()
            _, err, code = run_code("--resume", self.path)
            self.assertEqual(code, 0, err)
            self.assertEqual(len(self.case_requests(srv)), N_CASES - 4)
        self.assertEqual(self.types().count("resume"), 2)
        raw, keys = state.load_state(self.path)
        self.assertEqual(len(raw[keys[0]["key"]]), N_CASES)
        self.assertFalse(keys[0]["end"]["interrupted"])

    def test_an_errored_case_is_sent_again_and_the_last_result_wins(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv)
            lines = self.read().splitlines()
            bad = json.loads(lines[3])  # the third result
            bad.update(error="boom", eval_result="FAIL")
            lines[3] = json.dumps(bad)
            self.write("\n".join(lines) + "\n")
            srv.requests.clear()
            _, err, code = run_code("--resume", self.path)
            self.assertEqual(code, 0, err)
            self.assertEqual(len(self.case_requests(srv)), 1)
        self.assertIn("9 of 10 cases done, 1 to run", err)
        raw, keys = state.load_state(self.path)
        results = raw[keys[0]["key"]]
        self.assertEqual(len(results), N_CASES)
        self.assertNotIn("error", results[bad["id"]])
        self.assertNotEqual(results[bad["id"]]["eval_result"], "FAIL")

    def test_a_given_option_that_differs_is_rejected_by_name(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv, 3)
            srv.requests.clear()
            for extra, name in ((["--max-context", "64k"], "--max-context: recorded none, given 65536"),
                                (["--temperature", "0.5"], "--temperature: recorded none, given 0.5"),
                                (["--turns", "1-4"], "--turns"), (["--no-subagents"], "--no-subagents"),
                                (["--model", "nothing-like-it"], "--model")):
                _, err, code = run_code("--resume", self.path, *extra)
                self.assertEqual(code, 2, extra)
                self.assertIn(name, err)
                self.assertIn("recorded", err)
            self.assertEqual(srv.requests, [])
        self.assertEqual(self.types().count("resume"), 0)

    def test_all_differing_options_are_listed(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv, 3)
            _, err, code = run_code("--resume", self.path, "--turns", "1-4", "--max-context", "64k")
        self.assertEqual(code, 2)
        self.assertIn("--turns", err)
        self.assertIn("--max-context", err)

    def test_equal_options_and_presentation_options_are_allowed(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv, 3)
            srv.requests.clear()
            out, err, code = run_code("--resume", self.path, "--suite", "coding", "--fixture", self.fixture,
                                      "--endpoint", srv.url + "/", "--no-temperature", "--report-format", "markdown",
                                      "--no-mdcat", "--no-prompt-progress")
            self.assertEqual(code, 0, err)
            self.assertEqual(len(self.case_requests(srv)), N_CASES - 3)
        self.assertIn("## ", out)  # markdown, as given
        self.assertTrue(all("return_progress" not in r for r in srv.requests))

    def test_the_chat_transcript_has_only_the_new_requests(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv, 3)
            out, _, code = run_code("--resume", self.path, "--report-format", "chat")
        self.assertEqual(code, 0)
        self.assertNotIn("R001_main_e0_t000", out)
        self.assertIn("R004_S1_t001", out)

    def test_other_fixture_content_is_rejected_before_anything_is_sent(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv, 3)
            fx = synthetic_fixture()
            fx["turns"][0]["recorded_answer"]["content"] = "changed"
            write_fixture(fx, self.tmp.name)
            srv.requests.clear()
            _, err, code = run_code("--resume", self.path)
            self.assertEqual(code, 2)
            self.assertIn("fixture changed", err)
            self.assertEqual(srv.requests, [])

    def test_other_model_is_rejected(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv, 3)
            lines = self.read().splitlines()
            head = json.loads(lines[0])
            head["model"] = "some-other-model"
            lines[0] = json.dumps(head)
            self.write("\n".join(lines) + "\n")
            srv.requests.clear()
            _, err, code = run_code("--resume", self.path)
            self.assertEqual(code, 2)
            self.assertIn("some-other-model", err)
            self.assertEqual(srv.requests, [])

    def test_server_and_commit_differences_only_warn(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv, 3)
            lines = self.read().splitlines()
            head = json.loads(lines[0])
            head["server"] = dict(head["server"], version="0ld0ld0l")
            head["git_commit"] = "abcdef0"
            head["n_ctx"] = 1234
            lines[0] = json.dumps(head)
            self.write("\n".join(lines) + "\n")
            _, err, code = run_code("--resume", self.path)
        self.assertEqual(code, 0, err)
        self.assertIn("resuming with: " + os.path.basename(sys.argv[0]), err)
        for needle in ("server version differs", "git commit differs", "n_ctx differs: 1234"):
            self.assertEqual(sum(needle in ln for ln in err.splitlines()), 1, needle)

    def test_a_complete_file_sends_nothing_and_exits_zero(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv)
            before = self.read()
            srv.requests.clear()
            out, err, code = run_code("--resume", self.path)
            self.assertEqual(srv.requests, [])
        self.assertEqual(code, 0, err)
        self.assertIn("10 of 10 cases done, 0 to run", err)
        self.assertIn("Metrics", out)
        self.assertIn("R010_main_e1_t001", out)
        self.assertEqual(self.read(), before)

    def test_several_run_blocks_are_rejected(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv, 3)
            text = self.read()
            self.write(text + text)
            srv.requests.clear()
            _, err, code = run_code("--resume", self.path)
            self.assertEqual(code, 2)
            self.assertIn("resume works on single-model state files", err)
            self.assertEqual(srv.requests, [])

    def test_incompatible_options(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv, 3)
            for flag in (["--state-file", "x.jsonl"], ["--no-save-state"], ["--state", self.path],
                         ["--compare-state", self.path], ["--dry-run"], ["--list"]):
                _, err, code = run_code("--resume", self.path, *flag)
                self.assertEqual(code, 2, flag)
                self.assertIn("--resume cannot be combined with " + flag[0], err)

    def test_a_missing_file_is_a_user_error(self):
        _, err, code = run_code("--resume", os.path.join(self.tmp.name, "nope.jsonl"))
        self.assertEqual(code, 2)
        self.assertIn("state file not found", err)


class Torn(unittest.TestCase):
    def test_resume_and_interrupted_end_are_one_block_with_last_wins(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.jsonl")
            recs = [{"type": "run", "schema": 1, "backend": "h:1", "model": "m", "endpoint": "http://h:1/v1"},
                    {"type": "result", "id": "A", "content": "x", "eval_result": "PASS", "error": "boom"},
                    {"type": "end", "interrupted": True, "reason": "Ctrl+C"},
                    {"type": "resume", "at": "t", "argv": []},
                    {"type": "result", "id": "A", "content": "y", "eval_result": "PASS"},
                    {"type": "result", "id": "B", "content": "z", "eval_result": "FAIL"}]
            with open(path, "w") as f:
                f.write("\n".join(json.dumps(r) for r in recs) + "\n")
            blocks = state.read_blocks(path)
            self.assertEqual(len(blocks), 1)
            self.assertEqual(list(blocks[0]["results"]), ["A", "B"])
            self.assertNotIn("error", blocks[0]["results"]["A"])
            self.assertIsNone(blocks[0]["end"])  # killed again after resuming: no end yet
            with open(path, "a") as f:
                f.write(json.dumps({"type": "end", "interrupted": False}) + "\n")
            self.assertFalse(state.read_blocks(path)[0]["end"]["interrupted"])


class Classification(unittest.TestCase):
    def test_every_option_is_in_exactly_one_class(self):
        parser = replay_bench.build_parser()
        classes = [resume_mod.RESUME_MATCH, resume_mod.RESUME_PRESENTATION, resume_mod.RESUME_INCOMPATIBLE,
                   resume_mod.RESUME_OWN]
        for a in parser._actions:
            if a.dest == "help":
                continue
            self.assertEqual(sum(a.dest in c for c in classes), 1, "%s must be in exactly one RESUME_* set" % a.option_strings)
        dests = {a.dest for a in parser._actions}
        for c in classes:
            self.assertLessEqual(c, dests)  # no stale names




class ResumeHint(unittest.TestCase):
    def write(self, d, name, models):
        path = os.path.join(d, name)
        with open(path, "w") as f:
            for m in models:
                f.write(json.dumps({"type": "run", "schema": 1, "suite": "smoke", "model": m}) + "\n")
                f.write(json.dumps({"type": "result", "id": "A", "tokens": 1}) + "\n")
                f.write(json.dumps({"type": "result", "id": "B", "error": "boom"}) + "\n")
        return path

    def test_hint_is_a_copy_pasteable_command(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, "my state.jsonl", ["m"])
            text = display.resume_hint([path, path], "replay_bench.py", 5)
        self.assertIn("interrupted: 1 of 5 cases done. To continue:", text)
        self.assertEqual(text.count("--resume"), 1)
        self.assertIn("python3 replay_bench.py --resume '%s'" % path, text)

    def test_several_models_in_one_file_are_not_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            text = display.resume_hint([self.write(d, "x.jsonl", ["a", "b"])], "replay_bench.py", 5)
        self.assertNotIn("--resume '", text)
        self.assertIn("2 models", text)

    def test_nothing_without_state(self):
        self.assertEqual(display.resume_hint([], "replay_bench.py", 5), "")


if __name__ == "__main__":
    unittest.main()
