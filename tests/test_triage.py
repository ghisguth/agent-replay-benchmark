"""tools/triage_states.py on small synthetic states in a temp dir (no server)."""

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
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401
import triage_states  # noqa: E402
from test_replay import synthetic_fixture, write_fixture  # noqa: E402

from bench import paths, state  # noqa: E402
from bench.cases import grading  # noqa: E402


def header(fixture=None, suite="smoke"):
    return {"type": "run", "schema": 1, "suite": suite, "fixture": fixture, "started_at": "2026-11-07T10:00:00+00:00",
            "backend": "host:1", "model": "m/x", "device": {"name": "dev"}, "request": {"temperature": 0.6}}


def result(cid, verdict, reason=None, **kw):
    return {"type": "result", "id": cid, "eval_result": verdict, "fail_reason": reason, "prompt_tokens": 1500,
            "content": "x", **kw}


class Triage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.path = os.path.join(self.dir, "a.jsonl")
        self.write([header(), result("c1", "PASS"), result("c2", "FAIL", "empty"), result("c3", "WARN", "odd"),
                    {"type": "end", "interrupted": False}])

    def write(self, recs, path=None, mode="w"):
        with open(path or self.path, mode) as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    def run_cli(self, *args, files=True):
        out = io.StringIO()
        argv = ["--states-dir", self.dir] + list(args) + ([self.path] if files is True else list(files or []))
        with contextlib.redirect_stdout(out):
            triage_states.main(argv)
        return out.getvalue()

    def test_first_run_shows_everything(self):
        out = self.run_cli()
        self.assertIn("host:1 | dev | m/x | smoke | t0.6 | 2026-11-07T10:00:00+00:00 | finished", out)
        self.assertIn("1 PASS · 1 WARN · 1 FAIL · 0 SKIP", out)
        self.assertIn("FAIL c2", out)
        self.assertIn("WARN c3", out)
        self.assertNotIn("c1 ", out)

    def test_second_run_shows_nothing_new(self):
        self.run_cli()
        out = self.run_cli()
        self.assertEqual(out.strip(), "1 file unchanged")

    def test_appended_cases_only(self):
        self.run_cli()
        self.write([result("c4", "FAIL", "loop"), result("c5", "PASS")], mode="a")
        out = self.run_cli()
        self.assertIn("+2 cases", out)
        self.assertIn("FAIL c4", out)
        self.assertNotIn("c2", out)
        self.assertIn("2 PASS · 1 WARN · 2 FAIL", out)
        self.assertEqual(self.run_cli().strip(), "1 file unchanged")

    def test_quiet_warn(self):
        out = self.run_cli("--quiet-warn")
        self.assertIn("FAIL c2", out)
        self.assertNotIn("WARN c3", out)

    def test_no_save_and_all_and_reset(self):
        self.run_cli("--no-save")
        self.assertFalse(os.path.exists(os.path.join(self.dir, ".triage.json")))
        self.assertIn("FAIL c2", self.run_cli("--no-save"))
        self.run_cli()
        self.assertIn("FAIL c2", self.run_cli("--all"))
        self.assertIn("reset", self.run_cli("--reset", files=None))
        self.assertIn("FAIL c2", self.run_cli())

    def test_default_files_and_unreadable(self):
        out = self.run_cli(files=None)
        self.assertIn("FAIL c2", out)
        with open(os.path.join(self.dir, "b.jsonl"), "w") as f:
            f.write("not a state\n")
        self.assertIn("b.jsonl: skipped", self.run_cli(files=None))

    def test_regrade_detected(self):
        fx = synthetic_fixture()
        fx_path = write_fixture(fx, self.dir, "fx.json.gz")
        old_state_fx = state.FIXTURES_DIR
        old_paths_fx = paths.FIXTURES_DIR
        state.FIXTURES_DIR = self.dir
        paths.FIXTURES_DIR = self.dir
        self.addCleanup(setattr, state, "FIXTURES_DIR", old_state_fx)
        self.addCleanup(setattr, paths, "FIXTURES_DIR", old_paths_fx)
        state._fixture_cache.clear()
        self.addCleanup(state._fixture_cache.clear)
        h = header({"file": "fx.json.gz", "sha256": state.file_sha256(fx_path)}, suite="coding")
        self.write([h, result("R001_main_e0_t000", "PASS"), result("R002_S1_t000", "PASS")])
        with mock.patch.object(grading, "grade", return_value=("PASS", None)):
            out = self.run_cli()
        self.assertNotIn("regraded", out)
        with mock.patch.object(grading, "grade", return_value=("FAIL", "now broken")):
            out = self.run_cli()
        self.assertIn("regraded: R001_main_e0_t000 PASS -> FAIL (now broken)", out)
        self.assertIn("2 FAIL", out)

    def test_missing_fixture(self):
        h = header({"file": "conv-2.json.gz", "sha256": "0" * 64}, suite="conv-2")
        self.write([h, result("R001_main_e0_t000", "PASS"), result("R002_S1_t000", "WARN", "w")])
        out = self.run_cli()
        self.assertIn("not re-graded (fixture conv-2 not here)", out)
        self.assertIn("1 PASS · 1 WARN · 0 FAIL", out)


if __name__ == "__main__":
    unittest.main()
