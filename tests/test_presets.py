"""--preset: the named selections, the default, conflicts, state naming and --resume. Offline; mock servers on 127.0.0.1 only.

    python3 -m unittest tests.test_presets
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
from test_resume import run_code  # noqa: E402
from test_state import SplashLike, behaviour  # noqa: E402

import replay_bench  # noqa: E402
from bench import baseline, report, state  # noqa: E402
from bench.cases import replay as replay_cases  # noqa: E402

FIXTURE = os.path.join(ROOT, "fixtures", "conv-3.json.gz")
JUDGE = os.path.join(ROOT, "fixtures", "judge.json.gz")


def run_main(*argv):
    """(exit code or None, stderr, run_benchmark mock) of main(argv) with run_benchmark mocked."""
    err, code = io.StringIO(), None
    with mock.patch.object(replay_bench, "run_benchmark") as rb, contextlib.redirect_stderr(err):
        try:
            replay_bench.main(list(argv))
        except SystemExit as e:
            code = e.code
    return code, err.getvalue(), rb


def selection(name):
    """(entries, max prompt) the preset selects on its real fixture."""
    p = replay_bench.PRESETS[name]
    fixture = replay_cases.load_fixture(JUDGE if p["suite"] == "judge" else FIXTURE)
    sel = replay_cases.select_entries(
        fixture, max_context=replay_bench.parse_max_context(p["max_context"]), no_subagents=p["no_subagents"],
        turns=p["turns"])
    return sel, max(replay_cases.recorded_prompt_tokens(e) for _, e in sel)


class Counts(unittest.TestCase):
    def test_every_replay_preset_selects_what_the_help_says(self):
        help_text = replay_bench.presets_help()
        for name, p in replay_bench.PRESETS.items():
            if p["suite"] in ("smoke", "sanity"):
                continue
            sel, top = selection(name)
            n = len(sel)
            top_str = "%.1fk" % (top / 1024)
            matching_rows = [l for l in help_text.splitlines() if l.strip().startswith(name)]
            self.assertTrue(matching_rows, name)
            self.assertIn(str(n), matching_rows[0], name)
            self.assertIn(top_str, matching_rows[0], name)

            if p["no_subagents"]:
                self.assertTrue(all(e.get("stream") == "main" or e.get("kind") == "compaction" or "main" in e.get("label", "") for _, e in sel), name)
            if p["turns"]:
                self.assertEqual(len(sel), p["turns"][1] - p["turns"][0] + 1, name)
            if p["max_context"] != "full":
                mc = replay_bench.parse_max_context(p["max_context"])
                self.assertTrue(all(replay_cases.recorded_prompt_tokens(e) + replay_cases.SKIP_RESERVE <= mc for _, e in sel), name)

        # Fixed count guard for large
        sel_large, _ = selection("large")
        self.assertEqual(len(sel_large), 58)

    def test_compaction_preset_contains_the_compaction_call_and_the_restart(self):
        sel, _ = selection("compaction")
        self.assertEqual([e["kind"] for _, e in sel].count("compaction"), 1)
        self.assertEqual({e["epoch"] for _, e in sel if e["stream"] == "main"}, {0, 1})
        self.assertEqual(selection("main")[0][-1][1]["epoch"], 1)

    def test_xlarge_stops_before_the_compaction(self):
        sel, _ = selection("xlarge")
        self.assertNotIn("compaction", {e["kind"] for _, e in sel})

    def test_all_names_in_the_help_table(self):
        table = replay_bench.presets_help()
        for name in replay_bench.PRESETS:
            self.assertIn(name, table)
        self.assertIn("alias 128k", table)
        self.assertIn("alias 236k", table)


class Cli(unittest.TestCase):
    def test_presets_set_their_options(self):
        for name, p in replay_bench.PRESETS.items():
            if p["suite"] in ("smoke", "sanity"):
                continue
            code, err, rb = run_main("--preset", name)
            self.assertIsNone(code, (name, err))
            r = rb.call_args[1]["replay"]
            self.assertEqual(rb.call_args[0][0], p["suite"], name)
            self.assertEqual(rb.call_args[1]["preset"], name)
            self.assertEqual((r["max_context"], r["no_subagents"], r["turns"]),
                             (replay_bench.parse_max_context(p["max_context"]), p["no_subagents"], p["turns"]), name)
            self.assertIn("preset %s: " % name, err)

    def test_smoke_preset(self):
        code, err, rb = run_main("--preset", "smoke")
        self.assertIsNone(code)
        self.assertEqual(rb.call_args[0][0], "smoke")
        self.assertIsNone(rb.call_args[1]["replay"])
        self.assertEqual(rb.call_args[1]["temperature"], 0.0)
        self.assertEqual(rb.call_args[1]["preset"], "smoke")

    def test_sanity_preset(self):
        code, err, rb = run_main("--preset", "sanity")
        self.assertIsNone(code, err)
        self.assertEqual(rb.call_args[0][0], "sanity")
        self.assertIsNone(rb.call_args[1]["replay"])
        self.assertEqual(rb.call_args[1]["temperature"], 0.0)
        self.assertIn("preset sanity: sanity, 10 cases", err)
        code, err, rb = run_main("--suite", "sanity")
        self.assertIsNone(code, err)
        self.assertEqual(rb.call_args[0][0], "sanity")

    def test_sanity_takes_no_selection_options(self):
        code, err, _ = run_main("--preset", "sanity", "--turns", "1-2")
        self.assertIn("sanity takes no selection options", err)
        code, err, _ = run_main("--suite", "sanity", "--dry-run")
        self.assertEqual(code, 2)

    def test_quickstart_is_the_first_five_main_requests(self):
        sel, top = selection("quickstart")
        self.assertEqual([p for p, _ in sel], [1, 2, 3, 4, 5])
        self.assertTrue(all(e["stream"] == "main" for _, e in sel))
        self.assertLess(top, 24 * 1024)
        self.assertIn("quickstart", replay_bench.presets_help())
        code, err, rb = run_main("--preset", "quickstart", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("5 requests", err)

    def test_default_is_large(self):
        code, err, rb = run_main()
        self.assertIsNone(code)
        self.assertEqual(rb.call_args[1]["preset"], "large")
        self.assertEqual(rb.call_args[1]["replay"]["max_context"], 128 * 1024)
        self.assertTrue(rb.call_args[1]["replay"]["no_subagents"])
        self.assertIn("preset large: coding, ≤128k, main agent only, 58 requests", err)
        _, _, rb2 = run_main("--preset", "large")
        self.assertEqual(rb.call_args[1]["replay"], rb2.call_args[1]["replay"])

    def test_aliases(self):
        for alias, name in (("128k", "large"), ("236k", "xlarge"), ("LARGE", "large")):
            _, _, rb = run_main("--preset", alias)
            self.assertEqual(rb.call_args[1]["preset"], name)
        _, _, rb = run_main("--suite", "coding")
        self.assertEqual(rb.call_args[0][0], "coding")
        self.assertIsNone(rb.call_args[1]["preset"])
        self.assertIsNone(rb.call_args[1]["replay"]["max_context"])  # a suite alone is everything
        code, _, _ = run_main("--preset", "huge")
        self.assertEqual(code, 2)

    def test_conflicts_are_errors_naming_both(self):
        for argv, flag in ((["--preset", "large", "--max-context", "64k"], "--max-context 64k"),
                           (["--preset", "large", "--max-context", "full"], "--max-context full"),
                           (["--preset", "full", "--no-subagents"], "--no-subagents"),
                           (["--preset", "large", "--turns", "1-5"], "--turns 1-5"),
                           (["--preset", "large", "--suite", "judge"], "--suite judge"),
                           (["--preset", "smoke", "--suite", "coding"], "--suite coding")):
            code, _, rb = run_main(*argv)
            self.assertEqual(code, 2, argv)
            rb.assert_not_called()
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                replay_bench.main(argv)
            self.assertIn(flag, err.getvalue())
            self.assertIn("--preset %s" % argv[1], err.getvalue())

    def test_conflict_hints_smoke_vs_replay(self):
        # Smoke preset conflict: informs that smoke takes no selection options
        for argv in (["--preset", "smoke", "--turns", "1-2"], ["--preset", "smoke", "--suite", "coding"],
                     ["--preset", "smoke", "--no-subagents"], ["--preset", "smoke", "--max-context", "64k"]):
            code, err, rb = run_main(*argv)
            self.assertEqual(code, 2, argv)
            rb.assert_not_called()
            self.assertIn("smoke takes no selection options", err)
            self.assertNotIn("use --suite smoke with the options", err)

        # Replay preset conflict: suggests using --suite X with the options
        code, err, rb = run_main("--preset", "large", "--turns", "1-2")
        self.assertEqual(code, 2)
        rb.assert_not_called()
        self.assertIn("use --suite coding with the options", err)

        code, err, rb = run_main("--preset", "judge", "--turns", "1-2")
        self.assertEqual(code, 2)
        rb.assert_not_called()
        self.assertIn("use --suite judge with the options", err)

    def test_same_value_is_not_a_conflict_and_other_options_combine(self):
        code, _, rb = run_main("--preset", "large", "--max-context", "128k", "--no-subagents", "--suite", "coding",
                               "--model", "Q4", "--endpoint", "http://127.0.0.1:1/v1", "--temperature", "0.6",
                               "--fixture", "x.json.gz")
        self.assertIsNone(code)
        self.assertEqual(rb.call_args[1]["model_filters"], ["Q4"])
        self.assertEqual(rb.call_args[1]["temperature"], 0.6)
        self.assertEqual(rb.call_args[1]["replay"]["fixture"], "x.json.gz")

    def test_selection_options_alone_mean_the_coding_suite(self):
        _, _, rb = run_main("--max-context", "64k")
        self.assertEqual(rb.call_args[0][0], "coding")
        self.assertIsNone(rb.call_args[1]["preset"])
        self.assertEqual(rb.call_args[1]["replay"]["max_context"], 64 * 1024)

    def test_test_flag_and_state_do_not_start_a_preset(self):
        _, _, rb = run_main("--test", "TC01_GIT_COMMIT")
        self.assertEqual(rb.call_args[0][0], "smoke")
        self.assertIsNone(rb.call_args[1]["preset"])

    def test_dry_run_prints_the_selection(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            replay_bench.main(["--preset", "small", "--dry-run", "--fixture", FIXTURE])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("coding selection: 9 entries", out.getvalue())
        self.assertIn("preset small:", err.getvalue())

    def test_removed_suites(self):
        for s in ("1", "2"):
            code, _, _ = run_main("--suite", s)
            self.assertEqual(code, 2)


class States(unittest.TestCase):
    def test_file_name_uses_the_preset(self):
        target = {"kind": "splash", "version": "1a2b3c4d"}
        p = state.default_path("/nonexistent", target, "m", "coding", preset="large", request={"temperature": 0.6})
        self.assertRegex(p, r"_m_coding-large_t0\.6_\d{8}-\d{6}\.jsonl$")
        self.assertRegex(state.default_path("/nonexistent", target, "m", "coding"), r"_m_coding_\d{8}")
        self.assertRegex(state.default_path("/nonexistent", target, "m", "judge", preset="judge"), r"_m_judge_\d{8}")
        self.assertRegex(state.default_path("/nonexistent", target, "m", "smoke", preset="smoke"), r"_m_smoke_\d{8}")

    def test_header_has_the_preset(self):
        t = {"url": "http://x/v1", "backend": "b", "name": "m"}
        self.assertEqual(state.make_header("coding", t, preset="large")["preset"], "large")
        self.assertNotIn("preset", state.make_header("coding", t))

    def test_old_conv_3_states_are_the_coding_suite(self):
        self.assertTrue(state.same_suite("conv-3", "coding"))
        self.assertTrue(state.same_suite("3", "coding"))
        self.assertTrue(state.same_suite("4", "judge"))
        self.assertTrue(state.same_suite("single", "smoke"))
        self.assertTrue(state.same_suite("coding", "coding"))
        self.assertFalse(state.same_suite("conv-3", "judge"))
        self.assertFalse(state.same_suite("smoke", "coding"))
        with tempfile.TemporaryDirectory() as d:
            t = {"url": "http://x/v1/chat/completions", "backend": "b", "name": "m"}
            path = os.path.join(d, "old.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps(state.make_header("conv-3", t)) + "\n")
            bl = baseline.load_baseline(path)
            baseline.check_baseline(bl, "coding", None)  # does not raise
            with self.assertRaises(replay_bench.UserError):
                baseline.check_baseline(bl, "judge", None)


class Agreement(unittest.TestCase):
    def test_line_has_the_share_of_same_tools(self):
        from bench import results
        rs = [{"eval_result": "PASS", "matches_recorded": True}, {"eval_result": "PASS", "matches_recorded": False},
              {"eval_result": "PASS", "matches_recorded": True, "kind": "compaction"},   # not a turn
              {"eval_result": "SKIP", "matches_recorded": False}, {"eval_result": "FAIL", "error": "x"},
              {"eval_result": "PASS"}]  # smoke: nothing to compare
        self.assertEqual(results.tools_agreement(rs), 50.0)
        self.assertIsNone(results.tools_agreement([{"eval_result": "PASS"}]))
        line = results.counts_line(results.verdict_counts(rs), 50.0)
        self.assertTrue(line.endswith("same tools as recorded 50%"))
        self.assertTrue(line.markdown().endswith("same tools as recorded 50%"))
        self.assertNotIn("same tools", results.counts_line(results.verdict_counts(rs)))


class Regrade(unittest.TestCase):
    """--state re-grades replay results with the current grade() when the shipped fixture is there."""

    def write(self, d, results, header_fixture=FIXTURE):
        t = {"url": "http://x/v1/chat/completions", "backend": "b", "name": "m"}
        path = os.path.join(d, "s.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps(state.make_header("coding", t, fixture_path=header_fixture)) + "\n")
            for r in results:
                f.write(json.dumps({"type": "result", **r}) + "\n")
        return path

    def test_old_verdicts_are_replaced_but_skip_and_errors_stay(self):
        fixture = replay_cases.load_fixture(FIXTURE)
        (p1, e1), (p2, e2), (p3, e3) = [(p, e) for p, e in enumerate(fixture["turns"], 1) if e["kind"] != "compaction"][:3]
        good = lambda e: {"content": "", "tool_calls": e["recorded_answer"]["tool_calls"], "finish_reason": "tool_calls"}
        old = {"eval_result": "WARN", "fail_reason": "tools x vs recorded y"}
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, [
                {"id": replay_cases.case_id(p1, e1), **good(e1), **old},
                {"id": replay_cases.case_id(p2, e2), "eval_result": "SKIP", "fail_reason": "too big"},
                {"id": replay_cases.case_id(p3, e3), "error": "boom", "eval_result": "FAIL", "fail_reason": "boom"}])
            res = state.read_blocks(path)[0]["results"]
        r1, r2, r3 = res.values()
        self.assertEqual(r1["eval_result"], "PASS")
        self.assertTrue(r1["matches_recorded"])
        self.assertEqual((r2["eval_result"], r2["fail_reason"]), ("SKIP", "too big"))
        self.assertEqual((r3["eval_result"], r3["fail_reason"]), ("FAIL", "boom"))

    def test_other_fixture_keeps_stored_verdicts(self):
        with tempfile.TemporaryDirectory() as d:
            other = write_fixture(synthetic_fixture(), d)
            path = self.write(d, [{"id": "R001_main_e0_t000", "eval_result": "WARN", "fail_reason": "kept"}], other)
            res = state.read_blocks(path)[0]["results"]
        self.assertEqual(res["R001_main_e0_t000"]["fail_reason"], "kept")


class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: report.configure("tui", io.StringIO()))
        self.fixture = write_fixture(synthetic_fixture(), self.tmp.name)
        self.path = os.path.join(self.tmp.name, "s.jsonl")

    def start(self, srv, *argv):
        return run_code(*argv, "--fixture", self.fixture, "--endpoint", srv.url, "--state-file", self.path)

    def header(self):
        return state.read_blocks(self.path)[0]["header"]

    def test_header_records_the_preset_and_resume_needs_the_same_one(self):
        with SplashLike(behaviour) as srv:
            _, err, code = self.start(srv, "--preset", "full")
            self.assertEqual(code, 0, err)
            self.assertEqual(self.header()["preset"], "full")
            self.assertEqual(self.header()["suite"], "coding")
            _, err, code = run_code("--resume", self.path, "--preset", "main")
            self.assertEqual(code, 2)
            self.assertIn("--preset: recorded full, given main", err)
            _, err, code = run_code("--resume", self.path, "--max-context", "64k")
            self.assertEqual(code, 2)
            self.assertIn("conflicts with --preset full", err)
            _, err, code = run_code("--resume", self.path, "--preset", "full")
            self.assertEqual(code, 0, err)

    def test_preset_cannot_be_added_to_a_run_without_one(self):
        with SplashLike(behaviour) as srv:
            _, err, code = self.start(srv, "--suite", "coding")
            self.assertEqual(code, 0, err)
            self.assertNotIn("preset", self.header())
            _, err, code = run_code("--resume", self.path, "--preset", "full")
            self.assertEqual(code, 2)
            self.assertIn("--preset: recorded none, given full", err)
            _, err, code = run_code("--resume", self.path)  # a file written before presets still resumes
            self.assertEqual(code, 0, err)


if __name__ == "__main__":
    unittest.main()
