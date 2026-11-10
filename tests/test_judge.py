"""Offline tests for the judge suite: the fixture builder (tools/judge_fixture.py), selection and the verdict grading.

    python3 -m unittest discover -s tests
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401
import judge_fixture  # noqa: E402
import omp_fixture  # noqa: E402

import replay_bench  # noqa: E402
from bench.cases import (  # noqa: E402
    grading,  # noqa: E402
    judge,  # noqa: E402
    replay,  # noqa: E402
)

REAL_FIXTURE = os.path.join(ROOT, "fixtures", "judge.json.gz")

VERDICT = "STATUS: ACCEPT\nID: 42\nDESCRIPTION: Limes, raw\nCONFIDENCE: HIGH\nEXPLANATION: x"


def call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def synthetic_data():
    prefix = [{"role": "system", "content": "You judge."}, {"role": "user", "content": "shot"}]
    track = [
        {"role": "assistant", "content": None, "tool_calls": [call("fdc_get", {"fdc_id": 42}, "c1")]},
        {"role": "tool", "content": "{\"fdcId\":42}", "tool_call_id": "c1", "name": "fdc_get"},
        {"role": "assistant", "content": VERDICT},
    ]
    tool = {"type": "function", "function": {"name": "fdc_get", "description": "d", "parameters": {"type": "object"}}}
    return {"meta": {"built": "2026-10-25"}, "prefix": prefix, "tools": [tool],
            "request": {"temperature": 0, "max_tokens": 8192, "reasoning_effort": "xhigh", "tool_choice": "auto"},
            "cases": [{"row": 7, "id": "7_lime", "input": "line: lime", "track_model": "m",
                       "expect": {"status": "ACCEPT", "fdcIds": [42], "descriptionContains": "limes, raw"},
                       "track": track}]}


def build(tokens=None):
    return judge_fixture.build_judge_fixture(synthetic_data(), 7, tokens)


class Builder(unittest.TestCase):
    def test_turns_are_the_recorded_assistant_turns(self):
        fx = build({1: 100, 2: 150})
        self.assertEqual((fx["suite"], fx["prompt_tokens_recounted"], fx["counts"]["turns"]), ("judge", False, 2))
        pool = fx["messages"]
        t0, t1 = fx["turns"]
        self.assertEqual([m["role"] for m in (pool[i] for i in t0["messages"])], ["system", "user", "user"])
        self.assertEqual([m["role"] for m in (pool[i] for i in t1["messages"])],
                         ["system", "user", "user", "assistant", "tool"])
        self.assertEqual(replay.recorded_tool_names(t0), ["fdc_get"])
        self.assertEqual(replay.recorded_tool_names(t1), [])
        self.assertEqual((t0.get("judge_final"), t1.get("judge_final")), (False, True))
        # the request settings are the CLI's, max_tokens as in the replay (4096)
        self.assertEqual(t0["params"], {"stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
                                        "max_tokens": 4096, "reasoning_effort": "xhigh", "tool_choice": "auto"})
        self.assertEqual((t0["recorded_prompt_tokens"], t1["prompt_tokens"]), (100, 150))

    def test_check_fixture_clean(self):
        ok, errors = omp_fixture.check_fixture(build({1: 100, 2: 150}))
        self.assertTrue(ok, errors)

    def test_redaction_rules_applied(self):
        import re
        data = synthetic_data()
        data["cases"][0]["input"] = "line: lime for Zork"
        fx = judge_fixture.build_judge_fixture(data, 7, {1: 1, 2: 2}, [(re.compile("Zork"), "Anon")])
        self.assertNotIn("Zork", json.dumps(fx))
        self.assertIn("Anon", json.dumps(fx))

    def test_missing_tokens_are_tolerated_by_selection(self):
        fx = build()
        sel = replay.select_entries(fx)
        self.assertEqual(len(sel), 2)
        self.assertEqual(replay.recorded_prompt_tokens(sel[0][1]), 0)
        self.assertIn("2 entries", replay.summary_line(sel))
        replay.select_entries(fx, max_context=64 * 1024)

    def test_unknown_row(self):
        with self.assertRaises(ValueError):
            judge_fixture.build_judge_fixture(synthetic_data(), 99)

    def test_state_prompt_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            with open(path, "w") as f:
                json.dump({"targets": [{"results": {"MJ07_T01::iter1": {"prompt_tokens": 11},
                                                    "MJ07_T02::iter1": {"prompt_tokens": 12},
                                                    "MJ08_T01::iter1": {"prompt_tokens": 99},
                                                    "MJ07_T02": {"prompt_tokens": 12}}}]}, f)
            self.assertEqual(judge_fixture.state_prompt_tokens(path, 7), {1: 11, 2: 12})


class Grading(unittest.TestCase):
    def setUp(self):
        self.fx = build({1: 100, 2: 150})
        self.cases = [replay.make_case(self.fx, p, e) for p, e in replay.select_entries(self.fx)]

    def metrics(self, text="", calls=None):
        m = {"text": text}
        if calls is not None:
            m["tool_calls"] = calls
        return m

    def test_generic_names_on_the_first_turn(self):
        c = self.cases[0]
        self.assertEqual(grading.grade(c, self.metrics("", [call("fdc_get", {"fdc_id": 1}, "x")]))[0], "PASS")
        self.assertEqual(grading.grade(c, self.metrics("", []))[0], "FAIL")  # empty
        self.assertEqual(grading.grade(c, self.metrics("STATUS: ACCEPT\nID: 42", []))[0], "WARN")  # text where the recording called a tool

    def test_final_turn_verdict(self):
        c = self.cases[1]
        self.assertEqual(grading.grade(c, self.metrics(VERDICT, [])), ("PASS", ""))
        v = VERDICT.replace("ID: 42", "ID: 43")
        status, why = grading.grade(c, self.metrics(v, []))
        self.assertEqual(status, "FAIL")
        self.assertIn("ACCEPT/43", why)
        self.assertEqual(grading.grade(c, self.metrics(VERDICT.replace("ACCEPT", "REJECT"), []))[0], "FAIL")
        self.assertEqual(grading.grade(c, self.metrics("I think lime.", []))[0], "FAIL")
        self.assertEqual(grading.grade(c, self.metrics(VERDICT.replace("Limes, raw", "Lemons"), []))[0], "FAIL")
        # markup around the labels is tolerated
        self.assertEqual(grading.grade(c, self.metrics("**STATUS:** ACCEPT\n**ID:** 42\n**DESCRIPTION:** Limes, raw", []))[0],
                         "PASS")
        # a final turn that calls tools is graded generically (other tools than recorded: PASS)
        self.assertEqual(grading.grade(c, self.metrics("", [call("fdc_get", {"fdc_id": 1}, "x")]))[0], "PASS")

    def test_judge_module_direct(self):
        v = judge.parse_verdict("**STATUS:** ACCEPT\n**ID:** 42\n**DESCRIPTION:** Limes, raw")
        self.assertEqual(v["status"], "ACCEPT")
        self.assertEqual(v["fdcId"], 42)
        self.assertEqual(v["description"], "Limes, raw")
        ok, reason = judge.judge_verdict_check({"status": "ACCEPT", "fdcIds": [42]}, VERDICT)
        self.assertTrue(ok)
        self.assertEqual(reason, "")


class Cli(unittest.TestCase):
    def dry(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            replay_bench.main(list(argv))
        from test_replay import plain_dry_lines
        return cm.exception.code, plain_dry_lines(out.getvalue())

    def test_dry_run_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "judge.json.gz")
            omp_fixture.save_fixture(build({1: 100, 2: 150}), path)
            for suite in ("judge",):
                code, lines = self.dry("--suite", suite, "--dry-run", "--fixture", path)
                self.assertEqual(code, 0)
                self.assertIn("judge selection: 2 entries (main 2, subagent 0, compaction 0), max prompt 150", lines[-1])
                self.assertIn("fdc_get", lines[1])

    @unittest.skipUnless(os.path.exists(REAL_FIXTURE), "fixtures/judge.json.gz not built")
    def test_real_fixture_dry_run(self):
        code, lines = self.dry("--suite", "judge", "--dry-run")
        self.assertEqual(code, 0)
        self.assertEqual(len(lines) - 2, 6)
        self.assertTrue(lines[-1].startswith("judge selection: 6 entries (main 6, subagent 0, compaction 0)"), lines[-1])
        code, lines = self.dry("--suite", "judge", "--dry-run", "--turns", "2-3")
        self.assertEqual(len(lines) - 2, 2)

    @unittest.skipUnless(os.path.exists(REAL_FIXTURE), "fixtures/judge.json.gz not built")
    def test_real_fixture_is_clean_and_matches_replay_shape(self):
        fx = replay.load_fixture(REAL_FIXTURE)
        ok, errors = omp_fixture.check_fixture(fx)
        self.assertTrue(ok, errors)
        self.assertIs(fx["prompt_tokens_recounted"], False)
        finals = [t for t in fx["turns"] if t.get("judge_final")]
        self.assertEqual(len(finals), 1)
        self.assertIs(fx["turns"][-1].get("judge_final"), True)
        self.assertEqual(fx["turns"][0]["params"]["max_tokens"], 4096)
        self.assertEqual(replay.recorded_tool_names(fx["turns"][-1]), [])
        case = replay.make_case(fx, 6, fx["turns"][-1])
        self.assertEqual(grading.grade(case, {"text": "STATUS: ACCEPT\nID: %d\nDESCRIPTION: %s" % (
            fx["judge"]["expect"]["fdcIds"][0], fx["judge"]["expect"]["descriptionContains"])}), ("PASS", ""))


if __name__ == "__main__":
    unittest.main()
