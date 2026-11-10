import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from bench.cases import ALL_TEST_CASES, select_cases
from bench.cases.smoke import build_messages


class CaseSelectionTest(unittest.TestCase):
    def test_smoke_cases_nonempty(self):
        self.assertTrue(select_cases())

    def test_select_by_id(self):
        self.assertEqual([t["id"] for t in select_cases(["TT01_WEATHER_TOOL"])], ["TT01_WEATHER_TOOL"])
        with self.assertRaises(ValueError):
            select_cases(["nope"])

    def test_unique_ids(self):
        ids = [t["id"] for t in ALL_TEST_CASES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_build_messages_static(self):
        tc = {"system": "sys prompt", "prompt": "user prompt"}
        msgs = build_messages(tc)
        self.assertEqual(msgs, [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "user prompt"},
        ])

    def test_build_messages_explicit_messages(self):
        tc = {"messages": [{"role": "user", "content": "hello"}]}
        msgs = build_messages(tc)
        self.assertEqual(msgs, [{"role": "user", "content": "hello"}])
        # verify deepcopy
        msgs[0]["content"] = "changed"
        self.assertEqual(tc["messages"][0]["content"], "hello")

    def test_regrade_of_a_saved_smoke_result_is_the_live_grade(self):
        from bench.cases import smoke
        fenced = '```json\n{"route": "docker", "confidence": 0.9, "args": {}}\n```'
        live = smoke.evaluate_response(smoke.SMOKE_CASES[1], fenced)
        self.assertEqual(live[0], "WARN")  # recovered from the markdown fence
        results = {"TC02_INTENT_ROUTING": {"text": fenced, "eval_result": "FAIL", "fail_reason": "old"},
                   "TC01_GIT_COMMIT": {"text": "", "eval_result": "FAIL", "fail_reason": "server", "error": "timeout"}}
        smoke.regrade(results)
        self.assertEqual((results["TC02_INTENT_ROUTING"]["eval_result"], results["TC02_INTENT_ROUTING"]["fail_reason"]), live)
        self.assertEqual(results["TC01_GIT_COMMIT"]["fail_reason"], "server")  # errors keep their verdict


if __name__ == "__main__":
    unittest.main()
