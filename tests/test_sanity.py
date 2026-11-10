"""The sanity suite: ten cases, each graded by its validator through evaluate_response on canned good and bad answers.

    python3 -m unittest tests.test_sanity
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from bench.cases import SMOKE_CASES, SUITE_CASES, select_cases  # noqa: E402
from bench.cases.sanity import NEEDLE_CODE, SANITY_CASES, UNICODE_TEXT, tool_calls_in  # noqa: E402
from bench.cases.smoke import evaluate_response  # noqa: E402


def calls(*pairs):
    """The text transport produces for a turn with these (name, arguments dict) tool calls and no content."""
    return json.dumps([{"id": "c%d" % i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                       for i, (n, a) in enumerate(pairs)])


def case(cid):
    return next(c for c in SANITY_CASES if c["id"] == cid)


def grade(cid, text):
    return evaluate_response(case(cid), text)[0]


class Registry(unittest.TestCase):
    def test_ten_cases_and_smoke_is_the_first_four(self):
        self.assertEqual(len(SANITY_CASES), 10)
        self.assertEqual(SANITY_CASES[:4], SMOKE_CASES)
        self.assertTrue(all(a is b for a, b in zip(SANITY_CASES, SMOKE_CASES)))  # one definition
        self.assertEqual(len(select_cases()), 4)
        self.assertEqual(select_cases(suite="sanity"), SANITY_CASES)
        self.assertEqual(SUITE_CASES["sanity"], SANITY_CASES)

    def test_ids_are_unique_and_selectable_in_any_suite(self):
        ids = [c["id"] for c in SANITY_CASES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual([c["id"] for c in select_cases(["SC09_NEEDLE_8K"])], ["SC09_NEEDLE_8K"])

    def test_needle_prompt_is_about_8k_tokens(self):
        n = len(case("SC09_NEEDLE_8K")["prompt"])
        self.assertTrue(28000 < n < 45000, n)
        self.assertEqual(case("SC09_NEEDLE_8K")["prompt"].count(NEEDLE_CODE), 1)

    def test_tool_calls_in(self):
        self.assertEqual(tool_calls_in("no calls [1, 2]"), [])
        self.assertEqual(tool_calls_in("text\n" + calls(("a", {"x": 1}), ("b", {}))), [("a", {"x": 1}), ("b", {})])


class Cases(unittest.TestCase):
    def test_count_and_stop(self):
        self.assertEqual(grade("SC05_COUNT_AND_STOP", "1, 2, 3, 4, 5"), "PASS")
        self.assertEqual(grade("SC05_COUNT_AND_STOP", "1, 2, 3, 4, 5.\n"), "PASS")
        r = evaluate_response(case("SC05_COUNT_AND_STOP"), "1, 2, 3, 4, 5, 6, 7")
        self.assertEqual(r[0], "FAIL")
        self.assertIn("did not stop", r[1])
        self.assertEqual(grade("SC05_COUNT_AND_STOP", "Sure! 1, 2, 3"), "FAIL")
        self.assertEqual(grade("SC05_COUNT_AND_STOP", ""), "FAIL")

    def test_two_tool_calls(self):
        good = calls(("get_weather", {"location": "Paris"}), ("get_weather", {"location": "Tokyo, Japan"}))
        self.assertEqual(grade("SC06_TWO_TOOL_CALLS", good), "PASS")
        self.assertEqual(grade("SC06_TWO_TOOL_CALLS", calls(("get_weather", {"location": "Paris"}))), "FAIL")
        self.assertEqual(grade("SC06_TWO_TOOL_CALLS", calls(("get_weather", {"location": "Paris"}),
                                                           ("get_weather", {"location": "Paris"}))), "FAIL")
        self.assertEqual(grade("SC06_TWO_TOOL_CALLS", "Paris is sunny and Tokyo rainy."), "FAIL")

    def test_tool_choice(self):
        self.assertEqual(grade("SC07_TOOL_CHOICE", calls(("convert_currency", {"amount": 250, "from": "EUR", "to": "JPY"}))), "PASS")
        self.assertEqual(grade("SC07_TOOL_CHOICE", calls(("convert_currency", {"amount": "250", "from": "eur", "to": "jpy"}))), "PASS")
        self.assertEqual(grade("SC07_TOOL_CHOICE", calls(("get_weather", {"location": "Tokyo"}))), "FAIL")
        self.assertEqual(grade("SC07_TOOL_CHOICE", calls(("convert_currency", {"amount": 250, "from": "USD", "to": "JPY"}))), "FAIL")
        self.assertEqual(grade("SC07_TOOL_CHOICE", calls(("get_time", {"timezone": "Asia/Tokyo"}),
                                                        ("convert_currency", {"amount": 250, "from": "EUR", "to": "JPY"}))), "FAIL")
        self.assertEqual(grade("SC07_TOOL_CHOICE", "About 40000 yen."), "FAIL")

    def test_arithmetic(self):
        for good in ("36", "The answer is 36.", "Steps...\n9 bags x 4 = $36", "36 dollars"):
            self.assertEqual(grade("SC08_ARITHMETIC", good), "PASS", good)
        for bad in ("54", "The bags sell for 9 bags, 54 apples", "forty", "37"):
            self.assertEqual(grade("SC08_ARITHMETIC", bad), "FAIL", bad)

    def test_needle(self):
        self.assertEqual(grade("SC09_NEEDLE_8K", NEEDLE_CODE), "PASS")
        self.assertEqual(grade("SC09_NEEDLE_8K", "The code is %s." % NEEDLE_CODE), "PASS")
        self.assertEqual(grade("SC09_NEEDLE_8K", "AMBER-4471-HERON"), "FAIL")
        self.assertEqual(grade("SC09_NEEDLE_8K", "I could not find a code."), "FAIL")

    def test_unicode_echo(self):
        self.assertEqual(grade("SC10_UNICODE_ECHO", UNICODE_TEXT), "PASS")
        self.assertEqual(grade("SC10_UNICODE_ECHO", '"%s"' % UNICODE_TEXT), "PASS")
        self.assertEqual(grade("SC10_UNICODE_ECHO", UNICODE_TEXT.replace("ï", "i")), "FAIL")
        self.assertEqual(grade("SC10_UNICODE_ECHO", UNICODE_TEXT.replace("東京", "Tokyo")), "FAIL")
        self.assertEqual(grade("SC10_UNICODE_ECHO", "Here: " + UNICODE_TEXT), "FAIL")


if __name__ == "__main__":
    unittest.main()
