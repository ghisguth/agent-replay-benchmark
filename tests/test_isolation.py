"""Loads tests/isolation.py so unittest discovery applies it to every test module (see its docstring)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: F401, E402

from bench import report  # noqa: E402


class IsolationTest(unittest.TestCase):
    def test_process_streams_never_count_as_terminals(self):
        self.assertFalse(report.color_ok(sys.__stdout__))
        self.assertFalse(report.color_ok(sys.__stderr__))

    def test_environment_is_pinned(self):
        self.assertEqual(os.environ["COLUMNS"], "120")
        self.assertNotIn("NO_COLOR", os.environ)
        self.assertNotIn("REPLAY_BENCH_NERD_FONT", os.environ)


if __name__ == "__main__":
    unittest.main()
