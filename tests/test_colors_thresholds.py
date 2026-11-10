"""Threshold colors of the per-case table: terminal only, markdown numbers stay plain.

    python3 -m unittest tests.test_colors_thresholds
"""

import os
import re
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from bench import fmt as F  # noqa: E402
from bench import results  # noqa: E402
from bench.colors import Clr, configure  # noqa: E402
from bench.report import Table  # noqa: E402


class Thresholds(unittest.TestCase):
    def test_decode(self):
        s = lambda v: F.speed_style(v, F.DECODE_TPS)
        self.assertEqual([s(59.9), s(60), s(29.9), s(30), s(None)],
                         ["speed_ok", "speed_good", "speed_slow", "speed_ok", "dim"])

    def test_prefill(self):
        s = lambda v: F.speed_style(v, F.PREFILL_TPS)
        self.assertEqual([s(199.9), s(200), s(99.9), s(100), s(None)],
                         ["speed_ok", "speed_good", "speed_slow", "speed_ok", "dim"])

    def test_cache(self):
        self.assertEqual([F.cache_style(v) for v in (79, 80, 49, 50, 19, 20, None)],
                         ["cache_mid", "cache_hi", "cache_low", "cache_mid", "cache_bad", "cache_low", "dim"])


class Rendered(unittest.TestCase):
    def setUp(self):
        self.addCleanup(configure, False)
        configure(True)
        from unittest import mock
        p = mock.patch("bench.report.term_limit", return_value=None)
        p.start()
        self.addCleanup(p.stop)
        lay = results.CaseLayout(True)
        m = {"id": "c1", "ttft_ms": 500, "total_ms": 2000, "prompt_tokens": 1000, "cached_tokens": 900, "tokens": 50,
             "native_decode_tps": 72.5, "native_prefill_tps": 90.0, "eval_result": "PASS"}
        self.table = Table(lay.header, [lay.cells(m)], aligns=lay.aligns, dim_cols=lay.dim_cols)

    def test_tui_colors(self):
        out = self.table.tui()
        self.assertRegex(out, re.escape(Clr.BCYN) + r" *72\.5" + re.escape(Clr.RST))  # decode good, right-aligned
        self.assertRegex(out, re.escape(Clr.RED) + r" *90\.0" + re.escape(Clr.RST))  # prefill slow
        self.assertRegex(out, re.escape(Clr.BGRN) + r" *90%" + re.escape(Clr.RST))  # cache hit

    def test_markdown_plain(self):
        md = self.table.markdown()
        self.assertNotIn("\033", md)
        for v in ("72.5", "90.0", "90%"):
            self.assertRegex(md, r"\| +" + re.escape(v) + r" +\|")


class ClientClockFallback(unittest.TestCase):
    def setUp(self):
        self.lay = results.CaseLayout(has_stats=True)

    def test_no_native_stats_shows_tilde_values_and_dash_cache(self):
        # 100 output tokens over (2500 - 500) ms = 2.0s -> 50.0 tps decode
        # 1000 prompt tokens (cached is None) over 500 ms -> 2000.0 tps prefill
        m = {
            "id": "c1", "ttft_ms": 500, "total_ms": 2500,
            "prompt_tokens": 1000, "cached_tokens": None, "tokens": 100,
            "eval_result": "PASS",
        }
        cells = self.lay.cells(m)
        col_map = {h: str(c) for h, c in zip(self.lay.header, cells)}
        self.assertEqual(col_map["C%"], "–")
        self.assertEqual(col_map["Decode"], "~50.0")
        self.assertEqual(col_map["Prefill"], "~2000.0")

    def test_reported_cached_zero_shows_zero_pct(self):
        m = {
            "id": "c1", "ttft_ms": 500, "total_ms": 2500,
            "prompt_tokens": 1000, "cached_tokens": 0, "tokens": 100,
            "eval_result": "PASS",
        }
        cells = self.lay.cells(m)
        col_map = {h: str(c) for h, c in zip(self.lay.header, cells)}
        self.assertEqual(col_map["C%"], "0%")

    def test_missing_both_native_and_client_shows_warn(self):
        m = {
            "id": "c1", "ttft_ms": None, "total_ms": None,
            "prompt_tokens": 1000, "cached_tokens": None, "tokens": 0,
            "eval_result": "PASS",
        }
        cells = self.lay.cells(m)
        col_map = {h: str(c) for h, c in zip(self.lay.header, cells)}
        self.assertEqual(col_map["Decode"], "⚠")
        self.assertEqual(col_map["Prefill"], "⚠")

    def test_totals_with_mixed_results(self):
        # Result A: native stats
        ra = {
            "id": "r1", "eval_result": "PASS", "ttft_ms": 500, "total_ms": 2000,
            "prompt_tokens": 1000, "cached_tokens": 200, "tokens": 100,
            "native_decode_tps": 60.0, "native_prefill_tps": 300.0,
        }
        # Result B: client clock estimate (tokens=50 over 2s gen -> 25.0 decode, 1000 prompt / 1s -> 1000 prefill)
        rb = {
            "id": "r2", "eval_result": "PASS", "ttft_ms": 1000, "total_ms": 3000,
            "prompt_tokens": 1000, "cached_tokens": None, "tokens": 50,
        }
        tot = results.run_totals([ra, rb])
        # Decode: p50 of (60.0, 25.0) = 42.5, decode_est is True
        self.assertEqual(tot["decode"], 42.5)
        self.assertTrue(tot["decode_est"])
        # Prefill: p50 of (300.0, 1000.0) = 650.0, prefill_est is True
        self.assertEqual(tot["prefill"], 650.0)
        self.assertTrue(tot["prefill_est"])
        # Cache: only ra reported a count, so the share is over ra's prompt (200 / 1000 = 20%)
        self.assertEqual(tot["cache"], 20.0)
        self.assertEqual(tot["ttft"], 1.5)  # sum of TTFTs (0.5 + 1.0), not the median
        self.assertEqual(results.run_totals([ra, rb, dict(rb, id="r3", eval_result="SKIP")])["ttft"], 1.5)
        d = results.run_totals([ra, rb]), results.run_totals([ra])
        self.assertEqual(str(self.lay.delta_row(*d)[self.lay.header.index("TTFT")]), "+200.0%")

        # Totals row formatting
        row = self.lay.totals("Total", tot)
        col_map = {h: str(c) for h, c in zip(self.lay.header, row)}
        self.assertEqual(col_map["Decode"], "~42.5")
        self.assertEqual(col_map["Prefill"], "~650.0")
        self.assertEqual(col_map["C%"], "20%")

    def test_totals_with_all_unknown_cache_shows_dash(self):
        rb = {
            "id": "r1", "eval_result": "PASS", "ttft_ms": 1000, "total_ms": 3000,
            "prompt_tokens": 1000, "cached_tokens": None, "tokens": 50,
        }
        tot = results.run_totals([rb])
        self.assertIsNone(tot["cache"])
        row = self.lay.totals("Total", tot)
        col_map = {h: str(c) for h, c in zip(self.lay.header, row)}
        self.assertEqual(col_map["C%"], "–")


if __name__ == "__main__":
    unittest.main()
