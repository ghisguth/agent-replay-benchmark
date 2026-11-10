"""Perf tables (by context bucket, prefill by append size) on synthetic results. No server.

    python3 -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from bench import compare, perf, report  # noqa: E402
from bench import results as R  # noqa: E402

K = 1024


def r(i, prompt, cached=0, tokens=100, ttft=1000, total=3000, cycles=None, accepted=0, ev="PASS", **kw):
    d = {"id": "R%03d" % i, "eval_result": ev, "prompt_tokens": prompt, "cached_tokens": cached, "tokens": tokens,
         "ttft_ms": ttft, "total_ms": total}
    if cycles is not None:
        d["metrics"] = {"speculation": {"drafted_tokens": cycles * 4, "accepted_draft_tokens": accepted, "cycles": cycles}}
    d.update(kw)
    return d


def ent(name, results, backend="h:1"):
    return compare.entry(backend, name, {x["id"]: x for x in results})


class Buckets(unittest.TestCase):
    def test_context_labels(self):
        self.assertEqual([R.band_label(n) for n in (1, 32 * K, 32 * K + 1, 64 * K, 256 * K, 256 * K + 1)],
                         ["0-32k", "32-64k", "32-64k", "64-96k", "256-288k", "256-288k"])

    def test_empty_buckets_skipped_and_order(self):
        e = ent("m", [r(1, 100 * K), r(2, 10 * K), r(3, 12 * K)])
        rows = perf.context_rows([("m", list(e["results"].values()))])
        self.assertEqual([(x["bucket"], x["requests"]) for x in rows], [("0-32k", 2), ("96-128k", 1)])


class ContextNumbers(unittest.TestCase):
    def test_decode_prefill_and_cycles(self):
        rs = [r(1, 10000, cached=2000, tokens=100, ttft=2000, total=4000, cycles=40, accepted=60),
              r(2, 20000, cached=10000, tokens=300, ttft=3000, total=9000, cycles=100, accepted=140),
              r(3, 5000, tokens=10, ttft=500, total=1000, cycles=5, accepted=5)]  # < 50 tokens
        s = perf.context_stats(rs)
        self.assertEqual(s["requests"], 3)
        self.assertAlmostEqual(s["decode_tps"], 410 / (2 + 6 + 0.5))  # output-weighted: sum tokens / sum decode time
        self.assertAlmostEqual(s["prefill_tps"], (8000 + 10000 + 5000) / 5.5)
        self.assertAlmostEqual(s["tok_per_cycle"], 1 + 205 / 145)  # Splash: 1 + accepted / cycles, all requests
        self.assertAlmostEqual(s["cycles_per_s"], 140 / 8)  # only the requests with >= 50 output tokens
        self.assertAlmostEqual(s["ms_per_cycle"], 1000 / (140 / 8))

    def test_missing_telemetry_is_none(self):
        s = perf.context_stats([r(1, 1000), r(2, 2000, error="boom"), r(3, 3000, ev="SKIP")])
        self.assertEqual(s["requests"], 1)
        self.assertIsNone(s["tok_per_cycle"])
        self.assertIsNone(s["cycles_per_s"])
        self.assertIsNone(s["ms_per_cycle"])
        self.assertIsNotNone(s["decode_tps"])

    def test_malformed_speculation_ignored(self):
        s = perf.context_stats([r(1, 1000, metrics={"speculation": {"cycles": 0}}), r(2, 1000, metrics="x")])
        self.assertIsNone(s["tok_per_cycle"])


class Tables(unittest.TestCase):
    def setUp(self):
        self.a = ent("MLX", [r(i, 20000, cached=5000, ttft=2000, total=6000, cycles=50, accepted=150) for i in range(3)])
        self.b = ent("Q2", [r(i, 20000, cached=5000, ttft=1000, total=4000, cycles=60, accepted=120) for i in range(3)])
        self.nospec = ent("plain", [r(i, 20000, cached=5000) for i in range(3)])

    def md(self, entries, bi=0):
        return report.blocks_markdown(perf.perf_blocks(entries, bi))

    def test_has_spec(self):
        self.assertTrue(perf.has_spec([self.a]))
        self.assertFalse(perf.has_spec([self.nospec]))

    def test_run_names_fall_back_to_labels_when_models_repeat(self):
        a2 = ent("MLX", [r(1, 1000)], backend="h:2")
        self.assertEqual(perf.run_names(compare.run_labels([self.a, a2])), ["MLX · h:1", "MLX · h:2"])  # only the host differs
        self.assertEqual(perf.run_names([self.a, self.b]), ["MLX", "Q2"])

    def test_run_names_comparison_base_this_run(self):
        e1 = ent("m1 (baseline)", [r(1, 100)])
        e2 = ent("m2", [r(1, 100)])
        self.assertEqual(perf.run_names([e1, e2]), ["base", "this run"])

    def test_append_buckets_and_threshold(self):
        rs = [r(1, 100, cached=90, ttft=10),                       # 10 new: below the threshold
              r(2, 116, cached=100, ttft=10),                      # 16 new: not "more than 16"
              r(3, 117, cached=100, ttft=100),                     # 17 new: < 1k
              r(4, 2000, cached=0, ttft=2000),                     # 2000: 1-5k
              r(5, 6000, cached=0, ttft=3000),                     # >= 5K
              r(6, 1024 + 5, cached=5, ttft=500)]                  # exactly 1k new: 1-5k
        rows = perf.append_rows([("m", rs)])
        got = {x["bucket"]: (x["requests"], x["sum_new"], x["tps"]) for x in rows}
        self.assertEqual(got["< 1k"], (1, 17, 17 / 0.1))
        self.assertEqual(got["1–5k"], (2, 3024, 3024 / 2.5))
        self.assertEqual(got["≥ 5k"], (1, 6000, 2000.0))

    def test_append_table_markdown(self):
        text = self.md([self.a])
        self.assertIn("### Prefill by append size", text)
        self.assertIn("| New tokens |", text)

    def test_nothing_to_show(self):
        self.assertEqual(perf.perf_blocks([ent("m", [])]), [])
        self.assertEqual(perf.perf_blocks([ent("m", [r(1, 100, cached=100)])]), [])
        self.assertEqual(perf.perf_blocks([ent("m", [r(1, 1000, cached=100)])])[:2], ["## Performance", "### Prefill by append size"])

    def test_print_perf_returns_false_without_data(self):
        self.assertFalse(compare.print_perf([ent("m", [])]))


if __name__ == "__main__":
    unittest.main()
