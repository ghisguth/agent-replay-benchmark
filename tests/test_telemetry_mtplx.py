"""MTPLX telemetry shapes (bench/telemetry.py): mtplx_stats, mtplx_progress and timings. No server.

The chunks follow MTPLX 2.12.x's own tests and the timings recorded in fixtures/refs/mtplx-2.12.3_*.jsonl.
"""

import glob
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from bench import telemetry  # noqa: E402

REFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "samples")


class MtplxShapes(unittest.TestCase):
    def test_mtplx_stats_final_chunk(self):
        acc = telemetry.normalize({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                   "mtplx_stats": {"decode_tok_s": 42.0, "mtp_depth": 3, "request_elapsed_s": 1.25}},
                                  telemetry.init_acc())
        self.assertEqual(acc["native_decode_tps"], 42.0)
        self.assertIsNone(acc["native_ttft_ms"])
        self.assertIsNone(acc["native_prefill_tps"])

    def test_mtplx_stats_with_prefill_and_counts(self):
        acc = telemetry.normalize({"mtplx_stats": {"ttft_s": 0.12, "prefill_tok_s": 900.0, "decode_tok_s": 30.0,
                                                   "prompt_tokens": 1000, "cached_tokens": 200,
                                                   "completion_tokens": 50}}, telemetry.init_acc())
        self.assertAlmostEqual(acc["native_ttft_ms"], 120.0)
        self.assertEqual((acc["native_prefill_tps"], acc["native_decode_tps"]), (900.0, 30.0))
        self.assertEqual((acc["prompt_tokens"], acc["cached_tokens"], acc["exact_tokens"]), (1000, 200, 50))

    def test_mtplx_progress_heartbeat_and_growth(self):
        acc = telemetry.init_acc()
        telemetry.normalize({"mtplx_progress": {"phase": "generating", "completion_tokens": 0, "heartbeat": True}}, acc)
        self.assertEqual(acc["exact_tokens"], 0)
        telemetry.normalize({"mtplx_progress": {"phase": "generating", "completion_tokens": 7}}, acc)
        telemetry.normalize({"mtplx_progress": {"phase": "generating", "completion_tokens": 5}}, acc)
        self.assertEqual(acc["exact_tokens"], 7)  # never goes back

    def test_timings_shape_from_recorded_mtplx_run(self):
        files = sorted(glob.glob(os.path.join(REFS, "mtplx-*.jsonl")))
        self.assertTrue(files)
        with open(files[0]) as f:
            recs = [r for r in map(json.loads, f) if r.get("type") == "result" and r.get("timings")]
        self.assertTrue(recs)
        for r in recs[:20]:
            t = r["timings"]
            self.assertIn("draft_n_accepted", t)  # MTPLX's speculative-decoding counters
            acc = telemetry.normalize({"timings": t}, telemetry.init_acc())
            self.assertEqual(acc["native_decode_tps"], t["predicted_per_second"])
            self.assertEqual(acc["exact_tokens"], t["predicted_n"])
            self.assertAlmostEqual(acc["native_ttft_ms"], t["prompt_ms"])
            # MTPLX rates the whole prompt: with a cached prefix the new-token rate is recomputed, never above the raw one
            self.assertLessEqual(acc["native_prefill_tps"], t["prompt_per_second"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
