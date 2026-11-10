"""LM Studio reports no cached tokens: new tokens are estimated as prompt - previous prompt of the stream (the prefill
column, the append-size buckets, the context table), also for old state files. No server.

    python3 -m unittest discover -s tests
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from bench import compare, perf, state  # noqa: E402
from bench import results as R  # noqa: E402
from bench.outputs import legend  # noqa: E402


def res(i, prompt, ttft=4500, stream="main", kind="turn", epoch=0, cached=None, **kw):
    d = {"id": "R%03d" % i, "eval_result": "PASS", "prompt_tokens": prompt, "cached_tokens": cached, "tokens": 50,
         "ttft_ms": ttft, "total_ms": ttft + 2000, "stream": stream, "kind": kind, "epoch": epoch}
    d.update(kw)
    return d


class NewTokens(unittest.TestCase):
    def test_estimate_is_prompt_minus_previous_prompt_in_the_stream(self):
        rs = R.fill_prev_prompts([res(1, 18000), res(2, 18500), res(3, 5000, stream="sub1"), res(4, 19000)])
        self.assertEqual([R.new_tokens(r) for r in rs], [18000, 500, 5000, 500])

    def test_first_request_compaction_and_larger_previous_use_the_whole_prompt(self):
        rs = R.fill_prev_prompts([res(1, 18000), res(2, 9000), res(3, 20000, kind="compaction"), res(4, 21000)])
        self.assertEqual(R.new_tokens(rs[1]), 9000)  # previous prompt is larger
        self.assertEqual(R.new_tokens(rs[2]), 20000)  # a compaction expects no cache
        self.assertEqual(R.new_tokens(rs[3]), 12000)  # the compaction is not a previous request: R002 is
        self.assertFalse(R.estimates_new_tokens(rs[0]))

    def test_reported_cache_wins(self):
        rs = R.fill_prev_prompts([res(1, 18000, cached=0), res(2, 18500, cached=18000)])
        self.assertEqual([R.new_tokens(r) for r in rs], [18000, 500])
        self.assertFalse(any(R.estimates_new_tokens(r) for r in rs))

    def test_prefill_column_and_append_buckets_use_the_estimate(self):
        rs = R.fill_prev_prompts([res(1, 18000), res(2, 22104, ttft=4500)])
        speed, est = R.request_prefill_speed(rs[1])
        self.assertTrue(est)
        self.assertAlmostEqual(speed, 4104 * 1000 / 4500)  # 912 tok/s, not 22104 / 4.5 = 4912
        e = compare.entry("h:1", "m", {r["id"]: r for r in rs})
        rows = perf.append_rows([("m", list(e["results"].values()))])
        self.assertEqual([(x["bucket"], x["requests"]) for x in rows], [("1–5k", 1), ("≥ 5k", 1)])
        stats = perf.context_stats(rs)
        self.assertAlmostEqual(stats["prefill_tps"], (18000 + 4104) / 9.0)

    def test_legend_line_only_when_estimating(self):
        line = "no cache reported: new tokens ≈ prompt − previous prompt in the stream."
        self.assertIn(line, legend(False, est=True, width=200))
        self.assertNotIn("no cache reported", legend(False, width=200))
        rs = R.fill_prev_prompts([res(1, 100), res(2, 200)])
        self.assertTrue(perf.has_new_estimate([compare.entry("h:1", "m", {r["id"]: r for r in rs})]))
        self.assertFalse(perf.has_new_estimate([compare.entry("h:1", "m", {})]))

    def test_old_state_file_gets_the_estimate_when_read(self):
        header = {"type": "run", "schema": 1, "backend": "h:1", "model": "m", "suite": "coding"}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps(header) + "\n")
                for r in (res(1, 18000), res(2, 22104)):
                    f.write(json.dumps(dict(r, type="result")) + "\n")
            (block,) = state.read_blocks(path)
        second = block["results"]["R002"]
        self.assertEqual(second["prev_prompt_tokens"], 18000)
        self.assertEqual(R.new_tokens(second), 4104)


class DraftAcceptance(unittest.TestCase):
    def test_metrics_row_from_speculation_counts(self):
        rs = [res(i, 1000, metrics={"speculation": {"drafted_tokens": 49, "accepted_draft_tokens": 21}}) for i in (1, 2)]
        s = compare.stats(rs)
        self.assertAlmostEqual(s["draft_acc"], 100 * 21 / 49)
        self.assertIsNone(compare.stats([res(1, 1000)])["draft_acc"])
        labels = [row[0] for row in compare._client_rows(s)]
        self.assertIn("Draft acceptance", labels)
        self.assertNotIn("Draft acceptance", [row[0] for row in compare._client_rows(compare.stats([res(1, 1000)]))])


if __name__ == "__main__":
    unittest.main()
