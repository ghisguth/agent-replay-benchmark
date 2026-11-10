"""Unit tests for server telemetry extraction and normalization in bench/telemetry.py."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from bench import telemetry  # noqa: E402


class TestNormalizeLlamaCpp(unittest.TestCase):
    def test_timings_chunk(self):
        acc = telemetry.init_acc()
        chunk = {
            "choices": [{"delta": {}}],
            "timings": {
                "prompt_n": 100,
                "prompt_ms": 120.0,
                "prompt_per_second": 416.0,
                "predicted_n": 7,
                "predicted_ms": 80.0,
                "predicted_per_second": 75.0,
            },
        }
        telemetry.normalize(chunk, acc)
        self.assertEqual(acc["native_ttft_ms"], 120.0)
        self.assertEqual(acc["native_prefill_tps"], 416.0)
        self.assertEqual(acc["native_decode_tps"], 75.0)
        self.assertEqual(acc["prompt_tokens"], 100)
        self.assertEqual(acc["exact_tokens"], 7)
        self.assertEqual(acc["raw_timings"], chunk["timings"])


class TestNormalizeSplash(unittest.TestCase):
    def test_splash_metrics_and_usage(self):
        acc = telemetry.init_acc()
        chunk = {
            "choices": [],
            "usage": {
                "prompt_tokens": 16580,
                "completion_tokens": 146,
                "prompt_tokens_details": {"cached_tokens": 50},
                "completion_tokens_details": {"reasoning_tokens": 72},
            },
            "metrics": {
                "prefill": {"tokens": 16580},
                "decode": {"tokens": 145},
                "request_latency": {
                    "start_to_first_token_ms": 13739.577,
                    "first_token_to_done_ms": 1652.308,
                    "stream_tokens_per_second": 87.756,
                },
                "cache": {"matched_tokens": 50},
            },
        }
        telemetry.normalize(chunk, acc)
        self.assertEqual(acc["exact_tokens"], 146)
        self.assertEqual(acc["prompt_tokens"], 16580)
        self.assertEqual(acc["cached_tokens"], 50)
        self.assertEqual(acc["reasoning_tokens"], 72)
        self.assertEqual(acc["native_ttft_ms"], 13739.577)
        self.assertEqual(acc["native_decode_tps"], 87.756)
        self.assertEqual(acc["raw_metrics"], chunk["metrics"])
        self.assertEqual(acc["raw_usage"], chunk["usage"])


class TestNormalizeKapitsa(unittest.TestCase):
    def test_kapitsa_progress_streaming(self):
        acc = telemetry.init_acc()
        # Phase 0 (waiting) is ignored
        chunk0 = {"kapitsa_progress": {"phase": 0, "prompt_tokens": 1000, "cached_tokens": 200, "processed_tokens": 0}}
        telemetry.normalize(chunk0, acc, current_time=1.0, start_time=0.0)
        self.assertIsNone(acc["last_progress"])

        # Phase 1 (prefill) records progress
        chunk1 = {"kapitsa_progress": {"phase": 1, "prompt_tokens": 1000, "cached_tokens": 200, "processed_tokens": 600}}
        telemetry.normalize(chunk1, acc, current_time=1.5, start_time=1.0)
        self.assertEqual(acc["last_progress"], {"total": 1000, "cache": 200, "processed": 600, "time_ms": 500})

        # Regression is ignored
        chunk_reg = {"kapitsa_progress": {"phase": 1, "prompt_tokens": 1000, "cached_tokens": 200, "processed_tokens": 500}}
        telemetry.normalize(chunk_reg, acc, current_time=1.6, start_time=1.0)
        self.assertEqual(acc["last_progress"]["processed"], 600)

        # Once ttft is known, progress chunks are ignored
        chunk2 = {"kapitsa_progress": {"phase": 1, "prompt_tokens": 1000, "cached_tokens": 200, "processed_tokens": 800}}
        telemetry.normalize(chunk2, acc, current_time=2.0, start_time=1.0, ttft=0.5)
        self.assertEqual(acc["last_progress"]["processed"], 600)

    def test_kapitsa_final_usage(self):
        acc = telemetry.init_acc()
        chunk = {
            "choices": [],
            "usage": {
                "prompt_tokens": 18145,
                "completion_tokens": 206,
                "prompt_tokens_details": {"cached_tokens": 145},
                "completion_tokens_details": {"reasoning_tokens": 78},
                "kapitsa": {
                    "prefill_ms": 81764.1,
                    "decode_ms": 4800.0,
                },
            },
        }
        telemetry.normalize(chunk, acc)
        self.assertTrue(acc["is_kapitsa"])
        self.assertEqual(acc["native_ttft_ms"], 81764.1)
        self.assertAlmostEqual(acc["native_decode_tps"], 205 * 1000.0 / 4800.0)
        expected_prefill = (18145 - 145) * 1000.0 / 81764.1
        self.assertAlmostEqual(acc["native_prefill_tps"], expected_prefill, places=4)
        self.assertEqual(acc["reasoning_tokens"], 78)

    def test_kapitsa_root_object(self):
        acc = telemetry.init_acc()
        chunk = {
            "choices": [],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 10},
            "kapitsa": {
                "prefill_ms": 500.0,
                "decode_ms": 300.0,
            },
        }
        telemetry.normalize(chunk, acc)
        self.assertTrue(acc["is_kapitsa"])
        self.assertEqual(acc["native_ttft_ms"], 500.0)
        self.assertEqual(acc["native_decode_tps"], 30.0)  # (10 - 1) tokens over 300 ms
        self.assertEqual(acc["native_prefill_tps"], 2000.0)


class TestNormalizeMTPLX(unittest.TestCase):
    def test_mtplx_stats(self):
        acc = telemetry.init_acc()
        chunk = {
            "mtplx_stats": {
                "ttft_s": 0.25,
                "prefill_tok_s": 500.0,
                "decode_tok_s": 65.0,
            }
        }
        telemetry.normalize(chunk, acc)
        self.assertEqual(acc["native_ttft_ms"], 250.0)
        self.assertEqual(acc["native_prefill_tps"], 500.0)
        self.assertEqual(acc["native_decode_tps"], 65.0)


class TestNormalizePromptProgress(unittest.TestCase):
    def test_prompt_progress_stream(self):
        acc = telemetry.init_acc()
        chunk1 = {"prompt_progress": {"total": 1000, "cache": 200, "processed": 300, "time_ms": 100}}
        telemetry.normalize(chunk1, acc)
        self.assertEqual(acc["last_progress"], {"total": 1000, "cache": 200, "processed": 300, "time_ms": 100})

        # Monotonic non-regression
        chunk_reg = {"prompt_progress": {"total": 1000, "cache": 200, "processed": 250, "time_ms": 50}}
        telemetry.normalize(chunk_reg, acc)
        self.assertEqual(acc["last_progress"]["processed"], 300)
        self.assertEqual(acc["last_progress"]["time_ms"], 100)

        # Progression
        chunk2 = {"prompt_progress": {"total": 1000, "cache": 200, "processed": 800, "time_ms": 300}}
        telemetry.normalize(chunk2, acc)
        self.assertEqual(acc["last_progress"]["processed"], 800)
        self.assertEqual(acc["last_progress"]["time_ms"], 300)

        # Ignored after first token (ttft is not None)
        chunk_late = {"prompt_progress": {"total": 1000, "cache": 200, "processed": 1000, "time_ms": 500}}
        telemetry.normalize(chunk_late, acc, ttft=0.3)
        self.assertEqual(acc["last_progress"]["processed"], 800)


class TestNormalizeMtplx(unittest.TestCase):
    def test_mtplx_stats_prioritized_over_timings(self):
        acc = telemetry.init_acc()
        chunk = {
            "choices": [{"delta": {}}],
            "timings": {
                "prompt_n": 18679,
                "prompt_ms": 3515.0,
                "prompt_per_second": 5313.4,
                "predicted_n": 70,
                "predicted_per_second": 22.9,
            },
            "mtplx_stats": {
                "ttft_s": 4.37,
                "prefill_tok_s": 160.0,
                "decode_tok_s": 22.9,
                "prompt_tokens": 18679,
                "completion_tokens": 70,
                "cached_tokens": 18117,
            },
        }
        telemetry.normalize(chunk, acc)
        self.assertEqual(acc["native_ttft_ms"], 4370.0)
        self.assertEqual(acc["native_prefill_tps"], 160.0)
        self.assertEqual(acc["native_decode_tps"], 22.9)
        self.assertEqual(acc["prompt_tokens"], 18679)
        self.assertEqual(acc["cached_tokens"], 18117)
        self.assertEqual(acc["exact_tokens"], 70)

    def test_mtplx_progress_updates_exact_tokens(self):
        acc = telemetry.init_acc()
        chunk = {"mtplx_progress": {"completion_tokens": 42}}
        telemetry.normalize(chunk, acc)
        self.assertEqual(acc["exact_tokens"], 42)

    def test_timings_scaled_on_cache_hit(self):
        acc = telemetry.init_acc()
        acc["prompt_tokens"] = 10000
        acc["cached_tokens"] = 9000  # 1000 uncached (10%)
        chunk = {
            "timings": {
                "prompt_n": 10000,
                "prompt_ms": 1000.0,
                "prompt_per_second": 10000.0,
            }
        }
        telemetry.normalize(chunk, acc)
        # Scaled by uncached (1000) / prompt_tokens (10000) = 0.1 * 10000.0 = 1000.0
        self.assertEqual(acc["native_prefill_tps"], 1000.0)


class TestFinalize(unittest.TestCase):
    def test_cache_hit_drops_prefill_speed(self):
        acc = telemetry.init_acc()
        acc["prompt_tokens"] = 1000
        acc["cached_tokens"] = 995  # only 5 uncached (<10)
        acc["native_prefill_tps"] = 50000.0  # absurd KV illusion
        telemetry.finalize(acc, total_duration=1.0, ttft=0.1, token_count=5)
        self.assertIsNone(acc["native_prefill_tps"])

    def test_cache_miss_keeps_prefill_speed(self):
        acc = telemetry.init_acc()
        acc["prompt_tokens"] = 1000
        acc["cached_tokens"] = 900  # 100 uncached (>=10)
        acc["native_prefill_tps"] = 500.0
        telemetry.finalize(acc, total_duration=1.0, ttft=0.1, token_count=5)
        self.assertEqual(acc["native_prefill_tps"], 500.0)

    def test_ttft_derivation_from_native(self):
        acc = telemetry.init_acc()
        acc["native_ttft_ms"] = 250.0
        ttft = telemetry.finalize(acc, total_duration=1.0, ttft=None, token_count=5)
        self.assertEqual(ttft, 0.25)

    def test_kapitsa_decode_speed_estimation(self):
        acc = telemetry.init_acc()
        acc["is_kapitsa"] = True
        acc["native_decode_tps"] = None
        ttft = 0.5
        total_duration = 1.5  # decode duration = 1.0 s
        token_count = 11  # (11 - 1) / 1.0 = 10.0 tps
        telemetry.finalize(acc, total_duration=total_duration, ttft=ttft, token_count=token_count)
        self.assertEqual(acc["native_decode_tps"], 10.0)


class TestCachedTokensTelemetry(unittest.TestCase):
    def test_init_acc_cached_tokens_is_none(self):
        acc = telemetry.init_acc()
        self.assertIsNone(acc["cached_tokens"])

    def test_usage_without_cached_tokens_leaves_it_none(self):
        acc = telemetry.init_acc()
        chunk = {
            "choices": [{"delta": {}}],
            "usage": {
                "prompt_tokens": 334,
                "completion_tokens": 60,
                "total_tokens": 394,
            },
        }
        telemetry.normalize(chunk, acc)
        self.assertIsNone(acc["cached_tokens"])

    def test_usage_with_cached_tokens_zero_records_zero(self):
        acc = telemetry.init_acc()
        chunk = {
            "choices": [{"delta": {}}],
            "usage": {
                "prompt_tokens": 334,
                "completion_tokens": 60,
                "cached_tokens": 0,
            },
        }
        telemetry.normalize(chunk, acc)
        self.assertEqual(acc["cached_tokens"], 0)

    def test_lmstudio_stream_tail(self):
        # LM Studio (llama.cpp runtime 2.71.0, speculative decoding on), streamed /api/v0/chat/completions, last chunk
        # (choices: []): usage and the draft counters only, no speeds, no TTFT, no cached tokens.
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lmstudio_v0_stream_tail.json")) as f:
            chunk = json.load(f)
        acc = telemetry.init_acc()
        telemetry.normalize(chunk, acc)
        self.assertIsNone(acc["native_decode_tps"])
        self.assertIsNone(acc["native_ttft_ms"])
        self.assertIsNone(acc["native_prefill_tps"])
        self.assertIsNone(acc["cached_tokens"])
        self.assertEqual((acc["prompt_tokens"], acc["exact_tokens"], acc["reasoning_tokens"]), (59, 40, 40))
        self.assertEqual(acc["raw_stats"], chunk["stats"])
        self.assertEqual(acc["raw_metrics"], {"speculation": {"drafted_tokens": 49, "accepted_draft_tokens": 21}})

    def test_lmstudio_non_streaming_stats_still_read(self):
        acc = telemetry.normalize({"stats": {"tokens_per_second": 22.26, "time_to_first_token": 0.793,
                                             "generation_time": 2.2, "stop_reason": "eosFound"}}, telemetry.init_acc())
        self.assertEqual(acc["native_decode_tps"], 22.26)
        self.assertEqual(acc["native_ttft_ms"], 793.0)

    def test_lmstudio_draft_counts_from_usage(self):
        acc = telemetry.normalize({"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_draft_tokens_count": 10,
                                             "accepted_draft_tokens_count": 4}}, telemetry.init_acc())
        self.assertEqual(acc["raw_metrics"]["speculation"], {"drafted_tokens": 10, "accepted_draft_tokens": 4})


DSPARK = {"mode": "dflash", "accept_len": 5.714, "tokens_per_sec": 19.6, "decode_tokens_per_sec": 50.9, "target_forwards": 15,
          "cap": 7, "prompt_tokens": 334, "cached_tokens": 0, "completion_tokens": 80, "context_tokens": 414,
          "prefill_seconds": 2.517, "decode_seconds": 1.57, "ttft_seconds": 2.517, "prefill_tokens_per_sec": 132.7,
          "ceiling_tokens_per_sec": 22.7, "roofline_ratio": 2.249}


class TestMlxDspark(unittest.TestCase):
    def test_decode_prefill_and_ttft_keys_win_over_end_to_end_tokens_per_sec(self):
        acc = telemetry.normalize({"x_mlx_dspark": dict(DSPARK)}, telemetry.init_acc())
        self.assertEqual(acc["native_decode_tps"], 50.9)  # not tokens_per_sec (19.6, end-to-end)
        self.assertEqual(acc["native_prefill_tps"], 132.7)
        self.assertAlmostEqual(acc["native_ttft_ms"], 2517.0)
        self.assertEqual((acc["prompt_tokens"], acc["cached_tokens"], acc["exact_tokens"]), (334, 0, 80))

    def test_tokens_per_sec_alone_is_not_a_decode_speed(self):
        acc = telemetry.normalize({"x_mlx_dspark": {"tokens_per_sec": 2.3}}, telemetry.init_acc())
        self.assertIsNone(acc["native_decode_tps"])

    def test_accept_len_and_target_forwards_become_speculation(self):
        acc = telemetry.normalize({"x_mlx_dspark": dict(DSPARK)}, telemetry.init_acc())
        sp = acc["raw_metrics"]["speculation"]
        self.assertEqual(sp["cycles"], 15)
        self.assertAlmostEqual(1 + sp["accepted_draft_tokens"] / sp["cycles"], 5.714, places=1)  # tok/cycle = accept_len

    def test_final_chunk_after_usage_keeps_usage_counts(self):
        acc = telemetry.init_acc()
        telemetry.normalize({"usage": {"prompt_tokens": 334, "completion_tokens": 80,
                                       "prompt_tokens_details": {"cached_tokens": 300}}}, acc)
        telemetry.normalize({"x_mlx_dspark": dict(DSPARK)}, acc)
        self.assertEqual(acc["cached_tokens"], 300)


if __name__ == "__main__":
    unittest.main()
