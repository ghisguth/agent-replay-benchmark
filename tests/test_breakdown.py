import os
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from bench import fmt as F  # noqa: E402
from bench import report  # noqa: E402
from bench import results as R  # noqa: E402
from bench.report import Table  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def widest(text):
    return max(report.vlen(ANSI.sub("", l)) for l in text.splitlines())


class BreakdownTableTests(unittest.TestCase):
    def test_rows_and_counts_right(self):
        res = [
            # main <64k: 2 cases (1 PASS, 1 WARN)
            {
                "id": "R001_main_e0_t000", "stream": "main", "kind": "turn", "epoch": 0,
                "prompt_tokens": 10000, "tokens": 100, "native_decode_tps": 50.0, "native_prefill_tps": 150.0,
                "eval_result": "PASS", "matches_recorded": True, "in_line": True,
            },
            {
                "id": "R002_main_e0_t001", "stream": "main", "kind": "turn", "epoch": 0,
                "prompt_tokens": 20000, "tokens": 200, "native_decode_tps": 40.0, "native_prefill_tps": 100.0,
                "eval_result": "WARN", "matches_recorded": False, "in_line": True,
            },
            # main 64-128k: 3 cases (2 PASS, 1 FAIL)
            {
                "id": "R003_main_e0_t002", "stream": "main", "kind": "turn", "epoch": 0,
                "prompt_tokens": 70000, "tokens": 300, "native_decode_tps": 30.0, "native_prefill_tps": 80.0,
                "eval_result": "PASS", "matches_recorded": True, "in_line": True,
            },
            {
                "id": "R004_main_e0_t003", "stream": "main", "kind": "turn", "epoch": 0,
                "prompt_tokens": 80000, "tokens": 300, "native_decode_tps": 30.0, "native_prefill_tps": 80.0,
                "eval_result": "PASS", "matches_recorded": True, "in_line": True,
            },
            {
                "id": "R005_main_e0_t004", "stream": "main", "kind": "turn", "epoch": 0,
                "prompt_tokens": 90000, "tokens": 400, "native_decode_tps": 20.0, "native_prefill_tps": 70.0,
                "eval_result": "FAIL", "matches_recorded": False, "in_line": False,
            },
            # compaction ≥192k: 1 case
            {
                "id": "R157_main_c0", "stream": "main", "kind": "compaction", "epoch": 0,
                "prompt_tokens": 200000, "tokens": 4000, "native_decode_tps": 15.0, "native_prefill_tps": 50.0,
                "eval_result": "PASS", "matches_recorded": True, "in_line": True,
            },
            # subagent <64k: 2 cases (1 PASS, 1 SKIP)
            {
                "id": "R200_S1_t000", "stream": "S1", "kind": "turn", "epoch": 0,
                "prompt_tokens": 15000, "tokens": 120, "native_decode_tps": 60.0, "native_prefill_tps": 200.0,
                "eval_result": "PASS", "matches_recorded": True, "in_line": True,
            },
            {
                "id": "R201_S1_t001", "stream": "S1", "kind": "turn", "epoch": 0,
                "prompt_tokens": 25000, "tokens": 0,
                "eval_result": "SKIP",
            },
        ]

        table = R.breakdown_table(res)
        self.assertIsNotNone(table)
        self.assertIsInstance(table, Table)
        self.assertEqual(len(table.rows), 4)
        self.assertEqual(table.header[8], "In line N/M (%)")

        # Check row 0 (main 0-32k)
        r0 = [str(c) for c in table.rows[0]]
        self.assertEqual(r0, ["main", "0-32k", "2", "1", "1", "0", "0", "50%", "2/2 (100%)", "45.0", "125.0"])

        # Check row 1 (main 64-96k)
        r1 = [str(c) for c in table.rows[1]]
        self.assertEqual(r1, ["main", "64-96k", "3", "2", "0", "1", "0", "67%", "2/3 (67%)", "30.0", "80.0"])

        # Check row 2 (compaction 192-224k): compaction excluded from agreement & consensus
        r2 = [str(c) for c in table.rows[2]]
        self.assertEqual(r2, ["compaction", "192-224k", "1", "1", "0", "0", "0", F.DASH, F.DASH, "15.0", "50.0"])

        # Check row 3 (subagent 0-32k): 1 PASS, 1 SKIP
        r3 = [str(c) for c in table.rows[3]]
        self.assertEqual(r3, ["subagent", "0-32k", "2", "1", "0", "0", "1", "100%", "1/1 (100%)", "60.0", "200.0"])

    def test_empty_groups_absent(self):
        res = [
            {"id": "R001", "stream": "main", "prompt_tokens": 1000, "eval_result": "PASS"},
            {"id": "R002", "stream": "S1", "prompt_tokens": 150000, "eval_result": "PASS"},
        ]
        table = R.breakdown_table(res)
        self.assertIsNotNone(table)
        self.assertEqual(len(table.rows), 2)
        r0 = [str(c) for c in table.rows[0]]
        r1 = [str(c) for c in table.rows[1]]
        self.assertEqual((r0[0], r0[1]), ("main", "0-32k"))
        self.assertEqual((r1[0], r1[1]), ("subagent", "128-160k"))

    def test_skip_table_when_fewer_than_2_rows(self):
        self.assertIsNone(R.breakdown_table([]))
        self.assertIsNone(R.breakdown_table([{"id": "R001", "stream": "main", "prompt_tokens": 1000}]))
        # 2 cases in the same group and band -> 1 row -> skipped
        self.assertIsNone(R.breakdown_table([
            {"id": "R001", "stream": "main", "prompt_tokens": 1000},
            {"id": "R002", "stream": "main", "prompt_tokens": 2000},
        ]))

    def test_consensus_none_when_no_consensus_ids(self):
        res = [
            {"id": "R001", "stream": "main", "prompt_tokens": 1000, "eval_result": "PASS", "in_line": None},
            {"id": "R002", "stream": "S1", "prompt_tokens": 1000, "eval_result": "PASS", "in_line": None},
        ]
        table = R.breakdown_table(res)
        self.assertIsNotNone(table)
        # When all rows have no consensus, the column is omitted entirely
        self.assertNotIn("In line N/M (%)", table.header)
        self.assertNotIn("Path N/M (%)", table.header)

        # But when at least one row has consensus data, the column is kept
        res[0]["in_line"] = True
        table_with_con = R.breakdown_table(res)
        self.assertIsNotNone(table_with_con)
        self.assertIn("In line N/M (%)", table_with_con.header)

    def test_fits_80_columns(self):
        # Create multiple rows with long values across stream groups and bands
        res = [
            {"id": f"R00{i}", "stream": "main", "prompt_tokens": 10000, "eval_result": "PASS",
             "tokens": 1234, "native_decode_tps": 45.6, "native_prefill_tps": 1234.5,
             "matches_recorded": True, "in_line": True}
            for i in range(5)
        ] + [
            {"id": f"R01{i}", "stream": "F428FalsifyP1213R2", "prompt_tokens": 70000, "eval_result": "WARN",
             "tokens": 456, "native_decode_tps": 23.4, "native_prefill_tps": 567.8,
             "matches_recorded": False, "in_line": True}
            for i in range(5)
        ] + [
            {"id": "R157", "stream": "main", "kind": "compaction", "prompt_tokens": 200000, "eval_result": "PASS",
             "tokens": 4936, "native_decode_tps": 13.6, "native_prefill_tps": 50.3}
        ]

        table = R.breakdown_table(res)
        self.assertIsNotNone(table)
        with mock.patch.object(report, "term_limit", return_value=80):
            tui_text = table.tui()
        self.assertLessEqual(widest(tui_text), 80)

    def test_fallback_to_recorded_prompt_tokens(self):
        res = [
            {"id": "R001", "stream": "main", "prompt_tokens": None, "recorded_prompt_tokens": 80000, "eval_result": "PASS"},
            {"id": "R002", "stream": "main", "prompt_tokens": None, "recorded_prompt_tokens": 1000, "eval_result": "PASS"},
        ]
        table = R.breakdown_table(res)
        self.assertIsNotNone(table)
        self.assertEqual(len(table.rows), 2)
        # Both rows share "main" stream, so Stream column is omitted
        self.assertNotIn("Stream", table.header)
        bands = [str(r[0]) for r in table.rows]
        self.assertEqual(bands, ["0-32k", "64-96k"])

    def test_smoke_cases_ignored(self):
        # Cases without stream or compaction are ignored
        res = [
            {"id": "TC01", "eval_result": "PASS", "prompt_tokens": 1000},
            {"id": "TC02", "eval_result": "PASS", "prompt_tokens": 2000},
        ]
        self.assertIsNone(R.breakdown_table(res))

    def test_dict_input(self):
        res_dict = {
            "R001": {"id": "R001", "stream": "main", "prompt_tokens": 1000, "eval_result": "PASS"},
            "R002": {"id": "R002", "stream": "S1", "prompt_tokens": 1000, "eval_result": "PASS"},
        }
        table = R.breakdown_table(res_dict)
        self.assertIsNotNone(table)
        self.assertEqual(len(table.rows), 2)

    def test_markdown_rendering(self):
        res = [
            {
                "id": "R001", "stream": "main", "prompt_tokens": 1000, "eval_result": "PASS",
                "tokens": 100, "native_decode_tps": 50.0, "native_prefill_tps": 150.0,
                "matches_recorded": True, "in_line": True,
            },
            {
                "id": "R002", "stream": "S1", "prompt_tokens": 1000, "eval_result": "PASS",
                "tokens": 200, "native_decode_tps": 40.0, "native_prefill_tps": 100.0,
                "matches_recorded": False, "in_line": True,
            },
        ]
        table = R.breakdown_table(res)
        md = table.markdown()
        head = md.splitlines()[0]
        for col in ("Stream", "Band", "Cases", "PASS", "WARN", "FAIL", "SKIP", "Same tools %",
                    "In line N/M (%)", "Decode tok/s p50", "Prefill tok/s p50"):
            self.assertIn(col, head)
        self.assertNotIn("Out tok p50", head)
        self.assertNotIn("Consensus N/M (%)", head)
        self.assertNotIn("Path N/M (%)", head)
        self.assertIn("| `main`", md)
        self.assertIn("| `subagent`", md)

    def test_all_dash_column_omitted(self):
        # res where timings are absent across all rows
        res = [
            {"id": "R001", "stream": "main", "prompt_tokens": 1000, "eval_result": "PASS",
             "matches_recorded": True, "in_line": True},
            {"id": "R002", "stream": "S1", "prompt_tokens": 1000, "eval_result": "PASS",
             "matches_recorded": True, "in_line": True},
        ]
        table = R.breakdown_table(res)
        self.assertIsNotNone(table)
        for omitted in ("Decode tok/s p50", "Prefill tok/s p50", "Tok/cycle"):
            self.assertNotIn(omitted, table.header)
        for kept in ("Stream", "Band", "Cases", "PASS", "WARN", "FAIL", "SKIP", "Same tools %", "In line N/M (%)"):
            self.assertIn(kept, table.header)

    def test_stream_column_omitted_when_single_group(self):
        # 2 cases in different bands, both in stream "main"
        res_single = [
            {"id": "R001", "stream": "main", "prompt_tokens": 1000, "eval_result": "PASS"},
            {"id": "R002", "stream": "main", "prompt_tokens": 70000, "eval_result": "PASS"},
        ]
        table = R.breakdown_table(res_single)
        self.assertIsNotNone(table)
        self.assertNotIn("Stream", table.header)
        self.assertEqual(table.header[0], "Band")

        # 2 cases in different stream groups
        res_multi = [
            {"id": "R001", "stream": "main", "prompt_tokens": 1000, "eval_result": "PASS"},
            {"id": "R002", "stream": "S1", "prompt_tokens": 1000, "eval_result": "PASS"},
        ]
        table_multi = R.breakdown_table(res_multi)
        self.assertIsNotNone(table_multi)
        self.assertIn("Stream", table_multi.header)
        self.assertEqual(table_multi.header[0], "Stream")

    def test_speculation_columns_included_when_metrics_present(self):
        res = [
            {"id": "R001", "stream": "main", "prompt_tokens": 1000, "eval_result": "PASS",
             "tokens": 100, "total_ms": 2000, "ttft_ms": 500, "native_decode_tps": 50.0, "native_prefill_tps": 150.0,
             "metrics": {"speculation": {"cycles": 20, "accepted_draft_tokens": 60}}},
            {"id": "R002", "stream": "main", "prompt_tokens": 70000, "eval_result": "PASS",
             "tokens": 80, "total_ms": 1500, "ttft_ms": 300, "native_decode_tps": 40.0, "native_prefill_tps": 100.0},
        ]
        table = R.breakdown_table(res)
        self.assertIsNotNone(table)
        self.assertIn("Tok/cycle", table.header)
        self.assertIn("Cycles/s", table.header)
        self.assertIn("ms/cycle", table.header)

    def test_drop_order_prefers_dropping_cycles_before_prefill(self):
        res = [
            {
                "id": "R001", "stream": "main", "prompt_tokens": 1000, "eval_result": "PASS",
                "tokens": 100, "total_ms": 2000, "ttft_ms": 500, "native_decode_tps": 50.0, "native_prefill_tps": 150.0,
                "metrics": {"speculation": {"cycles": 20, "accepted_draft_tokens": 60}},
                "matches_recorded": True, "in_line": True,
            },
            {
                "id": "R002", "stream": "S1", "prompt_tokens": 70000, "eval_result": "PASS",
                "tokens": 200, "total_ms": 3000, "ttft_ms": 800, "native_decode_tps": 40.0, "native_prefill_tps": 100.0,
                "matches_recorded": False, "in_line": True,
            },
        ]
        table = R.breakdown_table(res)
        w = table.widths()
        total_w = sum(w) + 3 * len(w) + 1
        tight_limit = total_w - 5
        fitted = table.fitted(w, limit=tight_limit)
        kept_cols = [h for h, col_w in zip(table.header, fitted) if col_w is not None]
        self.assertNotIn("ms/cycle", kept_cols)
        self.assertIn("Prefill tok/s p50", kept_cols)


class BandLabelTests(unittest.TestCase):
    def test_specific_token_counts(self):
        self.assertEqual(R.band_label(0), "0-32k")
        self.assertEqual(R.band_label(32767), "0-32k")
        self.assertEqual(R.band_label(32768), "32-64k")
        self.assertEqual(R.band_label(131072), "128-160k")


class MergedTableTests(unittest.TestCase):
    def test_merged_table_without_baseline(self):
        res = [
            {"id": "R001", "stream": "main", "prompt_tokens": 1000, "eval_result": "PASS",
             "tokens": 50, "native_decode_tps": 40.0, "native_prefill_tps": 500.0},
            {"id": "R002", "stream": "main", "prompt_tokens": 70000, "eval_result": "PASS",
             "tokens": 60, "native_decode_tps": 30.0, "native_prefill_tps": 400.0},
        ]
        table = R.breakdown_table(res)
        self.assertIsNotNone(table)
        self.assertIn("Decode tok/s p50", table.header)
        self.assertIn("Prefill tok/s p50", table.header)
        self.assertNotIn("Δ", table.header)

    def test_merged_table_with_baseline(self):
        res = [
            {"id": "R001", "stream": "main", "prompt_tokens": 1000, "eval_result": "PASS",
             "tokens": 50, "native_decode_tps": 50.0, "native_prefill_tps": 600.0},
            {"id": "R002", "stream": "main", "prompt_tokens": 70000, "eval_result": "PASS",
             "tokens": 60, "native_decode_tps": 40.0, "native_prefill_tps": 500.0},
        ]
        base_res = [
            {"id": "R001", "stream": "main", "prompt_tokens": 1000, "eval_result": "PASS",
             "tokens": 50, "native_decode_tps": 40.0, "native_prefill_tps": 500.0},
            {"id": "R002", "stream": "main", "prompt_tokens": 70000, "eval_result": "PASS",
             "tokens": 60, "native_decode_tps": 30.0, "native_prefill_tps": 400.0},
        ]
        table = R.breakdown_table(res, base_results=base_res)
        self.assertIsNotNone(table)
        self.assertIn("Δ", table.header)
        self.assertEqual(table.header.count("Δ"), 2)


class DeltaHelperTests(unittest.TestCase):
    def test_pct_deltas(self):
        c = F.delta_cell(120, 100, "up")
        self.assertEqual(str(c), "+20.0%")
        self.assertEqual(c.style, "good")

        c = F.delta_cell(80, 100, "up")
        self.assertEqual(str(c), "-20.0%")
        self.assertEqual(c.style, "bad")

        c = F.delta_cell(120, 100, "down")
        self.assertEqual(c.style, "bad")

        c = F.delta_cell(100, 100, "up")
        self.assertEqual(str(c), "+0.0%")
        self.assertEqual(c.style, "dim")

        c = F.delta_cell(None, 100)
        self.assertEqual(str(c), F.DASH)
        self.assertEqual(c.style, "dim")

    def test_native_against_estimate_is_dim_and_approximate(self):
        c = R.delta_cell(120, 100, "up", approx=True)
        self.assertEqual((str(c), c.style), ("~+20.0%", "dim"))
        layout = R.CaseLayout(True, base=True)
        native = {"total_ms": 1000, "native_decode_tps": 20.0, "native_prefill_tps": 100.0}
        est = {"total_ms": 1000, "tokens": 20, "ttft_ms": 500, "prompt_tokens": 100}  # ~40 decode, ~200 prefill
        cells = layout.delta_cells(est, native)
        self.assertEqual([str(x) for x in cells[1:]], ["~+100.0%", "~+100.0%"])
        self.assertEqual({x.style for x in cells[1:]}, {"dim"})
        both = layout.delta_cells(native, native)
        self.assertEqual(str(both[1]), "+0.0%")

    def test_pt_deltas(self):
        c = F.delta_cell(75.0, 50.0, "up", unit="pt")
        self.assertEqual(str(c), "+25.0 pt")
        self.assertEqual(c.style, "good")

        c = F.delta_cell(50.0, 50.0, "up", unit="pt")
        self.assertEqual(str(c), "+0.0 pt")
        self.assertEqual(c.style, "dim")


if __name__ == "__main__":
    unittest.main()
