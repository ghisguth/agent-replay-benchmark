"""The terminal tables fit the terminal: the width comes from the stream written to (a tty, no COLUMNS in the
environment, as in zsh), the reason column gives way first, then Stream and Case (middle ellipsis), then columns drop."""

import contextlib
import io
import os
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from bench import report  # noqa: E402
from bench import results as R  # noqa: E402
from bench.report import Table, fit, fit_columns, term_limit  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")


class FakeTty(io.StringIO):
    def isatty(self):
        return True


def conv3_metrics(n=6):
    """Results shaped like a conv-3 state: long ids, long subagent stream names, a native-speed column set."""
    out = []
    for i in range(n):
        sub = i % 2
        out.append({"id": "R%03d_F428FalsifyP1213R2_t%03d" % (232 + i, 24 + i) if sub else "R%03d_main_e0_t%03d" % (i, i),
                    "stream": "F428FalsifyP1213R2" if sub else "main", "epoch": 0, "ttft_ms": 3500 + i, "total_ms": 9000,
                    "prompt_tokens": 180000 + i, "cached_tokens": 170000, "tokens": 120, "native_decode_tps": 25.5,
                    "native_prefill_tps": 140.0, "eval_result": "WARN" if sub else "PASS",
                    "fail_reason": "early report: the model stopped before the recorded session did" if sub else "",
                    "content": "I will read the file next and then run the tests to see what fails"})
    return out


def lines(text):
    return [ANSI.sub("", l) for l in text.splitlines()]


def widest(text):
    return max(report.vlen(l) for l in lines(text))


def state_table(results):
    layout = R.CaseLayout(True, False, True)
    rows = R.case_rows(layout, results) + R.totals_rows(layout, results)
    return layout, Table(layout.header, rows, aligns=layout.aligns, dim_cols=layout.dim_cols)


class TermLimit(unittest.TestCase):
    def test_tty_without_columns_env_uses_the_terminal(self):
        env = {k: v for k, v in os.environ.items() if k != "COLUMNS"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("os.get_terminal_size", side_effect=OSError), \
                mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((154, 24))):
            self.assertEqual(term_limit(FakeTty()), 154)

    def test_not_a_tty_uses_columns_then_no_limit(self):
        with mock.patch.dict(os.environ, {"COLUMNS": "90"}):
            self.assertEqual(term_limit(io.StringIO()), 90)
        env = {k: v for k, v in os.environ.items() if k != "COLUMNS"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(term_limit(io.StringIO()))


class FitText(unittest.TestCase):
    def test_middle_ellipsis_keeps_both_ends(self):
        s = fit("F428FalsifyP1213R2", 12, mid=True)
        self.assertEqual(report.vlen(s), 12)
        self.assertTrue(s.startswith("F428F") and s.endswith("1213R2"[-4:]) and "…" in s)

    def test_fit_columns_never_wider(self):
        w = [30, 20, 6, 6, 10, 9, 4, 6, 7, 7, 60]
        h = ["Case", "Stream", "TTFT", "Time", "In", "Out", "C%", "Decode", "Prefill", "Verdict", "Reason / Preview"]
        for limit in (200, 154, 120, 100, 80, 60):
            self.assertLessEqual(report.table_width(fit_columns(h, w, limit, Table.SHRINK, Table.DROP, Table.LATE)), limit)


class StateTables(unittest.TestCase):
    def test_final_table_fits_every_width(self):
        _, table = state_table(conv3_metrics())
        for width in (154, 120, 100, 80):
            with mock.patch.object(report, "term_limit", return_value=width):
                self.assertLessEqual(widest(table.tui()), width, width)

    def test_live_table_fits_every_width(self):
        res = conv3_metrics()
        layout, table = state_table(res)
        for width in (154, 120, 100, 80):
            w = layout.widths([m["id"] for m in res], ["model"], [R.stream_label(m) for m in res], term=width)
            out = [table.tui_top(w), table.tui_head(w)] + [table.tui_row(r, w) for r in R.case_rows(layout, res)]
            out += [table.tui_sep(w)] + [table.tui_row(r, w) for r in R.totals_rows(layout, res)] + [table.tui_bottom(w)]
            self.assertLessEqual(widest("\n".join(out)), width, width)

    def test_wide_terminal_and_no_limit_keep_everything(self):
        _, table = state_table(conv3_metrics())
        with mock.patch.object(report, "term_limit", return_value=None):
            text = ANSI.sub("", table.tui())
        for col in ("Stream", "C%", "Prefill", "Out"):
            self.assertIn(col, text)
        self.assertIn("F428FalsifyP1213R2", text)
        self.assertIn("R233_F428FalsifyP1213R2_t025", text)

    def test_ids_stay_whole_at_120_and_100(self):
        _, table = state_table(conv3_metrics())
        for width in (120, 100):
            with mock.patch.object(report, "term_limit", return_value=width):
                text = ANSI.sub("", table.tui())
            self.assertIn("R233_F428FalsifyP1213R2_t025", text, width)
            self.assertNotIn("Prev\u2026", text)

    def test_case_table_fills_the_terminal_width(self):
        """The live tui case table (CaseLayout.widths): the reason column takes what is left, whatever the width."""
        res = conv3_metrics()
        ids = [r["id"] for r in res]
        streams = [R.stream_label(r) for r in res]
        for width in (100, 160, 220, 300):
            layout = R.CaseLayout(True, False, True)
            _, table = state_table(res)
            widths = layout.widths(ids, ["m"], streams, term=width)
            rows = R.case_rows(layout, res) + R.totals_rows(layout, res)
            tbl = Table(layout.header, rows, aligns=layout.aligns, dim_cols=layout.dim_cols)
            text = "\n".join([ANSI.sub("", x) for x in (tbl.tui_top(widths), tbl.tui_head(widths)) +
                              tuple(tbl.tui_row(r, widths) for r in rows)])
            self.assertLessEqual(widest(text), width, width)
            if width >= 160:
                self.assertEqual(widest(text), width, width)  # full width, not capped
                self.assertIn("Reason", text)
            if width >= 220:
                self.assertGreater(layout.reason_limit, R.REASON_MAX)
                self.assertIn("I will read the file next and then run the tests to see what fails", text)
        # no terminal (a pipe): the natural width, the reason capped
        layout = R.CaseLayout(True, False, True)
        widths = layout.widths(ids, ["m"], streams, term=None)
        self.assertEqual(widths[layout.rcol], R.REASON_MAX)

    def test_narrow_reason_header_is_plain_reason(self):
        _, table = state_table(conv3_metrics())
        with mock.patch.object(report, "term_limit", return_value=80):
            head = ANSI.sub("", table.tui()).split("\n")[1]
        self.assertIn("Reason", head)
        self.assertNotIn("\u2026", head)

    def test_stream_shrinks_before_columns_drop(self):
        _, table = state_table(conv3_metrics())
        with mock.patch.object(report, "term_limit", return_value=154):
            text = ANSI.sub("", table.tui())
        self.assertIn("Stream", text)
        self.assertIn("Prefill", text)

    def test_other_tables_fit_too(self):
        t = Table(["Band", "Requests", "Note"], [["a" * 40, "12", "b" * 60]])
        with mock.patch.object(report, "term_limit", return_value=50):
            self.assertLessEqual(widest(t.tui()), 50)

    def test_tty_stdout_without_columns_env(self):
        """The --state report path: stdout is a tty, no COLUMNS, the table is fitted at print time."""
        _, table = state_table(conv3_metrics())
        env = {k: v for k, v in os.environ.items() if k != "COLUMNS"}
        out = FakeTty()
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("os.get_terminal_size", side_effect=OSError), \
                mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((100, 24))), \
                contextlib.redirect_stdout(out):
            text = table.tui()
        self.assertLessEqual(widest(text), 100)


if __name__ == "__main__":
    unittest.main()
