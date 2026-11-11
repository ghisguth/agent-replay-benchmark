"""--compare-state against another model or server: which block of the baseline file is used, how the baseline is
labelled, the per-request columns (ΔTime, ΔDecode, ΔPrefill, Base), width fitting and markdown. Synthetic state files in
a temp dir, no server.

    python3 -m unittest tests.test_baseline_compare
"""

import contextlib
import io
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

import replay_bench  # noqa: E402
from bench import baseline, report, state  # noqa: E402
from bench import results as R  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")
IDS = ["R%03d_main_e0_t%03d" % (i, i) for i in range(6)]


def target(model, kind="splash", version="1.6.0-m1", host="h:1"):
    return {"key": "%s::%s" % (host, model), "backend": host, "name": model, "url": "http://%s/v1/chat/completions" % host,
            "n_ctx": 262144, "backend_info": {"kind": kind, "version": version, "build": "x", "details": {}}}


def result(cid, speed, tools=("bash",), verdict="PASS", i=0):
    return {"id": cid, "stream": "main", "kind": "turn", "epoch": 0, "content": "", "text": "", "ttft_ms": 500 + 100 * i,
            "total_ms": 2000 + 100 * i, "tokens": 50, "prompt_tokens": 10000 + 1000 * i, "cached_tokens": 9000,
            "native_prefill_tps": 100.0 * speed / 40, "native_decode_tps": speed, "eval_result": verdict,
            "fail_reason": "boom" if verdict == "FAIL" else "", "tool_names": list(tools), "matches_recorded": True,
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": t, "arguments": "{}"}} for t in tools]}


def write(path, blocks, suite="coding"):
    """blocks: [(model, kwargs of target(), speed, {case id: (tools, verdict)}, note)]; one file, one block each."""
    w = state.StateWriter(path)
    for model, tkw, speed, special, note in blocks:
        h = state.make_header(suite, target(model, **tkw), request={"temperature": None, "reasoning_effort": None},
                              fixture_path=None, argv=[], selection=None, started_at="2026-11-07T10:00:00+00:00", note=note)
        w.write(h)
        raw = {}
        for i, cid in enumerate(IDS):
            tools, verdict = (special or {}).get(cid, (("bash",), "PASS"))
            raw[cid] = result(cid, speed, tools, verdict, i)
            w.write(state.result_record(raw[cid]))
        w.write(state.end_record(raw))
    w.close()


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            replay_bench.main(list(argv))
        except SystemExit as e:
            if e.code not in (0, None):
                raise
    return ANSI.sub("", out.getvalue()), err.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: report.configure("tui", io.StringIO()))

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def make(self, name, blocks, **kw):
        p = self.path(name)
        write(p, blocks, **kw)
        return p


class Selection(Base):
    def test_matching_backend_and_model_wins(self):
        p = self.make("b.jsonl", [("a", {"host": "h:2"}, 10, None, None), ("a", {"host": "h:1"}, 20, None, None)])
        bl = baseline.load_baseline(p)
        self.assertEqual(bl.pick("h:1", "a").header["backend"], "h:1")
        self.assertEqual(bl.pick("h:1", "a")[IDS[0]]["native_decode_tps"], 20)

    def test_same_model_on_another_server(self):
        p = self.make("b.jsonl", [("a", {"host": "h:2"}, 10, None, None), ("b", {"host": "h:1"}, 20, None, None)])
        self.assertEqual(baseline.load_baseline(p).pick("h:9", "b").header["model"], "b")

    def test_a_single_block_is_the_baseline_whatever_it_is(self):
        p = self.make("b.jsonl", [("other", {"kind": "mtplx", "version": "2.12.3"}, 10, None, None)])
        bl = baseline.load_baseline(p)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(bl.pick("h:1", "mine").header["model"], "other")
        self.assertEqual(err.getvalue(), "")

    def test_several_blocks_none_matching_takes_the_first_and_says_so_once(self):
        p = self.make("b.jsonl", [("x", {}, 10, None, None), ("y", {}, 20, None, None), ("z", {}, 30, None, None)])
        bl = baseline.load_baseline(p)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            first = bl.pick("h:1", "mine")
            bl.pick("h:1", "mine")
        self.assertEqual(first.header["model"], "x")
        lines = err.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("x on h:1", lines[0])
        self.assertIn("y on h:1; z on h:1", lines[0])

    def test_empty_file_picks_nothing(self):
        self.assertIsNone(baseline.Baseline().pick("h:1", "m"))

    def test_a_repeated_block_counts_once_the_last_complete_one(self):
        p = self.make("b.jsonl", [("a", {}, 10, None, None), ("a", {}, 20, None, None)])
        self.assertEqual(baseline.load_baseline(p).pick("h:1", "a")[IDS[0]]["native_decode_tps"], 20)
        with open(p) as f:
            lines = f.read().splitlines()
        cut = [i for i, l in enumerate(lines) if '"type": "end"' in l][-1]
        with open(p, "w") as f:  # the second block was interrupted: the first, complete one stands
            f.write("\n".join(lines[:cut]) + "\n")
        self.assertEqual(baseline.load_baseline(p).pick("h:1", "a")[IDS[0]]["native_decode_tps"], 10)

    def test_run_for_gives_results_header_path_and_label(self):
        p = self.make("b.jsonl", [("x", {}, 10, None, "MTP")])
        run = baseline.load_baseline(p).run_for(target("m"), {"model": "m", "backend": "h:1", "server": "Splash 1.6.0-m1"})
        self.assertEqual((run.label, run.path, run.header["model"]), ("x · MTP", p, "x"))
        self.assertIn(IDS[0], run.results)


class Labels(Base):
    def info(self, model="m", kind="splash", version="1.6.0-m1", note=None):
        return {"model": model, "backend": "h:1", "server": "%s %s" % ({"splash": "Splash", "mtplx": "MTPLX"}[kind], version),
                "note": note}

    def header(self, model="m", kind="splash", version="1.6.0-m1", note=None):
        return {"model": model, "backend": "h:1", "server": {"kind": kind, "version": version}, "note": note,
                "started_at": "2026-11-07T10:00:00+00:00"}

    def test_other_server_and_model(self):
        self.assertEqual(baseline.baseline_label(self.info("mine", "mtplx", "2.12.3"), self.header("other")),
                         "Splash 1.6.0-m1 · other")

    def test_same_server_is_just_the_model(self):
        self.assertEqual(baseline.baseline_label(self.info("mine"), self.header("other")), "other")

    def test_same_model_other_server(self):
        self.assertEqual(baseline.baseline_label(self.info("m", "mtplx", "2.12.3"), self.header("m")), "Splash 1.6.0-m1")

    def test_same_model_and_server_is_the_note_else_the_start_time(self):
        self.assertEqual(baseline.baseline_label(self.info("m"), self.header("m", note="MTP")), "MTP")
        self.assertEqual(baseline.baseline_label(self.info("m", note="MTP"), self.header("m", note="MTP")),
                         "started 2026-11-07 10:00")

    def test_the_host_only_when_it_is_the_only_difference(self):
        h = self.header("m")
        h["backend"] = "h:2"
        self.assertEqual(baseline.baseline_label(self.info("m"), h), "h:2")
        self.assertEqual(baseline.baseline_label(self.info("mine"), h), "m")

    def test_note_joins_the_other_parts(self):
        self.assertEqual(baseline.baseline_label(self.info("mine"), self.header("other", note="DFlash")), "other · DFlash")


class CrossModel(Base):
    def setUp(self):
        super().setUp()
        self.base = self.make("base.jsonl", [("vendor/slow", {"kind": "mtplx", "version": "2.12.3"}, 20.0,
                                              {IDS[2]: (("read",), "PASS"), IDS[3]: ((), "FAIL")}, None)])
        self.this = self.make("this.jsonl", [("vendor/fast", {}, 40.0, None, None)])

    def tui(self, cols="200"):
        with mock.patch.dict(os.environ, {"COLUMNS": cols}):
            return run("--state", self.this, "--compare-state", self.base, "--no-mdcat")[0]

    def test_a_different_model_is_compared(self):
        out = self.tui()
        self.assertIn("compared with: MTPLX 2.12.3 · vendor/slow (base.jsonl, 6 common requests)", out)
        self.assertIn("Baseline", out)
        self.assertIn("This run", out)
        self.assertRegex(out, r"Δ +│")  # the totals change row
        self.assertNotIn("## Comparison", out)
        self.assertNotIn("━ Comparison", out)  # the Metrics table is the run-level home of the baseline

    def test_case_rows_carry_the_deltas_and_the_base_column(self):
        out = self.tui()
        head = next(l for l in out.splitlines() if "ΔTime" in l)
        for col in ("ΔTime", "ΔDecode", "ΔPrefill", "Verdict", "Base"):
            self.assertIn(col, head)
        rows = {l.split("│")[1].strip(): l for l in out.splitlines() if l.startswith("│ R00")}
        self.assertEqual(len(rows), 6)
        r0 = rows[IDS[0]]
        self.assertIn("+100.0%", r0)  # decode 20 -> 40
        self.assertRegex(r0, r"│ =\s+│")  # same verdict and tools
        self.assertRegex(rows[IDS[2]], r"✅ read")  # the baseline read where this run called bash
        self.assertRegex(rows[IDS[3]], r"✗")  # the baseline failed

    def test_no_baseline_no_columns(self):
        with mock.patch.dict(os.environ, {"COLUMNS": "200"}):
            out = run("--state", self.this, "--no-mdcat")[0]
        for col in ("ΔTime", "ΔDecode", "│ Base ", "compared with"):
            self.assertNotIn(col, out)

    def test_metrics_table_has_the_baseline_column_and_the_change(self):
        out = self.tui()
        block = out.split("Metrics")[1].split("Performance")[0]
        for row in ("P/W/F", "Same tools as recorded %", "Wall time", "TTFT p50 / p90", "Native decode"):
            self.assertIn(row, block)
        self.assertIn("This run", block)
        self.assertIn("+100.0%", block)

    def test_totals_delta_row_is_never_truncated(self):
        for cols in ("200", "160", "120"):
            out = self.tui(cols)
            row = next(l for l in out.splitlines() if l.startswith("│ Δ "))
            self.assertNotIn("…", row, cols)
        self.assertRegex(self.tui("200").split("│ Δ ")[1].splitlines()[0], r"[+-]\d+\.\d pt")  # the cache change in points

    def test_metrics_and_perf_tables_name_the_baseline_by_its_label(self):
        out = self.tui()
        self.assertRegex(out, r"│ Metric +│ +MTPLX 2\.12\.3 · vendor/slow +│ +This run")
        self.assertIn("Baseline: MTPLX 2.12.3 · vendor/slow.", out)
        self.assertNotIn("baseline ·", out)

    def test_ids_stay_whole_and_the_width_holds_when_narrow(self):
        for cols in (200, 160, 120, 100):
            out = self.tui(str(cols))
            case_table = []
            for l in out.splitlines():
                if l.startswith("┌") and not case_table:
                    case_table = [l]
                elif case_table and (l[:1] in ("│", "├", "└")):
                    case_table.append(l)
                    if l.startswith("└"):
                        break
            self.assertTrue(all(report.vlen(l) <= cols for l in case_table), cols)
            for cid in IDS:
                self.assertIn(cid, "\n".join(case_table), cols)
            head = case_table[1]
            if cols <= 120:
                self.assertNotIn("ΔPrefill", head, cols)  # the baseline's prefill delta goes first
            if cols >= 200:
                self.assertIn("ΔPrefill", head, cols)
            # the comparison columns outlive In / Out / C% (and Prefill) at every width
            for col in ("ΔDecode", "ΔTime", "Verdict", "Base"):
                self.assertIn(col, head, cols)
            if cols <= 120:
                for col in ("In", "Out", "C%"):
                    self.assertNotIn(" %s " % col, head, cols)

    def test_markdown(self):
        out = run("--state", self.this, "--compare-state", self.base, "--report-format", "markdown", "--no-mdcat")[0]
        self.assertIn("*compared with: MTPLX 2.12.3 · vendor/slow (base.jsonl, 6 common requests)*", out)
        head = next(l for l in out.splitlines() if l.startswith("| Case"))
        for col in ("ΔTime", "ΔDecode", "ΔPrefill", "Base"):
            self.assertIn(col, head)
        self.assertRegex(out, r"\| ✅ read +\|")
        self.assertNotIn("## Comparison", out)
        self.assertIn("| **Baseline** ", out)

    def test_legend_mentions_the_case_columns(self):
        out = self.tui()
        self.assertIn("ΔTime, ΔDecode and ΔPrefill", out)
        self.assertIn("Base is the baseline's verdict", out)

    def test_no_overlap_says_why(self):
        other = self.make("other.jsonl", [("vendor/slow", {}, 20.0, None, None)])
        # same fixture and suite, but no request in common: rewrite the ids
        with open(other) as f:
            text = f.read().replace("R00", "Q00")
        with open(other, "w") as f:
            f.write(text)
        with mock.patch.dict(os.environ, {"COLUMNS": "200"}):
            out = run("--state", self.this, "--compare-state", other, "--no-mdcat")[0]
        self.assertIn("no common requests (baseline 6, this run 6), nothing to compare", out)
        self.assertFalse([l for l in out.splitlines() if l.startswith("│ Case") and "ΔTime" in l])

    def test_a_run_note_is_not_compared_with_itself(self):
        noted = self.make("noted.jsonl", [("vendor/fast", {}, 40.0, None, "MTP")])
        out = run("--state", noted, "--no-mdcat")[0]
        for word in ("Baseline", "compared with", "ΔTime"):
            self.assertNotIn(word, out)

    def test_two_states_and_the_old_baseline_option_are_errors(self):
        for argv in (("--state", self.base, "--state", self.this), ("--baseline", "x")):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                run(*argv)
            self.assertEqual(cm.exception.code, 2)

    def test_test_with_a_replay_state_is_rejected(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            replay_bench.main(["--state", self.this, "--test", "TC01_GIT_COMMIT"])
        self.assertIn("--test selects smoke or sanity cases", err.getvalue())


class SeveralBlocks(Base):
    def test_note_on_stderr_and_first_block_is_used(self):
        base = self.make("base.jsonl", [("x", {}, 20.0, None, None), ("y", {}, 30.0, None, None)])
        this = self.make("this.jsonl", [("z", {}, 40.0, None, None)])
        out, err = run("--state", this, "--compare-state", base, "--no-mdcat")
        self.assertIn("comparing with the first, x on h:1 (others: y on h:1)", err)
        self.assertIn("compared with: x", out)


class AgainstReference(Base):
    """--state A --compare-state B --reference C: both runs are judged against C."""

    def files(self):
        ref = self.make("c.jsonl", [("ref", {}, 30.0, None, None)])
        a = self.make("a.jsonl", [("a", {}, 30.0, {IDS[0]: (("read",), "PASS")}, None)])
        b = self.make("b.jsonl", [("b", {}, 30.0, {IDS[0]: (("read",), "PASS"), IDS[1]: (("read",), "PASS")}, None)])
        return a, b, ref

    def test_both_runs_are_judged_against_the_reference(self):
        a, b, ref = self.files()
        out, _ = run("--state", a, "--compare-state", b, "--reference", ref, "--match", "calls", "--no-mdcat")
        self.assertIn("5 PASS · 1 WARN · 0 FAIL", out)
        self.assertIn("vs reference ref: same first step 100% · same tools 83% · same targets 83% · same calls 83%", out)
        row = lambda label: next(ln for ln in out.splitlines() if ln.lstrip("│ ").startswith(label))
        cells = [c.strip() for c in row("Same calls %").strip("│").split("│")]
        self.assertEqual(cells, ["Same calls %", "67%", "83%", "+16.7 pt"])  # B (baseline), A, change
        self.assertEqual([c.strip() for c in row("Broken calls").strip("│").split("│")][1:], ["0", "0", "+0"])
        self.assertIn("R000_main_e0_t000  WARN  call 1: read, reference bash", out)

    def test_the_agreement_rows_have_the_targets_and_per_tool_rows(self):
        a, b, ref = self.files()
        out, _ = run("--state", a, "--compare-state", b, "--reference", ref, "--match", "targets", "--no-mdcat")
        row = lambda label: next(ln for ln in out.splitlines() if ln.lstrip("│ ").startswith(label))
        cells = [c.strip() for c in row("Same targets %").strip("│").split("│")]
        self.assertEqual(cells[0], "Same targets %")
        self.assertIn("pt", cells[3])
        for label in ("Same calls by first tool", "Same targets by first tool"):
            cells = [c.strip() for c in row(label).strip("│").split("│")]
            self.assertEqual(len(cells), 4)
            self.assertRegex(cells[1], r"\w+ \d+/\d+")   # baseline
            self.assertRegex(cells[2], r"\w+ \d+/\d+")   # this run
            self.assertEqual(cells[3], "")                 # no change cell
        names = [ln.lstrip("│ ") for ln in out.splitlines()]
        order = [next(i for i, n in enumerate(names) if n.startswith(x)) for x in ("Same tools %", "Same targets %", "Same calls %")]
        self.assertEqual(order, sorted(order))

    def test_verdict_changes_use_the_reference_verdicts(self):
        a, b, ref = self.files()
        out, _ = run("--state", a, "--compare-state", b, "--reference", ref, "--match", "calls", "--no-mdcat")
        changes = out.split("Verdict changes")[1].split("Anomalies")[0]
        self.assertIn("R001_main_e0_t001  WARN → PASS", changes)  # B differs from C there, A does not
        self.assertNotIn("R000_main_e0_t000", changes)           # both differ from C

    def test_the_stored_verdicts_are_not_changed_and_without_a_reference_nothing_is_added(self):
        a, b, ref = self.files()
        out, _ = run("--state", a, "--no-mdcat")
        self.assertNotIn("vs reference", out)
        self.assertNotIn("Same calls %", out)

    def test_reference_of_another_suite_is_refused(self):
        a, b, ref = self.files()
        sanity = self.make("s.jsonl", [("s", {}, 30.0, None, None)], suite="sanity")
        out = io.StringIO()
        with contextlib.redirect_stderr(out), self.assertRaises(SystemExit):
            replay_bench.main(["--state", a, "--reference", sanity])
        self.assertIn("is suite sanity", out.getvalue())

    def test_match_needs_a_reference(self):
        a, _, _ = self.files()
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            replay_bench.main(["--state", a, "--match", "calls"])
        self.assertIn("--match needs --reference", err.getvalue())

    def test_temperature_warning_with_calls_but_not_with_step(self):
        a, _, ref = self.files()
        _, err = run("--state", a, "--reference", ref, "--match", "calls", "--no-mdcat")
        self.assertIn("not sent at temperature 0", err)
        _, err = run("--state", a, "--reference", ref, "--no-mdcat")
        self.assertNotIn("temperature", err)


class BaseCell(unittest.TestCase):
    def test_cells(self):
        m = result("a", 10, ("bash",))
        self.assertEqual(R.base_cell(m, None), "–")
        self.assertEqual(R.base_cell(m, result("a", 10, ("bash",))), "=")
        self.assertEqual(R.base_cell(m, result("a", 10, ("read",))), "✅ read")
        self.assertEqual(R.base_cell(m, result("a", 10, ("bash",), "WARN")), "⚠")
        self.assertEqual(R.base_cell(m, result("a", 10, ("bash",), "FAIL")), "✗")
        self.assertEqual(R.base_cell(m, result("a", 10, ("a_very_long_tool_name",))), "✅ a_very…")

    def test_layout_columns_only_with_a_baseline(self):
        self.assertNotIn("Base", R.CaseLayout(True).header)
        h = R.CaseLayout(True, base=True).header
        self.assertEqual(h[-4:], ["ΔPrefill", "Verdict", "Base", "Reason / Preview"])
        self.assertIn("ΔPrefill", h)
        self.assertNotIn("ΔPrefill", R.CaseLayout(False, base=True).header)


if __name__ == "__main__":
    unittest.main()
