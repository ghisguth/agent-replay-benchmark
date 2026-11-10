"""Offline tests for bench/compare.py on synthetic results. No server.

    python3 -m unittest discover -s tests
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

import replay_bench  # noqa: E402
from bench import compare, report  # noqa: E402
from bench import fmt as F  # noqa: E402
from bench import results as R  # noqa: E402
from bench.baseline import Comparison  # noqa: E402


def summary(entries):
    """The multi-run comparison as markdown, None for fewer than two runs."""
    blocks = compare.summary_blocks(entries)
    return None if blocks is None else report.blocks_markdown(blocks)


def run_markdown(e, base_e=None, **kw):
    """One run's report blocks as markdown, against base_e when given."""
    return report.blocks_markdown(compare.run_blocks(e, Comparison(e, base_e) if base_e else None, **kw))


def res(tid, ev, total=1000, ttft=200, tokens=100, prompt=1000, reason=""):
    return {"id": tid, "eval_result": ev, "total_ms": total, "ttft_ms": ttft, "tokens": tokens,
            "prompt_tokens": prompt, "cached_tokens": 0, "fail_reason": reason}


def raw(rows):
    """rows: [(test id, [verdict], kwargs)] -> raw results dict keyed by case id."""
    return {tid: res(tid, verdicts[0], **kw) for tid, verdicts, kw in rows}


def ent(name, rows, backend="h:1"):
    return compare.entry(backend, name, raw(rows))


BASE = [("TC01_A", ["PASS"], {}), ("TC02_B", ["PASS"], {}), ("TT01_C", ["WARN"], {}), ("TT02_D", ["FAIL"], {})]
FAST = [("TC01_A", ["PASS"], {"total": 500, "ttft": 100}), ("TC02_B", ["FAIL"], {"total": 500, "ttft": 100,
        "reason": "wrong answer"}), ("TT01_C", ["WARN"], {"total": 500, "ttft": 100}),
        ("TT02_D", ["PASS"], {"total": 500, "ttft": 100}), ("TT03_E", ["SKIP"], {})]


class StatsTests(unittest.TestCase):
    def test_warn_counts_and_skip_excluded(self):
        s = compare.stats(list(ent("m", FAST)["results"].values()))
        self.assertEqual((s["requests"], s["pass"], s["warn"], s["fail"]), (4, 2, 1, 1))

    def test_perf_numbers(self):
        s = compare.stats(list(ent("m", BASE)["results"].values()))
        self.assertAlmostEqual(s["wall_s"], 4.0)
        self.assertEqual(s["out"], 400)
        self.assertAlmostEqual(s["client_tps"], 400 / 3.2)  # 400 tokens over 4 x (1000-200) ms

    def test_deltas(self):
        self.assertEqual(compare.delta(150, 100), "+50.0%")
        self.assertEqual(compare.delta(1, None), "\u2013")
        self.assertEqual(compare.pt_delta(75.0, 50.0), "+25.0 pt")


class SummaryContentTests(unittest.TestCase):
    def test_summary_content(self):
        es = [ent("beta-Q4", FAST), ent("alpha", BASE), ent("gamma", BASE)]
        text = summary(es)
        self.assertIn("Baseline: beta-Q4 (marked \\*)", text)  # the first run is the baseline
        self.assertIn("TC02_B failed 1×: wrong answer", text)
        self.assertNotIn("TT01_C warned", text)  # warnings are listed once, in each model's anomalies
        self.assertNotIn("Per suite", text)
        self.assertNotIn("Pass rate", text)


class SummaryTests(unittest.TestCase):
    def test_single_model_prints_nothing(self):
        self.assertIsNone(summary([ent("only", BASE)]))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertFalse(compare.print_summary([ent("only", BASE)]))
        self.assertEqual(buf.getvalue(), "")

    def test_faster_model_deltas(self):
        text = summary([ent("slow", BASE), ent("fast", FAST)])
        self.assertIn("-50.0%", text)  # wall time 4.0s -> 2.0s
        self.assertIn("base", text)

    def test_quality_is_verdict_counts(self):
        worse = [("TC01_A", ["FAIL"], {}), ("TC02_B", ["FAIL"], {}), ("TT01_C", ["WARN"], {}), ("TT02_D", ["FAIL"], {})]
        text = summary([ent("a", BASE), ent("b", worse)])
        self.assertRegex(text, r"\| \*\*b\*\*\s*\|\s*4 \|\s*0 \|\s*1 \|\s*\*\*3\*\* \|")  # requests, PASS, WARN, FAIL

    def test_no_spread_section(self):
        self.assertNotIn("Run-to-run", summary([ent("a", BASE), ent("b", FAST)]))


def write_state(path, targets):
    """A schema 1 JSONL state: one block (header line, a result line per case) per (backend, model, rows)."""
    with open(path, "w") as f:
        for b, n, rows in targets:
            f.write(json.dumps({"type": "run", "schema": 1, "backend": b, "model": n, "endpoint": None, "n_ctx": None}) + "\n")
            for r in raw(rows).values():
                f.write(json.dumps(dict(r, type="result")) + "\n")


class OfflineStateTests(unittest.TestCase):
    def test_old_json_state_is_refused(self):
        from bench.baseline import load_baseline
        from bench.state import load_state
        from bench.utils import UserError
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.json")
            with open(path, "w") as f:
                json.dump({"meta": {}, "targets": [{"key": "h:1::m", "backend": "h:1", "name": "m",
                           "results": {"TC01_A::iter1": res("TC01_A", "PASS")}}]}, f)
            for load in (load_state, load_baseline):
                with self.assertRaisesRegex(UserError, "is not a state file"):
                    load(path)

    def test_two_states_are_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = (os.path.join(d, x) for x in ("a.jsonl", "b.jsonl"))
            write_state(a, [("h:1", "m-one", BASE)])
            write_state(b, [("h:2", "m-two", FAST)])
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                replay_bench.main(["--state", a, "--state", b, "--no-mdcat"])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("--state takes one file", err.getvalue())
            self.assertIn("--compare-state", err.getvalue())

    def test_state_with_compare_state_compares_case_by_case(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = (os.path.join(d, x) for x in ("a.jsonl", "b.jsonl"))
            write_state(a, [("h:1", "m-one", BASE)])
            write_state(b, [("h:2", "m-two", FAST)])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                replay_bench.main(["--state", b, "--compare-state", a, "--no-mdcat"])
            text = out.getvalue()
            self.assertIn("compared with: m-one", text)
            self.assertIn("-50.0%", text)  # wall time 4.0s -> 2.0s on the 4 common requests
            self.assertIn("compared on 4 common requests (baseline 4, this run 5)", text)

    def test_baseline_option_is_gone(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            replay_bench.main(["--baseline", "x"])

    def test_single_state_not_compared_when_one_model(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.json")
            write_state(a, [("h:1", "m-one", BASE)])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                replay_bench.main(["--state", a, "--no-mdcat"])
            self.assertNotIn("Comparison", out.getvalue())

    def test_single_state_two_models_compared(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.json")
            write_state(a, [("h:1", "m-one", BASE), ("h:1", "m-two", FAST)])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                replay_bench.main(["--state", a, "--no-mdcat"])
            self.assertIn("Comparison", out.getvalue())
            self.assertIn("Baseline: m-one", out.getvalue())


def rich(tid, ev, prompt, dec=None, pre=None, cached=0, total=1000, ttft=200, tokens=100):
    r = res(tid, ev, total=total, ttft=ttft, tokens=tokens, prompt=prompt)
    r.update(cached_tokens=cached, native_decode_tps=dec, native_prefill_tps=pre)
    return r


def rent(name, rs, backend="h:1"):
    return compare.entry(backend, name, {r['id']: r for r in rs})


class ClientStatsTests(unittest.TestCase):
    def setUp(self):
        self.rs = [rich("TC01", "PASS", 1000, dec=10, pre=100, cached=500, total=2000, ttft=500, tokens=300),
                   rich("TC02", "WARN", 3000, dec=30, pre=300, cached=1000, total=1000, ttft=0, tokens=100),
                   rich("TC03", "FAIL", 2000, dec=20, pre=None, cached=0, total=1000, ttft=500, tokens=100),
                   rich("TC04", "SKIP", 9999)]
        self.s = compare.stats(self.rs)

    def test_counts(self):
        s = self.s
        self.assertEqual((s["requests"], s["pass"], s["warn"], s["fail"]), (3, 1, 1, 1))

    def test_times(self):
        s = self.s
        self.assertAlmostEqual(s["wall_s"], 4.0)
        self.assertAlmostEqual(s["ttft_p50"], 0.5)
        self.assertAlmostEqual(s["ttft_p90"], 0.5)
        self.assertAlmostEqual(s["e2e_p50"], 1.0)
        self.assertAlmostEqual(s["e2e_p90"], 1.8)

    def test_tokens_cache_rates(self):
        s = self.s
        self.assertEqual((s["in"], s["cached"], s["out"]), (6000, 1500, 500))
        self.assertAlmostEqual(s["cache"], 25.0)
        self.assertAlmostEqual(s["out_per_req"], 500 / 3)
        self.assertAlmostEqual(s["client_tps"], 500 / 3.0)  # gen time 1.5 + 1.0 + 0.5
        self.assertAlmostEqual(s["e2e_tps"], 500 / 4.0)

    def test_native_p50_skips_missing(self):
        self.assertAlmostEqual(self.s["native_dec_p50"], 20)
        self.assertAlmostEqual(self.s["native_pre_p50"], 200)  # [100, 300]; the None is left out

    def test_empty(self):
        s = compare.stats([])
        self.assertEqual(s["requests"], 0)
        self.assertIsNone(s["cache"])
        self.assertIsNone(s["native_dec_p50"])

    def test_every_client_row_renders(self):
        for label, key, f, _kind in compare.CLIENT_ROWS:
            cell = compare.client_cell(self.s, key, f)
            self.assertNotEqual(cell, "", label)
            self.assertNotIn("None", cell, label)


class RunTableTests(unittest.TestCase):
    def setUp(self):
        self.a = ent("alpha", BASE)
        self.b = ent("beta", FAST)

    def test_without_baseline_is_value_table(self):
        text = run_markdown(self.a)
        self.assertRegex(text, r"\| Metric\s*\|\s*Value \|")
        self.assertNotIn("Baseline", text)
        self.assertNotIn("Native", text)  # this server reports no native speeds: the rows are left out
        self.assertIn("Decode tok/s *(client)*", text)
        self.assertNotIn("Pass rate", text)

    def test_native_rows_shown_with_native_values(self):
        r = {"id": "x", "eval_result": "PASS", "ttft_ms": 100, "total_ms": 1000, "tokens": 10, "prompt_tokens": 100,
             "native_decode_tps": 20.0, "native_prefill_tps": 200.0}
        labels = [row[0] for row in compare._client_rows(compare.stats([r]))]
        self.assertIn("Native decode tok/s p25 / p50 / p75", labels)
        self.assertIn("Native prefill tok/s p50", labels)
        r = dict(r, native_prefill_tps=None)
        labels = [row[0] for row in compare._client_rows(compare.stats([r]))]
        self.assertIn("Native decode tok/s p25 / p50 / p75", labels)
        self.assertNotIn("Native prefill tok/s p50", labels)
        none = [row[0] for row in compare._client_rows(compare.stats([dict(r, native_decode_tps=None)]))]
        self.assertFalse([x for x in none if x.startswith("Native")])
        # a baseline that has them keeps the rows for both columns
        self.assertIn("Native decode tok/s p25 / p50 / p75",
                      [row[0] for row in compare._client_rows(compare.stats([dict(r, native_decode_tps=None)]),
                                                              compare.stats([r]))])

    def test_with_baseline_has_delta_columns(self):
        text = run_markdown(self.b, self.a)
        self.assertRegex(text, r"\| Metric\s*\|\s*alpha \|\s*This run \|\s*\u0394 \|")  # the baseline's label heads its column
        self.assertIn("Baseline: alpha.", text)
        # a faster wall time is a gain: bold
        self.assertRegex(text, r"Wall time \*\(sum of requests\)\*\s*\|\s*4.00s \|\s*2.00s \|\s*\*\*-50.0%\*\*")
        self.assertIn("2 PASS \u00b7 1 WARN \u00b7 **1 FAIL** \u00b7 1 SKIP", text)

    def test_anomalies_once(self):
        text = run_markdown(self.b)
        self.assertNotIn("Not clean passes", text)
        self.assertEqual(text.count("### Anomalies"), 1)
        self.assertIn("- **TC02_B** **FAIL**: wrong answer", text)
        self.assertIn("- **TT01_C** WARN", text)
        clean = run_markdown(ent("ok", [("TC01_A", ["PASS"], {})]))
        self.assertNotIn("Anomalies", clean)

    def test_counts_line_instead_of_rates(self):
        text = run_markdown(self.a)
        self.assertIn("2 PASS \u00b7 1 WARN \u00b7 1 FAIL \u00b7 0 SKIP".replace("1 FAIL", "**1 FAIL**"), text)
        for word in ("Pass rate", "Warn rate", "Fail rate"):
            self.assertNotIn(word, text)

    def test_no_per_suite_tables(self):
        self.assertNotIn("suite", run_markdown(self.b, self.a))

    def test_metrics_carry_the_verdict_shares_with_a_baseline(self):
        text = run_markdown(self.b, self.a)
        self.assertRegex(text, r"\| P/W/F\s*\|\s*2/1/1 \|\s*2/1/1 \|")
        self.assertNotIn("P/W/F", run_markdown(self.a))

    def test_single_model_run_prints_table(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            compare.emit(compare.run_blocks(self.a))
            self.assertFalse(compare.print_summary([self.a]))
        self.assertIn("Metrics", buf.getvalue())
        self.assertNotIn("Comparison", buf.getvalue())

    def test_state_replay_prints_per_run_tables(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.json")
            write_state(a, [("h:1", "m-one", BASE)])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                replay_bench.main(["--state", a, "--no-mdcat"])
            self.assertIn("Metrics", out.getvalue())
            self.assertIn("2 PASS", out.getvalue())


class MdcatTests(unittest.TestCase):
    def render(self, mdcat, which, returncode=0):
        buf = io.StringIO()
        with mock.patch.object(compare.shutil, "which", return_value=which), \
                mock.patch.object(compare.subprocess, "run",
                                  return_value=mock.Mock(returncode=returncode)) as run, \
                contextlib.redirect_stdout(buf):
            buf.isatty = lambda: True  # mdcat is only used on a terminal
            compare.render("# hi", mdcat)
        return run, buf.getvalue()

    def test_uses_mdcat_when_present_and_enabled(self):
        run, out = self.render(True, "/x/mdcat")
        run.assert_called_once_with(["mdcat", "--full-width"], input="# hi", text=True)
        self.assertEqual(out, "")

    def test_falls_back_when_missing(self):
        run, out = self.render(True, None)
        run.assert_not_called()
        self.assertEqual(out, "# hi\n")

    def test_falls_back_when_disabled(self):
        run, out = self.render(False, "/x/mdcat")
        run.assert_not_called()
        self.assertEqual(out, "# hi\n")

    def test_falls_back_on_nonzero_exit(self):
        run, out = self.render(True, "/x/mdcat", returncode=1)
        run.assert_called_once()
        self.assertEqual(out, "# hi\n")

    def test_no_mdcat_flag_threads_to_runner(self):
        for argv, want in (([], True), (["--no-mdcat"], False)):
            with mock.patch.object(replay_bench, "run_benchmark") as rb, \
                    mock.patch.object(replay_bench, "resolve_endpoints", return_value=[]), \
                    contextlib.redirect_stderr(io.StringIO()):  # the "preset large: ..." line
                replay_bench.main(argv)
            self.assertIs(rb.call_args.kwargs["mdcat"], want)


class BandTests(unittest.TestCase):
    K = 1024

    def rs(self, scale=1.0):
        return [rich("TC01", "PASS", 10 * self.K, dec=50 * scale, pre=1000, ttft=100),
                rich("TC02", "FAIL", 70 * self.K, dec=40 * scale, pre=900, ttft=400),
                rich("TC03", "WARN", 130 * self.K, dec=30 * scale, pre=800, ttft=800),
                rich("TC04", "PASS", 200 * self.K, dec=20 * scale, pre=700, ttft=1600)]

    def test_boundaries_use_1024(self):
        r = lambda n: rich("T", "PASS", n)
        got = {lb: len(p) for lb, p in compare.band_groups(
            [r(65535), r(65536), r(131071), r(131072), r(196607), r(196608)])}
        self.assertEqual(got, {"32-64k": 1, "64-96k": 1, "96-128k": 1, "128-160k": 1, "160-192k": 1, "192-224k": 1})

    def test_empty_bands_skipped_and_single_band_hidden(self):
        self.assertEqual([lb for lb, _ in compare.band_groups(self.rs()[:2])], ["0-32k", "64-96k"])
        self.assertIsNone(R.breakdown_table([rich("T", "PASS", 10), rich("U", "PASS", 20)]))
        self.assertNotIn("By context", run_markdown(rent("m", [rich("T", "PASS", 10)])))

    def test_band_table_values(self):
        text = R.breakdown_table(self.rs()).markdown()
        self.assertRegex(text, r"\| 64-96k\s*\|\s*1 \|\s*0\s*\|\s*0\s*\|\s*\*\*1\*\*")
        self.assertRegex(text, r"\| 128-160k\s*\|\s*1 \|\s*0\s*\|\s*1\s*\|\s*0")
        self.assertIn("Prefill tok/s p50", text)
        self.assertNotIn("\u0394", text)

    def test_band_table_with_baseline(self):
        text = R.breakdown_table([r for r in self.rs(0.5)], self.rs()).markdown()
        self.assertIn("-50.0%", text)  # native decode halved in every band
        self.assertRegex(text, r"\| 0-32k\s*\|\s*1\s*\|")

    def test_summary_band_section(self):
        text = summary([rent("slow", self.rs()), rent("fast", self.rs(2))])
        self.assertIn("### Per context band", text)
        self.assertRegex(text, r"\| 0-32k\s*\| \*\*fast\*\*\s*\|")
        self.assertIn("+100.0%", text)
        single = [rich("T", "PASS", 10), rich("U", "PASS", 20)]
        self.assertNotIn("Per context band", summary([rent("a", single), rent("b", single)]))


class QualityPerfTableTests(unittest.TestCase):
    def test_columns(self):
        text = summary([ent("a", BASE), ent("b", FAST)])
        for col in ("Requests", "PASS", "WARN", "FAIL", "Wall", "TTFT p50/p90",
                    "Request p50/p90", "Out tok", "Client tok/s", "E2E tok/s"):
            self.assertIn(col, text)
        self.assertNotIn("Pass (P+W)", text)
        self.assertEqual(sum(l.count("\u0394") for l in text.splitlines() if l.startswith("|")), 4)  # performance only
        self.assertIn("### Quality", text)

    def test_table_hides_empty_delta_column(self):
        head = ["Model", "tok/s", "\u0394"]
        rows = [["base", "10.0", "base"], ["other", "–", "–"]]
        t = compare.table(head, rows)
        self.assertNotIn("\u0394", t.header)
        self.assertEqual(t.header, ["Model", "tok/s"])

    def test_state_with_retried_case_counts_unique_ids(self):
        from bench import state
        with tempfile.TemporaryDirectory() as tmp:
            state_a = os.path.join(tmp, "a.jsonl")
            state_b = os.path.join(tmp, "b.jsonl")
            recs_a = [
                {"type": "run", "schema": 1, "suite": "smoke", "backend": "h:1", "model": "mA"},
                {"type": "result", "id": "R001", "eval_result": "FAIL", "error": "timeout"},
                {"type": "result", "id": "R001", "eval_result": "PASS", "tokens": 50, "prompt_tokens": 100},
                {"type": "result", "id": "R002", "eval_result": "PASS", "tokens": 60, "prompt_tokens": 120},
                {"type": "end", "interrupted": False},
            ]
            recs_b = [
                {"type": "run", "schema": 1, "suite": "smoke", "backend": "h:1", "model": "mB"},
                {"type": "result", "id": "R001", "eval_result": "PASS", "tokens": 50, "prompt_tokens": 100},
                {"type": "result", "id": "R002", "eval_result": "PASS", "tokens": 60, "prompt_tokens": 120},
                {"type": "end", "interrupted": False},
            ]
            with open(state_a, "w") as f:
                for r in recs_a:
                    f.write(json.dumps(r) + "\n")
            with open(state_b, "w") as f:
                for r in recs_b:
                    f.write(json.dumps(r) + "\n")

            raw_a, targets_a = state.load_state(state_a)
            raw_b, targets_b = state.load_state(state_b)
            ea = compare.entries_from_run(targets_a, raw_a)[0]
            eb = compare.entries_from_run(targets_b, raw_b)[0]
            cmp = Comparison(ea, eb)
            self.assertEqual(cmp.n_this, 2)
            self.assertEqual(cmp.n_base, 2)
            self.assertEqual(len(cmp.ids), 2)
            self.assertFalse(cmp.partial)

            from bench import results as R
            from bench.baseline import BaselineRun
            brun = BaselineRun(raw_b[targets_b[0]["key"]], {"backend": "h:1", "model": "mB"}, state_b, "mB")
            line = R.compared_line(brun, ["R001", "R001", "R002"])
            self.assertIn("2 common requests", line)

    def test_compared_on_line_position_in_run_blocks(self):
        ea = compare.entry("h:1", "mA", {"R001": res("R001", "PASS", prompt=1000), "R002": res("R002", "PASS", prompt=40000)})
        eb = compare.entry("h:1", "mB", {"R001": res("R001", "PASS", prompt=1000)})
        cmp = Comparison(ea, eb)
        self.assertTrue(cmp.partial)
        note = report.Note("compared with: mB (b.jsonl, 1 common request)")
        dummy_table = report.Table(["Case", "Verdict"], [["R001", "PASS"]])
        blocks = compare.run_blocks(ea, comparison=cmp, cases=[note, dummy_table])
        texts = [str(b) for b in blocks]
        # Find index of note and out_cmp
        note_idx = next(i for i, b in enumerate(blocks) if isinstance(b, report.Note) and "compared with:" in str(b))
        cmp_idx = next(i for i, b in enumerate(blocks) if isinstance(b, report.Line) and "compared on 1 common requests" in str(b))
        self.assertEqual(cmp_idx, note_idx + 1)
        by_ctx_idx = next(i for i, t in enumerate(texts) if t == "### By context")
        self.assertLess(cmp_idx, by_ctx_idx)
        metrics_idx = next(i for i, t in enumerate(texts) if t == "### Metrics")
        self.assertLess(cmp_idx, metrics_idx)

    def test_metrics_with_mixed_results_and_unknown_cache(self):
        r1 = {"id": "R001", "eval_result": "PASS", "prompt_tokens": 1000, "cached_tokens": 200,
              "tokens": 100, "ttft_ms": 500, "total_ms": 2500}
        r2 = {"id": "R002", "eval_result": "PASS", "prompt_tokens": 1000, "cached_tokens": None,
              "tokens": 50, "ttft_ms": 1000, "total_ms": 3000}
        s = compare.stats([r1, r2])
        self.assertEqual(s["cached"], 200)
        self.assertEqual(s["in"], 2000)
        self.assertEqual(s["cache"], 20.0)  # over the requests that reported a count only

        s_unknown = compare.stats([r2])
        self.assertIsNone(s_unknown["cached"])
        self.assertIsNone(s_unknown["cache"])
        c_cell = compare.client_cell(s_unknown, "cache", lambda v: F.pct(v, ".1f"))
        self.assertEqual(str(c_cell), "–")

        r3 = {"id": "R003", "eval_result": "PASS", "prompt_tokens": 1000, "cached_tokens": 0,
              "tokens": 50, "ttft_ms": 500, "total_ms": 2000}
        s_zero = compare.stats([r3])
        self.assertEqual(s_zero["cached"], 0)
        self.assertEqual(s_zero["cache"], 0.0)
        c_cell_zero = compare.client_cell(s_zero, "cache", lambda v: F.pct(v, ".1f"))
        self.assertEqual(str(c_cell_zero), "0.0%")


if __name__ == "__main__":
    unittest.main()
