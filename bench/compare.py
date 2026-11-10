"""Comparison summary of runs, printed to stdout (no files).

Rows are (endpoint, model) pairs compared against a baseline (the first), or one run against its --compare-state
baseline. Verdicts are counts (PASS, WARN, FAIL, SKIP), SKIP is left out of the numbers, wall time is the sum of the
requests, client tok/s is output tokens over (request time - TTFT).

Works on the results a run holds in memory or on saved state files, so it needs no server.
"""

import shutil
import subprocess
import sys

from . import backend, perf, report
from . import fmt as F
from . import results as R
from .report import Cell, Note, Table
from .state import _decoded as decoded_name
from .state import model_label, with_note


def entry(backend_label, name, results, label=None, suite=None, fixture_sha256=None, path=None, note=None, server=None,
          preset=None, model_info=None):
    """One comparison row from a target's raw results dict ({case id: metrics}) or list of results; `note` is the run note,
    `server` the server's label (`Splash 1.6.0-m1`). The label is the model and its note; run_labels() makes labels of runs
    of one model differ by what differs."""
    if isinstance(results, list):
        res_dict = {r["id"]: r for r in results if isinstance(r, dict) and "id" in r}
    else:
        res_dict = {k: v for k, v in (results or {}).items() if isinstance(v, dict)}
    m_lbl = model_label(name, model_info)
    return {"label": label or with_note(m_lbl, note), "note": note, "backend": backend_label, "name": name,
            "server": server, "preset": preset, "model_info": model_info,
            "results": res_dict,
            "suite": suite, "fixture_sha256": fixture_sha256, "path": path}


def run_labels(entries):
    """Set each entry's label: the model and its run note; where several runs have the same model name, also what tells
    them apart (server and version, preset, note), and the host (`host:port`) only when nothing else differs."""
    groups = {}
    for e in entries:
        base_name = model_label(e["name"], e.get("model_info"))
        groups.setdefault(base_name, []).append(e)
    for name, group in groups.items():
        if len(group) == 1:
            continue
        fields = [k for k in ("server", "preset", "note", "backend") if len({e.get(k) for e in group}) > 1]
        if "backend" in fields and len(fields) > 1:
            fields.remove("backend")  # the host only when it is the only difference
        seen = {}
        for e in group:
            parts = [e.get(k) for k in fields if e.get(k)]
            e["label"] = " \u00b7 ".join([name] + parts)
            n = seen[e["label"]] = seen.get(e["label"], 0) + 1
            if n > 1:
                e["label"] += " #%d" % n
    return entries


def _entry_of_header(backend_label, name, results, h, **kw):
    """An entry for results run under the state header `h` (suite, fixture, note, preset, server come from it)."""
    h = h or {}
    sha = (h.get("fixture") or {}).get("sha256") if isinstance(h.get("fixture"), dict) else None
    return entry(backend_label, name, results, suite=h.get("suite"), fixture_sha256=sha, note=h.get("note"),
                 preset=h.get("preset"), server=backend.label(h["server"]) if h.get("server") else None,
                 model_info=h.get("model_info"), **kw)


def entries_from_run(ordered_targets, raw_results):
    """The entries of the models of one run, labelled so that runs of one model differ by what differs (run_labels)."""
    return run_labels([_entry_of_header(t["backend"], t["name"], raw_results.get(t["key"], {}), t.get("header"))
                       for t in ordered_targets])


def baseline_entry(base_run, name=None):
    """The entry of a baseline.BaselineRun, labelled as the run says (`name`: the entry's model name, default the
    baseline's own); built with the same helper as the run's own."""
    h = base_run.header
    return _entry_of_header(h.get("backend", ""), name or h.get("model") or "", base_run.results, h, label=base_run.label,
                            path=base_run.path)


def pct(values, q):
    return F.pct(values, q)


def stats(results):
    """Client-side numbers from a list of per-request results."""
    ran = [r for r in results if r.get("eval_result") != "SKIP"]
    n = len(ran)
    count = lambda status: sum(1 for r in ran if r.get("eval_result") == status)
    s = {"requests": n, "pass": count("PASS"), "warn": count("WARN"), "fail": n - count("PASS") - count("WARN"),
         "skip": len(results) - n}
    ms = lambda r, k: r.get(k) or 0
    cl = perf.clock_ok(ran)  # client-clock figures leave out requests during which the server paused
    s["paused_note"] = perf.paused_note(ran)
    s["wall_s"] = sum(ms(r, "total_ms") for r in cl) / 1000
    s["ttft_p50"], s["ttft_p90"] = (pct([ms(r, "ttft_ms") / 1000 for r in cl], q) for q in (0.5, 0.9))
    s["e2e_p50"], s["e2e_p90"] = (pct([ms(r, "total_ms") / 1000 for r in cl], q) for q in (0.5, 0.9))
    s["in"] = sum(ms(r, "prompt_tokens") for r in ran)
    reported = [r for r in ran if r.get("cached_tokens") is not None]
    cached_vals = [r["cached_tokens"] for r in reported]
    s["cached"] = sum(cached_vals) if cached_vals else None
    reported_in = sum(ms(r, "prompt_tokens") for r in reported)  # the share counts only requests with a cache count
    s["out"] = sum(ms(r, "tokens") for r in ran)
    s["cache"] = (100 * s["cached"] / reported_in) if (reported_in and s["cached"] is not None) else None
    s["out_per_req"] = s["out"] / n if n else None
    gen_s = sum(max(ms(r, "total_ms") - ms(r, "ttft_ms"), 0) for r in cl) / 1000
    out_cl = sum(ms(r, "tokens") for r in cl)
    s["client_tps"] = out_cl / gen_s if gen_s else None  # output tokens over generation time, client clock
    s["e2e_tps"] = out_cl / s["wall_s"] if s["wall_s"] else None  # over the whole request time, prefill included
    s["drops"] = sum(1 for r in ran if r.get("eval_result") != "FAIL" and r.get("native_decode_tps") is None)
    s["native_dec_p25"], s["native_dec_p75"] = (pct([r.get("native_decode_tps") for r in ran], q) for q in (0.25, 0.75))
    s["native_dec_p50"] = pct([r.get("native_decode_tps") for r in ran], 0.5)
    s["native_pre_p50"] = pct([r.get("native_prefill_tps") for r in ran], 0.5)
    drafts = [d for d in map(R.draft_counts, ran) if d]
    s["draft_acc"] = 100 * sum(d[1] for d in drafts) / sum(d[0] for d in drafts) if drafts else None
    return s


def issues(results, status):
    """{test id: (count, [distinct reasons])} for one status (FAIL or WARN)."""
    out = {}
    for r in results or []:
        if r.get("eval_result") == status:
            n, reasons = out.get(r.get("id"), (0, []))
            reason = (r.get("fail_reason") or "").strip().replace("\n", " ")
            if reason and reason not in reasons:
                reasons.append(reason)
            out[r.get("id")] = (n + 1, reasons)
    return out


# -- formatting ---------------------------------------------------------------

fmt, secs, delta, pt_delta = F.flt, F.secs, F.signed_pct, F.signed_pt


def table(header, rows, dirs=None, aligns=None, dim_cols=()):
    """A report table (markdown or box-drawing, by the report format); see report.Table."""
    if rows and any(h in ("Δ", "\u0394") for h in header):
        keep = []
        for i, h in enumerate(header):
            if h in ("Δ", "\u0394") and all(str(r[i]).strip() in (F.DASH, "–", "-", "base", "") for r in rows):
                continue
            keep.append(i)
        if len(keep) < len(header):
            header = [header[i] for i in keep]
            aligns = [aligns[i] for i in keep] if aligns else None
            dirs = {new_i: dirs[old_i] for new_i, old_i in enumerate(keep) if dirs and old_i in dirs} if dirs else None
            rows = [[r[i] for i in keep] for r in rows]
    return Table(header, rows, dirs=dirs, aligns=aligns, dim_cols=dim_cols)


def labels(entries):
    """Short labels of the entries: the model names with their common prefix stripped (and the run note after), or the
    entries' own labels when two runs serve the same model name (run_labels)."""
    names = [decoded_name(e["name"], e.get("model_info")) for e in entries]
    if len(set(names)) != len(names):
        return [e["label"] for e in entries]
    return [with_note(n, e.get("note")) for n, e in zip(F.short_names(names), entries)]


# Context bands by prompt tokens (k = 1024, as in --max-context): (label, lower bound inclusive, upper exclusive).
BANDS = R.BANDS


def band_groups(results):
    """[(label, [results])] for the bands that have requests (SKIP left out), by each request's prompt_tokens."""
    ran = [r for r in results if r.get("eval_result") != "SKIP"]
    band_indices = sorted({((r.get("prompt_tokens") or 0) // R.BAND_STEP) for r in ran})
    out = []
    for b in band_indices:
        label = R.band_label(b * R.BAND_STEP)
        part = [r for r in ran if ((r.get("prompt_tokens") or 0) // R.BAND_STEP) == b]
        if part:
            out.append((label, part))
    return out


def _pwf(s):
    """PASS/WARN/FAIL counts of one group in a cell: 1/5/0."""
    return "%d/%d/%d" % (s["pass"], s["warn"], s["fail"])


def client_cell(s, key, f):
    if isinstance(key, tuple):
        return " / ".join(f(s.get(k)) for k in key)
    return f(s.get(key))


def client_delta(s, b, key, kind):
    """The change of a metric against the baseline's: signed text in a Cell styled by whether it is a gain.
    kind: pct (neutral), pct+ (higher is better), pct- (lower is better), pt (points, neutral)."""
    if kind is None or not b:
        return ""
    k = key[len(key) // 2] if isinstance(key, tuple) and len(key) == 3 else key[0] if isinstance(key, tuple) else key
    val = s.get(k)
    base_val = b.get(k)
    if kind == "pt":
        return F.delta_cell(val, base_val, higher_is_better=None, unit="pt")
    better = "up" if kind == "pct+" else "down" if kind == "pct-" else None
    return F.delta_cell(val, base_val, higher_is_better=better, unit="%")


# Client metrics: (label, key, format, compare): compare shows the change against a baseline (see client_delta).
CLIENT_ROWS = [
    ("Requests", "requests", F.num, None),
    ("Wall time (sum of requests)", "wall_s", secs, "pct-"),
    ("TTFT p50 / p90", ("ttft_p50", "ttft_p90"), secs, "pct-"),
    ("Request time p50 / p90", ("e2e_p50", "e2e_p90"), secs, "pct-"),
    ("Prompt tokens", "in", F.num, "pct"),
    ("Cache hit (cached \u00f7 prompt)", "cache", lambda v: F.pct(v, ".1f"), "pt"),
    ("Output tokens", "out", F.num, "pct"),
    ("Output per request", "out_per_req", F.num, "pct"),
    ("Decode tok/s (client)", "client_tps", lambda v: fmt(v, ".1f"), "pct+"),
    ("End-to-end tok/s", "e2e_tps", lambda v: fmt(v, ".1f"), "pct+"),
    ("Native decode tok/s p25 / p50 / p75", ("native_dec_p25", "native_dec_p50", "native_dec_p75"),
     lambda v: fmt(v, ".1f"), "pct+"),
    ("Native prefill tok/s p50", "native_pre_p50", lambda v: fmt(v, ".0f"), "pct+"),
    ("Draft acceptance", "draft_acc", lambda v: F.pct(v, ".1f"), "pt"),
]


NATIVE_ROWS = {"Native decode tok/s p25 / p50 / p75": "native_dec_p50", "Native prefill tok/s p50": "native_pre_p50",
               "Draft acceptance": "draft_acc"}


def _client_rows(*runs):
    """CLIENT_ROWS without the native speed rows no run in `runs` has a value for (a server that reports none)."""
    return [row for row in CLIENT_ROWS
            if row[0] not in NATIVE_ROWS or any(r and r.get(NATIVE_ROWS[row[0]]) is not None for r in runs)]


def _label_cell(label):
    """A metric name; its parenthetical (the unit or definition) is italic in markdown."""
    head, sep, rest = label.partition(" (")
    return Cell(label, md=head + " *(" + rest + "*" if sep else None)


def _extra_rows(s):
    """Skipped requests, and requests whose server telemetry was dropped (not when the server gives none at all)."""
    drops = s["drops"] if s["drops"] < s["requests"] else 0
    return [(label, F.num(n)) for label, n in (("Skipped", s["skip"]), ("Telemetry drops", drops)) if n]


def client_table(s, b, name, base_name, shares=None):
    """Per-run vertical metrics view, against the baseline's numbers when there is one. `shares`: rows for the verdicts and
    the agreement shares of both runs, [(label, baseline cell, this cell, change)], shown with a baseline."""
    extra = _extra_rows(s)
    if b:
        rows = [[_label_cell(label), client_cell(b, key, f), client_cell(s, key, f), client_delta(s, b, key, kind)]
                for label, key, f, kind in _client_rows(s, b)]
        rows[1:1] = [[_label_cell(label), bc, tc, d] for label, bc, tc, d in shares or []]
        rows += [[l, "", v, ""] for l, v in extra]
        head = R.clip(base_name, 32) if base_name else "Baseline"
        return [Table(["Metric", head, name, "\u0394"], rows, aligns=["l", "r", "r", "r"]),
                Note("Baseline: %s." % base_name)]
    rows = [[_label_cell(label), client_cell(s, key, f)] for label, key, f, _ in _client_rows(s)] + [[l, v] for l, v in extra]
    return [Table(["Metric", "Value"], rows, aligns=["l", "r"])]


def share_rows(this_results, base_results):
    """The Metrics rows only the verdicts can give: P/W/F, the share of turns with the recorded tools, the share of
    accepted first steps; the last two with their change in percentage points."""
    def pwf(rs):
        c = R.verdict_counts(rs)
        return "%d/%d/%d" % (c["PASS"], c["WARN"], c["FAIL"])
    il = lambda rs: (R.in_line_share(rs) or (None, None, None))[2]
    rows = [("P/W/F", pwf(base_results), pwf(this_results), "")]
    for label, f in (("Same tools as recorded %", R.tools_agreement), ("In line %", il)):
        b, t = f(base_results), f(this_results)
        rows.append((label, F.pct(b), F.pct(t), Cell(F.signed_pt(t, b), "dim") if None not in (b, t) else ""))
    return rows


def fail_lines(results, title):
    """`- <title><test> failed N\u00d7: reasons` for each test with a FAIL."""
    lines = []
    for test, (n, reasons) in issues(results, "FAIL").items():
        why = "; ".join(r[:110] + ("\u2026" if len(r) > 110 else "") for r in reasons) or "no reason given"
        lines.append(f"- {title}{test} failed {n}\u00d7: {why}")
    return lines


def run_blocks(e, comparison=None, cases=None, anomalies=None, titles=True, interrupted=None):
    """Report blocks for one finished (endpoint, model): its heading and `### Results` (the verdict counts, then the
    case table when given), the context breakdown, `### Metrics` (with the baseline's column and the change when there is
    a baseline.Comparison), the context bands and the anomalies. titles=False leaves out the two headings (the live
    terminal table printed them already)."""
    all_res = list(e["results"].values())
    res, bres, out_cmp = all_res, None, None
    if comparison is not None:
        if not comparison:
            out_cmp = report.Line("no common requests with the baseline %s (baseline %d, this run %d): nothing to compare"
                                  % (comparison.base["label"], comparison.n_base, comparison.n_this))
        else:
            res, bres = comparison.this_results, comparison.base_results
            if comparison.partial:
                out_cmp = report.Line("compared on %d common requests (baseline %d, this run %d)"
                                      % (len(comparison.ids), comparison.n_base, comparison.n_this))
    out = []
    if titles:
        out += ["## " + decoded_name(e["name"], e.get("model_info")), "### Results"]
    out.append(R.counts_line(R.verdict_counts(all_res), R.tools_agreement(all_res), R.in_line_share(all_res)))
    if interrupted:
        out.append(R.interrupted_line(interrupted, len(all_res)))
    if cases is not None:
        c_list = cases if isinstance(cases, list) else [cases]
        if out_cmp:
            if c_list and isinstance(c_list[0], report.Note) and "compared with:" in str(c_list[0]):
                out += [c_list[0], out_cmp] + c_list[1:]
            else:
                out += [out_cmp] + c_list
            out_cmp = None
        else:
            out += c_list
    elif out_cmp:
        out.append(out_cmp)
        out_cmp = None
    bd = R.breakdown_table(all_res, base_results=bres)
    if bd is not None:
        out += ["### By context", bd]
    s = stats(res)
    b = stats(bres) if bres else None
    if b and not b["requests"]:
        b = None
    base_label = comparison.base["label"] if b else None
    out += ["### Metrics"] + client_table(s, b, "This run" if b else e["label"], base_label,
                                          share_rows(res, bres) if b else None)
    if s["paused_note"]:
        out.append(Note(s["paused_note"]))
    changes = R.verdict_changes(res, bres) if bres else []
    if changes:
        out += ["### Verdict changes", R.changes_block(changes)]
    items = anomalies if anomalies is not None else R.anomalies(res)
    if items and report.is_chat():  # the transcript already showed each one under its turn: list only the FAILs
        warns = sum(1 for _, v, _ in items if v == "WARN")
        items = [it for it in items if it[1] != "WARN"]
        if warns:
            out += ["### Anomalies", Note("%d WARN shown under their turns above%s." % (
                warns, "; the FAILs follow" if items else ""))]
            if items:
                out.append(R.anomaly_block(items))
            return out
    if items:
        out += ["### Anomalies", R.anomaly_block(items)]
    return out


def md_renderer():
    """The terminal markdown renderer's argv: `mdcat --full-width` if installed, else `glow` at the terminal width,
    else None."""
    if shutil.which("mdcat"):
        return ["mdcat", "--full-width"]
    if shutil.which("glow"):
        return ["glow", "-w", str(shutil.get_terminal_size((100, 24)).columns), "-"]
    return None


def mdcat_usable(mdcat):
    return bool(mdcat and sys.stdout.isatty() and md_renderer())


def render(text, mdcat):
    """Markdown to stdout: through mdcat (or glow) when asked, installed and stdout is a terminal, else as is."""
    argv = md_renderer() if mdcat and sys.stdout.isatty() else None
    if argv:
        sys.stdout.flush()
        # the renderer writes to our stdout itself, so it sees the terminal and its width.
        if subprocess.run(argv, input=text, text=True).returncode == 0:
            return
    print(text)


def emit(blocks, mdcat=False):
    """Print report blocks in the selected report format: box-drawing tables, or markdown (through mdcat on a
    terminal when asked). The chat formats print markdown through mdcat when it can be used, else box-drawing tables."""
    if report.is_markdown() or (report.is_chat() and mdcat_usable(mdcat)):
        render("\n" + report.blocks_markdown(blocks) + "\n", mdcat)
    else:
        print("\n" + report.blocks_tui(blocks) + "\n")


# -- whole-run summary --------------------------------------------------------

def _runs_table(entries, bi, columns, cells, dirs=None):
    """Table with one row per model: `columns` after Model; cells(s, b, is_base) for each."""
    st = [stats(list(e["results"].values())) for e in entries]
    names = labels(entries)
    rows = []
    for i, (n, s) in enumerate(zip(names, st)):
        rows.append([Cell(n + (" *" if i == bi else ""), "id")] + cells(s, st[bi], i == bi))
    return table(["Model"] + columns, rows, dirs=dirs, aligns=["l"] + ["r"] * len(columns))


def client_quality(entries, bi):
    def cells(c, bc, is_base):
        if not c["requests"]:
            return ["0"] + [F.DASH] * 3
        return [F.num(c["requests"]), F.num(c["pass"]), F.num(c["warn"]), Cell(F.num(c["fail"]), "fail" if c["fail"] else "")]
    return _runs_table(entries, bi, ["Requests", "PASS", "WARN", "FAIL"], cells)


def client_perf(entries, bi):
    def cells(c, bc, is_base):
        d = lambda k: "base" if is_base else delta(c[k], bc[k])
        return [secs(c["wall_s"]), d("wall_s"), f"{secs(c['ttft_p50'])} / {secs(c['ttft_p90'])}",
                f"{secs(c['e2e_p50'])} / {secs(c['e2e_p90'])}", F.num(c["out"]), d("out"),
                fmt(c["client_tps"], ".1f"), d("client_tps"), fmt(c["e2e_tps"], ".1f"), d("e2e_tps")]
    return _runs_table(entries, bi, ["Wall", "\u0394", "TTFT p50/p90", "Request p50/p90", "Out tok", "\u0394",
                                     "Client tok/s", "\u0394", "E2E tok/s", "\u0394"], cells,
                       dirs={2: "down", 8: "up", 10: "up"})


def bands_summary(entries, bi):
    """Per-band comparison across models (rows: band x model), deltas vs the baseline's same band; None when
    fewer than two bands have requests anywhere."""
    per = [dict(band_groups(list(e["results"].values()))) for e in entries]
    names = labels(entries)
    all_bands = {}
    for p in per:
        for lb in p:
            lo = int(lb.split("-")[0])
            all_bands[lo] = lb
    bands = [all_bands[lo] for lo in sorted(all_bands)]
    if len(bands) < 2:
        return None
    rows = []
    for lb in bands:
        bs = stats(per[bi][lb]) if lb in per[bi] else None
        for i, (n, p) in enumerate(zip(names, per)):
            if lb not in p:
                continue
            s = stats(p[lb])
            is_base = i == bi
            d = lambda k, better: "base" if is_base else (F.delta_cell(s[k], bs[k], better) if bs else Cell(F.DASH, "dim"))
            rows.append([lb, Cell(n + (" *" if is_base else ""), "id"), F.num(s["requests"]), _pwf(s),
                         secs(s["ttft_p50"]), d("ttft_p50", "down"), fmt(s["client_tps"], ".1f"), d("client_tps", "up"),
                         fmt(s["native_dec_p50"], ".1f"), d("native_dec_p50", "up"),
                         fmt(s["native_pre_p50"], ".0f"), d("native_pre_p50", "up")])
    return table(["Context", "Model", "Requests", "P/W/F", "TTFT p50", "\u0394", "Client tok/s", "\u0394",
                  "Native decode p50", "\u0394", "Native prefill p50", "\u0394"], rows,
                 dirs={5: "down", 7: "up", 9: "up", 11: "up"}, aligns=["l", "l"] + ["r"] * 10)


def summary_blocks(entries):
    """The comparison of several runs of one live run (or file) against the first, or None for fewer than two."""
    if len(entries) < 2:
        return None
    bi = 0
    base_e = entries[bi]
    def _count(e):
        res = e["results"]
        return len(res) if isinstance(res, dict) else len({r["id"] for r in res if isinstance(r, dict) and "id" in r})
    common = set.intersection(*(set(e["results"]) for e in entries))
    names = labels(entries)
    if len(entries) == 2:
        cmp_line = f"compared on {len(common)} common requests (baseline {_count(base_e)}, this run {_count(entries[1])})"
    else:
        parts = [f"baseline {_count(base_e)}"] + [f"{name} {_count(e)}" for name, e in zip(names[1:], entries[1:])]
        cmp_line = f"compared on {len(common)} common requests ({', '.join(parts)})"
    if not common:
        return [report.Line(cmp_line)]

    out = ["## Comparison"]
    if any(_count(e) != len(common) for e in entries):
        out.append(report.Line(cmp_line))
    common_entries = [dict(e, results={k: v for k, v in e["results"].items() if k in common}) for e in entries]
    names = labels(common_entries)
    out += [Note("Baseline: %s (marked *). SKIP is left out." % common_entries[bi]["label"]),
            "### Quality", client_quality(common_entries, bi),
            "### Performance (client side)", client_perf(common_entries, bi)]
    bt = bands_summary(common_entries, bi)
    if bt:
        out += ["### Per context band", bt]
    fails = []
    for e, n in zip(common_entries, names):
        fails += fail_lines(list(e["results"].values()), n + ": ")
    if fails:
        out += ["### Failures", "\n".join(fails)]
    return out


def print_summary(entries, mdcat=False):
    """Print the comparison to stdout; returns False when there was nothing to compare."""
    blocks = summary_blocks(entries)
    if blocks is None:
        return False
    emit(blocks, mdcat)
    return True


def print_perf(entries, mdcat=False):
    """Print the two perf tables (by context, prefill by append size) for all runs, the first being the baseline; deltas
    only with several runs. Returns False when there was nothing to show."""
    if len(entries) > 1:
        common = set.intersection(*(set(e["results"]) for e in entries))
        if not common:
            return False
        entries = [dict(e, results={k: v for k, v in e["results"].items() if k in common}) for e in entries]
    blocks = perf.perf_blocks(entries, 0)
    if not blocks:
        return False
    emit(blocks, mdcat)
    return True
