"""The pieces of a model's report that both formats share: the per-case row, the verdict counts line, the anomaly
list, the run header / "Run" section. Pure functions over result dicts (the runner's metrics, or those loaded from a
state) and report blocks; no printing here.
"""

import json
import os

from . import backend, report
from . import fmt as F
from .colors import Clr
from .report import Cell, Line, Table

VERDICTS = ("PASS", "WARN", "FAIL", "SKIP")
MARKS = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL", "SKIP": "⏭️ SKIP"}
REASON_MAX = 60


# -- cells ---------------------------------------------------------------------------------------------------

def verdict_cell(v):
    return Cell(MARKS.get(v, v or "?"), (v or "").lower())


def stream_label(m):
    """main/e0, a subagent's name, main/c0 for a compaction; '' for cases that are not replay requests."""
    stream = m.get("stream")
    if not stream:
        return ""
    if m.get("kind") == "compaction":
        return "%s/c%s" % (stream, m.get("epoch") or 0)
    return "%s/e%s" % (stream, m.get("epoch") or 0) if stream == "main" else str(stream)


def code_names(names):
    """Comma-separated tool names with each in inline code (markdown)."""
    parts = [n.strip() for n in names.split(",")]
    return ", ".join("`%s`" % n if n not in ("", "none") else n for n in parts)


def clip(text, n=REASON_MAX):
    text = " ".join(str(text or "").replace("`", "'").split())
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def preview(m, limit=REASON_MAX):
    """One line of what the model answered: its prose, else the tool calls it made."""
    content = (m.get("content") or "").strip()
    if not content and not m.get("tool_calls"):
        content = (m.get("text") or "").strip()
        if content.startswith("[{"):
            try:
                calls = json.loads(content)
                names = [(c.get("function") or {}).get("name", "?") for c in calls]
                return "call " + ", ".join(names)
            except (ValueError, AttributeError):
                pass
    if content:
        return clip(content.replace("```json", "").replace("```", ""), limit)
    names = [(c.get("function") or {}).get("name", "?") for c in m.get("tool_calls") or []]
    return "call " + ", ".join(names) if names else ""


def reason_cell(m, limit=REASON_MAX):
    """WARN, FAIL and SKIP: the reason; PASS: the reason when it has one, else a one-line preview of the answer."""
    v, reason = m.get("eval_result"), m.get("fail_reason") or ""
    if v in ("WARN", "FAIL", "SKIP"):
        return Cell(clip(reason, limit), "dim" if v == "SKIP" else "")
    text = preview(m, limit)
    if text.startswith("call "):
        return Cell(clip(text, limit), md="call " + code_names(text[5:]))
    return Cell(clip(text, limit), "dim")


def live_line(m, label):
    """The stderr line of a finished case in the markdown format: `R004 main/e0 12.6k ctx · 0.57s · WARN siblings≠fdc`."""
    why = clip(m.get("fail_reason"), 50)
    return "%s %s %s ctx \u00b7 %s \u00b7 %s%s" % (str(m.get("id", "?")).split("_")[0], label, tokens_k(m.get("prompt_tokens")),
                                                  F.ms(m.get("ttft_ms")), m.get("eval_result") or "?",
                                                  " " + why if why else "")


def tokens_k(n):
    """A token count as 1536 -> '1.5k' (k = 1024, plain below 1024)."""
    n = n or 0
    return "%d" % n if n < 1024 else "%.1fk" % (n / 1024)


# -- the case table ----------------------------------------------------------------------------------------

def request_decode_speed(r):
    """(speed, is_estimate) for a result: native_decode_tps if present, else client-clock decode tok/s.
    Returns (None, False) if neither is available.
    """
    if r.get("native_decode_tps") is not None:
        return float(r["native_decode_tps"]), False
    out = r.get("tokens") or 0
    ttft = r.get("ttft_ms")
    total = r.get("total_ms")
    if out > 0 and ttft is not None and total is not None and total > ttft:
        gen_s = (total - ttft) / 1000.0
        return out / gen_s, True
    return None, False


MIN_NEW_TOKENS = 17  # "more than 16 new tokens": below it a prefill speed (and an append size) says nothing


def new_tokens(r):
    """New (uncached) prompt tokens of a result. With the server's cache count: prompt - cached. A server that reports
    none (LM Studio): prompt - the previous prompt of the same stream (`prev_prompt_tokens`, see fill_prev_prompts),
    clamped at 0; the whole prompt for a first request or when the previous prompt is larger."""
    prompt = r.get("prompt_tokens") or 0
    cached = r.get("cached_tokens")
    if cached is not None:
        return max(0, prompt - cached)
    prev = r.get("prev_prompt_tokens")
    if isinstance(prev, int) and 0 <= prev <= prompt:
        return prompt - prev
    return prompt


def estimates_new_tokens(r):
    """True when new_tokens(r) is the previous-prompt estimate (no cache reported, a previous prompt known)."""
    prev = r.get("prev_prompt_tokens")
    return r.get("cached_tokens") is None and isinstance(prev, int) and 0 <= prev <= (r.get("prompt_tokens") or 0)


def fill_prev_prompts(results):
    """Set `prev_prompt_tokens` on each result (in run order): the prompt of the previous request of its stream and
    epoch, absent for a first request and for a compaction (it expects no cache, as runner.note_finished). Computed
    again whenever a state is read, so old states get it too."""
    last = {}
    for r in results:
        r.pop("prev_prompt_tokens", None)
        prompt = r.get("prompt_tokens") or 0
        if r.get("error") or r.get("eval_result") == "SKIP" or prompt <= 0:
            continue
        if not r.get("stream"):
            continue  # smoke and sanity cases are independent requests, not one growing conversation
        key = (r.get("stream"), r.get("epoch"))
        if r.get("kind") == "compaction":
            continue
        if key in last:
            r["prev_prompt_tokens"] = last[key]
        last[key] = prompt
    return results


def request_prefill_speed(r):
    """(speed, is_estimate) for a result: native_prefill_tps if present, else client-clock prefill tok/s.
    Returns (None, False) if neither is available, or when the new tokens are under MIN_NEW_TOKENS.
    """
    if r.get("native_prefill_tps") is not None:
        return float(r["native_prefill_tps"]), False
    prompt = r.get("prompt_tokens")
    ttft = r.get("ttft_ms")
    if prompt is not None and ttft is not None and ttft > 0 and new_tokens(r) >= MIN_NEW_TOKENS:
        return (new_tokens(r) * 1000.0) / ttft, True
    return None, False  # a (nearly) fully cached request has no prefill to time


class CaseLayout:
    """Columns of the per-case table. The Model column only with several models, Stream only for replay requests,
    In / Out / C% / Prefill only when the server reported token counts."""

    def __init__(self, has_stats, multi=False, stream=False, reason_limit=REASON_MAX, base=False):
        self.has_stats, self.multi, self.stream, self.base = has_stats, multi, stream, base
        self.reason_limit = reason_limit
        h = ["Case"] + (["Model"] if multi else []) + (["Stream"] if stream else []) + ["TTFT", "Time"]
        h += ["In", "Out", "C%", "Decode", "Prefill"] if has_stats else ["Toks", "Decode"]
        # with a baseline: the per-request change right of the metrics, the baseline's verdict next to this one's
        self.delta_cols = (["\u0394Time", "\u0394Decode"] + (["\u0394Prefill"] if has_stats else [])) if base else []
        self.header = h + self.delta_cols + ["Verdict"] + (["Base"] if base else []) + ["Reason / Preview"]
        left = {"Case", "Model", "Stream", "Verdict", "Base", "Reason / Preview"}
        self.aligns = ["l" if c in left else "r" for c in self.header]
        self.dim_cols = [i for i, c in enumerate(self.header) if c in ("Model", "Time", "Out", "Toks")]
        self.rcol = len(self.header) - 1

    def cells(self, m, model="", suffix="", base=None):
        """The row of one result; `base`: the baseline's result of the same request (None: none), used with a baseline."""
        dec, dec_est = request_decode_speed(m)
        pre, pre_est = request_prefill_speed(m)
        verdict = m.get("eval_result")
        is_ok = verdict in ("PASS", "WARN")

        out = [Cell(str(m.get("id", "?")) + suffix, "dim" if suffix else "id")]
        if self.multi:
            out.append(model)
        if self.stream:
            out.append(Cell(stream_label(m), "code"))
        out += [F.ms(m.get("ttft_ms")), F.ms(m.get("total_ms"))]
        if self.has_stats:
            p = m.get("prompt_tokens") or 0
            c = m.get("cached_tokens")
            if c is None:
                c_cell = Cell(F.DASH, "dim")
            else:
                pc = c / p * 100 if p else 0
                c_cell = Cell(F.pct(pc), F.cache_style(pc))
            out += [F.num(p), F.num(m.get("tokens")), c_cell]
        else:
            out.append(F.num(m.get("tokens")))

        if dec is not None:
            dec_text = ("~" if dec_est else "") + F.flt(dec)
            out.append(Cell(dec_text, F.speed_style(dec, F.DECODE_TPS)))
        elif is_ok:
            out.append(Cell("⚠", "warn"))
        else:
            out.append(Cell(F.DASH, "dim"))

        if self.has_stats:
            if pre is not None:
                pre_text = ("~" if pre_est else "") + F.flt(pre)
                out.append(Cell(pre_text, F.speed_style(pre, F.PREFILL_TPS)))
            elif is_ok:
                out.append(Cell("⚠", "warn"))
            else:
                out.append(Cell(F.DASH, "dim"))

        if self.base:
            out += self.delta_cells(m, base)
        out.append(verdict_cell(m.get("eval_result")))
        if self.base:
            out.append(base_cell(m, base))
        return out + [reason_cell(m, self.reason_limit)]

    def delta_cells(self, m, b):
        """ΔTime, ΔDecode and (with token counts) ΔPrefill of one request against the baseline's, `-` without it."""
        if not b:
            return [Cell(F.DASH, "dim")] * len(self.delta_cols)
        m_dec, m_dec_est = request_decode_speed(m)
        b_dec, b_dec_est = request_decode_speed(b)
        out = [delta_cell(m.get("total_ms"), b.get("total_ms"), "down"),
               delta_cell(m_dec, b_dec, "up", approx=m_dec_est != b_dec_est)]
        if self.has_stats:
            m_pre, m_pre_est = request_prefill_speed(m)
            b_pre, b_pre_est = request_prefill_speed(b)
            out.append(delta_cell(m_pre, b_pre, "up", approx=m_pre_est != b_pre_est))
        return out

    def totals(self, label, t, note=""):
        """The row of a run's totals: TTFT and time and tokens summed, cache share, native or fallback p50 speeds, P/W/F."""
        out = [Cell(label, "bold")] + [""] * (self.multi + self.stream)
        out += [F.secs(t["ttft"]), F.secs(t["time"])]
        if self.has_stats:
            c_text = F.pct(t["cache"])
            c_style = F.cache_style(t["cache"]) if t["cache"] is not None else "dim"
            out += [F.num(t["in"]), F.num(t["out"]), Cell(c_text, c_style)]
        else:
            out.append(F.num(t["out"]))
        dec_str = (("~" if t.get("decode_est") else "") + F.flt(t["decode"])) if t["decode"] is not None else F.DASH
        out.append(Cell(dec_str, F.speed_style(t["decode"], F.DECODE_TPS)))
        if self.has_stats:
            pre_str = (("~" if t.get("prefill_est") else "") + F.flt(t["prefill"])) if t["prefill"] is not None else F.DASH
            out.append(Cell(pre_str, F.speed_style(t["prefill"], F.PREFILL_TPS)))
        out += [""] * len(self.delta_cols) + [Cell(t["pwf"], "bold")] + [""] * self.base
        return out + [Cell(clip(note, self.reason_limit), "dim") if note else ""]

    def delta_row(self, t, base):
        """The change of a run's totals against the baseline's: signed, never truncated (the widths leave room), the
        cache share in percentage points."""
        d = lambda k, better: delta_cell(t[k], base[k], better)
        pt = delta_cell(t["cache"], base["cache"], None, unit="pt")
        out = [Cell("\u0394", "bold")] + [""] * (self.multi + self.stream)
        out += [d("ttft", "down"), d("time", "down")]
        out += [d("in", None), d("out", None), pt] if self.has_stats else [d("out", None)]
        out.append(d("decode", "up"))
        if self.has_stats:
            out.append(d("prefill", "up"))
        return out + [""] * (len(self.delta_cols) + 1 + self.base + 1)

    # With a baseline the comparison columns outlive the metrics: the reason column shrinks further, then In/Out/C%,
    # Prefill and ΔPrefill go, Verdict, Base, ΔTime and ΔDecode last.
    SHRINK_WITH_BASE = (("Reason / Preview", 12), ("Stream", 10))
    DROP_WITH_BASE = ("Out", "Toks", "In", "C%", "Prefill", "Model", "\u0394Prefill", "Time", "Stream")

    def widths(self, ids, models, streams, term=None):
        """Fixed column widths for a table whose rows arrive one by one: the cases are known up front. `term`: the
        terminal's columns (None: no limit); the table is fitted to it (report.fit_columns), None marks a dropped column."""
        w = {"Case": max([len("Baseline")] + [len(i) + 1 for i in ids]), "Model": max([len("Model")] + [len(x) for x in models]),
             "Stream": max([len("Stream")] + [len(s) for s in streams]), "TTFT": 6, "Time": 6, "In": 10, "Out": 9,
             "Toks": 6, "C%": 4, "Decode": 6, "Prefill": 7, "Verdict": 7, "\u0394Time": 7, "\u0394Decode": 8,
             "\u0394Prefill": 9, "Base": 11}
        if self.base:  # the totals' change row holds signed deltas: +136.2%, -12.3 pt
            w.update({"TTFT": 8, "Time": 8, "Decode": 8, "Prefill": 8, "C%": 9})
        fixed = [w[c] for c in self.header[:-1]]
        # The reason column takes what the terminal has left (no cap: a wide terminal is filled); without a terminal
        # (a pipe) the table has its natural width, REASON_MAX for the reason.
        rest = term - sum(fixed) - 3 * len(self.header) - 1 if term else REASON_MAX
        w["Reason / Preview"] = max(24 if not self.base else 12, rest if term else REASON_MAX)
        shrink, drop = (self.SHRINK_WITH_BASE, self.DROP_WITH_BASE) if self.base else (Table.SHRINK, Table.DROP)
        fitted = report.fit_columns(self.header, [w[c] for c in self.header], term, shrink, drop, Table.LATE)
        if term and fitted[self.rcol]:
            self.reason_limit = fitted[self.rcol]  # the cells are clipped to the column they get
        return fitted


def delta_cell(value, baseline, higher_is_better=None, unit="%", approx=False):
    """A signed change of `value` against `baseline` in a styled Cell.

    higher_is_better: True / 'up' (positive is good), False / 'down' (negative is good), None (neutral/dim).
    unit: '%' (relative change in percent, e.g. +12.3%) or 'pt' (percentage points, e.g. +2.0 pt).
    approx: one side is an estimate and the other a native figure: the change is shown dim with a `~` prefix, never
    as better or worse.
    """
    if unit == "pt":
        if value is None or baseline is None:
            return Cell(F.DASH, "dim")
        diff = value - baseline
        text = "%+.1f pt" % diff
        if abs(diff) < 0.05:
            return Cell("+0.0 pt", "dim")
        if higher_is_better is None:
            return Cell(text, "dim")
        good = (diff > 0) == (higher_is_better is True or higher_is_better == "up")
        return Cell(text, "good" if good else "bad")

    if value is None or baseline is None or not baseline:
        return Cell(F.DASH, "dim")
    pct_val = (value - baseline) / abs(baseline) * 100
    if abs(pct_val) < 0.05:
        return Cell("+0.0%", "dim")
    text = "%+.1f%%" % pct_val
    if approx:
        return Cell("~" + text, "dim")
    if higher_is_better is None:
        return Cell(text, "dim")
    good = (pct_val > 0) == (higher_is_better is True or higher_is_better == "up")
    return Cell(text, "good" if good else "bad")


def run_totals(results):
    """The numbers of the totals row of a set of results (SKIP and errors left out; requests during which the server
    paused are left out of the client-clock times, as in the Metrics table)."""
    from . import perf  # perf imports this module
    ran = [r for r in results if r.get("eval_result") != "SKIP" and not r.get("error")]
    cl = perf.clock_ok(ran)
    ttft = sum((r.get("ttft_ms") or 0) / 1000 for r in cl)  # time spent waiting for first tokens
    prompt = sum(r.get("prompt_tokens") or 0 for r in ran)

    dec_pairs = [request_decode_speed(r) for r in ran]
    dec_vals = [s for s, _ in dec_pairs if s is not None]
    dec_any_est = any(est for s, est in dec_pairs if s is not None)

    pre_pairs = [request_prefill_speed(r) for r in ran]
    pre_vals = [s for s, _ in pre_pairs if s is not None]
    pre_any_est = any(est for s, est in pre_pairs if s is not None)

    reported = [r for r in ran if r.get("cached_tokens") is not None]
    cached = sum(r["cached_tokens"] for r in reported)
    reported_prompt = sum(r.get("prompt_tokens") or 0 for r in reported)  # only requests that gave a cache count
    cache = (100 * cached / reported_prompt) if reported_prompt else (0.0 if reported else None)

    c = verdict_counts(results)
    return {
        "ttft": ttft,
        "time": sum(r.get("total_ms") or 0 for r in cl) / 1000,
        "in": prompt,
        "out": sum(r.get("tokens") or 0 for r in ran),
        "cache": cache,
        "decode": F.p50(dec_vals),
        "decode_est": dec_any_est,
        "prefill": F.p50(pre_vals),
        "prefill_est": pre_any_est,
        "pwf": "%d/%d/%d" % (c["PASS"], c["WARN"], c["FAIL"]),
    }


def totals_rows(layout, results, comparison=None, label=""):
    """Rows closing the case table: this run's totals, and with a baseline.Comparison its totals above and the change
    below, over the requests both have (the labels of the two runs in the last column)."""
    if not comparison:
        return [layout.totals("Total", run_totals(results))]
    b = run_totals(comparison.base_results)
    t = run_totals(comparison.this_results)
    return [layout.totals("Baseline", b, comparison.base["label"]), layout.totals("This run", t, label),
            layout.delta_row(t, b)]


GLYPHS = {"PASS": "\u2705", "WARN": "\u26a0", "FAIL": "\u2717", "SKIP": "\u23ed"}
BASE_STEP_MAX = 7


def first_step(m):
    """The name of a result's first step: its first tool call, else `text`; None when it has neither."""
    names = m.get("tool_names")
    if names:
        return str(names[0])
    return "text" if (m.get("content") or m.get("text") or "").strip() else None


def base_cell(m, b):
    """The Base column: `=` when the baseline got the same verdict with the same tools; else its verdict glyph, and its
    first step when it took other tools than this run (`\u2705 read` while this run called bash). `-` without a baseline
    result."""
    if not b:
        return Cell(F.DASH, "dim")
    bv, v = b.get("eval_result"), m.get("eval_result")
    bt, t = b.get("tool_names"), m.get("tool_names")
    other = bt is not None and t is not None and sorted(bt) != sorted(t)
    if bv == v and not other:
        return Cell("=", "dim")
    text = GLYPHS.get(bv, "?")
    step = first_step(b) if other else None
    if step:
        text += " " + clip(step, BASE_STEP_MAX)
    return Cell(text, (bv or "").lower())


def compared_line(base_run, ids):
    """`compared with: <label> (<file>, N common requests)` under the run header; says so when nothing is common."""
    unique_ids = list(dict.fromkeys(ids))
    n = base_run.common(unique_ids)
    where = os.path.basename(base_run.path or "")
    n_base = len(base_run.results) if isinstance(base_run.results, dict) else len({r["id"] for r in base_run.results if isinstance(r, dict) and "id" in r})
    if not n:
        return "compared with: %s (%s): no common requests (baseline %d, this run %d), nothing to compare" % (
            base_run.label, where, n_base, len(unique_ids))
    return "compared with: %s (%s, %d common request%s)" % (base_run.label, where, n, "" if n == 1 else "s")


def verdict_changes(results, base_results):
    """[(case id, baseline verdict, verdict)] where they differ."""
    base = {r.get("id"): r.get("eval_result") for r in base_results or []}
    return [(r.get("id"), base[r.get("id")], r.get("eval_result")) for r in results
            if r.get("id") in base and base[r.get("id")] != r.get("eval_result")]


def changes_block(items):
    return Line("\n".join("  %s  %s \u2192 %s" % (i, a, b) for i, a, b in items),
                md="\n".join("- **%s** %s \u2192 %s" % (i, a, "**FAIL**" if b == "FAIL" else b) for i, a, b in items),
                tui="\n".join("  %s%s%s  %s \u2192 %s%s%s" % (Clr.WHT, i, Clr.RST, a, {"FAIL": Clr.BRED, "WARN": Clr.BYEL}.get(b, Clr.BGRN), b, Clr.RST)
                              for i, a, b in items))


def case_rows(layout, results, model="", base=None):
    """Rows of the table for finished results; `base`: the baseline's results by request id (with a baseline)."""
    base = base or {}
    return [layout.cells(m, model, base=base.get(m.get("id"))) for m in results]


# -- counts and anomalies --------------------------------------------------------------------------------------

def verdict_counts(results):
    out = {v: 0 for v in VERDICTS}
    for r in results:
        if r.get("eval_result") in out:
            out[r["eval_result"]] += 1
    return out


def tools_agreement(results):
    """Share (0-100) of the graded turns whose tool calls are the same set as the recording's, or None without any.
    Compactions, SKIP and errors are left out; so are results without grade()'s `matches_recorded` (smoke cases)."""
    graded = [r for r in results if r.get("matches_recorded") is not None and r.get("kind") != "compaction"
              and r.get("eval_result") != "SKIP" and not r.get("error")]
    return 100 * sum(1 for r in graded if r["matches_recorded"]) / len(graded) if graded else None


def in_line_share(results):
    """(matching, graded, percent) of the graded turns whose first step is one the reference models took
    (grade()'s `in_line`; None where the id has no entry), or None without any. Same turns as
    tools_agreement."""
    graded = [r for r in results if r.get("in_line") is not None and r.get("kind") != "compaction" and r.get("eval_result") != "SKIP" and not r.get("error")]
    if not graded:
        return None
    n = sum(1 for r in graded if r["in_line"])
    return n, len(graded), 100 * n / len(graded)


def counts_line(counts, agreement=None, in_line=None):
    """`1 PASS · 5 WARN · 0 FAIL · 0 SKIP` (a nonzero FAIL is bold in markdown; colored in the terminal), then
    `same tools as recorded 91%` when `agreement` (tools_agreement) is given, then `in line with reference models 310/470 (66%)` when
    `in_line` (in_line_share) is."""
    sep = " · "
    col = {"PASS": Clr.BGRN, "WARN": Clr.BYEL, "FAIL": Clr.BRED, "SKIP": Clr.DIM}
    plain = sep.join("%d %s" % (counts[v], v) for v in VERDICTS)
    md = sep.join(("**%d FAIL**" if v == "FAIL" and counts[v] else "%d %s") % ((counts[v],) if v == "FAIL" and counts[v]
                                                                                 else (counts[v], v)) for v in VERDICTS)
    tui = sep.join("%s%d %s%s" % (col[v] if counts[v] else Clr.DIM, counts[v], v, Clr.RST) for v in VERDICTS)
    if agreement is not None:
        tail = "%s%d%%" % ("same tools as recorded ", round(agreement))
        plain, md, tui = (x + sep + tail for x in (plain, md, tui))
    if in_line is not None:
        tail = "in line with reference models %d/%d (%d%%)" % (in_line[0], in_line[1], round(in_line[2]))
        plain, md, tui = (x + sep + tail for x in (plain, md, tui))
    return Line(plain, md=md, tui=tui)


# -- per-stream and per-context-band breakdown ----------------------------------------------------------------

BAND_STEP = 32768  # 32 * 1024 (k = 1024, matching --max-context)


def band_label(prompt_tokens):
    """Context band label: '0-32k', '32-64k', '64-96k', ... (k = 1024, matching --max-context)."""
    pt = max(0, prompt_tokens or 0)
    b = pt // BAND_STEP
    return "%d-%dk" % (b * 32, (b + 1) * 32)


BANDS = [("%d-%dk" % (i * 32, (i + 1) * 32), i * BAND_STEP, (i + 1) * BAND_STEP) for i in range(8)]
STREAM_GROUPS = ("main", "compaction", "subagent")

BREAKDOWN_HEADER = [
    "Stream", "Band", "Cases", "PASS", "WARN", "FAIL", "SKIP",
    "Same tools %", "In line N/M (%)", "Decode tok/s p50", "Prefill tok/s p50"
]
BREAKDOWN_ALIGNS = ["l", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r"]
BREAKDOWN_SHRINK = (("Stream", 8),)
BREAKDOWN_DROP = ("Δ", "ms/cycle", "Cycles/s", "Tok/cycle", "Prefill tok/s p50", "Decode tok/s p50", "In line N/M (%)", "SKIP", "Same tools %")


def speculation(r):
    """(cycles, accepted_draft_tokens) from Splash's metrics.speculation, or None."""
    m = r.get("metrics")
    sp = m.get("speculation") if isinstance(m, dict) else None
    if not isinstance(sp, dict) or not sp.get("cycles"):
        return None
    return sp["cycles"], sp.get("accepted_draft_tokens") or 0


def draft_counts(r):
    """(drafted_tokens, accepted_draft_tokens) of a result from metrics.speculation, or None without a drafted count."""
    m = r.get("metrics")
    sp = m.get("speculation") if isinstance(m, dict) else None
    if not isinstance(sp, dict) or not (sp.get("drafted_tokens") or 0) > 0:
        return None
    return sp["drafted_tokens"], sp.get("accepted_draft_tokens") or 0


def _stream_group(r):
    if r.get("kind") == "compaction":
        return "compaction"
    stream = r.get("stream")
    if not stream or stream == "main":
        return "main"
    return "subagent"


def _prompt_tokens(r):
    pt = r.get("prompt_tokens")
    if pt is not None:
        return pt
    return r.get("recorded_prompt_tokens") or 0


def breakdown_groups(results):
    """[(stream_group, band_label, [subset_results])] for combinations that have cases."""
    if isinstance(results, dict):
        results = list(results.values())
    out = []
    for sg in STREAM_GROUPS:
        sg_res = [r for r in results if _stream_group(r) == sg]
        if not sg_res:
            continue
        band_indices = sorted({(_prompt_tokens(r) // BAND_STEP) for r in sg_res})
        for b in band_indices:
            label = "%d-%dk" % (b * 32, (b + 1) * 32)
            part = [r for r in sg_res if (_prompt_tokens(r) // BAND_STEP) == b]
            if part:
                out.append((sg, label, part))
    return out


def breakdown_rows(results, base_results=None):
    """Rows of the (stream group, context band) breakdown table."""
    base_by_id = {r["id"]: r for r in base_results} if base_results else {}
    has_base = base_results is not None
    res_list = list(results.values()) if isinstance(results, dict) else results
    has_spec = any(speculation(r) is not None for r in res_list)
    rows = []
    for sg, label, part in breakdown_groups(results):
        counts = verdict_counts(part)
        agr = tools_agreement(part)
        con = in_line_share(part)
        ran = [r for r in part if r.get("eval_result") != "SKIP"]
        dec_tps = sorted(r["native_decode_tps"] for r in ran if r.get("native_decode_tps") is not None)
        pre_tps = sorted(r["native_prefill_tps"] for r in ran if r.get("native_prefill_tps") is not None)
        p50_dec = F.p50(dec_tps)
        p50_pre = F.p50(pre_tps)

        row = [
            Cell(sg, "code"),
            Cell(label),
            Cell(F.num(len(part))),
            Cell(F.num(counts["PASS"]), "pass" if counts["PASS"] else ""),
            Cell(F.num(counts["WARN"]), "warn" if counts["WARN"] else ""),
            Cell(F.num(counts["FAIL"]), "fail" if counts["FAIL"] else ""),
            Cell(F.num(counts["SKIP"]), "skip" if counts["SKIP"] else ""),
            Cell(F.pct(agr, ".0f") if agr is not None else F.DASH),
            Cell("%d/%d (%d%%)" % (con[0], con[1], round(con[2])) if con is not None else F.DASH),
            Cell(F.flt(p50_dec, ".1f") if p50_dec is not None else F.DASH, F.speed_style(p50_dec, F.DECODE_TPS)),
        ]
        if has_base:
            base_part = [base_by_id[r["id"]] for r in part if r.get("id") in base_by_id]
            base_ran = [b for b in base_part if b.get("eval_result") != "SKIP"]
            b_dec = sorted(b["native_decode_tps"] for b in base_ran if b.get("native_decode_tps") is not None)
            base_p50_dec = F.p50(b_dec) if b_dec else None
            row.append(F.delta_cell(p50_dec, base_p50_dec, higher_is_better="up"))
        row.append(Cell(F.flt(p50_pre, ".1f") if p50_pre is not None else F.DASH, F.speed_style(p50_pre, F.PREFILL_TPS)))
        if has_base:
            b_pre = sorted(b["native_prefill_tps"] for b in base_ran if b.get("native_prefill_tps") is not None)
            base_p50_pre = F.p50(b_pre) if b_pre else None
            row.append(F.delta_cell(p50_pre, base_p50_pre, higher_is_better="up"))
        if has_spec:
            dec_s = lambda r: max((r.get("total_ms") or 0) - (r.get("ttft_ms") or 0), 0) / 1000
            spec = [(r, speculation(r)) for r in part if not r.get("error") and r.get("eval_result") != "SKIP"]
            spec = [(r, s) for r, s in spec if s]
            cycles = sum(s[0] for _, s in spec)
            tok_per_cycle = 1 + sum(s[1] for _, s in spec) / cycles if cycles else None
            fast = [(r, s) for r, s in spec if (r.get("tokens") or 0) >= 50 and dec_s(r) > 0]
            fsecs = sum(dec_s(r) for r, _ in fast)
            cycles_per_s = sum(s[0] for _, s in fast) / fsecs if fsecs else None
            ms_per_cycle = 1000 / cycles_per_s if cycles_per_s else None
            row.append(Cell(F.flt(tok_per_cycle, ".2f") if tok_per_cycle is not None else F.DASH))
            row.append(Cell(F.flt(cycles_per_s, ".2f") if cycles_per_s is not None else F.DASH))
            row.append(Cell(F.flt(ms_per_cycle, ".1f") if ms_per_cycle is not None else F.DASH))
        rows.append(row)
    return rows


def breakdown_table(results, base_results=None):
    """Table broken down by stream group and context band, or None when fewer than 2 rows."""
    rows = breakdown_rows(results, base_results=base_results)
    if len(rows) < 2:
        return None
    drop_stream = len({str(r[0]) for r in rows}) <= 1
    has_base = base_results is not None
    res_list = list(results.values()) if isinstance(results, dict) else results
    has_spec = any(speculation(r) is not None for r in res_list)
    base_header = (
        ["Stream", "Band", "Cases", "PASS", "WARN", "FAIL", "SKIP",
         "Same tools %", "In line N/M (%)", "Decode tok/s p50", "Δ", "Prefill tok/s p50", "Δ"]
        if has_base else list(BREAKDOWN_HEADER)
    )
    base_aligns = (
        ["l", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"]
        if has_base else list(BREAKDOWN_ALIGNS)
    )
    if has_spec:
        base_header += ["Tok/cycle", "Cycles/s", "ms/cycle"]
        base_aligns += ["r", "r", "r"]
    keep_indices = []
    for i in range(len(base_header)):
        if i == 0 and drop_stream:
            continue
        if all(str(r[i]).strip() in (F.DASH, "–", "-") for r in rows):
            continue
        keep_indices.append(i)

    header = [base_header[i] for i in keep_indices]
    aligns = [base_aligns[i] for i in keep_indices]
    filtered_rows = [[r[i] for i in keep_indices] for r in rows]
    shrink = tuple((col, w) for col, w in BREAKDOWN_SHRINK if col in header)
    drop = tuple(col for col in BREAKDOWN_DROP if col in header)
    dirs = {i: "up" for i, col in enumerate(header) if col == "Δ"}
    return Table(header, filtered_rows, aligns=aligns, shrink=shrink, drop=drop, dirs=dirs)


def anomaly_block(items):
    """`- **id** WARN: reason` lines for [(case id, verdict, reason)]; the terminal form is aligned and colored."""
    width = max(len(i) for i, _, _ in items)
    md, tui = [], []
    for cid, v, reason in items:
        reason = " ".join(str(reason or "no reason given").split())
        md.append("- **%s** %s: %s" % (cid, "**FAIL**" if v == "FAIL" else v, reason.replace("`", "'")))
        col = Clr.BYEL if v == "WARN" else Clr.BRED
        tui.append("  %s%s%s  %s%-4s%s  %s%s%s" % (Clr.WHT, cid.ljust(width), Clr.RST, col, v, Clr.RST, Clr.DIM, reason, Clr.RST))
    return Line("\n".join("  %s  %s  %s" % (i.ljust(width), v, r) for i, v, r in items), md="\n".join(md), tui="\n".join(tui))


def anomalies(results):
    """[(case id, verdict, reason)] of the WARN and FAIL results, in order."""
    return [(r.get("id"), r["eval_result"], r.get("fail_reason")) for r in results
            if r.get("eval_result") in ("WARN", "FAIL")]


# -- run description -----------------------------------------------------------------------------------------

def run_info(mode, target, request=None, selection=None, summary=None, note=None, preset=None):
    """What one model's run was, for the header and the Run section: from a live target or from a state header."""
    h = target.get("header") or {}
    server = target.get("backend_info") or h.get("server")
    model_info = target.get("model_info") or h.get("model_info")
    return {"model": target["name"], "backend": target["backend"], "suite": h.get("suite") or mode,
            "server": backend.label(server) if server else None,
            "device": ((target.get("device") or h.get("device")) or {}).get("name"),
            "ctx": target.get("n_ctx") or h.get("n_ctx"), "request": request or h.get("request") or {},
            "selection": selection if selection is not None else h.get("selection"), "summary": summary,
            "note": note or h.get("note"), "preset": preset or h.get("preset"),
            "model_info": model_info}


def model_info_text(mi):
    """Format model_info for display: 'Q4_K_M (gguf, 27.3B)' or 'Q4_K_M'."""
    if not isinstance(mi, dict) or not mi:
        return None
    quant = mi.get("quant")
    details = [str(mi[k]) for k in ("format", "parameter_size", "runner", "mode") if mi.get(k)]
    if mi.get("drafter"):
        details.append("drafter %s" % mi["drafter"])
    if quant and details:
        return "%s (%s)" % (quant, ", ".join(details))
    if quant:
        return str(quant)
    if details:
        return ", ".join(details)
    return ", ".join("%s: %s" % (k, v) for k, v in mi.items() if v is not None and k not in ("digest", "label")) or None


def selection_text(sel):
    if not sel:
        return "full"
    parts = []
    if sel.get("max_context"):
        parts.append("max context %s" % F.ctx_k(sel["max_context"]))
    if sel.get("no_subagents"):
        parts.append("no subagents")
    if sel.get("turns"):
        parts.append("requests %s-%s" % tuple(sel["turns"]))
    return ", ".join(parts) or "full"


def sampling_text(req):
    req = req or {}
    t, e = req.get("temperature"), req.get("reasoning_effort")
    th = req.get("thinking")
    return "temperature %s, reasoning effort %s%s" % ("server default" if t is None else t, "none sent" if e is None else e,
                                                     "" if th is None else ", thinking %s" % th)


def headline(infos, n_cases):
    """`Agent replay · suite judge (6 requests) · Splash 4b2dee69 · ctx 256k` (several models: `3 models`)."""
    suite = infos[0]["suite"] if infos else "?"
    parts = ["Agent replay", "suite %s (%s)" % (suite, ("%d requests" % n_cases) if n_cases != 1 else "1 request")]
    if len(infos) == 1:
        i = infos[0]
        mi_str = model_info_text(i.get("model_info"))
        note = i.get("note")
        if note and note == (i.get("model_info") or {}).get("mode"):
            note = None  # mlx-dspark: the default note is the mode, already in the model details
        parts += [p for p in (i["server"], mi_str, note, i.get("device"),
                              ("ctx %s" % F.ctx_k(i["ctx"])) if i["ctx"] else None) if p]
    elif infos:
        parts.append("%d models" % len(infos))
    return " · ".join(parts)


def run_blocks(infos, labels, n_cases):
    """The markdown document head: the title and the `## Run` section (a field table for one model, one row per model
    for several, named by `labels`, which carry the run notes)."""
    suite = infos[0]["suite"] if infos else "?"
    out = ["# Agent replay — suite %s" % suite, "## Run"]
    if len(infos) == 1:
        i = infos[0]
        mi = i.get("model_info")
        mi_str = model_info_text(mi) if mi else None
        rows = [("Server", i["server"]), ("Model", Cell((i.get("model_info") or {}).get("label") or i["model"], "code")),
                ("Model info", mi_str),
                ("Note", i.get("note")), ("Device", i.get("device")), ("Endpoint", i["backend"]),
                ("Context", ("%s tokens (%s)" % (F.num(i["ctx"]), F.ctx_k(i["ctx"]))) if i["ctx"] else None),
                ("Suite", "%s, %d requests" % (suite, n_cases)), ("Requests", i.get("summary")),
                ("Selection", selection_text(i["selection"])), ("Sampling", sampling_text(i["request"]))]
        out.append(Table(["Field", "Value"], [(k, v) for k, v in rows if v], aligns=["l", "l"]))
    else:
        rows = [[Cell(lb, "code"), i["server"] or F.DASH, ("%s" % F.ctx_k(i["ctx"])) if i["ctx"] else F.DASH,
                 selection_text(i["selection"]), sampling_text(i["request"])] for i, lb in zip(infos, labels)]
        out.append(Table(["Model", "Server", "Ctx", "Selection", "Sampling"], rows, aligns=["l"] * 5))
        out.append(report.Note("%s, %d requests per model" % (suite, n_cases)))
    return out


def interrupted_line(reason, n):
    text = "Interrupted (%s) after %d request%s." % (reason, n, "" if n == 1 else "s")
    return Line(text, md="**Interrupted** " + text[len("Interrupted "):], tui="%s%s%s" % (Clr.BYEL, text, Clr.RST))
