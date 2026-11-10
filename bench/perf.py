"""Performance by context bucket and prefill by append size, from client-side results plus the server telemetry
stored per result. Rows are data (`context_rows`, `append_rows`); `perf_blocks` renders them as report tables.

Definitions (k = 1024):
- Context bucket: the request's prompt tokens, <=32k, <=64k, ... <=256k (and >256k); empty buckets are skipped.
- Decode time = request time - TTFT, on the client clock, for every figure (decode tok/s, cycles/s).
- Decode tok/s = sum(output tokens) / sum(decode time), output-weighted.
- Prefill tok/s = sum(prompt - cached) / sum(TTFT): cache hits included, a request with nothing new adds time only.
- A speculation cycle is one draft-and-verify step. Tok/cycle = 1 + accepted_draft_tokens / cycles, Splash's own
  definition (server/diagnostics.py); cycles/s = cycles / decode time over requests with >= 50 output tokens.
  Both come from `metrics.speculation` of Splash; a result without it contributes nothing, and a column nobody
  has data for is dropped.
- New tokens = prompt - cached; a server that reports no cache (LM Studio): prompt - the previous prompt of the stream
  (results.new_tokens), the whole prompt for the first request of a stream.
- Append size: new tokens, only requests with more than 16 new tokens, in < 1k, 1-5k, >= 5k (k = 1024).
"""

from . import fmt as F
from . import results as R
from .report import Cell, Table
from .state import with_note

K = 1024
MIN_CYCLE_TOKENS = 50
MIN_NEW_TOKENS = R.MIN_NEW_TOKENS  # "more than 16 new tokens"
APPEND_BUCKETS = [("< 1k", 0, 1 * K), ("1–5k", 1 * K, 5 * K), ("≥ 5k", 5 * K, None)]


def completed(results):
    """Results that produced a response: not SKIP, no error, with a prompt count."""
    return [r for r in results if r.get("eval_result") != "SKIP" and not r.get("error")
            and (r.get("prompt_tokens") or 0) > 0]


PAUSE_MIN_MS = 60_000


def server_paused_ms(r):
    """Milliseconds the server was frozen mid-request (client TTFT minus the server's own TTFT), or None. A request
    paused when the client TTFT exceeds the native one by more than max(60 s, 2 x native). Computed from the stored
    ttft_ms / native_ttft_ms, so old states get it too."""
    if r.get("error") or r.get("eval_result") == "SKIP":
        return None
    ttft, native = r.get("ttft_ms") or 0, r.get("native_ttft_ms")
    if not ttft or native is None:
        return None
    extra = ttft - native
    return extra if extra > max(PAUSE_MIN_MS, 2 * native) else None


def clock_ok(results):
    """The results whose client-clock figures are trustworthy (the server did not pause)."""
    return [r for r in results if server_paused_ms(r) is None]


def paused_note(results):
    """One sentence naming the requests excluded from client-clock figures, or None."""
    paused = [r for r in results if server_paused_ms(r) is not None]
    if not paused:
        return None
    n = len(paused)
    worst = max(paused, key=server_paused_ms)
    return "%d request%s excluded from client-clock figures: the server paused (client TTFT %.0fs vs server %.1fs)%s" % (
        n, "" if n == 1 else "s", worst["ttft_ms"] / 1000, worst["native_ttft_ms"] / 1000,
        "" if n == 1 else " in the worst case")


def speculation(r):
    return R.speculation(r)


def context_stats(results):
    """Numbers of one group of results; None for a figure without data."""
    all_rs = completed(results)
    rs = clock_ok(all_rs)  # client-clock figures leave out requests during which the server paused
    dec_s = lambda r: max((r.get("total_ms") or 0) - (r.get("ttft_ms") or 0), 0) / 1000
    out = {"requests": len(all_rs)}
    toks = sum(r.get("tokens") or 0 for r in rs if dec_s(r) > 0)
    secs = sum(dec_s(r) for r in rs if (r.get("tokens") or 0) > 0)
    out["decode_tps"] = toks / secs if secs and toks else None
    new = sum(R.new_tokens(r) for r in rs if (r.get("ttft_ms") or 0) > 0)
    ttft = sum((r.get("ttft_ms") or 0) for r in rs) / 1000
    out["prefill_tps"] = new / ttft if ttft and new else None
    spec = [(r, speculation(r)) for r in all_rs]
    spec = [(r, s) for r, s in spec if s]
    cycles = sum(s[0] for _, s in spec)
    drafts = [d for d in map(R.draft_counts, all_rs) if d]
    out["draft_acc"] = 100 * sum(d[1] for d in drafts) / sum(d[0] for d in drafts) if drafts else None
    out["tok_per_cycle"] = 1 + sum(s[1] for _, s in spec) / cycles if cycles else None
    fast = [(r, s) for r, s in spec if (r.get("tokens") or 0) >= MIN_CYCLE_TOKENS and dec_s(r) > 0]
    fsecs = sum(dec_s(r) for r, _ in fast)
    out["cycles_per_s"] = sum(s[0] for _, s in fast) / fsecs if fsecs else None
    out["ms_per_cycle"] = 1000 / out["cycles_per_s"] if out["cycles_per_s"] else None
    return out


def context_rows(runs):
    """[{bucket, run, index, **context_stats}] for runs = [(name, results list)], bucket-major, runs in order."""
    band_indices = sorted({
        ((r.get("prompt_tokens") or 0) // R.BAND_STEP)
        for _, results in runs
        for r in completed(results)
    })
    rows = []
    for b in band_indices:
        label = R.band_label(b * R.BAND_STEP)
        for i, (name, results) in enumerate(runs):
            part = [r for r in completed(results) if R.band_label(r.get("prompt_tokens") or 0) == label]
            if part:
                rows.append(dict(context_stats(part), bucket=label, run=name, index=i))
    return rows


def append_rows(runs):
    rows = []
    for label, lo, hi in APPEND_BUCKETS:
        for i, (name, results) in enumerate(runs):
            part = []
            for r in clock_ok(completed(results)):
                new = R.new_tokens(r)
                if new >= MIN_NEW_TOKENS and new >= lo and (hi is None or new < hi):
                    part.append((new, (r.get("ttft_ms") or 0) / 1000))
            if part:
                s_new, s_ttft = sum(p[0] for p in part), sum(p[1] for p in part)
                rows.append({"bucket": label, "run": name, "index": i, "requests": len(part), "sum_new": s_new,
                             "ttft_s": s_ttft, "tps": s_new / s_ttft if s_ttft else None})
    return rows


def run_names(entries):
    """Short run names: the model names with their common prefix stripped when distinct across entries, else the
    full labels; a run note follows as `name \u00b7 note`. A --compare-state run joined as a baseline is named by its
    label (what differs from the run: `Splash 1.6.0-m1 \u00b7 mlx-community/Qwen3.8-27B-4bit`, note included)."""
    names = [e["name"] for e in entries]
    if any(n.endswith(" (baseline)") for n in names):  # --compare-state runs joined as baselines
        if len(entries) == 2:
            return ["base", "this run"]
        rest = iter(F.short_names([n for n in names if not n.endswith(" (baseline)")]))
        out = ["base" if n.endswith(" (baseline)") else next(rest) for n in names]
        return [o if n.endswith(" (baseline)") else with_note(o, e.get("note")) for o, n, e in zip(out, names, entries)]
    elif len(set(names)) != len(names):
        return [e["label"] for e in entries]  # the labels carry the run notes
    else:
        out = F.short_names(names)
    return [with_note(n, e.get("note")) for n, e in zip(out, entries)]


def _f(v, spec):
    return F.DASH if v is None else format(v, spec)


def append_table(entries):
    runs = list(zip(run_names(entries), [list(e["results"].values()) for e in entries]))
    rows = append_rows(runs)
    if not rows:
        return None
    multi = len(entries) > 1
    body = [[r["bucket"]] + ([Cell(r["run"], "id")] if multi else [])
            + [F.num(r["requests"]), F.num(r["sum_new"]), F.secs(r["ttft_s"]), _f(r["tps"], ".1f")] for r in rows]
    head = ["New tokens"] + (["Run"] if multi else []) + ["Requests", "Sum new", "TTFT", "tok/s"]
    return Table(head, body, aligns=["l"] + ["l"] * multi + ["r"] * 4)


def has_new_estimate(entries):
    """True when some request's new tokens are estimated from the previous prompt (the server reports no cache)."""
    return any(R.estimates_new_tokens(r) for e in entries for r in e["results"].values())


def has_spec(entries):
    """True when some request carries speculative-decoding cycle figures."""
    for e in entries:
        for r in e["results"].values():
            if speculation(r):
                return True
    return False


def perf_blocks(entries, baseline_index=0):
    """Report blocks of the prefill-by-append table, or [] when there is nothing to show."""
    out = []
    at = append_table(entries)
    if at:
        out += ["## Performance", "### Prefill by append size", at]
    return out
