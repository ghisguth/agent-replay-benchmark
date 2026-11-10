"""Live terminal line, prefill progress gauge, and status rendering."""

import sys

from . import glyphs
from .colors import Clr
from .fmt import fmt_clock, fmt_number, live_vlen, term_width


def stream_color(label):
    """A stable color per stream: main cyan, compactions yellow, subagents spread over the others."""
    label = str(label)
    if label.startswith("main/c"):
        return Clr.BYEL
    if label.startswith("main"):
        return Clr.BCYN
    palette = [Clr.BMAG, Clr.BGRN, Clr.BBLU, Clr.MAG]
    return palette[sum(map(ord, label)) % len(palette)]


def bar(frac, width=10, color=None):
    """A progress bar: dim ends, `color` filled blocks, dim empty ones."""
    n = round(min(1.0, max(0.0, frac)) * width)
    return "%s\u2595%s%s%s%s%s%s\u258f%s" % (Clr.DIM, Clr.RST, color or Clr.BGRN, "\u2588" * n, Clr.RST, Clr.DIM,
                                           "\u2591" * (width - n), Clr.RST)


def term_cols(stream=None):
    """The current width of the terminal `stream` (default stderr; 100 when unknown), read on every call so a resize
    is picked up."""
    return term_width(stream or sys.stderr, 100)


def _word(icon, word, text, col, after=False):
    """A value with its label: the icon when icons are on, else the word (before the value, or after it)."""
    ico = glyphs.prefix(icon)
    if ico:
        return f"{ico}{col}{text}{Clr.RST}"
    return f"{col}{text}{Clr.RST} {Clr.DIM}{word}{Clr.RST}" if after and word else \
        (f"{Clr.DIM}{word}{Clr.RST} {col}{text}{Clr.RST}" if word else f"{col}{text}{Clr.RST}")


def fmt_live_tokens(n):
    """Format downloaded token count for the live progress bar.

    - < 10000: raw count with up to 4 digits (e.g. '0', '999', '1000', '9000', '9999')
    - >= 10000 and < 100K: 1 decimal place with 'K', keeping trailing .0 (e.g. '10.0K', '10.1K', '11.0K', '11.2K', '99.9K')
    - >= 100K: whole count with 'K' and no float digit (e.g. '100K', '101K')
    """
    n = int(n or 0)
    if n < 10000:
        return str(n)
    val = n / 1000
    if round(val, 1) < 100.0:
        return f"{val:.1f}K"
    return f"{round(val)}K"


def prefill_info(st, now):
    """Prefill progress of a request that has not produced its first token: {frac, est, elapsed, remaining} (seconds;
    `remaining` None when unknown, shown as "?"), or None when there is nothing to base it on (the plain timer is shown then).

    Real: the server's last `prompt_progress` (st["pp"] = {total, cache, processed, time_ms}); the fraction is
    (processed - cache) / (total - cache) and the remaining time extrapolates the server's own rate
    (uncached tokens left / (processed - cache) / server elapsed; before 0.5 s or any progress st["est_tps"] marks it `<~`, else
    unknown). A reading with "chunk" (MTPLX, one reading per 2048-token chunk) is interpolated by the server rate since its "at"
    time, capped at the next chunk boundary and 99%, never backwards; st["pp_expected"]
    (the server is known to send progress) without a pp yet: 0%, remaining unknown, never the estimate. Estimate: st["est_tokens"]
    (uncached prompt tokens expected) over st["est_tps"] (recent prefill speed), never shown above 99%."""
    elapsed = max(0.0, now - st["start"])
    pp = st.get("pp")
    if pp and pp["total"] - pp["cache"] > 0:
        span = pp["total"] - pp["cache"]
        done = pp["processed"] - pp["cache"]
        spent = pp["time_ms"] / 1000.0 if pp["time_ms"] > 0 else elapsed
        rate = done / spent if done > 0 and spent >= 0.5 else None
        est_rate = st["est_tps"] if st.get("est_tps") and st["est_tps"] > 0 else None
        cur = pp["processed"]
        if pp.get("chunk") and pp.get("at") is not None:  # MTPLX: readings come once per chunk, interpolate between them
            step = rate or est_rate
            if step:
                cap = min(pp["processed"] + pp["chunk"], pp["total"], pp["cache"] + int(0.99 * span))
                cur = max(cur, min(cap, pp["processed"] + step * max(0.0, now - pp["at"])))
            cur = max(cur, pp.get("shown", 0))  # never backwards for this reading
            pp["shown"] = cur
        frac = min(1.0, max(0.0, (cur - pp["cache"]) / span))
        use = rate or est_rate
        remaining = max(0.0, (pp["total"] - cur) / use) if use else None
        return {"frac": frac, "est": False, "elapsed": elapsed, "remaining": remaining, "rest_est": rate is None and use is not None}
    if st.get("pp_expected"):  # the server sends progress: 0% until the first chunk, never the estimate
        return {"frac": 0.0, "est": False, "elapsed": elapsed, "remaining": None}
    if st.get("est_tokens") and st.get("est_tps"):
        if st["est_tokens"] < 10 or st["est_tps"] <= 0:
            return None
        guess = st["est_tokens"] / st["est_tps"]
        return {"frac": min(0.99, elapsed / guess), "est": True, "elapsed": elapsed,
                "remaining": max(0.0, guess - elapsed)}
    return None


def _prefill_seg(info, drop):
    """`prefill ▕███░░░▏ ~52% 12s<~11s`: bar, percent ("~" when estimated) and elapsed<remaining (tqdm style,
    `<?` when unknown)."""
    out = []
    if "pbar" not in drop:
        out.append(bar(info["frac"], 6, Clr.BYEL))
    pct = "%s%d%%" % ("~" if info["est"] else "", int(info["frac"] * 100))
    t = fmt_clock(info["elapsed"])
    if "peta" not in drop:
        r = info["remaining"]
        t += "<?" if r is None else "<%s%s" % ("~" if info["est"] or info.get("rest_est") else "", fmt_clock(r))
    out += [f"{Clr.BYEL}{pct}{Clr.RST}", f"{Clr.BYEL}{t}{Clr.RST}"]
    return (glyphs.prefix("prefill") or f"{Clr.DIM}prefill{Clr.RST} ") + " ".join(out)


def live_parts(tc, st, now, progress=None, n_ctx=None):
    """The pieces of the live line, each with its optional extras: {bar, count, eta, run, id, stream, gauge, size, cache,
    ttft, out, speed, elapsed}; None where there is nothing to show. `live_compose` decides which fit."""
    entry = tc.get("entry") or {}
    ctx = tc.get("recorded_prompt_tokens") or entry.get("prompt_tokens") or 0
    p = {"id": str(tc.get("id", "?")).split("_")[0] if entry else str(tc.get("id", "?")),
         "stream": entry.get("label")}
    if progress:
        done, total = progress["done"], progress["total"]
        p["bar"] = bar(done / max(1, total))
        p["count"] = "%d/%d" % (done + 1, total)
        if done > 0:
            p["eta"] = fmt_clock((progress["t_last"] - progress["t0"]) / done * (total - done))
        p["run"] = fmt_clock(now - progress["t0"])
        if progress.get("cache") is not None:
            p["cache"] = "%.0f%%" % progress["cache"]
        n_ctx = n_ctx or progress.get("n_ctx")
    if entry:
        p["size"] = fmt_number(ctx).lower() + ("/" + fmt_number(n_ctx).lower() if n_ctx else "")
        if n_ctx:
            p["gauge"] = bar(ctx / n_ctx, 6, Clr.BYEL if ctx > 0.8 * n_ctx else Clr.BCYN)
    st = st or {}
    if "start" in st:
        if not st.get("first"):
            p["ttft"] = ("prefill", fmt_clock(now - st["start"]))
            p["prefill"] = prefill_info(st, now)
        else:
            gen = now - st["first"]
            p["ttft"] = ("TTFT", "%.1fs" % (st["first"] - st["start"]))
            p["out"] = "%s" % fmt_live_tokens(st.get("down", 0))
            if gen > 0.5 and st.get("down"):
                p["speed"] = "%.0f" % (st["down"] / gen)
            p["elapsed"] = fmt_clock(now - st["start"])
    return p


# Detail levels, most to least: each level drops one more piece. The order keeps the spinner, the run bar and count,
# the case id, the context gauge and tok/s longest.
DROP_ORDER = ("eta", "cache", "out", "run", "elapsed", "peta", "pbar", "ttft", "gauge", "bar", "stream", "size", "count")


def live_render(p, drop=()):
    """The live line from `live_parts`, without the pieces named in `drop`."""
    with glyphs.live():
        return _live_render(p, drop)


def _live_render(p, drop):
    dot = f" {Clr.DIM}\u00b7{Clr.RST} "
    g = lambda k: None if k in drop else p.get(k)
    head = []
    if g("bar"):
        head.append(p["bar"])
    if g("count"):
        count = f"{Clr.DIM}{p['count']}{Clr.RST}"
        if g("run"):
            count += dot + _word("delta", "run", p["run"], Clr.DIM)
        if g("eta"):
            count += dot + _word("eta", "ETA", p["eta"], Clr.DIM)
        head.append(count)
    ident = f"{Clr.BLD}{p['id']}{Clr.RST}"
    if g("stream"):
        ident += f" {stream_color(p['stream'])}{p['stream']}{Clr.RST}"
    segs = [" ".join(head + [ident]) if head else ident]
    if p.get("size") and "size" not in drop:
        ctx = ""
        if g("gauge"):
            ctx += p["gauge"] + " "
        ctx += f"{Clr.BCYN}{p['size']}{Clr.RST}"
        if not glyphs.enabled:
            ctx += f" {Clr.DIM}ctx{Clr.RST}"
        else:
            ctx = glyphs.prefix("ctx") + ctx
        if g("cache"):
            ctx += " " + _word("cache", "cache", p["cache"], Clr.BBLU)
        segs.append(ctx)
    if g("ttft"):
        word, text = p["ttft"]
        if p.get("prefill"):
            segs.append(_prefill_seg(p["prefill"], drop))
        else:
            segs.append(_word("prefill" if word == "prefill" else "ttft", word, text, Clr.BYEL))
    if g("out"):
        segs.append(_word("out", "tok", p["out"], Clr.BGRN, after=True))
    if p.get("speed"):
        segs.append(f"{glyphs.prefix('speed')}{Clr.BMAG}{p['speed']} tok/s{Clr.RST}")
    if g("elapsed"):
        segs.append(_word("clock", "", p["elapsed"], Clr.DIM))
    return dot.join(segs)


def live_compose(tc, st, now, progress=None, width=None, n_ctx=None):
    """The live line, fitted to `width` display columns: pieces are dropped in DROP_ORDER until it fits, never cut."""
    p = live_parts(tc, st, now, progress, n_ctx)
    drop = []
    line = live_render(p, drop)
    for key in DROP_ORDER:
        if width is None or live_vlen(line) <= width:
            break
        drop.append(key)
        line = live_render(p, drop)
    return line


def live_label(tc, n_ctx=None, progress=None):
    """`R123 main/e0 · 204k ctx` for a replay case (the plain id for other cases); with `progress` it starts with the
    run bar and `case/total`, with `n_ctx` the size gets a gauge."""
    if not tc.get("entry") and not progress:
        return tc.get("id", "?")
    return live_render(live_parts(tc, None, 0, progress, n_ctx), ("eta", "cache", "run"))


def live_line(label, st, now, progress=None):
    """The label plus what a request in flight shows (kept for plain callers; the spinner uses `live_compose`)."""
    with glyphs.live():
        return _live_line(label, st, now, progress)


def _live_line(label, st, now, progress):
    st = st or {}
    if "start" not in st:
        return label
    dot = f" {Clr.DIM}\u00b7{Clr.RST} "
    parts = []
    if not st.get("first"):
        info = prefill_info(st, now)
        parts.append(_prefill_seg(info, ()) if info else _word("prefill", "prefill", fmt_clock(now - st["start"]), Clr.BYEL))
    else:
        gen = now - st["first"]
        parts.append(_word("ttft", "TTFT", "%.1fs" % (st["first"] - st["start"]), Clr.BYEL))
        parts.append(_word("out", "tok", "%s" % fmt_live_tokens(st.get("down", 0)), Clr.BGRN, after=True))
        if gen > 0.5 and st.get("down"):
            parts.append(f"{glyphs.prefix('speed')}{Clr.BMAG}{st['down'] / gen:.0f} tok/s{Clr.RST}")
        parts.append(_word("clock", "", fmt_clock(now - st["start"]), Clr.DIM))
    if progress and progress.get("cache") is not None:
        parts.append(_word("cache", "cache", "%.0f%%" % progress["cache"], Clr.BBLU))
    return label + dot + dot.join(parts)
