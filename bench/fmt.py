"""Number and label formatting shared by every report table, so one quantity looks the same everywhere.

- Durations: 0.90s, 8.57s (under 10 s: two decimals), 24.6s (under a minute: one), 1m11s, 2h05m. Never raw ms.
- Counts: thousands separators (12,174); missing values are an en dash.
- Labels: several models get short labels (the common prefix of their names stripped at a separator).
- Width and alignment: Unicode display width (ASCII, CJK wide, emoji presentation, Nerd Font PUA), fitting, clipping.
"""

import os
import re
import shutil
import sys
import unicodedata

DASH = "–"
_BOUNDARY = "/-_ :"

ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def cw(ch, icon_width=1):
    """Display width of a single character: 0 for combining / zero-width, 2 for wide / fullwidth, 1 otherwise.
    Nerd Font icons (Private Use Area) count `icon_width` columns; the live line passes 2, the worst case."""
    if not ch:
        return 0
    o = ord(ch)
    if unicodedata.combining(ch):
        return 0
    if 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF:  # Variation Selectors
        return 0
    if 0x200B <= o <= 0x200D or o == 0xFEFF:  # Zero-width spaces / joiners
        return 0
    if 0xE000 <= o <= 0xF8FF or o >= 0xF0000:  # Nerd Font and other Private Use Area characters
        return icon_width
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def vlen(text, icon_width=1):
    """Display width of `text` without ANSI codes, accounting for wide characters and emoji presentation."""
    if not text:
        return 0
    clean = ANSI.sub("", text)
    w = 0
    prev_w = 0
    for ch in clean:
        if ch == "\ufe0f":
            # VS16 promotes a preceding narrow character (width 1) to emoji presentation (width 2)
            if prev_w == 1:
                w += 1
                prev_w = 2
            continue
        char_w = cw(ch, icon_width)
        w += char_w
        if char_w > 0:
            prev_w = char_w
    return w


def live_cw(ch):
    """cw() for the live line: Nerd Font icons count 2 columns."""
    return cw(ch, 2)


def live_vlen(text):
    """vlen() for the live line: each Nerd Font icon counts 2 columns (the worst case)."""
    return vlen(text, 2)


def _head(text, width):
    """The longest prefix of `text` within `width` display columns."""
    cut = ""
    w = 0
    prev_w = 0
    for ch in text:
        if ch == "\ufe0f":
            extra = 1 if prev_w == 1 else 0
            if w + extra > width:
                break
            cut += ch
            w += extra
            if extra:
                prev_w = 2
            continue
        char_w = cw(ch)
        if w + char_w > width:
            break
        cut += ch
        w += char_w
        if char_w > 0:
            prev_w = char_w
    return cut


def clip(text, width):
    """One line cut to `width` columns with an ellipsis."""
    text = " ".join(str(text).split())
    if vlen(text) <= width:
        return text
    return _head(text, max(0, width - 1)) + "…"


def fit(text, width, align="l", mid=False):
    """`text` cut with an ellipsis to `width` display columns, then padded to it; `mid` keeps both ends
    (`F428Fal…P1213R2`), for ids whose tail tells them apart."""
    if vlen(text) > width:
        if mid and width >= 5:
            tail = ""
            for ch in reversed(text):
                if vlen(tail) + cw(ch) > (width - 1) // 2:
                    break
                tail = ch + tail
            text = _head(text, width - 1 - vlen(tail)) + "…" + tail
        else:
            text = _head(text, max(0, width - 1)) + "…"
    gap = " " * max(0, width - vlen(text))
    return gap + text if align == "r" else text + gap


def hard_wrap(line, width):
    """`line` cut into pieces of at most `width` display columns (at least one piece)."""
    line = line.expandtabs(4).rstrip("\r")
    out, cur, w = [], "", 0
    for ch in line:
        char_w = cw(ch)
        if w + char_w > width:
            out.append(cur)
            cur, w = "", 0
        cur += ch
        w += char_w
    return out + [cur]


def term_limit(stream=None):
    """Display columns of the terminal `stream` (default stdout) is, else $COLUMNS, else None (no limit: a pipe or
    file). A tty is asked directly (zsh does not export COLUMNS), then the other standard streams."""
    stream = stream if stream is not None else sys.stdout
    try:
        tty = stream.isatty()
    except (AttributeError, ValueError):
        tty = False
    if tty:
        for f in (stream, sys.__stdout__, sys.stderr):
            try:
                n = os.get_terminal_size(f.fileno()).columns
            except (AttributeError, ValueError, OSError):
                continue
            if n > 0:
                return n
        n = shutil.get_terminal_size((0, 0)).columns
        if n > 0:
            return n
    try:
        n = int(os.environ["COLUMNS"])
        return n if n > 0 else None
    except (KeyError, ValueError):
        return None


def term_width(stream=None, default=120):
    """The terminal width when `stream` (default stdout) is a terminal, else $COLUMNS, else `default`."""
    cols = term_limit(stream)
    return cols if cols is not None else default


def fmt_number(n):
    """omp's formatNumber: 999, 1.2K, 12K, 1.2M."""
    n = int(n or 0)
    trim = lambda x: ("%.1f" % x).removesuffix(".0")
    if n < 1000:
        return str(n)
    if n < 10000:
        return trim(n / 1000) + "K"
    if n < 1000000:
        return "%dK" % round(n / 1000)
    if n < 10000000:
        return trim(n / 1000000) + "M"
    return "%dM" % round(n / 1000000)


def fmt_clock(seconds):
    """Fixed-width duration for the live line: " 5s", "12s", "1m08s", "10m00s", "1h02m" (whole seconds, floored)."""
    t = max(0, int(seconds or 0))
    if t < 60:
        return "%2ds" % t
    if t < 3600:
        return "%dm%02ds" % (t // 60, t % 60)
    return "%dh%02dm" % (t // 3600, t % 3600 // 60)


def fmt_duration(ms):
    """omp's formatDuration, minutes and up zero-padded: 450ms, 8.2s, 1m05s, 5h02m, 2d03h."""
    if not ms or ms <= 0:
        return "0ms"
    if ms < 1000:
        return "%dms" % ms
    if ms < 60000:
        tenths = round(ms / 100)
        return "%.1fs" % (tenths / 10) if tenths < 600 else "1m00s"
    if ms < 3600000:
        m, s = int(ms // 60000), int(ms % 60000 // 1000)
        return "%dm%02ds" % (m, s)
    if ms < 86400000:
        h, m = int(ms // 3600000), int(ms % 3600000 // 60000)
        return "%dh%02dm" % (h, m)
    d, h = int(ms // 86400000), int(ms % 86400000 // 3600000)
    return "%dd%02dh" % (d, h)


def secs(v):
    """Seconds as a duration; None is a dash."""
    if v is None:
        return DASH
    if v < 10:
        return "%.2fs" % v
    if v < 60:
        return "%.1fs" % v
    if v < 3600:
        m, s = divmod(int(round(v)), 60)
        return "%dm%02ds" % (m, s)
    h, m = divmod(int(round(v)) // 60, 60)
    return "%dh%02dm" % (h, m)


def ms(v):
    """Milliseconds as a duration."""
    return DASH if v is None else secs(v / 1000.0)


def num(v):
    """An integer count with thousands separators (rounded when it is not whole)."""
    return DASH if v is None else format(int(round(v)), ",")


def flt(v, spec=".1f"):
    return DASH if v is None else format(v, spec)


def quantile(values, q):
    """Continuous linear interpolation quantile (q between 0 and 1) over non-None values, or None."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    i = (len(vals) - 1) * q
    lo = int(i)
    return vals[lo] + (vals[min(lo + 1, len(vals) - 1)] - vals[lo]) * (i - lo)


def p50(values):
    """50th percentile (median) of numeric values, None if empty."""
    return quantile(values, 0.5)


def pct(v, spec_or_q=".0f"):
    """Format a percentage string ('42%'), or if given a list/tuple, compute its quantile (like compare.pct)."""
    if isinstance(v, (list, tuple, set)):
        q = spec_or_q if isinstance(spec_or_q, (int, float)) else 0.5
        return quantile(v, q)
    return DASH if v is None else format(v, spec_or_q) + "%"


def delta_cell(value, baseline, higher_is_better=None, unit="%", approx=False):
    """A signed change of `value` against `baseline` in a styled Cell."""
    from .results import delta_cell as _dc
    return _dc(value, baseline, higher_is_better=higher_is_better, unit=unit, approx=approx)


def signed_pct(new, old):
    """Relative change of new against old, always signed: +12.3%, -4.0%."""
    if new is None or old is None or not old:
        return DASH
    return "%+.1f%%" % (100 * (new / old - 1))


def signed_pt(new, old):
    return DASH if new is None or old is None else "%+.1f pt" % (new - old)


def ctx_k(n):
    """A context size in k (k = 1024): 262144 -> '256k'."""
    return "%dk" % round(n / 1024) if n else DASH


def short_names(names):
    """Short labels for several names: the common prefix stripped back to a separator (`ISTA/Qwen-A`, `ISTA/Qwen-B`
    become `A`, `B`). One name, no common prefix, or a name that would become empty: unchanged."""
    names = list(names)
    if len(names) < 2:
        return names
    p = names[0]
    for n in names[1:]:
        while not n.startswith(p):
            p = p[:-1]
    if p in names:  # one name is the prefix of another: stripping would mangle it
        return names
    cut = max((p.rfind(c) for c in _BOUNDARY), default=-1) + 1
    out = [n[cut:] for n in names]
    return names if not all(out) else out


# Threshold colors for agentic coding (terminal only): (good, acceptable) floors for tok/s; cache hit % tiers.
DECODE_TPS = (60, 30)
PREFILL_TPS = (200, 100)
CACHE_PCT = (80, 50, 20)


def speed_style(v, floors):
    """Style name for a tok/s value against (good, acceptable) floors; None is dim."""
    if v is None:
        return "dim"
    return "speed_good" if v >= floors[0] else "speed_ok" if v >= floors[1] else "speed_slow"


def cache_style(v):
    """Style name for a cache hit percentage; None is dim."""
    if v is None:
        return "dim"
    hi, mid, low = CACHE_PCT
    return "cache_hi" if v >= hi else "cache_mid" if v >= mid else "cache_low" if v >= low else "cache_bad"
