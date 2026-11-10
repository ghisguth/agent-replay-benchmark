"""Chat transcripts of a replay: `--report-format chat` (how an omp session looks in the terminal) and `chat-compact`
(how Claude Code shows a compact transcript). Pure renderers: a transcript object takes one finished result at a time
and returns the text to print; the format classes in runner.py print it.

Per request: the model's answer (prose, tool calls), the omp stats line, a verdict. The tool results shown under
bash calls are the *recorded* ones: replay executes nothing, so the fixture's next request of the same stream holds the
result messages of the recorded call. They are shown only where the model's call equals the recorded call (the intent
argument `i` is ignored); otherwise the recorded call is shown dimmed under the model's.

omp stats line (packages/tui/src/overlays/usage-row.ts, formatUsageRow / usageRowSpecs): wall-clock timestamp
`YYYY-MM-DD HH:MM:SS`, `Δ <turn time>` (user prompt to the response's completion, formatDuration), new input tokens
(input + cache writes: the prompt minus what was read from the cache), output tokens, cache-read tokens (only when
> 0), time to first token (only when > 0, seconds), output tokens per second over the whole request (only when the
request took > 100 ms). Numbers go through formatNumber (1.2K, 123K), durations through formatDuration.
"""

import difflib
import json
import re
import textwrap

from . import glyphs, paths
from .colors import Clr
from .fmt import (
    clip,
    cw,
    fmt_clock,
    fmt_duration,
    fmt_number,
    hard_wrap,
    live_cw,
    live_vlen,
    term_width,
    vlen,
)
from .live import (
    DROP_ORDER,
    _live_line,
    _live_render,
    _prefill_seg,
    _word,
    bar,
    fmt_live_tokens,
    live_compose,
    live_label,
    live_line,
    live_parts,
    live_render,
    prefill_info,
    stream_color,
    term_cols,
)

FIXTURE_DIRS = [paths.FIXTURES_DIR]
OUTPUT_ROWS = 10   # omp shows the last 10 lines of a bash output
USER_ROWS = 8


def recorded_text(text, width):
    """`(no call)`, or a one-line preview of the text the recorded answer had instead of a call."""
    text = " ".join(str(text or "").split())
    return 'text "%s"' % clip(text, max(10, width - 7)) if text else "(no call)"


def message_text(msg):
    """The text of a chat message whose content is a string or a list of parts."""
    c = (msg or {}).get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


_INLINE = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`")


def inline_style(line):
    return _INLINE.sub(lambda m: f"{Clr.BLD}{m.group(1)}{Clr.RST}" if m.group(1) else f"{Clr.CYN}{m.group(2)}{Clr.RST}", line)


def render_prose(text, width, indent=""):
    """Markdown-ish text as wrapped lines: headings bold, bullets with a hanging indent, code fences and code dim,
    `code` and **bold** styled. Every line is prefixed with `indent`."""
    out, in_code = [], False
    w = max(20, width - len(indent))
    for line in (text or "").strip("\n").split("\n"):
        line = line.rstrip()
        if line.lstrip().startswith("```"):
            in_code = not in_code
            out.append(f"{indent}{Clr.DIM}{line}{Clr.RST}")
        elif in_code:
            out += [f"{indent}{Clr.DIM}{p}{Clr.RST}" for p in hard_wrap(line, w)]
        elif not line.strip():
            out.append("")
        elif re.match(r"^#{1,6}\s", line):
            out += [f"{indent}{Clr.BLD}{Clr.MAG}{p}{Clr.RST}" for p in textwrap.wrap(line, w) or [line]]
        elif line.lstrip().startswith("|"):
            out += [f"{indent}{p}" for p in hard_wrap(line, w)]
        else:
            m = re.match(r"^(\s*(?:[-*+]|\d+[.)])\s+)(.*)$", line)
            lead, body = (m.group(1), m.group(2)) if m else ("", line)
            pieces = textwrap.wrap(body, max(10, w - len(lead))) or [""]
            for i, p in enumerate(pieces):
                out.append(indent + (lead if i == 0 else " " * len(lead)) + inline_style(p))
    return out


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------

def _rule(left, label, right, width, color):
    label = (" %s " % label) if label else ""
    body = "───" + label if label else ""
    body = body[: width - 2]
    return f"{color}{left}{body}{'─' * (width - 2 - vlen(body))}{right}{Clr.RST}"


def box(width, rows, sections=(), title=None, footer=None, color=None):
    """An omp tool box. `rows` are the body lines (already at most width-4 columns); `sections` [(label, lines)]
    follow behind a `├─── label ───┤` divider; `title` sits in the top border, `footer` in the bottom one."""
    color = Clr.DIM if color is None else color
    bar = f"{color}│{Clr.RST}"
    line = lambda s: f"{bar} {s}{' ' * max(0, width - 4 - vlen(s))} {bar}"
    out = [_rule("╭", title, "╮", width, color)] + [line(r) for r in rows]
    for label, lines in sections:
        out.append(_rule("├", label, "┤", width, color))
        out += [line(r) for r in lines]
    out.append(_rule("╰", footer, "╯", width, color))
    return out


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------

# Other agents' tool names (Qwen Code) mapped to the omp ones the renderers know.
ALIASES = {"run_shell_command": "bash", "read_file": "read", "write_file": "write", "replace": "edit",
           "todo_write": "todo", "grep_search": "grep", "list_directory": "glob", "web_fetch": "fetch",
           "read_many_files": "read"}


def call_name(call):
    name = (call.get("function") or {}).get("name", "") or "?"
    return ALIASES.get(name, name)


def arg_path(args):
    """The file or target of a call: path, file_path, absolute_path, skill or url."""
    for k in ("path", "file_path", "absolute_path", "skill", "url"):
        if args.get(k):
            return str(args[k])
    return ""


def call_args(call):
    """The arguments dict of a tool call, or None when the arguments are not a JSON object."""
    raw = (call.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        args = json.loads(raw or "{}")
    except ValueError:
        return None
    return args if isinstance(args, dict) else None


def task_list(args):
    """The subagent specs of a `task` call: `tasks` is a list, or a JSON string holding one (omp sends both)."""
    args = args or {}
    tasks = args.get("tasks")
    if isinstance(tasks, str):
        try:  # raw_decode: models sometimes leave a stray bracket after the list
            tasks = json.JSONDecoder().raw_decode(tasks.strip())[0]
        except ValueError:
            return []
    if tasks is None and (args.get("name") or args.get("agent")):
        tasks = [args]  # a model that put one spec's fields at the top level
    if isinstance(tasks, dict):
        tasks = [tasks]
    return [t for t in tasks if isinstance(t, dict)] if isinstance(tasks, list) else []


def same_call(a, b):
    """True when two calls have the same tool and arguments (the intent argument `i` does not count)."""
    if call_name(a) != call_name(b):
        return False
    x, y = call_args(a), call_args(b)
    if x is None or y is None:
        return (a.get("function") or {}).get("arguments") == (b.get("function") or {}).get("arguments")
    strip = lambda d: {k: v for k, v in d.items() if k != "i"}
    return strip(x) == strip(y)


_SEPS = (("&&", "&&"), ("||", "||"), (";", ";"))


def split_shell(command):
    """A shell command cut at top-level `;`, `&&`, `||` (quotes respected; heredocs left whole) into
    [(text, separator before it)]."""
    if "<<" in command:
        return [(command, "")]
    parts, cur, quote, sep, i = [], "", None, "", 0
    while i < len(command):
        ch = command[i]
        if quote:
            cur += ch
            if ch == "\\" and quote == '"' and i + 1 < len(command):
                cur += command[i + 1]
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            cur += ch
        elif ch == "\\" and i + 1 < len(command):
            cur += command[i:i + 2]
            i += 1
        else:
            hit = next((s for tok, s in _SEPS if command.startswith(tok, i)), None)
            if hit:
                parts.append((cur.strip(), sep))
                cur, sep = "", hit
                i += len(hit) - 1
            else:
                cur += ch
        i += 1
    parts.append((cur.strip(), sep))
    return [(t, s) for t, s in parts if t]


def diff_rows(old, new, max_rows=40):
    """[(sign, line number, text)] of an edit's old/new text. Numbers count within the snippet (replay has no file)."""
    a, b = old.split("\n"), new.split("\n")
    rows = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            run = [(" ", i + 1, a[i]) for i in range(i1, i2)]
            rows += run if len(run) <= 6 else run[:2] + [(" ", None, "…")] + run[-2:]
        else:
            rows += [("-", i + 1, a[i]) for i in range(i1, i2)] + [("+", j + 1, b[j]) for j in range(j1, j2)]
    return rows


def _tail(text, n=OUTPUT_ROWS):
    lines = (text or "").rstrip("\n").split("\n") if (text or "").strip() else []
    if len(lines) > n:
        return ["… (%d earlier lines, showing %d of %d)" % (len(lines) - n, n, len(lines))] + lines[-n:]
    return lines


def _first_line(text, width):
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return clip(lines[0], width) if lines else ""


def describe_call(call, width=100):
    """One plain line naming a call, e.g. `bash: git diff` (for 'recorded: ...' and summaries)."""
    name, args = call_name(call), call_args(call)
    if args is None:
        return clip("%s: (unparsable arguments)" % name, width)
    key = {"bash": "command", "read": "path", "edit": "path", "write": "path", "grep": "pattern", "glob": "path",
           "find": "query", "todo": "op", "wait": "i", "skill": "skill", "fetch": "url"}.get(name)
    if key in ("path", "skill", "url") and args.get(key) is None:
        args = dict(args, **{key: arg_path(args)})
    if name == "task":
        key_text = ", ".join(str(t.get("name") or t.get("agent") or "?") for t in task_list(args))
    else:
        key_text = str(args.get(key, "")) if key else ", ".join("%s=%s" % kv for kv in list(args.items())[:2])
    return clip("%s: %s" % (name, key_text), width)


def _read_target(args):
    path = arg_path(args)
    off, lim = args.get("offset"), args.get("limit")
    if isinstance(off, int) and isinstance(lim, int) and ":" not in path:
        return "%s:%d-%d" % (path, off, off + lim - 1)
    if isinstance(off, int) and ":" not in path:
        return "%s:%d-" % (path, off)
    return path


def render_call(call, result, width):
    """The lines of one tool call as omp draws it. `result` is the recorded result text (a str) or None."""
    name, args = call_name(call), call_args(call)
    inner = width - 4
    dim = lambda s: f"{Clr.DIM}{s}{Clr.RST}"
    ico = glyphs.prefix(name)
    # with an icon the icon is the label; the word stays in the plain fallback
    line = lambda label, text, intent=None: "  %s%s%s%s%s" % (
        ico, Clr.BLD, "" if ico else label, Clr.RST,
        (" " if not ico else "") + clip(text, width - 6 - len(label) - (len(intent) + 3 if intent else 0)) + (
            dim("  · " + intent) if intent else ""))
    tail = lambda: [dim("  → " + clip(_first_line(result, width - 20) + ("  (%d lines)" % len(result.strip().splitlines())
                                                                       if len(result.strip().splitlines()) > 1 else ""), width - 6))] \
        if result and result.strip() else []
    if args is None:
        return [line(name, "invalid arguments: " + str((call.get("function") or {}).get("arguments")))]
    intent = args.get("i") if isinstance(args.get("i"), str) else None
    if name == "bash":
        rows, parts = [], split_shell(str(args.get("command", "")))
        for text, sep in parts:
            if sep:
                rows.append("  " + sep)
            for ln in text.split("\n"):
                rows += ["  " + p for p in hard_wrap(ln, inner - 2)]
        if len(parts) == 1 and rows:
            rows[0] = "$ " + rows[0][2:]
        if len(rows) > 14:
            rows = rows[:14] + ["  … (%d more lines)" % (len(rows) - 14)]
        sections = []
        if result is not None:
            sections.append(("Output", [clip(l.expandtabs(4), inner) for l in _tail(result)]))
        foot = []
        if args.get("timeout") is not None:
            foot.append("Timeout: %ss" % args["timeout"])
        if args.get("async"):
            foot.append("Async")
        return box(width, rows, sections, title=ico.strip() or None, footer=" · ".join(foot) or None)
    if name == "edit" and isinstance(args.get("old_string"), str) and isinstance(args.get("new_string"), str):
        rows = diff_rows(args["old_string"], args["new_string"])
        add, rem = sum(r[0] == "+" for r in rows), sum(r[0] == "-" for r in rows)
        g = max([len(str(r[1])) for r in rows if r[1]] + [1])
        body = []
        for sign, num, text in rows[:40]:
            col = Clr.BGRN if sign == "+" else Clr.BRED if sign == "-" else Clr.DIM
            gutter = "%s%s│" % (sign, str(num if num else "").rjust(g))
            pieces = hard_wrap(text, inner - g - 2)
            body += [f"{col}{gutter if k == 0 else ' ' * (g + 1) + '│'}{p}{Clr.RST}" for k, p in enumerate(pieces)]
        if len(rows) > 40:
            body.append(dim("… (%d more diff lines)" % (len(rows) - 40)))
        title = "%s%s ⟨+%d/-%d⟩%s" % (ico or "Edit:  ", clip(arg_path(args), inner - 30), add, rem,
                                           " (all)" if args.get("replace_all") else "")
        return box(width, body, title=title) + tail()
    if name == "write" and isinstance(args.get("content"), str):
        lines = args["content"].split("\n")
        g = len(str(min(len(lines), 10)))
        body = [f"{Clr.BGRN}+{str(i + 1).rjust(g)}│{clip(l.expandtabs(4), inner - g - 2)}{Clr.RST}" for i, l in enumerate(lines[:10])]
        if len(lines) > 10:
            body.append(dim("… (%d more lines)" % (len(lines) - 10)))
        return box(width, body, title="%s%s ⟨+%d⟩" % (ico or "Write:  ", clip(arg_path(args), inner - 24), len(lines))) + tail()
    if name == "read":
        return [line("Read", _read_target(args), intent)] + tail()
    if name == "grep":
        return [line("Grep", "‹%s› %s" % (args.get("pattern", ""), args.get("path", "")), intent)] + tail()
    if name in ("glob", "find"):
        return [line(name.capitalize(), str(args.get("path") or args.get("query") or ""), intent)] + tail()
    if name == "wait":
        return [line("Wait", intent or "")] + tail()
    if name == "todo":
        out = [line("Todo", " ".join(str(x) for x in (args.get("op"), args.get("task")) if x), intent)]
        for ph in (args.get("list") or [])[:6]:
            if isinstance(ph, dict):
                out.append("    %s %s" % (ph.get("phase", ""), dim("· " + clip("; ".join(map(str, ph.get("items") or [])), width - 20))))
        return out
    if name == "task":
        subs = task_list(args)
        out = [line("Task", "%d subagent%s" % (len(subs), "" if len(subs) == 1 else "s"), intent)]
        for t in subs:
            who = str(t.get("name") or "?") + (" [%s]" % t["agent"] if t.get("agent") else "")
            out.append("    ▸ %s%s%s" % (who, Clr.RST, "  " + dim(clip(_first_line(t.get("task", ""), width), width - 12 - len(who)))))
        return out
    return [line(name, json.dumps({k: v for k, v in args.items() if k != "i"}, ensure_ascii=False), intent)] + tail()


# ---------------------------------------------------------------------------
# Stats line, verdict
# ---------------------------------------------------------------------------

def stats_line(m, when, delta_ms):
    """The omp usage row for one request (see the module docstring)."""
    prompt, cached = m.get("prompt_tokens") or 0, m.get("cached_tokens") or 0
    out_tokens, total_ms, ttft = m.get("tokens") or 0, m.get("total_ms") or 0, m.get("ttft_ms") or 0
    parts = [when.strftime("%Y-%m-%d %H:%M:%S"), "Δ " + fmt_duration(delta_ms)]
    if prompt:
        parts.append(fmt_number(max(0, prompt - cached)))
    parts.append(fmt_number(out_tokens))
    if cached > 0:
        parts.append(fmt_number(cached))
    if ttft > 0:
        parts.append("%.1fs" % (ttft / 1000))
    if total_ms > 100 and out_tokens > 0:
        parts.append("%.1f/s" % (out_tokens / total_ms * 1000))
    return "   ".join(parts)


STAT_FIELDS = (("clock", 19, "DIM"), ("delta", 9, "DIM"), ("in", 6, "BCYN"), ("out", 6, "BGRN"),
               ("cache", 6, "BBLU"), ("ttft", 6, "BYEL"), ("speed", 8, "BMAG"))


def stat_values(m, when, delta_ms):
    """The values of the omp usage row, one per STAT_FIELDS key (None where omp leaves the field out)."""
    prompt, cached = m.get("prompt_tokens") or 0, m.get("cached_tokens") or 0
    out_tokens, total_ms, ttft = m.get("tokens") or 0, m.get("total_ms") or 0, m.get("ttft_ms") or 0
    return {"clock": when.strftime("%Y-%m-%d %H:%M:%S"),
            "delta": ("" if glyphs.enabled else "Δ ") + fmt_duration(delta_ms),
            "in": fmt_number(max(0, prompt - cached)) if prompt else None, "out": fmt_number(out_tokens),
            "cache": fmt_number(cached) if cached > 0 else None, "ttft": "%.1fs" % (ttft / 1000) if ttft > 0 else None,
            "speed": "%.1f/s" % (out_tokens / total_ms * 1000) if total_ms > 100 and out_tokens > 0 else None}


def stats_row(m, when, delta_ms):
    """The usage row as fixed-width columns: every field right-aligned in its own width (plus its icon), a missing field
    keeps its blank slot, so the rows of a transcript stack into columns."""
    vals, cells = stat_values(m, when, delta_ms), []
    for key, width, col in STAT_FIELDS:
        icon = glyphs.g(key)
        slot = width + (2 if icon else 0)
        v = vals.get(key)
        if v is None:
            cells.append(" " * slot)
            continue
        col = getattr(Clr, col)
        body = (icon + " " if icon else "") + v.rjust(width)
        cells.append(f"{col}{body}{Clr.RST}" if col else body)
    return "  ".join(cells)


_BADGE_COLORS = {"PASS": "BGRN", "WARN": "BYEL", "FAIL": "BRED"}


def badge(v):
    """The verdict as a fixed-width colored badge (` PASS `, with its icon when icons are on)."""
    v = v or "?"
    col = getattr(Clr, _BADGE_COLORS[v], Clr.DIM) if v in _BADGE_COLORS else Clr.DIM
    icon = glyphs.g(v)
    return f"{Clr.REV}{col} {icon + ' ' if icon else ''}{v.ljust(4)} {Clr.RST}"


def verdict_text(m, width):
    """(`R003_main_e0_t001 PASS`, reason or '') colored by the verdict."""
    v = m.get("eval_result") or "?"
    reason = m.get("fail_reason") or ""
    return f"{badge(v)} {Clr.BLD}{m.get('id', '?')}{Clr.RST}", ("" if v == "PASS" else clip(reason, width - 8))


# ---------------------------------------------------------------------------
# Transcripts (bench/transcript.py)
# ---------------------------------------------------------------------------

from .transcript import (  # noqa: E402
    ChatTranscript,
    CompactTranscript,
    Transcript,
    load_header_fixture,
    tidy,
)

__all__ = [
    "FIXTURE_DIRS",
    "OUTPUT_ROWS",
    "USER_ROWS",
    "badge",
    "stats_line",
    "stats_row",
    "render_call",
    "same_call",
    "verdict_text",
    "cw",
    "vlen",
    "live_cw",
    "live_vlen",
    "hard_wrap",
    "clip",
    "fmt_number",
    "fmt_clock",
    "fmt_duration",
    "term_width",
    "stream_color",
    "bar",
    "term_cols",
    "_word",
    "fmt_live_tokens",
    "prefill_info",
    "_prefill_seg",
    "live_parts",
    "DROP_ORDER",
    "live_render",
    "_live_render",
    "live_compose",
    "live_label",
    "live_line",
    "_live_line",
    "load_header_fixture",
    "tidy",
    "Transcript",
    "ChatTranscript",
    "CompactTranscript",
]




