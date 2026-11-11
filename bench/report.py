"""Report output: the two formats (`tui` box-drawing tables with colors, `markdown`), as blocks that both render.

A block is a markdown-ish string (a `#`/`##`/`###` heading, a paragraph, a `- ` list), a `Line` (a paragraph with
its own markdown and terminal forms), a `Note` (a one-line footnote), or a `Table`. `blocks_markdown` joins them into
valid markdown (tables have a header separator row); `blocks_tui` draws the tables with box characters and the
headings colored.

Table cells are strings or `Cell`s, a string with a style: `id` (bold in markdown, white in the terminal), `bold`,
`dim` (italic / dim: secondary values, units), `code` (inline code / cyan), `good` and `bad` (a delta: bold /
italic in markdown, green / red in the terminal), `pass` `warn` `fail` `skip` (verdicts). A `Cell` may carry a
separate markdown form (`md=`), e.g. tool names in code.

Colors follow the usual rules: none when the output is not a terminal, none with NO_COLOR set, none in markdown.
"""

import os
import re
import shutil
import sys

from . import colors, glyphs, paths
from .colors import Clr
from .fmt import (
    cw,
    fit,
    term_limit,
    vlen,
)

_cw = cw
CHAT_FORMATS = ("chat", "chat-compact")  # a transcript of the replay, then the tables
FORMATS = ("tui", "markdown") + CHAT_FORMATS
STDERR_CHATTER = ("markdown",) + CHAT_FORMATS  # formats whose stdout is a document: discovery and notices go to stderr
_format = "tui"


_nerd_font = None  # None: follow REPLAY_BENCH_NERD_FONT (off when unset)


def set_nerd_font(on):
    """--icons: True turns Nerd Font icons on, --no-nerd-font: False keeps plain Unicode, None follows the environment."""
    global _nerd_font
    _nerd_font = None if on is None else bool(on)


def _icons_wanted():
    if _nerd_font is not None:
        return _nerd_font
    return bool(glyphs.env_setting())


def configure(fmt="tui", stream=None):
    """Select the report format and set colors (and Nerd Font icons) for `stream` (default stdout)."""
    global _format
    if fmt not in FORMATS:
        raise ValueError("unknown report format %r (choose from %s)" % (fmt, ", ".join(FORMATS)))
    _format = fmt
    stream = stream if stream is not None else sys.stdout
    colors.configure(fmt != "markdown" and color_ok(stream))
    glyphs.configure(fmt in CHAT_FORMATS and _icons_wanted() and color_ok(stream))
    colors.configure_live(color_ok(sys.stderr))
    glyphs.configure_live(_icons_wanted() and color_ok(sys.stderr))


def color_ok(stream):
    """True when `stream` is a terminal and NO_COLOR is not set."""
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


REPO = paths.ROOT


def rel_path(path):
    """A path relative to the repository when it is inside it (states/x.jsonl), else as given with home shown as ~."""
    ap = os.path.abspath(path)
    return os.path.relpath(ap, REPO) if ap.startswith(REPO + os.sep) else paths.tilde(path)


def current_format():
    return _format


def is_markdown():
    return _format == "markdown"


def is_chat():
    return _format in CHAT_FORMATS


def error(msg):
    """A user-facing error: one line on stderr."""
    print(f"error: {msg}", file=sys.stderr)


def warn(msg):
    """A warning that does not stop the run: one line on stderr."""
    print(f"warning: {msg}", file=sys.stderr)


def info(text="", **kw):
    """Chatter that is not part of the report (discovery, state path, errors): stdout in tui, stderr in markdown so
    that stdout stays a clean markdown document."""
    print(text, file=sys.stderr if _format in STDERR_CHATTER else sys.stdout, **kw)


DASH = "\u2013"
_STYLE_TUI = {"id": lambda: Clr.BLD + Clr.WHT, "bold": lambda: Clr.BLD, "dim": lambda: Clr.DIM,
              "code": lambda: Clr.CYN, "good": lambda: Clr.BGRN, "bad": lambda: Clr.BRED,
              "pass": lambda: Clr.BGRN, "warn": lambda: Clr.BYEL, "fail": lambda: Clr.BRED,
              "skip": lambda: Clr.DIM,
              "speed_good": lambda: Clr.BCYN, "speed_ok": lambda: Clr.BYEL, "speed_slow": lambda: Clr.RED,
              "cache_hi": lambda: Clr.BGRN, "cache_mid": lambda: Clr.BYEL, "cache_low": lambda: Clr.YEL,
              "cache_bad": lambda: Clr.RED}
_STYLE_MD = {"id": "**%s**", "bold": "**%s**", "dim": "*%s*", "code": "`%s`", "good": "**%s**", "bad": "*%s*",
             "fail": "**%s**", "skip": "*%s*"}



def table_width(w):
    """Display columns of a boxed table with column widths `w` (None: a dropped column)."""
    w = [x for x in w if x is not None]
    return sum(w) + 3 * len(w) + 1


def fit_columns(header, w, limit, shrink=(), drop=(), late=()):
    """Column widths `w` made to fit `limit` display columns, never wider: each (name, minimum) of `shrink` is narrowed
    towards its minimum in turn; when that is not enough the `drop` columns go one by one (width None), the shrunk
    ones getting their room back after each; only then the `late` ones (ids that should stay whole) shrink. Last resort: the widest column loses a column at a time."""
    w = list(w)
    if not limit:
        return w
    names = list(header)

    def squeeze(base, order=shrink):
        cur = list(base)
        for name, lo in order:
            i = names.index(name) if name in names else None
            if i is None or cur[i] is None:
                continue
            over = table_width(cur) - limit
            if over > 0:
                cur[i] = max(min(lo, cur[i]), cur[i] - over)
        return cur

    cur = squeeze(w)
    for name in drop:
        if table_width(cur) <= limit:
            return cur
        if name in names and w[names.index(name)] is not None:
            w[names.index(name)] = None
            cur = squeeze(w)
    cur = squeeze(cur, late)
    while table_width(cur) > limit:  # last resort: shave the widest column
        live = [i for i, x in enumerate(cur) if x is not None and x > 1]
        if not live:
            break
        cur[max(live, key=lambda i: cur[i])] -= 1
    return cur


class Cell(str):
    """A table cell: the plain text, a style, and optionally its own markdown form."""

    def __new__(cls, text="", style="", md=None):
        obj = super().__new__(cls, text)
        obj.style, obj.md_text = style, md
        return obj

    def markdown(self):
        if self.md_text is not None:
            return self.md_text.replace("|", "\\|").replace("\n", " ")
        text = str(self).replace("|", "\\|").replace("\n", " ").replace("*", "\\*")
        fmt = _STYLE_MD.get(self.style)
        return fmt % text if fmt and text.strip() not in ("", DASH, "-") else text

    def tui(self, text):
        col = _STYLE_TUI.get(self.style, lambda: "")()
        return f"{col}{text}{Clr.RST}" if col else text


class Line(str):
    """A paragraph block with its own markdown and terminal forms (both default to the plain text)."""

    def __new__(cls, text="", md=None, tui=None):
        obj = super().__new__(cls, text)
        obj.md_text, obj.tui_text = md, tui
        return obj

    def markdown(self):
        return str(self) if self.md_text is None else self.md_text

    def tui(self):
        return str(self) if self.tui_text is None else self.tui_text


class Note(Line):
    """A one-line footnote or legend: italic in markdown, dim in the terminal."""

    def markdown(self):
        if self.md_text is not None:
            return self.md_text
        text = str(self).replace("*", "\\*")
        if "\n" not in text:
            return "*%s*" % text
        return "\n".join("*%s*" % l if l.strip() else "" for l in text.split("\n"))

    def tui(self):
        return f"{Clr.DIM}{self}{Clr.RST}"


_NUM = re.compile(r"^(base|[-+\u00b1\u2265<>\u2264]?\s*[\d.,]+\s*[a-zA-Z%]{0,3}(\s*pt)?|[-\u2013\u2014])$")
_SIGNED = re.compile(r"^[+\-\u2212]\d")
_ZERO = re.compile(r"^[+\-\u2212]0(\.0*)?\s*(%|pt)?$")


class Table:
    """header: column names; rows: lists of cells (strings or Cells); aligns: per column 'l'/'r' (default: right when
    every cell looks numeric); dirs: {column index: 'up' | 'down'} styles a delta column good/bad by sign (up: + is
    good); dim_cols: columns of secondary values (dim)."""

    def __init__(self, header, rows, aligns=None, dirs=None, dim_cols=(), shrink=None, drop=None, late=None):
        self.header = [str(h) for h in header]
        self.dirs = dirs or {}
        self.rows = [[self._cell(i, c) for i, c in enumerate(r)] for r in rows]
        self.dim_cols = set(dim_cols)
        if aligns is None:
            aligns = ["r" if self.rows and all(_NUM.match(r[i].strip()) for r in self.rows) else "l"
                      for i in range(len(self.header))]
        self.aligns = aligns
        self.shrink = shrink
        self.drop = drop
        self.late = late

    def _cell(self, i, c):
        if isinstance(c, Cell) and c.style:
            return c
        text = str(c)
        t = text.strip()
        d = self.dirs.get(i)
        if i < len(self.header) and (d or self.header[i].startswith("\u0394")):
            if d and _SIGNED.match(t) and not _ZERO.match(t):
                good = (t[0] == "+") == (d == "up")
                return Cell(text, "good" if good else "bad")
            return Cell(text, "dim")
        return c if isinstance(c, Cell) else Cell(text)

    def _style(self, i, c):
        if c.style or c.md_text is not None:
            return c
        if i in self.dim_cols:
            return Cell(str(c), "dim")
        return c

    def widths(self, markdown=False):
        cells = lambda r: [self._style(i, c) for i, c in enumerate(r)]
        text = (lambda c: c.markdown()) if markdown else str
        return [max([vlen(h)] + [vlen(text(cells(r)[i])) for r in self.rows]) for i, h in enumerate(self.header)]

    def markdown(self):
        w = self.widths(markdown=True)
        cell = lambda t, i: (" " * (w[i] - vlen(t)) + t) if self.aligns[i] == "r" else (t + " " * (w[i] - vlen(t)))
        esc = lambda c: c.replace("|", "\\|").replace("\n", " ")
        line = lambda cells: "| " + " | ".join(cell(c, i) for i, c in enumerate(cells)) + " |"
        sep = "|" + "|".join(("-" * (x + 1) + ":") if a == "r" else ("-" * (x + 2)) for x, a in zip(w, self.aligns)) + "|"
        body = [line([self._style(i, c).markdown() for i, c in enumerate(r)]) for r in self.rows]
        return "\n".join([line([esc(h) for h in self.header]), sep] + body)

    # -- terminal: whole table, or the parts of a table whose rows arrive one by one ------------------------
    MID_COLS = ("Case", "Stream")
    SHRINK = (("Reason / Preview", 24), ("Stream", 10))
    LATE = (("Reason / Preview", 8), ("Case", 14))  # after every droppable column is gone
    # after the reason column has shrunk, the baseline columns go first: the prefill delta, then the baseline's verdict
    DROP = ("\u0394Prefill", "Base", "C%", "Out", "Toks", "Prefill", "In", "Model", "Time", "\u0394Time", "Stream")

    def fitted(self, w, limit=None):
        """Column widths `w` narrowed to the terminal (`limit`, default stdout's): the reason column first, then the
        Stream and Case ids (ellipsis in the middle), then columns drop (DROP) and the last resort. Never wider."""
        limit = term_limit() if limit is None else limit
        if self.drop is not None:
            return fit_columns(self.header, w, limit, self.shrink or (), self.drop, self.late or ())
        if "Reason / Preview" in self.header:
            return fit_columns(self.header, w, limit, self.SHRINK, self.DROP, self.LATE)
        text = [i for i, a in enumerate(self.aligns) if a == "l"]
        wide = max(text, key=lambda i: w[i]) if text else None  # any other table: its widest text column gives way
        return fit_columns(self.header, w, limit, [(self.header[wide], 12)] if wide is not None else ())

    def _rule(self, w, l, m, r):
        line = m.join("\u2500" * (x + 2) for x in w if x is not None)
        return f"{Clr.BLD}{Clr.CYN}{l}{line}{r}{Clr.RST}"

    def tui_top(self, w):
        return self._rule(w, "\u250c", "\u252c", "\u2510")

    def tui_head(self, w):
        bar = f"{Clr.BLD}{Clr.CYN}\u2502{Clr.RST}"
        short = lambda h, x: "Reason" if h == "Reason / Preview" and x < vlen(h) else h
        cells = [f" {Clr.BLD}{fit(short(h, w[i]), w[i], self.aligns[i])}{Clr.RST} "
                 for i, h in enumerate(self.header) if w[i] is not None]
        return bar + bar.join(cells) + bar + "\n" + self._rule(w, "\u251c", "\u253c", "\u2524")

    def tui_row(self, row, w):
        bar = f"{Clr.BLD}{Clr.CYN}\u2502{Clr.RST}"
        cells = []
        for i, c in enumerate(row):
            if w[i] is None:
                continue
            c = self._style(i, c if isinstance(c, Cell) else Cell(str(c)))
            if not c.style and i == 0:
                c = Cell(str(c), "id")
            mid = i < len(self.header) and self.header[i] in self.MID_COLS
            cells.append(" " + c.tui(fit(str(c), w[i], self.aligns[i], mid)) + " ")
        return bar + bar.join(cells) + bar

    def tui_sep(self, w):
        return self._rule(w, "\u251c", "\u253c", "\u2524")

    def tui_bottom(self, w):
        return self._rule(w, "\u2514", "\u2534", "\u2518")

    def tui(self, w=None):
        w = self.fitted(w or self.widths())
        return "\n".join([self.tui_top(w), self.tui_head(w)] + [self.tui_row(r, w) for r in self.rows]
                         + [self.tui_bottom(w)])


def blocks_markdown(blocks):
    return "\n\n".join(b.markdown() if hasattr(b, "markdown") else str(b) for b in blocks)


def _rule_width():
    try:
        return max(40, min(120, shutil.get_terminal_size((100, 24)).columns))
    except (ValueError, OSError):
        return 100


def _tui_text(b):
    """A block as terminal text: markdown headings become TUI headings (`# Title` a double rule, `## Section` a bold
    rule, `### Part` a cyan marker), never markdown syntax."""
    if isinstance(b, Line):
        return b.tui()
    b = str(b)
    m = re.match(r"^(#{1,3}) (.*)$", b)
    if m:
        n, title = len(m.group(1)), m.group(2)
        w = _rule_width()
        if n == 1:
            bar = "\u2550" * w
            return f"{Clr.BLD}{Clr.MAG}{title}{Clr.RST}\n{Clr.MAG}{bar}{Clr.RST}"
        if n == 2:
            head = "\u2501\u2501 %s " % title
            bar = "\u2501" * max(3, w - vlen(head))
            return f"{Clr.BLD}{Clr.MAG}{head}{bar}{Clr.RST}"
        return f"{Clr.BLD}{Clr.CYN}\u25b8 {title}{Clr.RST}"
    return b


def blocks_tui(blocks):
    """The blocks as terminal text; a footnote sits directly under what it explains."""
    out = ""
    for b in blocks:
        text = b.tui() if isinstance(b, Table) else _tui_text(b)
        out += (("\n" if isinstance(b, Note) else "\n\n") if out else "") + text
    return out
