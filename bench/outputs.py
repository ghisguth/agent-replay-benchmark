"""Report format outputs: TUI, Markdown, Chat, ChatCompact, and renderer dispatch."""

import contextlib
import io
import sys
import textwrap
import time

from . import chat, colors, compare, perf, report, state
from . import fmt as F
from . import results as R
from .cases import replay as replay_cases
from .colors import Clr
from .report import Table


class _chatter:
    """Context for output that is not part of the report (discovery): stdout in tui, stderr in the chat formats, held
    back in markdown (the `## Run` section describes the run; `.text` is what was printed, for when nothing runs)."""

    def __enter__(self):
        fmt = report.current_format()
        self.buf = io.StringIO() if fmt == "markdown" else None
        self._cm = (contextlib.redirect_stdout(self.buf) if self.buf is not None else
                    contextlib.redirect_stdout(sys.stderr) if fmt in report.STDERR_CHATTER else contextlib.nullcontext())
        return self._cm.__enter__()

    def __exit__(self, *a):
        return self._cm.__exit__(*a)

    @property
    def text(self):
        return self.buf.getvalue() if self.buf is not None else ""


def legend(shows_delta, spec=False, width=None, est=False):
    """The closing legend: every definition the tables rely on, once, only with what the report shows (Δ with a
    baseline, the cycle terms with speculative-decoding figures). The case table's Decode / Prefill are the server's own
    tok/s; its Total row is sums for TTFT, Time, In and Out, medians for Decode and Prefill, cached ÷ prompt for C%."""
    w = width or F.term_width()
    ideas = [
        " · ".join(["TTFT = time to first token", "C% = share of the prompt served from cache",
                    "P/W/F = PASS/WARN/FAIL counts", "k = 1024 (context sizes, prompt and append sizes)"]),
        "case table: Decode / Prefill are the server's own tok/s; Total row: TTFT, Time, In and Out are sums "
        "(TTFT = time spent waiting for first tokens), Decode and Prefill are medians, C% is cached ÷ prompt.",
        "~ = measured on the client clock (the server reports no speed of its own).",
        "Metrics, context and comparison tables use the client clock: decode tok/s = output tokens ÷ (request time − TTFT),",
        "end-to-end = output tokens ÷ request time, prefill = new prompt tokens ÷ TTFT (native = the server's own figures).",
        "prefill by append size: requests with more than 16 new tokens (prompt − cached).",
    ]
    if est:
        ideas.append("no cache reported: new tokens ≈ prompt − previous prompt in the stream.")
    if spec:
        ideas.append("a cycle is one draft-and-verify step: tok/cycle = tokens kept per step, cycles/s (requests with "
                     "≥ %d output tokens) is the raw engine speed." % perf.MIN_CYCLE_TOKENS)
    if shows_delta:
        ideas += [
            "Δ = relative change against the baseline, bold when better, italic when worse (times: + is slower; tok/s: + is faster);",
            "case table with a baseline: ΔTime, ΔDecode and ΔPrefill are the change of that request,",
            "Base is the baseline's verdict, followed by its first step when it took other tools (= same verdict and tools, – no baseline result).",
        ]
    lines = []
    for idea in ideas:
        wrapped = textwrap.wrap(idea, width=w)
        lines.extend(wrapped or [idea])
    return "\n".join(lines)


class TuiOutput:
    """The terminal report: a header, then per model a table whose rows appear as the cases finish, the verdict
    counts, metrics, anomalies; at the end the comparison and performance tables."""

    layout_needs_stats = True  # the table columns depend on whether the server reports token counts

    base_run, base_line = None, None  # this model's --compare-state BaselineRun (only when it has results for some of the cases), the `compared with` line

    def __init__(self):
        self.multi, self.labels, self.layout, self.widths = False, {}, None, None
        self.infos = {}

    def use_baseline(self, base_run, cases):
        """Take the baseline of the model about to run (a BaselineRun, or None); the per-case columns appear only when it
        has results for some of the cases. Sets base_run and the `compared with` line."""
        self.base_run, self.base_line = None, None
        if base_run is None:
            return False
        ids = list(dict.fromkeys(c["id"] for c in cases))
        self.base_line = R.compared_line(base_run, ids)
        if base_run.common(ids):
            self.base_run = base_run
        return self.base_run is not None

    def set_fixture(self, fixture):
        pass

    def live(self, tc, label, progress=None):
        """(label, formatter) of the spinner (stderr) while a request runs: the run bar, bold id, colored stream and
        values, context gauge, cache share, elapsed time and ETA."""
        with colors.live():
            label = chat.live_label(tc, (progress or {}).get("n_ctx"), progress)
        return (label,
                lambda lbl, st: chat.live_compose(tc, st, time.time(), progress, chat.term_cols() - 4))

    def begin(self, infos, targets, n_cases, mdcat):
        """Start of the report. infos: results.run_info per target."""
        self.multi = len(targets) > 1
        names = compare.labels(compare.run_labels([
            compare.entry(t["backend"], t["name"], {}, note=i.get("note"), server=i.get("server"), preset=i.get("preset"),
                          model_info=t.get("model_info") or i.get("model_info"))
            for t, i in zip(targets, infos)]))
        self.labels = {t["key"]: n for t, n in zip(targets, names)}
        self.infos = {t["key"]: i for t, i in zip(targets, infos)}
        self.header(infos, names, n_cases, mdcat)

    def header(self, infos, names, n_cases, mdcat):
        print(f"{Clr.BLD}{Clr.MAG}{R.headline(infos, n_cases)}{Clr.RST}")

    def model_line(self, target):
        i = self.infos.get(target["key"], {})
        m_name = state.model_label(target["name"], target.get("model_info") or i.get("model_info"))
        more = [x for x in (i.get("server") if self.multi else None, ("ctx %s" % F.ctx_k(i["ctx"])) if self.multi and i.get("ctx") else None,
                            target["backend"]) if x]
        return f"{Clr.BLD}{Clr.BCYN}{m_name}{Clr.RST}" + (f" {Clr.DIM}· {' · '.join(more)}{Clr.RST}" if more else "")

    def target_start(self, target, has_stats, cases, base_run=None):
        """Before a model's first case: its line and the top of the live table."""
        self.layout = R.CaseLayout(has_stats, self.multi, any(c.get("stream") for c in cases),
                                   base=self.use_baseline(base_run, cases))
        self.table = Table(self.layout.header, [], aligns=self.layout.aligns, dim_cols=self.layout.dim_cols)
        label = self.labels.get(target["key"], target["name"])
        self.widths = self.layout.widths([c["id"] for c in cases], [label], [R.stream_label(c) for c in cases],
                                         term=report.term_limit())
        print(f"\n{self.model_line(target)}")
        if self.base_line:
            print(f"{Clr.DIM}{self.base_line}{Clr.RST}")
        print(self.table.tui_top(self.widths))
        print(self.table.tui_head(self.widths))

    def case_done(self, target, metrics, has_stats):
        label = self.labels.get(target["key"], target["name"])
        for row in R.case_rows(self.layout, [metrics], label, self.base_results()):
            print(self.table.tui_row(row, self.widths), flush=True)

    def base_results(self):
        """The baseline's results by request id for case_rows (None without a baseline)."""
        return self.base_run.results if self.base_run else None

    def case_loaded(self, target, metrics, has_stats):
        """A result carried over by --resume: shown like a finished case (the chat transcript has only new requests)."""
        self.case_done(target, metrics, has_stats)

    def target_done(self, target, entry, comparison, has_stats, mdcat, interrupted=None):
        """After a model's last case: close the table, then counts, metrics, context bands, anomalies. `comparison`: a
        baseline.Comparison of this run with its baseline, or None."""
        print(self.table.tui_sep(self.widths))
        for row in R.totals_rows(self.layout, list(entry["results"].values()), comparison):
            print(self.table.tui_row(row, self.widths))
        print(self.table.tui_bottom(self.widths))
        self.tail(entry, comparison, mdcat, interrupted, cases=None, titles=False)

    def tail(self, entry, comparison, mdcat, interrupted, cases, titles):
        compare.emit(compare.run_blocks(entry, comparison, cases=cases, titles=titles, interrupted=interrupted), mdcat)

    def done(self, state_paths, mdcat, interrupted=None, shows_delta=False, spec=False, est=False):
        compare.emit([report.Note(legend(shows_delta, spec, est=est))], mdcat)
        saved = ", ".join(dict.fromkeys(report.rel_path(p) for p in state_paths))
        head = (f"{Clr.BYEL}⚠ interrupted ({interrupted}){Clr.RST}" if interrupted else f"{Clr.BGRN}✔ done{Clr.RST}")
        print(f" {head}" + (f" {Clr.DIM}· state: {saved}{Clr.RST}" if saved else "") + "\n")

    def dry_run(self, selected, summary_line, mdcat):
        compare.emit([replay_cases.dry_run_table(selected), report.Note(summary_line)], mdcat)


class MarkdownOutput(TuiOutput):
    """All output as one markdown document: the run section, then per model its finished section (cases, metrics,
    anomalies) rendered at once. A table cannot be piped row by row, so stdout stays clean and every finished case is
    one line on stderr."""

    def header(self, infos, names, n_cases, mdcat):
        compare.emit(R.run_blocks(infos, names, n_cases), mdcat)

    live_run = False  # the per-case stderr lines belong to live runs, not to a re-displayed state

    def target_start(self, target, has_stats, cases, base_run=None):
        self.layout = R.CaseLayout(has_stats, self.multi, any(c.get("stream") for c in cases), reason_limit=40,
                                   base=self.use_baseline(base_run, cases))
        self.finished = []

    def case_done(self, target, metrics, has_stats):
        self.finished.append(metrics)
        if self.live_run:
            print(R.live_line(metrics, R.stream_label(metrics)), file=sys.stderr, flush=True)

    def case_loaded(self, target, metrics, has_stats):
        self.finished.append(metrics)

    def target_done(self, target, entry, comparison, has_stats, mdcat, interrupted=None):
        label = self.labels.get(target["key"], target["name"])
        rows = R.case_rows(self.layout, self.finished, label, self.base_results())
        rows += R.totals_rows(self.layout, self.finished, comparison)
        table = Table(self.layout.header, rows, aligns=self.layout.aligns, dim_cols=self.layout.dim_cols)
        self.tail(entry, comparison, mdcat, interrupted,
                  cases=([report.Note(self.base_line)] if self.base_line else []) + [table], titles=True)

    def done(self, state_paths, mdcat, interrupted=None, shows_delta=False, spec=False, est=False):
        blocks = [report.Note(legend(shows_delta, spec, est=est))]
        saved = ", ".join(f"`{report.rel_path(p)}`" for p in dict.fromkeys(state_paths))
        if interrupted:
            blocks.append(report.Line("Interrupted (%s)." % interrupted, md="**Interrupted** (%s)." % interrupted))
        if saved:
            blocks.append(f"State: {saved}")
        compare.emit(blocks, mdcat)


class ChatOutput(MarkdownOutput):
    """`chat`: a transcript of the replay on stdout, one block per request (bench/chat.py), then the metrics,
    comparison and perf tables (markdown through mdcat when it can be used, else box-drawing tables)."""

    layout_needs_stats = False
    transcript = chat.ChatTranscript

    def __init__(self):
        super().__init__()
        self.tr = self.transcript()

    def set_fixture(self, fixture):
        self.tr.set_fixture(fixture)

    def header(self, infos, names, n_cases, mdcat):
        print(f"{Clr.BLD}{Clr.MAG}{R.headline(infos, n_cases)}{Clr.RST}\n")

    def target_start(self, target, has_stats, cases, base_run=None):
        self.tr.start_target(target.get("header"))
        print(f"{self.model_line(target)}\n")

    def case_done(self, target, metrics, has_stats):
        text = self.tr.case(metrics)
        if text:
            print(text, flush=True)

    def case_loaded(self, target, metrics, has_stats):
        pass

    def target_done(self, target, entry, comparison, has_stats, mdcat, interrupted=None):
        tail = self.tr.finish()
        if tail:
            print(tail, flush=True)
        self.tail(entry, comparison, mdcat, interrupted, cases=None, titles=True)


class ChatCompactOutput(ChatOutput):
    """`chat-compact`: Claude Code's compact transcript; while a request runs, one live line on stderr."""
    transcript = chat.CompactTranscript


OUTPUTS = {"tui": TuiOutput, "markdown": MarkdownOutput, "chat": ChatOutput, "chat-compact": ChatCompactOutput}


def output():
    """The renderer of the configured report format."""
    return OUTPUTS[report.current_format()]()
