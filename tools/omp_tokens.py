#!/usr/bin/env python3
"""Count the prompt tokens of rebuilt omp requests offline, exactly as Splash counts them, and check them against the
transcript's recorded `message.contextSnapshot.promptTokens` (PLAN, "Proof that a rebuilt turn is exact", 2).

Counting reuses Splash's own request path (--splash, read-only): the tokenizer files are derived from the GGUF
metadata by Splash's adapter (`install/gguf.py` `tokenizer_files`, the same bytes the installed assembly holds), loaded
with transformers' `AutoTokenizer` as `server/server.py` loads them, the chat templates probed and patched by
`server/chat_templates.py` `ChatTemplates` ("render later system messages in place"), and the count is
`server/frontend.py` `Frontend.count_tokens` (`_prepare_prompt`: message and tool normalization, chat_template_kwargs;
`_render_prompt`: template options, generation prompt, `PromptTokenizer`). No engine and no model server: the
Frontend gets a stand-in backend that only announces the context. The server's `--default-reasoning-effort` changes
the rendered prompt (the template reads `reasoning_effort`); case A's server ran with `medium` (its log:
"effort absent → medium").

Splash's log also prints each request's tool signature (`Done · input N · … · tools 12·ecf4d8da`: count and the first
8 hex of sha1 over the normalized tools, `Frontend.chat_job`); `check --splash-log` compares the rebuilt tools'
signature with the log's entries of the same input size.

Needs jinja2, tokenizers and transformers: run with Splash's venv python,
`python3 tools/omp_tokens.py …`. The comparison and report functions are stdlib only.
Offline only: nothing here opens a network connection.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omp_requests  # noqa: E402
import omp_session  # noqa: E402

DEFAULT_SPLASH = os.environ.get("OMP_TOKENS_SPLASH") or os.environ.get("SPLASH_DIR", "")
DEFAULT_GGUF_GLOB = os.environ.get("OMP_TOKENS_GGUF") or os.environ.get("GGUF_GLOB", "")
DEFAULT_FAMILY = "Qwen3.8-Flash-Next"
DEFAULT_EFFORT = "medium"
# The engine's Ready for a Qwen3.8 family model (dev/tests/server_fixtures.py ENGINE_TOKENS): vocabulary rows and
# stop tokens. Counting reads neither; the Frontend only needs a Ready.
QWEN_READY = (248_320, 9, (248_044, 248_046))


class SplashImportError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------- Splash counter

class SplashCounter:
    """Splash's prompt count for chat-completions bodies (see the module doc)."""

    def __init__(self, splash_dir: Optional[str] = None, gguf_path: Optional[str] = None,
                 family: str = DEFAULT_FAMILY, default_reasoning_effort: Optional[str] = DEFAULT_EFFORT,
                 max_context: int = 262_144):
        splash_dir = splash_dir or DEFAULT_SPLASH
        if not splash_dir:
            raise SplashImportError("Splash source checkout required: pass --splash or set $OMP_TOKENS_SPLASH / $SPLASH_DIR")
        if splash_dir not in sys.path:
            sys.path.insert(0, splash_dir)
        try:
            from types import SimpleNamespace

            import transformers
            from install import families, gguf
            from server import frontend, protocol
            from server.chat_templates import ChatTemplates
            from server.thinking import ThinkingCodec
            from server.tool_schema import normalize_tools
            transformers.logging.set_verbosity_error()
            from transformers import AutoTokenizer
        except Exception as exc:  # ImportError, or a Splash module this python cannot run (e.g. dataclass slots)
            raise SplashImportError(
                f"cannot import Splash's request path from {splash_dir} ({type(exc).__name__}: {exc}); run this "
                f"tool with Splash's venv "
                f"python, e.g. {os.path.join(splash_dir, '.venv/bin/python')} tools/omp_tokens.py …") from exc
        self.gguf_path = gguf_path or default_gguf()
        self.splash_dir = splash_dir
        self._normalize_tools = normalize_tools
        # Tokenizer files from the GGUF metadata only (no tensor is read), loaded as server.py loads them.
        files = gguf.tokenizer_files(gguf.Metadata(self.gguf_path))
        self._tmp = tempfile.TemporaryDirectory(prefix="omp_tokens_")
        for name, data in files.items():
            path = os.path.join(self._tmp.name, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
        self.tokenizer_files = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(self._tmp.name, "tokenizer"), local_files_only=True,
                                                  trust_remote_code=False)
        self.chat_templates = ChatTemplates(tokenizer)
        fam = families.named(family)
        if fam is None:
            raise ValueError(f"Splash serves no model family {family!r}")
        vocab, rows, stops = QWEN_READY
        ready = protocol.ReadyEvent(1, max_context, False, vocab, rows, stops)
        backend = SimpleNamespace(runtime=SimpleNamespace(capabilities=ready))
        self.frontend = frontend.Frontend(
            tokenizer, backend, "omp-tokens", None, constraint_factory=None, chat_templates=self.chat_templates,
            chat_format=fam.chat_format, thinking_codec=ThinkingCodec(), response_store=None,
            default_reasoning_effort=default_reasoning_effort)

    def describe(self) -> str:
        return f"Chat template · {self.chat_templates.describe()}"

    def count(self, body: dict) -> int:
        """Prompt tokens of a chat-completions body, generation prompt included (what Splash logs as `input` and
        returns as usage.prompt_tokens). The body's model is not checked."""
        body = {k: v for k, v in body.items() if k != "model"}
        return self.frontend.count_tokens(body, deadline=time.monotonic() + 3600)

    def tools_signature(self, body: dict) -> Optional[str]:
        """`N·hex8` as Splash's log prints it (Frontend.chat_job), None without tools."""
        tools, _policy = self._normalize_tools(body.get("tools"), body.get("tool_choice"),
                                               body.get("parallel_tool_calls", True), None)
        return tools_signature(tools)


def default_gguf() -> str:
    if not DEFAULT_GGUF_GLOB:
        raise FileNotFoundError("no GGUF path: pass --gguf or set $OMP_TOKENS_GGUF / $GGUF_GLOB")
    found = sorted(glob.glob(DEFAULT_GGUF_GLOB))
    if not found:
        raise FileNotFoundError(f"no GGUF at {DEFAULT_GGUF_GLOB}; pass --gguf")
    return found[0]  # a split model's metadata is its first file's


def tools_signature(tools: Optional[list]) -> Optional[str]:
    if not tools:
        return None
    digest = hashlib.sha1(json.dumps(tools, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
    return f"{len(tools)}·{digest}"


# --------------------------------------------------------------------------------------------- Splash log

_DONE = re.compile(r"^(\d\d:\d\d:\d\d) (?:Done|Cancelled) · input ([\d,]+) ·(.*)$")
_TOOLS = re.compile(r"(?:^|· )tools (\d+·[0-9a-f]+)")


def parse_splash_log(path: str) -> list:
    """[{"time", "input", "tools"}] for each finished request line of a Splash log."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _DONE.match(line.rstrip("\n"))
            if not m:
                continue
            t = _TOOLS.search(m.group(3).strip())
            out.append({"time": m.group(1), "input": int(m.group(2).replace(",", "")),
                        "tools": t.group(1) if t else None})
    return out


def log_signatures_by_input(entries: list) -> dict:
    """{input tokens: set of tool signatures (None for none)}."""
    out: dict = {}
    for e in entries:
        out.setdefault(e["input"], set()).add(e["tools"])
    return out


# --------------------------------------------------------------------------------------------- comparison

def stream_label(stream: str, epoch: Optional[int]) -> str:
    return f"main/e{epoch or 0}" if stream == "main" else stream


def summarize(rows: list) -> list:
    """Per stream label (main per compaction epoch, each subagent), in first-seen order:
    {"stream", "turns", "exact", "min", "max", "median"} over delta = counted - recorded (rows without a recorded
    count or with an error count as not exact and stay out of the deltas)."""
    groups: dict = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    out = []
    for label, rs in groups.items():
        deltas = [r["delta"] for r in rs if r.get("delta") is not None]
        out.append({"stream": label, "turns": len(rs), "exact": sum(1 for d in deltas if d == 0),
                    "min": min(deltas) if deltas else None, "max": max(deltas) if deltas else None,
                    "median": statistics.median(deltas) if deltas else None})
    return out


def mismatches(rows: list) -> list:
    return [r for r in rows if r.get("delta") != 0 or r.get("tools_ok") is False]


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v:+d}" if isinstance(v, int) else f"{v:+.1f}"


def format_report(rows: list, limit: int = 200) -> str:
    summary = summarize(rows)
    lines = [f"{'stream':<22} {'turns':>5} {'exact':>5} {'min Δ':>7} {'max Δ':>7} {'med Δ':>7}"]
    for s in summary:
        lines.append(f"{s['stream']:<22} {s['turns']:>5} {s['exact']:>5} {_fmt(s['min']):>7} {_fmt(s['max']):>7} "
                     f"{_fmt(s['median']):>7}")
    total = sum(s["turns"] for s in summary)
    exact = sum(s["exact"] for s in summary)
    lines.append(f"{'total':<22} {total:>5} {exact:>5}")
    tools_checked = [r for r in rows if r.get("tools_ok") is not None]
    if tools_checked:
        lines.append(f"tool signatures equal to the Splash log's: "
                     f"{sum(1 for r in tools_checked if r['tools_ok'])}/{len(tools_checked)}")
    bad = mismatches(rows)
    if bad:
        lines.append(f"mismatching turns ({len(bad)}):")
        for r in bad[:limit]:
            extra = f" error: {r['error']}" if r.get("error") else ""
            tools = "" if r.get("tools_ok") is not False else f" tools {r.get('tools')} not in log {r.get('log_tools')}"
            lines.append(f"  {r['label']:<22} #{r['turn']:<4} counted {r.get('counted')} recorded {r.get('recorded')} "
                         f"Δ {_fmt(r.get('delta'))}{tools}{extra}")
        if len(bad) > limit:
            lines.append(f"  … {len(bad) - limit} more")
    return "\n".join(lines)


def check_session(session_path: str, snap: dict, counter, options: Optional[dict] = None,
                  log_entries: Optional[list] = None, notes_out: Optional[list] = None,
                  include_compactions: bool = True) -> list:
    """Rebuild and count every turn of every stream (and compactions). `counter` has `count(body)` and
    `tools_signature(body)`.
    Rows: {"label", "stream", "turn", "epoch", "recorded", "counted", "delta", "tools", "tools_ok", "log_tools"}."""
    sess = omp_session.load_session(session_path)
    by_input = log_signatures_by_input(log_entries) if log_entries is not None else None
    rows = []
    for s in sess.streams():
        for t in s.turns:
            row = {"label": stream_label(s.name, t.compaction_epoch), "stream": s.name, "turn": t.index,
                   "epoch": t.compaction_epoch, "recorded": t.prompt_tokens, "counted": None, "delta": None}
            notes: list = []
            try:
                body = omp_requests.build_request(s, t.index, snap, options, notes)
                row["counted"] = counter.count(body)
                row["tools"] = counter.tools_signature(body)
            except Exception as exc:  # a rebuild or render error is a finding, not a crash
                row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                continue
            if t.prompt_tokens is not None:
                row["delta"] = row["counted"] - t.prompt_tokens
            if by_input is not None and t.prompt_tokens is not None:
                seen = by_input.get(t.prompt_tokens)
                row["log_tools"] = sorted(x or "-" for x in seen) if seen else None
                row["tools_ok"] = None if not seen else row["tools"] in seen
            if notes_out is not None:
                notes_out.extend(f"{row['label']} #{t.index}: {n}" for n in notes)
            rows.append(row)

        if include_compactions:
            for i, c in enumerate(s.compactions):
                row = {"label": f"{s.name}/c{i}", "stream": s.name, "turn": i,
                       "epoch": i, "recorded": None, "counted": None, "delta": None}
                notes: list = []
                try:
                    body = omp_requests.build_compaction_request(s, i, snap, options, notes)
                    row["counted"] = counter.count(body)
                    row["tools"] = counter.tools_signature(body)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
                    continue

                if by_input is not None and row["counted"] in by_input:
                    seen = by_input[row["counted"]]
                    row["recorded"] = row["counted"]
                    row["delta"] = 0
                    row["log_tools"] = sorted(x or "-" for x in seen) if seen else None
                    row["tools_ok"] = None if not seen else row["tools"] in seen
                elif log_entries:
                    matched = next((e for e in log_entries if e.get("input") == row["counted"]), None)
                    if matched:
                        row["recorded"] = matched["input"]
                        row["delta"] = row["counted"] - row["recorded"]
                        row["tools_ok"] = (row["tools"] == matched.get("tools"))
                        row["log_tools"] = [matched.get("tools") or "-"]
                    elif c.tokens_before is not None:
                        row["recorded"] = c.tokens_before
                        row["delta"] = row["counted"] - c.tokens_before
                elif c.tokens_before is not None:
                    row["recorded"] = c.tokens_before
                    row["delta"] = row["counted"] - c.tokens_before

                if notes_out is not None:
                    notes_out.extend(f"{row['label']} #{i}: {n}" for n in notes)
                rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------- CLI

def _counter(args) -> SplashCounter:
    effort = None if args.default_reasoning_effort in ("", "none-set") else args.default_reasoning_effort
    try:
        return SplashCounter(args.splash, args.gguf, args.family, effort)
    except SplashImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def _cmd_count(args) -> int:
    counter = _counter(args)
    print(counter.describe(), file=sys.stderr)
    for path in args.request:
        with open(path, encoding="utf-8") as f:
            body = json.load(f)
        print(f"{path}: {counter.count(body)} prompt tokens, tools {counter.tools_signature(body)}")
    return 0


def _cmd_check(args) -> int:
    snap = omp_requests.load_snapshot(args.snapshot)
    options: dict = {}
    if args.tool_source:
        options["tool_sources"] = {t: p for p, t in (x.split("=", 1) for x in args.tool_source)}
    counter = _counter(args)
    print(counter.describe())
    log_entries = parse_splash_log(args.splash_log) if args.splash_log else None
    notes: list = []
    rows = check_session(args.session, snap, counter, options, log_entries, notes,
                         include_compactions=not args.no_compactions)
    print(format_report(rows, args.limit))
    if args.notes:
        for n in notes:
            print(f"note: {n}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "summary": summarize(rows), "notes": notes}, f, ensure_ascii=False, indent=1)
    return 0 if not mismatches(rows) else 1


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Count rebuilt omp requests' prompt tokens as Splash does (offline) and check them against the "
                    "transcript. Run with Splash's venv python (jinja2, tokenizers, transformers): "
                    "python3 tools/omp_tokens.py …")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--splash", default=DEFAULT_SPLASH or None, help="Splash source checkout (default: $OMP_TOKENS_SPLASH or $SPLASH_DIR)")
    common.add_argument("--gguf", help="model GGUF (default: $OMP_TOKENS_GGUF or $GGUF_GLOB)")
    common.add_argument("--family", default=DEFAULT_FAMILY, help="Splash model family (its chat format)")
    common.add_argument("--default-reasoning-effort", default=DEFAULT_EFFORT,
                        help="the server's --default-reasoning-effort ('' for none); case A's was medium")
    p = sub.add_parser("count", parents=[common], help="count built request bodies (JSON files)")
    p.add_argument("request", nargs="+")
    p.set_defaults(fn=_cmd_count)
    p = sub.add_parser("check", parents=[common], help="rebuild + count every turn of a session, compare with "
                       "contextSnapshot.promptTokens, per-stream table and mismatching turns")
    p.add_argument("--session", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--tool-source", action="append", default=[], metavar="POOL=TOOL",
                   help="take a subagent tool's schema from this snapshot pool (e.g. main=grep)")
    p.add_argument("--splash-log", help="Splash server log of the session's day: compare tool signatures")
    p.add_argument("--json", help="write rows + summary here (keep it out of the repo: sources/ or /tmp)")
    p.add_argument("--notes", action="store_true", help="print the rebuild notes")
    p.add_argument("--limit", type=int, default=200, help="mismatching turns to list")
    p.add_argument("--no-compactions", action="store_true", help="do not check compaction calls")
    p.set_defaults(fn=_cmd_check)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
