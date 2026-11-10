"""Transcripts of benchmark runs: Transcript, ChatTranscript, CompactTranscript."""

import bisect
import os
import re
from datetime import datetime, timedelta

from . import chat, glyphs
from . import state as state_mod
from .cases import replay as replay_cases
from .cases.grading import calls_from
from .chat import (
    _BADGE_COLORS,
    _first_line,
    arg_path,
    call_args,
    call_name,
    describe_call,
    message_text,
    recorded_text,
    render_call,
    render_prose,
    same_call,
    stats_row,
    task_list,
    verdict_text,
)
from .colors import Clr
from .fmt import clip, hard_wrap, term_width, vlen

USER_ROWS = chat.USER_ROWS
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def load_header_fixture(header):
    """The fixture a state header names ({"file", "sha256"}), found in chat.FIXTURE_DIRS and checked against the sha256;
    None when it is not there or is not the same file."""
    fx = (header or {}).get("fixture") or {}
    if not fx.get("file"):
        return None
    for d in chat.FIXTURE_DIRS:
        path = os.path.join(d, fx["file"])
        if not os.path.exists(path):
            continue
        try:
            if fx.get("sha256") and state_mod.file_sha256(path) != fx["sha256"]:
                continue
            return replay_cases.load_fixture(path)
        except (OSError, ValueError):
            continue
    return None


def tidy(lines):
    """Lines with single blank separators and one trailing blank line; '' when nothing is left."""
    out = []
    for ln in lines:
        if ln == "" and (not out or out[-1] == ""):
            continue
        out.append(ln)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n" if out else ""


class Transcript:
    """Shared state of both chat formats: the fixture (optional), the run clock, the current stream."""

    def __init__(self, width=None):
        self.width = width or term_width()
        self.fixture = None
        self.t0 = datetime.now().astimezone()
        self.clock_ms = 0
        self.stream = None
        self.stream_started = {}
        self._positions = None

    # -- setup ---------------------------------------------------------
    def set_fixture(self, fixture):
        self.fixture = fixture
        self._positions = None

    def start_target(self, header=None):
        """New model: the clock restarts at the run's start time (state header) or now."""
        self.clock_ms, self.stream, self.stream_started = 0, None, {}
        self.t0 = datetime.now().astimezone()
        started = (header or {}).get("started_at")
        if started:
            try:
                self.t0 = datetime.fromisoformat(started.replace("Z", "+00:00")).astimezone()
            except ValueError:
                pass
        if self.fixture is None and header:
            self.set_fixture(load_header_fixture(header))

    # -- fixture lookups -----------------------------------------------
    def entry_of(self, m):
        """(index in fixture turns, entry) of a result, or (None, None) without a matching fixture."""
        mt = re.match(r"R(\d+)_", str(m.get("id", "")))
        turns = (self.fixture or {}).get("turns") or []
        if mt and 1 <= int(mt.group(1)) <= len(turns):
            i = int(mt.group(1)) - 1
            if replay_cases.case_id(i + 1, turns[i]) == m.get("id"):
                return i, turns[i]
        return None, None

    def recorded_calls(self, entry):
        if entry is None:
            return None
        return (entry.get("recorded_answer") or {}).get("tool_calls") or []

    def recorded_results(self, idx, entry):
        """{recorded call id: result text} from the next fixture request of the same stream."""
        ids = [c.get("id") for c in self.recorded_calls(entry) or []]
        if not ids:
            return {}
        if self._positions is None:
            self._positions = {}
            for i, e in enumerate(self.fixture["turns"]):
                self._positions.setdefault(e["label"], []).append(i)
        pos = self._positions.get(entry["label"], [])
        k = bisect.bisect_right(pos, idx)
        if k >= len(pos):
            return {}
        pool, found = self.fixture["messages"], {}
        for mi in reversed(self.fixture["turns"][pos[k]]["messages"]):
            msg = pool[mi]
            if msg.get("role") == "tool" and msg.get("tool_call_id") in ids:
                found[msg["tool_call_id"]] = message_text(msg)
                if len(found) == len(ids):
                    break
        return found

    def first_user_prompt(self, entry):
        if entry is None:
            return None
        pool = self.fixture["messages"]
        for mi in entry["messages"]:
            if pool[mi].get("role") == "user":
                text = message_text(pool[mi])
                return _REMINDER.sub("", text).strip() or text.strip()
        return None

    # -- one result ----------------------------------------------------
    def label_of(self, m, entry):
        if entry is not None:
            return entry["label"]
        mt = re.match(r"R\d+_(.*?)(?:_t\d+)?$", str(m.get("id", "")))
        return mt.group(1) if mt else str(m.get("id", "?"))

    def kind_text(self, m, entry):
        kind = (entry or {}).get("kind") or m.get("kind")
        stream = (entry or {}).get("stream") or m.get("stream")
        return "compaction" if kind == "compaction" else "main agent" if stream == "main" else "subagent"

    def answer(self, m):
        """(content, tool_calls) of a result."""
        content = m.get("content")
        return content or "", calls_from(m) or []

    def advance(self, m):
        """Move the run clock past this request; return (time of completion, Δ since the stream's first request)."""
        first = self.stream_started.setdefault(self.stream, self.clock_ms)
        self.clock_ms += m.get("total_ms") or 0
        return self.t0 + timedelta(milliseconds=self.clock_ms), self.clock_ms - first

    def stream_header(self, label, kind):
        icon = glyphs.prefix({"main agent": "main", "subagent": "subagent", "compaction": "compaction"}.get(kind, ""))
        text = f"── {icon}{label}" + ("" if icon else f" · {kind}") + " "
        return f"{Clr.BLD}{Clr.MAG}{text}{'─' * max(0, self.width - vlen(text))}{Clr.RST}"

    def case(self, m):
        raise NotImplementedError

    def finish(self):
        """Text still pending at the end of a model's run."""
        return ""


class ChatTranscript(Transcript):
    """`chat`: omp's layout, one block per request."""

    def user_block(self, text):
        lines = text.splitlines()
        shown = lines[:USER_ROWS]
        out = []
        for i, ln in enumerate(shown):
            for k, p in enumerate(hard_wrap(ln, self.width - 6) or [""]):
                out.append(f" {Clr.BCYN}{'❯' if i == 0 and k == 0 else ' '}{Clr.RST} {Clr.DIM}{p}{Clr.RST}")
        if len(lines) > USER_ROWS:
            out.append(f"   {Clr.DIM}… ({len(lines) - USER_ROWS} more lines){Clr.RST}")
        return out

    def calls_block(self, calls, rec, results, rec_text=""):
        """The model's calls with recorded results where they match, recorded calls dimmed where they differ."""
        out = []
        dim = lambda s: f"{Clr.DIM}{s}{Clr.RST}"
        for i, c in enumerate(calls):
            r = rec[i] if rec is not None and i < len(rec) else None
            ok = r is not None and same_call(c, r)
            out += render_call(c, results.get(r.get("id")) if ok else None, self.width)
            if rec is not None and not ok:
                out.append(dim("  recorded: " + (describe_call(r, self.width - 14) if r else
                                                            recorded_text(rec_text, self.width - 14))))
            out.append("")
        for r in (rec or [])[len(calls):]:
            out += [dim("  recorded, not called by the model: " + describe_call(r, self.width - 40)), ""]
        return out

    def case(self, m):
        idx, entry = self.entry_of(m)
        label = self.label_of(m, entry)
        out = []
        new_stream = label != self.stream
        first_seen = label not in self.stream_started
        if new_stream:
            self.stream = label
            out += [self.stream_header(label, self.kind_text(m, entry)), ""]
        if first_seen and entry is not None:
            prompt = self.first_user_prompt(entry)
            if prompt:
                out += self.user_block(prompt) + [""]
        when, delta = self.advance(m)
        v, reason = verdict_text(m, self.width)
        if m.get("eval_result") == "SKIP" or ("error" in m and not m.get("tokens")):
            return tidy(out + [f" {v}  {Clr.DIM}{clip(reason or m.get('error', ''), self.width - 30)}{Clr.RST}"])
        content, calls = self.answer(m)
        if content.strip():
            out += render_prose(content, self.width, " ") + [""]
        rec = self.recorded_calls(entry)
        out += self.calls_block(calls, rec, self.recorded_results(idx, entry) if rec else {},
                                ((entry or {}).get("recorded_answer") or {}).get("content") or "")
        row = f" {stats_row(m, when, delta)}  {v}"
        if reason and vlen(row) + 2 + vlen(reason) <= self.width:
            row += f"  {Clr.DIM}{reason}{Clr.RST}"
            reason = ""
        out.append(row)
        if reason:
            out.append(f"   {Clr.DIM}↳ {reason}{Clr.RST}")
        return tidy(out)


_SUMMARY = {"bash": ("Ran", "shell command", False), "read": ("Read", "file", True), "edit": ("Edited", "file", True),
            "write": ("Wrote", "file", True), "grep": ("Searched", "time", False), "glob": ("Searched", "time", False),
            "find": ("Searched", "time", False), "fetch": ("Fetched", "page", True), "skill": ("Loaded", "skill", True)}


_FIXED = {"todo": "Updated todo list", "wait": "Waited for background work"}


class CompactTranscript(Transcript):
    """`chat-compact`: Claude Code's compact layout. Consecutive tool calls (also across requests of one stream) are
    one summary line."""

    def __init__(self, width=None):
        super().__init__(width)
        self.pending, self.under_message, self.last_verdict = [], False, None

    def start_target(self, header=None):
        super().start_target(header)
        self.pending, self.under_message, self.last_verdict = [], False, None

    @staticmethod
    def bullet(v):
        """The message bullet, colored by the verdict of the request."""
        return f"{getattr(Clr, _BADGE_COLORS.get(v, ''), Clr.DIM) if v in _BADGE_COLORS else Clr.DIM}●{Clr.RST}"

    def summary_line(self):
        groups, unknown, skills = {}, {}, []
        for c in self.pending:
            name = call_name(c)
            if name in _FIXED:
                continue
            if name not in _SUMMARY:
                unknown[name] = unknown.get(name, 0) + 1
                continue
            verb, noun, by_path = _SUMMARY[name]
            args = call_args(c) or {}
            key = (verb, noun)
            groups.setdefault(key, set() if by_path else [])
            if by_path:
                groups[key].add(arg_path(args))
            else:
                groups[key].append(name)
            if name == "skill":
                skills.append(arg_path(args))
        phrases = [_FIXED[n] for n in dict.fromkeys(call_name(c) for c in self.pending) if n in _FIXED]
        for (verb, noun), items in groups.items():
            n = len(items)
            if noun == "skill" and n == 1:
                phrases.append("Loaded skill %s" % skills[0])
            else:
                phrases.append("%s %d %s%s" % (verb, n, noun, "" if n == 1 else "s"))
        if unknown:
            phrases.append("Called " + ", ".join(n + (" \u00d7%d" % k if k > 1 else "") for n, k in unknown.items()))
        text = ", ".join(p if i == 0 else p[0].lower() + p[1:] for i, p in enumerate(phrases))
        names = [call_name(c) for c in self.pending]
        icon = glyphs.prefix(names[0] if len(set(names)) == 1 and names[0] in glyphs.ICONS else "tool")
        return icon + f"{Clr.DIM}{text}{Clr.RST}"

    def flush(self):
        if not self.pending:
            return []
        text, self.pending = self.summary_line(), []
        if self.under_message:  # tucked under the assistant message that made the calls
            self.under_message = False
            return [f"  {Clr.DIM}⎿{Clr.RST} {text}", ""]
        return [f"{self.bullet(self.last_verdict)} {text}", ""]

    def finish(self):
        return tidy(self.flush())

    def user_block(self, text):
        lines = [p for ln in text.splitlines()[:USER_ROWS] for p in (hard_wrap(ln, self.width - 4) or [""])]
        out = [f"  {Clr.DIM}{p}{Clr.RST}" for p in lines]
        if len(text.splitlines()) > USER_ROWS:
            out.append(f"  {Clr.DIM}… ({len(text.splitlines()) - USER_ROWS} more lines){Clr.RST}")
        return out

    def case(self, m):
        idx, entry = self.entry_of(m)
        label = self.label_of(m, entry)
        out = []
        first_seen = label not in self.stream_started
        if label != self.stream:
            out += self.flush()
            self.stream = label
            out += [self.stream_header(label, self.kind_text(m, entry)), ""]
        if first_seen and entry is not None:
            prompt = self.first_user_prompt(entry)
            if prompt:
                out += self.user_block(prompt) + [""]
        self.advance(m)
        v = m.get("eval_result")
        reason = m.get("fail_reason") or m.get("error") or ""
        content, calls = self.answer(m)
        if v == "SKIP":
            return tidy(out + self.flush() + [f"  {Clr.DIM}skipped {m.get('id')}: {clip(reason, self.width - 20)}{Clr.RST}"])
        self.last_verdict = v
        if content.strip():
            lines = render_prose(content, self.width - 2)
            first = next((i for i, l in enumerate(lines) if l.strip()), 0)
            lines = [(self.bullet(v) + " " + l) if i == first else ("  " + l if l else l) for i, l in enumerate(lines)]
            out += self.flush() + lines + [""]
            self.under_message = True
        for c in calls:
            if call_name(c) == "task":
                out += self.flush()
                subs = task_list(call_args(c))
                for t in subs or [{}]:
                    out += ["%s●%s Launched subagent %s%s" % (Clr.BGRN, Clr.RST, t.get("name") or t.get("agent") or "?",
                                                            Clr.DIM + ": " + clip(_first_line(t.get("task", ""), 200), self.width - 30) + Clr.RST
                                                            if t.get("task") else ""), ""]
            else:
                self.pending.append(c)
        if v in ("WARN", "FAIL"):
            out += self.flush()
            col = Clr.BYEL if v == "WARN" else Clr.BRED
            out += [f"  {Clr.DIM}{col}{glyphs.g(v, '⚠' if v == 'WARN' else '✗')} {m.get('id')} {v}{Clr.RST}{Clr.DIM}: {clip(reason, self.width - 30)}{Clr.RST}", ""]
        return tidy(out)
