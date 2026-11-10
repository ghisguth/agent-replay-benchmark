#!/usr/bin/env python3
# Derived in part from oh-my-pi (MIT); see THIRD_PARTY_NOTICES.md at the repository root.
"""Rebuild the OpenAI chat-completions request omp sent for an assistant turn, from the session transcript.

What the transcript stores is turned into the request the way omp 18.17.0 does it (oh-my-pi sources; 18.17.0's
bundled JS where it differs from the 18.13 checkout):

* Context (`session/session-context.ts` `buildSessionContext`, non-transcript mode) over the active branch up to the
  entry before assistant turn k: `message` entries, `custom_message` entries (skill prompts, nudges, async results),
  `branch_summary`; after the latest `compaction` on that path: the summary first (`compactionSummary`), then the kept
  entries from `firstKeptEntryId` up to the compaction, then everything after it. Then omp's clean-ups: dangling tool
  calls stripped, aborted/error assistant turns dropped with their results, `retryRecovery` / empty error turns
  skipped.
* Tool results pruned later (`prunedAt`, `agent/compaction/pruning.ts`) are blanked in the session file; a request
  that started before `prunedAt` saw the original. `wait` results over running jobs are re-rendered from `details`
  (`async/job-control.ts` `buildJobResult`); anything else is reported as not reconstructable.
* `convertToLlm` (`session/messages.ts`, `agent/compaction/messages.ts`): custom messages become `developer` turns
  (content as text parts), a user-invoked `skill-prompt` a `user` turn, the compaction summary a `user` turn rendered
  through `compaction-summary-context.md` / `handoff-summary-context.md` (omp's prompt `format` post-pass included).
* Date/cwd reminder (`session/date-cwd-reminder.ts`): the session's `custom` `date-cwd-reminder` records say which
  turn of which conversation (keyed `role:timestamp:attribution` of its first user turn) got the reminder, as a text
  prefix part (`placement: prefix`) or as a `developer` message after a turn (`control`).
* Provider body (`ai/providers/openai-completions.ts` `buildParams` + `convertMessages`) with the compat of a local
  llama.cpp-style backend: one `system` message (blocks joined by "\\n\\n"), developer turns sent as `user`,
  thinking replayed as `reasoning_content`, assistant text joined, `content: ""` next to tool calls, tool results as
  `role: tool` messages with joined text.

What the transcript does not store comes from a *snapshot* (a JSON-able dict, see `snapshot_from_capture`): the main
system prompt blocks, the tool schemas, the request parameters and their key order, the recorded provider/model.
Subagent system prompts come from the stream's `session_init` entry.

Stdlib only. CLI: `python3 tools/omp_requests.py --help`.
"""
from __future__ import annotations

import argparse
import calendar
import copy
import glob
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omp_session  # noqa: E402

SNAPSHOT_FORMAT = "omp-request-snapshot/1"

# --------------------------------------------------------------------------------------------- JS helpers

# ECMAScript WhiteSpace + LineTerminator (String.prototype.trim, \s).
_JS_WS = ("\u0009\u000a\u000b\u000c\u000d           "
          "       　﻿")
_JS_NON_WS = re.compile("[^" + re.escape(_JS_WS) + "]")


def js_trim(s: str) -> str:
    return s.strip(_JS_WS)


def js_is_blank(s: str) -> bool:
    return js_trim(s) == ""


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def well_formed(s: str) -> str:
    """String.prototype.toWellFormed: lone surrogates -> U+FFFD (Python strings from json can hold them)."""
    return _LONE_SURROGATE.sub("�", s)


_ARRAY_INDEX = re.compile(r"^(0|[1-9][0-9]*)$")


def _js_key_order(d: dict) -> list:
    """JS own-property order: array-index keys ascending first, then the rest in insertion order."""
    idx = [k for k in d if isinstance(k, str) and _ARRAY_INDEX.match(k) and int(k) < 2 ** 32 - 1]
    if not idx:
        return list(d)
    rest = [k for k in d if k not in set(idx)]
    return sorted(idx, key=int) + rest


def _js_number(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if v != v or v in (float("inf"), float("-inf")):
        return "null"
    if v == int(v) and abs(v) < 1e21:
        return str(int(v))
    r = repr(v)
    if "e" in r:  # JS: 1e+21, 1.5e-7
        mant, exp = r.split("e")
        sign = "-" if exp.startswith("-") else "+"
        r = f"{mant}e{sign}{exp.lstrip('+-').lstrip('0') or '0'}"
    return r


def js_stringify(v) -> str:
    """JSON.stringify(v) (compact) for JSON-parsed data."""
    if v is None:
        return "null"
    if isinstance(v, (bool, int, float)):
        return _js_number(v)
    if isinstance(v, str):
        return json.dumps(well_formed(v), ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ",".join(js_stringify(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + js_stringify(v[k])
                              for k in _js_key_order(v)) + "}"
    raise TypeError(f"not JSON data: {type(v)}")


def iso_to_ms(ts: str) -> int:
    """`new Date(iso).getTime()` for omp's `2026-11-06T20:25:59.011Z` timestamps (exact, no float rounding)."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    off = dt.utcoffset()
    base = calendar.timegm(dt.replace(tzinfo=None).timetuple()) - (off.total_seconds() if off else 0)
    return int(base) * 1000 + dt.microsecond // 1000


# --------------------------------------------------------------------------------------------- wyhash / digests

_M64 = (1 << 64) - 1
_WY_SECRET = (0xa0761d6478bd642f, 0xe7037ed1a0b428db, 0x8ebc6af09c88c6e3, 0x589965cc75374cc3)


def _wy_mum(a: int, b: int) -> tuple:
    x = a * b
    return x & _M64, (x >> 64) & _M64


def _wy_mix(a: int, b: int) -> int:
    lo, hi = _wy_mum(a, b)
    return lo ^ hi


def wyhash(data: bytes, seed: int = 0) -> int:
    """Zig std.hash.Wyhash, which is what Bun.hash / Bun.hash.wyhash compute (strings as UTF-8)."""
    rd = lambda b, i, n: int.from_bytes(b[i:i + n], "little")  # noqa: E731
    seed &= _M64
    s0 = seed ^ _wy_mix(seed ^ _WY_SECRET[0], _WY_SECRET[1])
    state = [s0, s0, s0]
    n = len(data)
    if n <= 16:
        if n >= 4:
            end, q = n - 4, (n >> 3) << 2
            a = (rd(data, 0, 4) << 32) | rd(data, q, 4)
            b = (rd(data, end, 4) << 32) | rd(data, end - q, 4)
        elif n > 0:
            a, b = (data[0] << 16) | (data[n >> 1] << 8) | data[n - 1], 0
        else:
            a = b = 0
    else:
        i = 0
        if n >= 48:
            while i + 48 < n:
                for k in range(3):
                    state[k] = _wy_mix(rd(data, i + 16 * k, 8) ^ _WY_SECRET[k + 1],
                                       rd(data, i + 16 * k + 8, 8) ^ state[k])
                i += 48
            state[0] ^= state[1] ^ state[2]
        j, rest = 0, n - i
        while j + 16 < rest:
            state[0] = _wy_mix(rd(data, i + j, 8) ^ _WY_SECRET[1], rd(data, i + j + 8, 8) ^ state[0])
            j += 16
        a, b = rd(data, n - 16, 8), rd(data, n - 8, 8)
    a, b = _wy_mum(a ^ _WY_SECRET[1], b ^ state[0])
    return _wy_mix(a ^ _WY_SECRET[0] ^ n, b ^ _WY_SECRET[1])


def bun_hash_hex(s: str, seed: int = 0) -> str:
    return "%x" % wyhash(well_formed(s).encode("utf-8"), seed)


def system_prompt_digest(blocks: list) -> str:
    """omp `system-prompt-digest` (session-tools.ts): wyhash chain seeded with the block count."""
    h = len(blocks)
    for b in blocks:
        h = wyhash(well_formed(b).encode("utf-8"), h)
    return "%x" % h


def split_blocks_for_digest(text: str, digest: str, max_blocks: int = 3) -> Optional[list]:
    """Find how a joined system prompt (blocks joined by "\\n\\n") splits into blocks with the given digest."""
    if system_prompt_digest([text]) == digest:
        return [text]
    seps = [m.start() for m in re.finditer("\n\n", text)]
    for i in seps:
        blocks = [text[:i], text[i + 2:]]
        if system_prompt_digest(blocks) == digest:
            return blocks
    if max_blocks >= 3:
        for x, i in enumerate(seps):
            for j in seps[x + 1:]:
                if j < i + 2:
                    continue
                blocks = [text[:i], text[i + 2:j], text[j + 2:]]
                if system_prompt_digest(blocks) == digest:
                    return blocks
    return None


# --------------------------------------------------------------------------------------------- omp prompt format

_TABLE_ROW = re.compile(r"^\|.*\|$", re.S)
_TABLE_SEP = re.compile(r"^\|[-:" + re.escape(_JS_WS) + r"|]+\|$")
_OPENING_XML = re.compile(r"^<([a-z_-]+)(?:[" + re.escape(_JS_WS) + r"]+[^>]*)?>$", re.S)


def _closing_tag_name(s: str) -> Optional[str]:
    n = len(s)
    if n < 4 or s[-1] != ">":
        return None
    for c in s[2:n - 1]:
        if not ("a" <= c <= "z" or c in "-_"):
            return None
    return s[2:n - 1]


def _opening_tag_name(s: str) -> Optional[str]:
    n = len(s)
    if n < 3 or s[-1] != ">":
        return None
    j = 1
    while j < n - 1 and ("a" <= s[j] <= "z" or s[j] in "-_"):
        j += 1
    if j == 1:
        return None
    if j == n - 1:
        return s[1:j]
    c = s[j]
    if c not in " \t":
        if ord(c) < 128:
            return None
        m = _OPENING_XML.match(s)
        return m.group(1) if m else None
    return s[1:j] if s.find(">", j + 1) == n - 1 else None


def _compact_table_row(line: str) -> str:
    return "|".join(js_trim(c) for c in line.split("|"))


def _compact_table_sep(line: str) -> str:
    out = []
    for c in line.split("|"):
        t = js_trim(c)
        if not t:
            continue
        left, right = t.startswith(":"), t.endswith(":")
        out.append(":---:" if left and right else ":---" if left else "---:" if right else "---")
    return "|" + "|".join(out) + "|"


def _js_ord0(s: str) -> float:
    """charCodeAt(0) on a UTF-16 string: NaN for empty, surrogate for astral."""
    if not s:
        return float("nan")
    o = ord(s[0])
    return 0xD800 if o > 0xFFFF else o


def prompt_format(content: str) -> str:
    """pi-utils `prompt.format(content, {renderPhase: "post-render"})` (18.13 TS == 18.17 bundle)."""
    lines = content.split("\n")
    result: list = []
    in_code = False
    tags: list = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        last = _js_ord0(raw[-1:]) if raw else float("nan")
        line = raw.rstrip(_JS_WS) if (last <= 32 or last >= 128) else raw
        s = 0
        while s < len(line) and line[s] in " \t":
            s += 1
        first = _js_ord0(line[s:])
        if first >= 128:
            s = len(line) - len(line.lstrip(_JS_WS))
            first = _js_ord0(line[s:])
        if first in (96, 126) and (line.startswith("```", s) or line.startswith("~~~", s)):
            in_code = not in_code
            result.append(line)
            i += 1
            continue
        if in_code:
            result.append(line)
            i += 1
            continue
        closing = False
        if first == 60:
            ts = line if s == 0 else line[s:]
            if ts[1:2] == "/":
                name = _closing_tag_name(ts)
                if name is not None:
                    closing = True
                    if tags and tags[-1] == name:
                        tags.pop()
            elif s == 0 and not ts.endswith("/>"):
                name = _opening_tag_name(ts)
                if name is not None:
                    tags.append(name)
        elif first == 124:
            ts = line if s == 0 else line[s:]
            if _TABLE_SEP.match(ts):
                line = line[:s] + _compact_table_sep(ts)
            elif _TABLE_ROW.match(ts):
                line = line[:s] + _compact_table_row(ts)
        if s >= len(line):
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            if nxt is None or len(nxt) == 0 or not _JS_NON_WS.search(nxt):
                while result and len(result[-1]) == 0:
                    result.pop()
                j = i + 1
                while j < len(lines) and (len(lines[j]) == 0 or not _JS_NON_WS.search(lines[j])):
                    j += 1
                i = j
                continue
            if not result or len(result[-1]) == 0:
                i += 1
                continue
        if closing:
            while result and len(result[-1]) == 0:
                result.pop()
        result.append(line)
        i += 1
    while result and len(result[-1]) == 0:
        result.pop()
    return "\n".join(result)


# Templates as embedded in the omp 18.17.0 binary (identical to the 18.13 checkout).
COMPACTION_SUMMARY_TEMPLATE = ("Prior model work/tool state available.\nMUST build on prior work; NEVER duplicate prior "
                               "work.\n\n<summary>\n{{summary}}\n</summary>\n")
HANDOFF_SUMMARY_TEMPLATE = (
    "Context replaced. The <handoff> below is a handoff document a prior instance of you wrote from the full "
    "conversation. It is your own working memory, not user input.\n"
    "- First person inside it refers to you (the prior instance).\n"
    "- \"Next Steps\" is your own resumed plan; re-check it against the latest user message before acting.\n"
    "- The handoff already exists and is complete: NEVER write another handoff document unless the user explicitly "
    "asks.\nMUST build on prior work; NEVER duplicate prior work.\n\n<handoff>\n{{summary}}\n</handoff>\n")
BRANCH_SUMMARY_TEMPLATE = "Branch-return summary:\n\n<summary>\n{{summary}}\n</summary>\n"


def render_summary_template(template: str, summary: str) -> str:
    """`prompt.render(template, {summary})`: handlebars with noEscape, then the post-render format pass."""
    return prompt_format(template.replace("{{summary}}", summary))


# Prompts as embedded in omp 18.17.0 (src/compaction/prompts/ and src/compaction/compaction.ts).
AUTO_HANDOFF_THRESHOLD_FOCUS = "Threshold-triggered maintenance: preserve critical implementation state and immediate next actions."

HANDOFF_DOCUMENT_PROMPT_TEMPLATE = """<critical>
Write a handoff document for another instance of yourself.
The handoff MUST be sufficient for seamless continuation without access to this conversation.
Output ONLY the handoff document. No preamble, no commentary, no wrapper text.
</critical>

<instruction>
Capture exact technical state, not abstractions.
- File paths, symbol names, commands run
- Test results, observed failures
- Decisions made
- Partial work affecting the next step
Register: address the successor directly in the imperative ("Fix X", "Run Y") — never first person ("I need to…", "my attempt…").
The handoff mechanism is invisible to the document: NEVER list writing, generating, or delivering a handoff/summary/context document as progress or a next step. Progress and Next Steps cover the user's task only.
</instruction>

<output>
Use exactly this structure:

## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]

## Progress
### Done
- [x] [Completed tasks with specifics]

### In Progress
- [ ] [Current work if any]

### Pending
- [ ] [Tasks mentioned but not started]

## Key Decisions
- **[Decision]**: [Rationale]

## Critical Context
- Code snippets, file paths, function/type names, error messages, data essential to continue
- Repository state if relevant

## Next Steps
1. [What should happen next]
</output>"""


def render_handoff_prompt(focus: Optional[str] = AUTO_HANDOFF_THRESHOLD_FOCUS) -> str:
    text = HANDOFF_DOCUMENT_PROMPT_TEMPLATE
    if focus:
        text = f"{text}\n\n<instruction>\nAdditional focus: {focus}\n</instruction>"
    return prompt_format(text)


# --------------------------------------------------------------------------------------------- snapshot

DEFAULT_COMPAT = {
    # catalog/src/compat/resolve.ts for a local openai-compatible backend (llama.cpp / Splash).
    "supports_developer_role": False,
    "supports_multiple_system_messages": False,
    "requires_assistant_after_tool_result": False,
    "requires_tool_result_name": False,
    "replay_reasoning_content": True,
    "reasoning_content_field": "reasoning_content",
    "uses_openai_tool_call_id_limit": False,
}


def _capture_pairs(capture_dir: str) -> list:
    """[(ms, req_path, resp_path_or_None)] sorted by the capture's ms file stem."""
    out = []
    for req in sorted(glob.glob(os.path.join(capture_dir, "*_req.json"))):
        stem = os.path.basename(req)[:-len("_req.json")]
        resp = os.path.join(capture_dir, stem + "_resp.json")
        try:
            ms = int(stem)
        except ValueError:
            ms = None
        out.append((ms, req, resp if os.path.exists(resp) else None))
    return out


def resp_usage(resp_path: Optional[str]) -> dict:
    """The `usage` object of a captured SSE (or plain JSON) response; {} if none."""
    if not resp_path:
        return {}
    usage = {}
    with open(resp_path, encoding="utf-8") as f:
        text = f.read()
    if text.lstrip().startswith("{"):
        try:
            return json.loads(text).get("usage") or {}
        except json.JSONDecodeError:
            pass
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(chunk, dict) and chunk.get("usage"):
            usage = chunk["usage"]
    return usage


def match_captures(session: "omp_session.Session", capture_dir: str, max_skew_ms: int = 2000) -> list:
    """Pair captured requests with transcript turns: request start (file stem ms vs message.timestamp) within
    `max_skew_ms` and, when the response has usage, prompt_tokens == contextSnapshot.promptTokens.
    Returns [{"req", "resp", "ms", "turn": Turn|None, "prompt_tokens"}]."""
    turns = [t for s in session.streams() for t in s.turns]
    caps = [(ms, req, resp, resp_usage(resp).get("prompt_tokens")) for ms, req, resp in _capture_pairs(capture_dir)]
    cand = []
    for ci, (ms, _req, _resp, pt) in enumerate(caps):
        if ms is None:
            continue
        for ti, t in enumerate(turns):
            skew = abs(t.start * 1000 - ms)
            if skew > max_skew_ms or (pt is not None and t.prompt_tokens is not None and pt != t.prompt_tokens):
                continue
            cand.append((skew, ci, ti))
    cand.sort()
    by_cap: dict = {}
    used_turns = set()
    for _skew, ci, ti in cand:  # closest pairs first, each capture and each turn used once
        if ci in by_cap or ti in used_turns:
            continue
        by_cap[ci] = turns[ti]
        used_turns.add(ti)
    return [{"req": req, "resp": resp, "ms": ms, "turn": by_cap.get(ci), "prompt_tokens": pt}
            for ci, (ms, req, resp, pt) in enumerate(caps)]


def _digest_entry_before(stream: "omp_session.Stream", pos: int) -> Optional[str]:
    for e in reversed(stream.branch[:pos]):
        if e.get("type") == "custom" and e.get("customType") == "system-prompt-digest":
            return e.get("data")
    return None


def _agent_key(stream: "omp_session.Stream", pos: Optional[int] = None) -> str:
    if stream.kind == "main":
        return "main"
    init = session_init_at(stream, pos)
    return (init or {}).get("agent") or "subagent"


def snapshot_from_capture(capture_dir: str, session_path: str, omp_version: Optional[str] = None) -> dict:
    """Build a snapshot from a proxy capture and the omp session it recorded.

    main system prompt: the captured main request's system message, split into blocks so that it reproduces the
    session's `system-prompt-digest`. tools: per agent (`main`, and each captured subagent's agent name) the
    captured tool list; `tool_pool` keeps every captured schema by agent and name. params: the request keys
    other than messages/tools, with the body key order."""
    sess = omp_session.load_session(session_path)
    matches = match_captures(sess, capture_dir)
    streams = {s.name: s for s in sess.streams()}
    snap: dict = {
        "format": SNAPSHOT_FORMAT, "omp_version": omp_version, "source": os.path.basename(os.path.normpath(capture_dir)),
        "provider": None, "model": None, "body_keys": None, "params": None, "system_prompt": {}, "tools": {},
        "tool_pool": {}, "compat": dict(DEFAULT_COMPAT), "notes": [],
    }
    for m in matches:
        t = m["turn"]
        if t is None:
            continue
        body = _load_json(m["req"])
        s = streams[t.stream]
        agent = _agent_key(s, t.branch_pos)
        if snap["params"] is None:
            snap["body_keys"] = list(body.keys())
            snap["params"] = {k: v for k, v in body.items() if k not in ("messages", "tools", "model")}
            snap["model"] = body.get("model")
            snap["provider"] = t.provider
        if agent not in snap["tools"]:
            snap["tools"][agent] = [tool["function"]["name"] for tool in body.get("tools") or []]
            snap["tool_pool"][agent] = {tool["function"]["name"]: tool for tool in body.get("tools") or []}
        if s.kind == "main" and "main" not in snap["system_prompt"]:
            sysmsg = [x for x in body["messages"] if x.get("role") in ("system", "developer")]
            text = "\n\n".join(x["content"] for x in sysmsg)
            digest = _digest_entry_before(s, t.branch_pos)
            blocks = split_blocks_for_digest(text, digest) if digest else None
            if blocks is None:
                blocks = [text]
                snap["notes"].append(f"main system prompt digest {digest} not reproduced; kept as one block")
            snap["system_prompt"]["main"] = blocks
            snap["system_prompt_digest"] = system_prompt_digest(blocks)
    subs = [k for k in snap["tools"] if k != "main"]
    if subs:
        # Tools a subagent's own capture lacks come from here. grep/glob descriptions depend on the session's spawn
        # policy (scout spawnable or not) and the model's delegation bias; a captured subagent with the same
        # `session_init.spawns` renders them the same way (tools/grep.ts, tools/glob.ts).
        snap["default_subagent_pool"] = subs[0]
    if not snap["omp_version"]:
        for e in sess.main.branch:
            v = (e.get("runtime") or {}).get("version")
            if v:
                snap["omp_version"] = v
                break
    return snap


_MODEL_LINE = re.compile(r"^- Model: (.+)$", re.M)


def snapshot_from_capture_only(capture_dir: str, subagent_agent: Optional[str] = None,
                               digest: Optional[str] = None, omp_version: Optional[str] = None) -> dict:
    """`snapshot_from_capture` where the capture's omp session is not at hand (another Mac).

    Requests are grouped by their tool-name list: the group with the most tools is the main agent, a second group is
    the subagent named `subagent_agent` (its `session_init.agent`, e.g. `sonic`). Every request of a group must carry
    the same system message and tool schemas. params, body key order and model come from the first request; the
    provider from the system prompt's `- Model: PROVIDER/MODEL` line. The main prompt is split into blocks with
    `digest` when given (`split_blocks_for_digest`), else kept as one block. Equal to the session-based snapshot
    on the golden capture apart from `source` notes (tests)."""
    groups: dict = {}
    first_body = None
    for _ms, req, _resp in _capture_pairs(capture_dir):
        body = _load_json(req)
        if first_body is None:
            first_body = body
        names = tuple(t["function"]["name"] for t in body.get("tools") or [])
        sysmsg = "\n\n".join(x["content"] for x in body["messages"] if x.get("role") in ("system", "developer"))
        g = groups.setdefault(names, {"system": sysmsg, "tools": body.get("tools") or []})
        if g["system"] != sysmsg or g["tools"] != (body.get("tools") or []):
            raise ValueError(f"{req}: system prompt or tool schemas differ within the {len(names)}-tool group")
    if first_body is None:
        raise ValueError(f"{capture_dir}: no *_req.json")
    order = sorted(groups, key=len, reverse=True)
    if len(order) > 2 or (len(order) == 2 and not subagent_agent):
        raise ValueError(f"{len(order)} tool groups in the capture; name the subagent with subagent_agent")
    model = first_body.get("model")
    m = _MODEL_LINE.search(groups[order[0]]["system"])
    if not m or not m.group(1).endswith("/" + model):
        raise ValueError(f"main system prompt has no '- Model: PROVIDER/{model}' line")
    snap: dict = {
        "format": SNAPSHOT_FORMAT, "omp_version": omp_version, "source": os.path.basename(os.path.normpath(capture_dir)),
        "provider": m.group(1)[:-len(model) - 1], "model": model, "body_keys": list(first_body.keys()),
        "params": {k: v for k, v in first_body.items() if k not in ("messages", "tools", "model")},
        "system_prompt": {}, "tools": {}, "tool_pool": {}, "compat": dict(DEFAULT_COMPAT), "notes": [],
    }
    for names, agent in zip(order, ["main", subagent_agent]):
        snap["tools"][agent] = list(names)
        snap["tool_pool"][agent] = {t["function"]["name"]: t for t in groups[names]["tools"]}
    text = groups[order[0]]["system"]
    blocks = split_blocks_for_digest(text, digest) if digest else [text]
    if blocks is None:
        blocks = [text]
        snap["notes"].append(f"main system prompt digest {digest} not reproduced; kept as one block")
    snap["system_prompt"]["main"] = blocks
    snap["system_prompt_digest"] = system_prompt_digest(blocks)
    if subagent_agent and len(order) == 2:
        snap["default_subagent_pool"] = subagent_agent
    snap["notes"].append("built from the capture alone (no omp session)")
    return snap


def model_line(provider: str, model: str) -> str:
    return f"- Model: {provider}/{model}"


def retarget_snapshot(snap: dict, provider: str, model: str) -> dict:
    """Same snapshot for another recorded provider/model: the system prompt's `- Model:` line and the body model."""
    out = copy.deepcopy(snap)
    old = model_line(snap["provider"], snap["model"])
    new = model_line(provider, model)
    for agent, blocks in out["system_prompt"].items():
        if not any(old in b for b in blocks):
            raise ValueError(f"system prompt for {agent} has no line {old!r}")
        out["system_prompt"][agent] = [b.replace(old, new) for b in blocks]
        if agent == "main":
            out["system_prompt_digest"] = system_prompt_digest(out["system_prompt"][agent])
    out["provider"], out["model"] = provider, model
    out.setdefault("notes", []).append(f"retargeted from {snap['provider']}/{snap['model']}")
    return out


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_snapshot(path: str) -> dict:
    snap = _load_json(path)
    if snap.get("format") != SNAPSHOT_FORMAT:
        raise ValueError(f"{path}: not a {SNAPSHOT_FORMAT} snapshot")
    return snap


# --------------------------------------------------------------------------------------------- stream helpers

def session_init_at(stream: "omp_session.Stream", pos: Optional[int] = None) -> Optional[dict]:
    """Latest `session_init` on the branch before `pos` (omp appends a newer one when the prompt base changes)."""
    found = None
    for e in stream.branch[: pos if pos is not None else len(stream.branch)]:
        if e.get("type") == "session_init":
            found = e
    return found


def stream_tools(stream: "omp_session.Stream", snap: dict, pos: Optional[int] = None,
                 tool_sources: Optional[dict] = None) -> list:
    """Tool schemas for a stream's request.

    main: `snap["tools"]["main"]`. subagent: the names of `session_init.tools` minus `mountedTools` (both recorded),
    in that order (checked on the golden subagent); each schema from the pool named by `tool_sources[name]`, else
    the agent's own captured pool, else the default subagent pool (`snap["default_subagent_pool"]`), else main.
    Descriptions that depend on the other tools are re-rendered for this tool list (`adapt_tool_descriptions`)."""
    tool_sources = tool_sources or {}
    if stream.kind == "main":
        names = snap["tools"]["main"]
        agent = "main"
    else:
        init = session_init_at(stream, pos) or {}
        mounted = set(init.get("mountedTools") or [])
        names = [n for n in init.get("tools") or [] if n not in mounted]
        agent = init.get("agent") or "subagent"
    pools = snap["tool_pool"]
    default_sub = snap.get("default_subagent_pool") or next((k for k in pools if k != "main"), "main")
    out = []
    for n in names:
        order = [tool_sources.get(n), agent, default_sub, "main"] if stream.kind != "main" else ["main"]
        for p in order:
            if p and p in pools and n in pools[p]:
                out.append(pools[p][n])
                break
        else:
            raise KeyError(f"no schema for tool {n!r} (agent {agent}) in snapshot pools {list(pools)}")
    captured = list(pools.get(agent) or pools.get(default_sub) or {})
    return adapt_tool_descriptions(out, names, captured)


# Tool descriptions that render differently by which other tools the agent has (`prompts/tools/*.md`). Each entry:
# (tool, other tool, text when it is active, text when it is not).
_TOOL_DEPENDENT_LINES = [
    # tools/bash.ts `hasEval: isToolActive("eval", …)`, prompts/tools/bash.md line 2.
    ("bash", "eval", "Scripts/heredocs/`$(…)`/complex pipelines → `eval`.",
     "Scripts/heredocs/`$(…)`/complex flow → dedicated tool or checked-in script."),
    # tools/grep.ts, tools/glob.ts `hasFind`: a `{{#if hasFind}}` line before the eager-delegation line (an empty line
    # when off). Checked by count and Splash tool signature (case D, main and falsifiers).
    ("grep", "find", "(`src;!**/*.test.ts`).\nBehavior/unknown symbol → `find`; literals/regex → `grep`.\nMulti-round",
     "(`src;!**/*.test.ts`).\n\nMulti-round"),
    ("glob", "find", "dirs end `/`.\nBehavior search → `find`.\nMulti-round", "dirs end `/`.\n\nMulti-round"),
]

# omp 18.17.0 `tools/builtin-names.ts` BUILTIN_TOOL_NAMES: the order tools are created in (the request's tool order).
BUILTIN_TOOL_ORDER = (
    "read", "bash", "edit", "ast_grep", "ast_edit", "ask", "debug", "ida", "eval", "github", "glob", "grep", "find",
    "lsp", "checkpoint", "rewind", "context_notes", "new_context", "security_scan", "task", "wait", "todo",
    "web_search", "write", "memory_edit", "retain", "recall", "reflect", "learn", "manage_skill",
)

# Tools a snapshot can gain that the capture did not have: (inventory label, schema). `find` (omp 18.17.0
# `tools/jfind/index.ts`): `find.md` is imported raw (no render, trailing newline kept), schema
# `{query: string, grep_keywords: string[], path?: string}` plus the intent field; proven by Splash tool signature
# on case D's falsifiers and main agent.
FIND_DESCRIPTION = (
    "Describe behavior, get implementing files and line ranges. MUST use first for unknown locations; known "
    "strings/regex/symbols → `grep`, names → `glob`.\n"
    "`query`: plain language, not regex; quoted phrases match whole. `grep_keywords`: likely verbatim terms, `[]` if "
    "unsure.\n"
    "`path`: one directory or file, host path or internal URL (`omp://`, `omp://<file>.md`, `skill://<name>`, "
    "`local://…`); omitted = workspace root. Scope known subsystem; batch related questions. Searches live files, no "
    "index; no `:start-end` selector (judges whole files).\n"
    "Hits strongest first: `path:start-end score snippet` (workspace-relative, or URL under URL scope); read returned "
    "ranges. Scores: absolute comparable 0–1 probability; below ~0.4 = weak evidence, so widen query or use `grep` "
    "before concluding absence.\n")
EXTRA_TOOLS = {
    "find": ("Find", {"type": "function", "function": {"name": "find", "description": FIND_DESCRIPTION, "parameters": {
        "type": "object", "properties": {"i": {"type": "string", "description": "concise intent"},
                                         "query": {"type": "string"},
                                         "grep_keywords": {"type": "array", "items": {"type": "string"}},
                                         "path": {"type": "string"}},
        "required": ["query", "grep_keywords", "i"], "additionalProperties": False}}}),
}

_SESSION_TOOL_NAMES = re.compile(r"^Session tool names: (`[^`]+`(?:, `[^`]+`)*)\.$", re.M)


def adapt_tool_descriptions(tools: list, active: list, captured: Optional[list] = None) -> list:
    """A pool schema was captured for an agent with other tools: re-render the parts that depend on them for the
    `active` tool names: the lines in `_TOOL_DEPENDENT_LINES`, and eval's "Session tool names" list (the active
    tools in order, then the names that were not tools of the captured agent, `captured`: its xd:// devices).
    Checked by count and Splash tool signature on case A's falsifiers (bash without eval) and case D (find)."""
    out = []
    for tool in tools:
        fn = tool.get("function") or {}
        desc = fn.get("description")
        for name, other, on, off in _TOOL_DEPENDENT_LINES:
            if fn.get("name") != name or not isinstance(desc, str):
                continue
            want, have = (on, off) if other in active else (off, on)
            if have in desc:
                desc = desc.replace(have, want)
        if fn.get("name") == "eval" and isinstance(desc, str) and captured is not None:
            m = _SESSION_TOOL_NAMES.search(desc)
            if m:
                listed = re.findall(r"`([^`]+)`", m.group(1))
                names = list(active) + [n for n in listed if n not in captured and n not in active]
                desc = desc[:m.start(1)] + ", ".join(f"`{n}`" for n in names) + desc[m.end(1):]
        if desc != fn.get("description"):
            tool = dict(tool, function=dict(fn, description=desc))
        out.append(tool)
    return out


# --------------------------------------------------------------------------------------------- snapshot edits

# System prompt lines that depend on whether a tool is active (omp 18.17.0 `prompts/system/system-prompt.md`
# `{{#has tools "find"}}`): (tool, text when active, text when not). The Tool Inventory list is rebuilt separately.
_PROMPT_TOOL_LINES = [
    ("find", "- Unknown behavior/location: descriptive `find` FIRST; NEVER guess `grep`/`glob` targets.\n"
             "- Regex/literal/known-symbol search: `grep`,", "- Regex/target search: `grep`,"),
    ("find", "NEVER open guessed files. Read `find` hits only.", "NEVER open guessed files."),
    ("find", "decide after your own first `find`/`grep`/`read`,", "decide after your own first `grep`/`read`,"),
    ("find", "Scope with `find`/`grep`/`read`/`glob` yourself;", "Scope with `grep`/`read`/`glob` yourself;"),
]

_INVENTORY = re.compile(r"^# Tool Inventory\n((?:- [^\n]*\n?)+)", re.M)
_INVENTORY_LINE = re.compile(r"^- (?:(.+): )?`([^`]+)`$")


def adapt_prompt_tools(text: str, tools: list, labels: Optional[dict] = None) -> str:
    """Re-render a main system prompt for the tool list `tools` (request order): the Tool Inventory (labels from the
    prompt's own list, else `labels`, else EXTRA_TOOLS) and the `_PROMPT_TOOL_LINES`."""
    m = _INVENTORY.search(text)
    if m:
        known = {}
        for line in m.group(1).splitlines():
            lm = _INVENTORY_LINE.match(line)
            if lm:
                known[lm.group(2)] = lm.group(1)
        lines = []
        for n in tools:
            label = known.get(n) or (labels or {}).get(n) or EXTRA_TOOLS.get(n, (None,))[0]
            lines.append(f"- {label}: `{n}`" if label else f"- `{n}`")
        block = "\n".join(lines) + ("\n" if m.group(1).endswith("\n") else "")
        text = text[:m.start(1)] + block + text[m.end(1):]
    for name, on, off in _PROMPT_TOOL_LINES:
        want, have = (on, off) if name in tools else (off, on)
        if have in text and not (have in want and want in text):  # an "off" text can be part of its "on" text
            text = text.replace(have, want)
    return text


_FILE_BLOCK = '<file path="{path}">\n'


def replace_context_file(text: str, path: str, content: str) -> str:
    """Swap a context file (AGENTS.md, …) embedded as `<file path="PATH">\\n…\\n</file>` for `content` as omp
    renders it (the prompt's `format` post-pass)."""
    head = _FILE_BLOCK.format(path=path)
    i = text.find(head)
    if i < 0:
        raise ValueError(f"no context file {path!r} in the system prompt")
    j = text.find("\n</file>", i + len(head))
    if j < 0:
        raise ValueError(f"context file {path!r} has no closing </file>")
    return text[:i + len(head)] + prompt_format(content) + text[j:]


_AGENT_LINE = "- `{name}`"
_BLOCKING_TAG = " (BLOCKING; inline result)"
_BLOCKING_INTRO = " BLOCKING agents return inline."


def set_agent_blocking(desc: str, agent: str, blocking: bool) -> str:
    """The task tool description with `agent` blocking or not (`prompts/tools/task.md`: the agent's
    `(BLOCKING; inline result)` tag and the first line's `BLOCKING agents return inline.` while any agent blocks)."""
    lines = desc.split("\n")
    head = _AGENT_LINE.format(name=agent)
    idx = next((k for k, ln in enumerate(lines) if ln.startswith(head) and ln[len(head):len(head) + 1] in (" ", ":")),
               None)
    if idx is None:
        raise ValueError(f"task description lists no agent {agent!r}")
    ln = lines[idx]
    if blocking and _BLOCKING_TAG not in ln:
        rest = ln[len(head):]
        ro = " (READ-ONLY; investigation only, no edits)"
        cut = len(head) + (len(ro) if rest.startswith(ro) else 0)
        lines[idx] = ln[:cut] + _BLOCKING_TAG + ln[cut:]
    elif not blocking:
        lines[idx] = ln.replace(_BLOCKING_TAG, "", 1)
    any_blocking = any(_BLOCKING_TAG in x for x in lines)
    first = lines[0].replace(_BLOCKING_INTRO, "")
    if any_blocking and first.endswith("immediately."):
        first += _BLOCKING_INTRO
    lines[0] = first
    return "\n".join(lines)


def adapt_snapshot(snap: dict, context_files: Optional[dict] = None, enable_tools: tuple = (),
                   disable_tools: tuple = (), agent_blocking: Optional[dict] = None,
                   digest: Optional[str] = None) -> dict:
    """The snapshot for a session whose checkout differed from the capture's: context files (`{path: text}`, e.g.
    AGENTS.md at the session's commit), main tools enabled (`EXTRA_TOOLS` or already pooled) or disabled by the
    project's `.omp/config.yml`, task agents blocking or not (`{agent: bool}`, `.omp/agents/*.md` `blocking:`).
    The main prompt is re-rendered for the new tool list and split into the blocks giving `digest` when given."""
    out = copy.deepcopy(snap)
    text = "\n\n".join(out["system_prompt"]["main"])
    for path, content in (context_files or {}).items():
        text = replace_context_file(text, path, content)
    pool = out["tool_pool"]["main"]
    names = [n for n in out["tools"]["main"] if n not in disable_tools]
    for n in enable_tools:
        if n in names:
            continue
        if n not in pool:
            if n not in EXTRA_TOOLS:
                raise ValueError(f"no schema for tool {n!r} (known: {sorted(EXTRA_TOOLS)})")
            pool[n] = copy.deepcopy(EXTRA_TOOLS[n][1])
        names.append(n)
    rank = {n: i for i, n in enumerate(BUILTIN_TOOL_ORDER)}
    names.sort(key=lambda n: rank.get(n, len(rank)))  # stable: non-builtins keep their place after the builtins
    if names != out["tools"]["main"]:
        text = adapt_prompt_tools(text, names)
        out["tools"]["main"] = names
    for agent, blocking in (agent_blocking or {}).items():
        fn = pool["task"]["function"]
        pool["task"] = dict(pool["task"], function=dict(fn, description=set_agent_blocking(
            fn["description"], agent, blocking)))
    blocks = [text]
    if digest:
        blocks = split_blocks_for_digest(text, digest)
        if blocks is None:
            raise ValueError(f"no block split of the adapted main prompt gives digest {digest}")
    elif len(snap["system_prompt"]["main"]) == 2:  # keep the capture's block boundary (before <project-context>)
        k = text.find("\n\n<project-context>")
        blocks = [text[:k], text[k + 2:]] if k >= 0 else [text]
    out["system_prompt"]["main"] = blocks
    out["system_prompt_digest"] = system_prompt_digest(blocks)
    out.setdefault("notes", []).append(
        f"adapted: context files {sorted(context_files or {})}, tools +{list(enable_tools)} -{list(disable_tools)}, "
        f"blocking {agent_blocking or {}}")
    return out


def stream_system_blocks(stream: "omp_session.Stream", snap: dict, pos: Optional[int] = None) -> list:
    if stream.kind == "main":
        return list(snap["system_prompt"]["main"])
    init = session_init_at(stream, pos)
    sp = (init or {}).get("systemPrompt")
    if sp is None:
        raise KeyError(f"{stream.name}: no session_init.systemPrompt")
    return [sp] if isinstance(sp, str) else list(sp)


# --------------------------------------------------------------------------------------------- context

class RebuildError(Exception):
    pass


def _entry_ms(e: dict) -> int:
    return iso_to_ms(e["timestamp"])


def _custom_message(e: dict) -> Optional[dict]:
    content = e.get("content")
    if not (isinstance(content, str) or isinstance(content, list)):
        return None
    msg = {"role": "custom", "customType": e.get("customType"), "content": content, "display": e.get("display"),
           "details": e.get("details"), "timestamp": _entry_ms(e)}
    if e.get("attribution") is not None:
        msg["attribution"] = e["attribution"]
    return msg


def _is_empty_error_turn(m: dict) -> bool:
    if m.get("stopReason") != "error":
        return False
    for b in m.get("content") or []:
        t = b.get("type")
        if t == "text" and not js_is_blank(b.get("text") or ""):
            return False
        if t == "thinking" and (not js_is_blank(b.get("thinking") or "")
                                or not js_is_blank(b.get("thinkingSignature") or "")):
            return False
        if t == "redactedThinking" and not js_is_blank(b.get("data") or ""):
            return False
        if t not in ("text", "thinking", "redactedThinking", "fallback"):
            return False
    return True


def _render_running_wait(details: dict) -> Optional[str]:
    """`buildJobResult` text (18.17.0) for a wait whose jobs are all still running."""
    jobs = (details or {}).get("jobs")
    if not jobs or any(j.get("status") != "running" for j in jobs) or details.get("cancelled") or details.get("agents"):
        return None
    if any(j.get("type") == "task" for j in jobs):
        return None  # running task rows carry a progress suffix the details may not hold
    lines = [f"## Still Running ({len(jobs)})\n"]
    for j in jobs:
        intent = f"{j['intent']} — " if j.get("intent") else ""
        lines.append(f"- `{j['id']}` [{j['type']}] — {intent}{j.get('label', '')}")
    return "\n".join(lines).rstrip(_JS_WS)


WAIT_LADDER_MS = (5_000, 10_000, 30_000, 60_000, 300_000)
WAIT_LADDER_RESET_MS = 60_000
WAIT_NOTICE_RUNG = 1

WAIT_IDLE_NOTICE = (
    "<system-notice>\n"
    "Nothing new after {waited} of back-to-back waits. Waiting is not progress: each response MUST advance the task. "
    "Make as much progress as you can on everything that does not need what you are waiting on; call `wait` again "
    "only once nothing else is left. Results and messages arrive on their own.\n"
    "</system-notice>"
)


def format_duration(ms: int) -> str:
    if ms <= 0:
        return "0ms"
    if ms < 1000:
        return f"{int(ms)}ms"
    if ms < 60_000:
        tenths = round(ms / 100)
        return f"{tenths / 10:.1f}s" if tenths < 600 else "1m"
    if ms < 3_600_000:
        mins = ms // 60_000
        secs = (ms % 60_000) // 1000
        return f"{mins}m{secs}s" if secs > 0 else f"{mins}m"
    if ms < 86_400_000:
        hours = ms // 3_600_000
        mins = (ms % 3_600_000) // 60_000
        return f"{hours}h{mins}m" if mins > 0 else f"{hours}h"
    days = ms // 86_400_000
    hours = (ms % 86_400_000) // 3_600_000
    return f"{days}d{hours}h" if hours > 0 else f"{days}d"


def _unprune(m: dict, request_ms: Optional[int], notes: list, notice: Optional[str] = None) -> dict:
    """A tool result as a request at `request_ms` saw it: before `prunedAt`, the original content."""
    pruned = m.get("prunedAt")
    if pruned is None or request_ms is None or request_ms >= pruned:
        return m
    text = _render_running_wait(m.get("details")) if m.get("toolName") == "wait" else None
    if text is None:
        notes.append(f"tool result {m.get('toolCallId')} ({m.get('toolName')}) pruned at {pruned}: original "
                     f"content not stored, sent as pruned")
        return m
    notes.append(f"tool result {m.get('toolCallId')} (wait) pruned at {pruned}: re-rendered from details")
    out = {k: v for k, v in m.items() if k != "prunedAt"}
    if notice:
        out["content"] = [{"type": "text", "text": text}, {"type": "text", "text": notice}]
    else:
        out["content"] = [{"type": "text", "text": text}]
    return out


def build_agent_messages(path: list, request_ms: Optional[int] = None, notes: Optional[list] = None) -> list:
    """`buildSessionContext(entries, leaf)` (non-transcript) over `path` (root -> leaf entries): AgentMessages."""
    notes = notes if notes is not None else []
    compaction = None
    for e in path:
        if e.get("type") == "compaction":
            compaction = e
    reset_idx = max((i for i, e in enumerate(path) if e.get("type") == "reset_boundary"), default=-1)
    messages: list = []

    rewind_ids = set()
    for i in range(1, len(path)):
        if (path[i].get("type") == "custom_message" and path[i].get("customType") == "rewind-report"
                and path[i - 1].get("type") == "branch_summary"):
            rewind_ids.add(path[i - 1]["id"])

    # Track idle wait streak over the active branch up to each wait result.
    idle: Optional[dict] = None
    notices: dict = {}
    for e in path:
        if e.get("type") == "message":
            m = e.get("message") or {}
            if m.get("role") == "toolResult" and m.get("toolName") == "wait":
                details = m.get("details") or {}
                jobs = details.get("jobs") or []
                is_idle = bool(jobs and all(j.get("status") == "running" for j in jobs)
                               and not details.get("cancelled") and not details.get("agents")
                               and not details.get("interrupted"))
                if is_idle:
                    ts = m.get("timestamp") or _entry_ms(e)
                    if idle is not None and (ts - idle["endedAt"]) < WAIT_LADDER_RESET_MS:
                        rung = min(idle["rung"] + 1, len(WAIT_LADDER_MS) - 1)
                        waited_ms = idle["waitedMs"] + WAIT_LADDER_MS[rung]
                    else:
                        rung = 0
                        waited_ms = WAIT_LADDER_MS[0]
                    idle = {"rung": rung, "endedAt": ts, "waitedMs": waited_ms}
                    if rung >= WAIT_NOTICE_RUNG:
                        notice = WAIT_IDLE_NOTICE.format(waited=format_duration(waited_ms))
                    else:
                        notice = None
                else:
                    idle = None
                    notice = None
                if e.get("id"):
                    notices[e["id"]] = notice
                notices[id(m)] = notice

    def append(e: dict) -> None:
        t = e.get("type")
        if t == "message":
            m = e["message"]
            if m.get("role") == "assistant" and (m.get("retryRecovery") or _is_empty_error_turn(m)):
                return
            if m.get("role") == "toolResult":
                notice = notices.get(id(m)) or notices.get(e.get("id"))
                m = _unprune(m, request_ms, notes, notice)
            messages.append(m)
        elif t == "custom_message":
            if e.get("customType") in ("prewalk-plan", "vibe-mode-context"):
                return
            cm = _custom_message(e)
            if cm:
                messages.append(cm)
        elif t == "branch_summary" and e.get("summary") and e["id"] not in rewind_ids:
            messages.append({"role": "branchSummary", "summary": e["summary"], "fromId": e.get("fromId"),
                             "timestamp": _entry_ms(e)})

    comp_idx = next((i for i, e in enumerate(path) if compaction is not None and e is compaction), -1)
    if reset_idx >= 0 and reset_idx > comp_idx:
        for e in path[reset_idx + 1:]:
            append(e)
    elif compaction is not None:
        pd = compaction.get("preserveData") or {}
        if pd.get("openaiRemoteCompaction") or pd.get("anthropicCompaction") or pd.get("snapcompact"):
            raise RebuildError(f"compaction {compaction['id']}: provider/snapcompact payloads not supported")
        details = compaction.get("details")
        if isinstance(details, dict) and details.get("kind") == "experimental-context-rollover":
            raise RebuildError(f"compaction {compaction['id']}: context-rollover not supported")
        messages.append({"role": "compactionSummary", "summary": compaction.get("summary") or "",
                         "tokensBefore": compaction.get("tokensBefore"), "method": compaction.get("method"),
                         "timestamp": _entry_ms(compaction)})
        first_kept = next((i for i, e in enumerate(path[:comp_idx])
                           if e.get("id") == compaction.get("firstKeptEntryId")), -1)
        if first_kept >= 0:
            for e in path[first_kept:comp_idx]:
                append(e)
        for e in path[comp_idx + 1:]:
            append(e)
    else:
        for e in path:
            append(e)

    # Strip dangling tool calls (no result on the path) and neutralise that turn's signatures.
    paired = {m.get("toolCallId") for m in messages if m.get("role") == "toolResult"}
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or []
        if not any(b.get("type") == "toolCall" and b.get("id") not in paired for b in content):
            continue
        kept = []
        for b in content:
            if (b.get("type") == "toolCall" and b.get("id") not in paired) or b.get("type") == "redactedThinking":
                continue
            if b.get("type") == "thinking" and b.get("thinkingSignature"):
                b = dict(b, thinkingSignature=None)
            kept.append(b)
        notes.append(f"stripped dangling tool calls from assistant at {m.get('timestamp')}")
        if not kept:
            del messages[i]
        else:
            messages[i] = dict(m, content=kept)

    # Drop aborted / error assistant turns and their paired results.
    i = len(messages) - 1
    while i >= 0:
        m = messages[i]
        if m.get("role") == "assistant" and m.get("stopReason") in ("aborted", "error"):
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            if not (nxt and nxt.get("role") == "custom" and nxt.get("customType") == "interrupted-thinking"):
                dropped = {b.get("id") for b in m.get("content") or [] if b.get("type") == "toolCall"}
                del messages[i]
                if dropped:
                    for j in range(len(messages) - 1, i - 1, -1):
                        if messages[j].get("role") == "toolResult" and messages[j].get("toolCallId") in dropped:
                            del messages[j]
                notes.append(f"dropped {m.get('stopReason')} assistant turn at {m.get('timestamp')}")
        i -= 1
    return messages


def _text_parts(content) -> list:
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


def convert_to_llm(agent_messages: list, notes: Optional[list] = None) -> list:
    """`convertToLlm` (coding-agent session/messages.ts + agent compaction/messages.ts): pi-ai Messages."""
    notes = notes if notes is not None else []
    out = []
    for idx, m in enumerate(agent_messages):
        role = m.get("role")
        if role == "custom":
            if m.get("customType") == "skill-prompt" and m.get("attribution") == "user":
                out.append({"role": "user", "content": _text_parts(m["content"]), "attribution": "user",
                            "timestamp": m["timestamp"]})
                continue
            if m.get("customType") == "collab-prompt" and m.get("attribution") == "user":
                raise RebuildError("collab steering messages are not supported")
            if any(isinstance(b, dict) and b.get("type") == "image" for b in _text_parts(m["content"])):
                raise RebuildError(f"image-bearing custom message {m.get('customType')} not supported")
            msg = {"role": "developer", "content": _text_parts(m["content"]), "timestamp": m["timestamp"]}
            if m.get("attribution") is not None:
                msg["attribution"] = m["attribution"]
            out.append(msg)
        elif role == "compactionSummary":
            tpl = HANDOFF_SUMMARY_TEMPLATE if m.get("method") == "handoff" else COMPACTION_SUMMARY_TEMPLATE
            out.append({"role": "user", "content": [{"type": "text", "text": render_summary_template(tpl, m["summary"])}],
                        "attribution": "agent", "timestamp": m["timestamp"]})
        elif role == "branchSummary":
            out.append({"role": "user", "content": [{"type": "text", "text": render_summary_template(
                BRANCH_SUMMARY_TEMPLATE, m["summary"])}], "attribution": "agent", "timestamp": m["timestamp"]})
        elif role == "user":
            out.append(dict(m, attribution=m.get("attribution") or "user"))
        elif role == "developer":
            out.append(dict(m, attribution=m.get("attribution") or "agent"))
        elif role == "assistant":
            nxt = agent_messages[idx + 1] if idx + 1 < len(agent_messages) else None
            if (nxt and nxt.get("role") == "custom" and nxt.get("customType") == "interrupted-thinking") or \
                    m.get("stopReason") == "aborted":
                raise RebuildError("interrupted assistant turns are not supported")
            out.append(m)
        elif role == "toolResult":
            r = dict(m, attribution=m.get("attribution") or "agent")
            if m.get("prunedAt") is not None:
                content = m.get("content") or []
                text = "".join(b.get("text", "") for b in content if b.get("type") == "text") or "[Output truncated]"
                first_text = next((k for k, b in enumerate(content) if b.get("type") == "text"), -1)
                if first_text < 0:
                    r["content"] = [{"type": "text", "text": text}] + list(content)
                else:
                    r["content"] = [b if b.get("type") != "text" else {"type": "text", "text": text}
                                    for k, b in enumerate(content) if b.get("type") != "text" or k == first_text]
            out.append(r)
        elif role in ("bashExecution", "pythonExecution", "fileMention", "hookMessage"):
            raise RebuildError(f"message role {role} not supported")
        else:
            notes.append(f"dropped message with role {role}")
    return out


# --------------------------------------------------------------------------------------------- date/cwd reminder

def _message_keys(messages: list) -> list:
    counts: dict = {}
    keys = []
    for m in messages:
        if m["role"] in ("user", "developer"):
            key = f"{m['role']}:{_js_number(m['timestamp'])}:{m.get('attribution') or ''}"
        else:
            key = f"{m['role']}:{_js_number(m['timestamp'])}"
        c = counts.get(key, 0)
        counts[key] = c + 1
        keys.append(key if c == 0 else f"{key}#{c}")
    return keys


def _inject_reminder(m: dict, reminder: str) -> dict:
    c = m["content"]
    return dict(m, content=f"{reminder}\n\n{c}" if isinstance(c, str) else [{"type": "text", "text": reminder}] + c)


def first_turn_hash(m: dict) -> str:
    """date-cwd-reminder `firstTurn`: Bun.hash(JSON.stringify([attribution ?? "", content])).toString(16)."""
    return bun_hash_hex(js_stringify([m.get("attribution") or "", m["content"]]))


def apply_date_cwd_reminders(messages: list, records: list, notes: Optional[list] = None) -> list:
    """Apply the recorded `date-cwd-reminder` placements of this conversation (records in branch order)."""
    notes = notes if notes is not None else []
    first = next((i for i, m in enumerate(messages) if m["role"] == "user"), -1)
    if first < 0:
        return messages
    keys = _message_keys(messages)
    root = keys[first]
    conv = root
    if not any(r.get("conversation") == root and r.get("turn") == root for r in records):
        fh = first_turn_hash(messages[first])
        hit = [r for r in records if r.get("firstTurn") == fh]
        if not hit:
            raise RebuildError(f"no date-cwd-reminder record for conversation {root}")
        conv = hit[-1]["conversation"]
        notes.append(f"date-cwd conversation {conv} matched by firstTurn hash for root {root}")
    prefix: dict = {}
    controls: dict = {}
    for r in records:
        if r.get("conversation") != conv:
            continue
        turn = root if r.get("turn") == conv else r.get("turn")
        if r.get("placement") == "control":
            controls.setdefault(turn, []).append(r["reminder"])
        else:
            prefix[turn] = r["reminder"]
    out = []
    for i, m in enumerate(messages):
        k = keys[i]
        if k in prefix and m["role"] == "user":
            rem = prefix[k]
            c = m["content"]
            already = (c.startswith(rem) if isinstance(c, str)
                       else bool(c) and c[0].get("type") == "text" and c[0].get("text") == rem)
            out.append(m if (already and k == root) else _inject_reminder(m, rem))
        else:
            out.append(m)
        for rem in controls.get(k, []):
            out.append({"role": "developer", "content": rem, "synthetic": True, "timestamp": None})
    return out


# --------------------------------------------------------------------------------------------- provider body

def convert_messages_openai(system_blocks: list, messages: list, compat: dict) -> list:
    """openai-completions `convertMessages` for a same-model replay (thinking replayed, ids kept)."""
    params: list = []
    blocks = [well_formed(b) for b in system_blocks if not js_is_blank(b)]
    if blocks:
        if compat.get("supports_multiple_system_messages"):
            params += [{"role": "system", "content": b} for b in blocks]
        else:
            params.append({"role": "system", "content": "\n\n".join(blocks)})
    dev_as_user = not compat.get("supports_developer_role")
    rfield = compat.get("reasoning_content_field") or "reasoning_content"
    limit = 40 if compat.get("uses_openai_tool_call_id_limit") else None
    norm_id = (lambda x: x[:limit]) if limit else (lambda x: x)
    last_role = None
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg["role"]
        if compat.get("requires_assistant_after_tool_result") and last_role == "toolResult" and role in ("user", "developer"):
            params.append({"role": "assistant", "content": "I have processed the tool results."})
        if role in ("user", "developer"):
            r = "developer" if (not dev_as_user and role == "developer") else "user"
            c = msg["content"]
            if isinstance(c, str):
                text = well_formed(c)
                if not js_is_blank(text):
                    params.append({"role": r, "content": text})
            else:
                parts = []
                for item in c:
                    if item.get("type") == "text":
                        text = well_formed(item.get("text") or "")
                        if not js_is_blank(text):
                            parts.append({"type": "text", "text": text})
                    else:
                        raise RebuildError("image content parts are not supported")
                if parts:
                    params.append({"role": "user", "content": parts})
            last_role = "system" if role == "developer" else role
        elif role == "assistant":
            am: dict = {"role": "assistant", "content": None}
            content = msg.get("content") or []
            texts = [b for b in content if b.get("type") == "text" and b.get("text") and not js_is_blank(b["text"])]
            if texts:
                am["content"] = "".join(well_formed(b["text"]) for b in texts)
            thinking = [b for b in content if b.get("type") == "thinking" and b.get("thinking")
                        and not js_is_blank(b["thinking"])]
            if thinking and compat.get("replay_reasoning_content"):
                sig = thinking[0].get("thinkingSignature")
                field = sig if sig in ("reasoning_content", "reasoning", "reasoning_text") else rfield
                am[field] = "\n".join(b["thinking"] for b in thinking)
            calls = [b for b in content if b.get("type") == "toolCall"]
            has_reasoning = any(k in am for k in ("reasoning_content", "reasoning", "reasoning_text"))
            if calls:
                am["tool_calls"] = [{"id": norm_id(tc["id"]), "type": "function",
                                     "function": {"name": tc["name"], "arguments": _serialize_args(tc.get("arguments"))}}
                                    for tc in calls]
                if any(tc.get("thoughtSignature") for tc in calls):
                    raise RebuildError("tool-call thought signatures are not supported")
            if am["content"] is None and (has_reasoning or am.get("tool_calls")):
                am["content"] = ""
            if not am["content"] and not am.get("tool_calls") and not has_reasoning:
                i += 1
                continue
            params.append(am)
            last_role = "assistant"
        elif role == "toolResult":
            j = i
            while j < len(messages) and messages[j]["role"] == "toolResult":
                tm = messages[j]
                if any(b.get("type") == "image" for b in tm.get("content") or []):
                    raise RebuildError("tool-result images are not supported")
                text = "\n".join(b.get("text", "") for b in tm.get("content") or [] if b.get("type") == "text")
                t = {"role": "tool", "content": well_formed(text), "tool_call_id": norm_id(tm["toolCallId"])}
                if compat.get("requires_tool_result_name") and tm.get("toolName"):
                    t["name"] = tm["toolName"]
                params.append(t)
                j += 1
            i = j
            last_role = "toolResult"
            continue
        i += 1
    return params


def _serialize_args(value) -> str:
    if isinstance(value, dict):
        return js_stringify(value)
    if isinstance(value, str):
        t = js_trim(value)
        if t:
            try:
                parsed = json.loads(t)
                if isinstance(parsed, dict):
                    return js_stringify(parsed)
            except json.JSONDecodeError:
                pass
    return "{}"


# --------------------------------------------------------------------------------------------- request

def turn_request_ms(stream: "omp_session.Stream", turn_index: int) -> int:
    e = stream.branch[stream.turns[turn_index].branch_pos]
    ts = (e.get("message") or {}).get("timestamp")
    return int(ts) if isinstance(ts, (int, float)) else _entry_ms(e)


def build_request(stream: "omp_session.Stream", turn_index: int, snap: dict, options: Optional[dict] = None,
                  notes: Optional[list] = None) -> dict:
    """The request body omp sent for assistant turn `turn_index` of `stream` (everything before that turn).

    options: `tool_sources` ({tool name: snapshot pool}) for subagent tools whose schema was not captured for that
    agent; `compat` overrides."""
    options = options or {}
    notes = notes if notes is not None else []
    turn = stream.turns[turn_index]
    pos = turn.branch_pos
    path = stream.branch[:pos]
    req_ms = turn_request_ms(stream, turn_index)
    compat = dict(DEFAULT_COMPAT, **(snap.get("compat") or {}), **(options.get("compat") or {}))
    agent_msgs = build_agent_messages(path, req_ms, notes)
    llm = convert_to_llm(agent_msgs, notes)
    records = [e["data"] for e in path if e.get("type") == "custom" and e.get("customType") == "date-cwd-reminder"
               and isinstance(e.get("data"), dict)]
    llm = apply_date_cwd_reminders(llm, records, notes)
    messages = convert_messages_openai(stream_system_blocks(stream, snap, pos), llm, compat)
    tools = stream_tools(stream, snap, pos, options.get("tool_sources"))
    model = turn.model or snap["model"]
    if snap.get("model") and model != snap["model"]:
        notes.append(f"turn model {model} differs from snapshot model {snap['model']}")
    values = dict(snap["params"], model=model, messages=messages)
    if tools:
        values["tools"] = tools
    keys = [k for k in snap["body_keys"] if k in values] + [k for k in values if k not in snap["body_keys"]]
    return {k: values[k] for k in keys}


def build_compaction_request(stream: "omp_session.Stream", compaction_index: int, snap: dict,
                             options: Optional[dict] = None, notes: Optional[list] = None) -> dict:
    """The request body omp sent to generate compaction `compaction_index` on `stream`.

    In omp, handoff compaction calls the model with the conversation history before the compaction entry,
    plus a synthetic user turn containing the handoff instructions, tool schemas provided, but
    `tool_choice: 'none'`."""
    options = options or {}
    notes = notes if notes is not None else []
    comp = stream.compactions[compaction_index]
    pos = comp.branch_pos
    path = stream.branch[:pos]
    req_ms = _entry_ms(stream.branch[pos])
    agent_msgs = build_agent_messages(path, req_ms, notes)
    if comp.method == "handoff":
        prompt_text = render_handoff_prompt(AUTO_HANDOFF_THRESHOLD_FOCUS)
        handoff_user_msg = {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
            "attribution": "agent",
            "timestamp": req_ms,
        }
        agent_msgs = agent_msgs + [handoff_user_msg]
    else:
        raise RebuildError(f"compaction method {comp.method!r} not supported")
    llm = convert_to_llm(agent_msgs, notes)
    records = [e["data"] for e in path if e.get("type") == "custom" and e.get("customType") == "date-cwd-reminder"
               and isinstance(e.get("data"), dict)]
    llm = apply_date_cwd_reminders(llm, records, notes)
    compat = dict(DEFAULT_COMPAT, **(snap.get("compat") or {}), **(options.get("compat") or {}))
    messages = convert_messages_openai(stream_system_blocks(stream, snap, pos), llm, compat)
    tools = stream_tools(stream, snap, pos, options.get("tool_sources"))
    values = dict(snap["params"], model=snap["model"], messages=messages)
    if tools:
        values["tools"] = tools
        values["tool_choice"] = "none"
    keys = [k for k in snap["body_keys"] if k in values] + [k for k in values if k not in snap["body_keys"]]
    return {k: values[k] for k in keys}


# --------------------------------------------------------------------------------------------- golden / diff

def json_diff(a, b, path: str = "$", out: Optional[list] = None, limit: int = 20) -> list:
    """Precise structural differences (JSON paths), at most `limit`."""
    out = out if out is not None else []
    if len(out) >= limit:
        return out
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                       and not isinstance(a, bool) and not isinstance(b, bool)):
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    elif isinstance(a, dict):
        for k in a.keys() | b.keys():
            if k not in a:
                out.append(f"{path}.{k}: missing in rebuilt")
            elif k not in b:
                out.append(f"{path}.{k}: extra in rebuilt")
            else:
                json_diff(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            json_diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        if isinstance(a, str) and isinstance(b, str):
            k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            out.append(f"{path}: strings differ at char {k} (len {len(a)} vs {len(b)}): "
                       f"{a[max(0, k - 30):k + 50]!r} vs {b[max(0, k - 30):k + 50]!r}")
        else:
            out.append(f"{path}: {a!r} != {b!r}")
    return out


def golden_check(session_path: str, capture_dir: str, snap: Optional[dict] = None) -> dict:
    """Rebuild every captured request that maps to a transcript turn and compare JSON-equal."""
    sess = omp_session.load_session(session_path)
    snap = snap or snapshot_from_capture(capture_dir, session_path)
    streams = {s.name: s for s in sess.streams()}
    rows = []
    for m in match_captures(sess, capture_dir):
        row = {"capture": os.path.basename(m["req"]), "prompt_tokens": m["prompt_tokens"]}
        t = m["turn"]
        if t is None:
            row["status"] = "unmatched"
            rows.append(row)
            continue
        row.update(stream=t.stream, turn=t.index)
        captured = _load_json(m["req"])
        notes: list = []
        try:
            rebuilt = build_request(streams[t.stream], t.index, snap, notes=notes)
        except (RebuildError, KeyError) as exc:
            row.update(status="error", error=str(exc))
            rows.append(row)
            continue
        diffs = json_diff(rebuilt, captured)
        row.update(status="match" if not diffs else "diff", diffs=diffs, notes=notes,
                   key_order=list(rebuilt) == list(captured),
                   bytes_equal=json.dumps(rebuilt, ensure_ascii=False) == json.dumps(captured, ensure_ascii=False))
        rows.append(row)
    matched = [r for r in rows if r["status"] != "unmatched"]
    return {"rows": rows, "matched": len(matched), "equal": sum(r["status"] == "match" for r in matched),
            "unmatched": [r["capture"] for r in rows if r["status"] == "unmatched"], "snapshot": snap}


def rebuild_all(session_path: str, snap: dict, options: Optional[dict] = None) -> dict:
    """Rebuild every turn of every stream (and compactions); collect errors and notes (no comparison)."""
    sess = omp_session.load_session(session_path)
    out = {"streams": {}, "errors": [], "notes": []}
    for s in sess.streams():
        ok = 0
        for t in s.turns:
            notes: list = []
            try:
                body = build_request(s, t.index, snap, options, notes)
                ok += 1
            except (RebuildError, KeyError) as exc:
                out["errors"].append(f"{s.name} #{t.index}: {exc}")
                continue
            for n in notes:
                out["notes"].append(f"{s.name} #{t.index}: {n}")
            del body
        comp_ok = 0
        for i, _c in enumerate(s.compactions):
            notes = []
            try:
                body = build_compaction_request(s, i, snap, options, notes)
                comp_ok += 1
            except (RebuildError, KeyError) as exc:
                out["errors"].append(f"{s.name} c#{i}: {exc}")
                continue
            for n in notes:
                out["notes"].append(f"{s.name} c#{i}: {n}")
            del body
        entry = {"turns": len(s.turns), "rebuilt": ok}
        if s.compactions:
            entry.update(compactions=len(s.compactions), compactions_rebuilt=comp_ok)
        out["streams"][s.name] = entry
    return out


# --------------------------------------------------------------------------------------------- CLI

def _cmd_snapshot(args) -> int:
    if bool(args.capture) == bool(args.from_snapshot):
        print("give exactly one of --capture and --from", file=sys.stderr)
        return 2
    if args.from_snapshot:
        snap = load_snapshot(args.from_snapshot)
    elif args.session:
        snap = snapshot_from_capture(args.capture, args.session, args.omp_version)
    else:
        snap = snapshot_from_capture_only(args.capture, args.subagent_agent, omp_version=args.omp_version)
    if args.retarget:
        prov, _, model = args.retarget.partition("/")
        snap = retarget_snapshot(snap, prov, model)
    if args.context_file or args.enable_tool or args.disable_tool or args.agent_blocking:
        files = {}
        for spec in args.context_file:
            path, _, src = spec.partition("=")
            with open(src, encoding="utf-8") as f:
                files[path] = f.read()
        blocking = {}
        for spec in args.agent_blocking:
            name, _, val = spec.partition("=")
            if val not in ("on", "off"):
                print(f"--agent-blocking {spec}: want NAME=on|off", file=sys.stderr)
                return 2
            blocking[name] = val == "on"
        try:
            snap = adapt_snapshot(snap, files, tuple(args.enable_tool), tuple(args.disable_tool), blocking,
                                  args.digest)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
    elif args.digest:  # split the (retargeted) main prompt into the blocks that give this digest
        blocks = split_blocks_for_digest("\n\n".join(snap["system_prompt"]["main"]), args.digest)
        if blocks is None:
            print(f"no block split of the main prompt gives digest {args.digest}", file=sys.stderr)
            return 2
        snap["system_prompt"]["main"] = blocks
        snap["system_prompt_digest"] = system_prompt_digest(blocks)
    if args.split_main_at is not None:
        text = "\n\n".join(snap["system_prompt"]["main"])
        k = args.split_main_at
        snap["system_prompt"]["main"] = [text[:k], text[k + 2:]] if text[k:k + 2] == "\n\n" else None
        if snap["system_prompt"]["main"] is None:
            print(f"no \\n\\n at offset {k}", file=sys.stderr)
            return 2
        snap["system_prompt_digest"] = system_prompt_digest(snap["system_prompt"]["main"])
    data = json.dumps(snap, ensure_ascii=False, indent=1)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(data + "\n")
        print(f"wrote {args.output}: provider {snap['provider']} model {snap['model']}, main prompt "
              f"{len(snap['system_prompt'].get('main', []))} blocks digest {snap.get('system_prompt_digest')}, "
              f"tools {[(k, len(v)) for k, v in snap['tools'].items()]}")
    else:
        print(data)
    return 0


def _cmd_golden(args) -> int:
    snap = load_snapshot(args.snapshot) if args.snapshot else None
    res = golden_check(args.session, args.capture, snap)
    for r in res["rows"]:
        where = f"{r.get('stream')}#{r.get('turn')}" if "stream" in r else "-"
        extra = ""
        if r["status"] == "match":
            extra = f"key order {'same' if r['key_order'] else 'DIFFERENT'}, bytes {'equal' if r['bytes_equal'] else 'differ'}"
        print(f"{r['capture']:<26} {where:<24} prompt_tokens={r['prompt_tokens']} {r['status']} {extra}")
        for d in r.get("diffs") or []:
            print(f"    {d}")
        if r.get("error"):
            print(f"    {r['error']}")
        for n in r.get("notes") or []:
            print(f"    note: {n}")
    print(f"golden: {res['equal']}/{res['matched']} rebuilt requests JSON-equal; unmatched captures: "
          f"{res['unmatched']}")
    return 0 if res["equal"] == res["matched"] and res["matched"] > 0 else 1


def _cmd_build(args) -> int:
    snap = load_snapshot(args.snapshot)
    options: dict = {}
    if args.tool_source:  # POOL=tool -> {tool: POOL}
        options["tool_sources"] = {t: p for p, t in (x.split("=", 1) for x in args.tool_source)}
    sess = omp_session.load_session(args.session)
    stream = sess.main if args.stream == "main" else sess.subagents[args.stream]
    if args.all:
        res = rebuild_all(args.session, snap, options)
        for name, st in res["streams"].items():
            comp_str = f", {st['compactions_rebuilt']}/{st['compactions']} compactions" if "compactions" in st else ""
            print(f"{name:<24} {st['rebuilt']}/{st['turns']} rebuilt{comp_str}")
        for n in res["notes"]:
            print(f"note: {n}")
        for e in res["errors"]:
            print(f"ERROR: {e}")
        return 1 if res["errors"] else 0
    notes: list = []
    if args.compaction is not None:
        body = build_compaction_request(stream, args.compaction, snap, options, notes)
    else:
        body = build_request(stream, args.turn, snap, options, notes)
    for n in notes:
        print(f"note: {n}", file=sys.stderr)
    json.dump(body, sys.stdout, ensure_ascii=False, indent=None if args.compact else 1)
    print()
    return 0


def _cmd_digest(args) -> int:
    snap = load_snapshot(args.snapshot)
    print(system_prompt_digest(snap["system_prompt"]["main"]))
    return 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild omp's chat-completions requests from a session transcript.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("snapshot", help="build a snapshot (system prompt, tools, params) from a proxy capture")
    p.add_argument("--capture", help="capture dir with <ms>_req.json / <ms>_resp.json pairs")
    p.add_argument("--from", dest="from_snapshot", metavar="SNAPSHOT",
                   help="start from this snapshot instead of a capture (with the edits below)")
    p.add_argument("--context-file", action="append", default=[], metavar="PATH=FILE",
                   help="replace the context file embedded as <file path=\"PATH\"> with FILE's text (e.g. AGENTS.md "
                   "at the session's commit)")
    p.add_argument("--enable-tool", action="append", default=[], metavar="NAME",
                   help="main agent has this tool too (find: omp 18.17.0 schema); re-renders the prompt's tool lines")
    p.add_argument("--disable-tool", action="append", default=[], metavar="NAME", help="main agent lacks this tool")
    p.add_argument("--agent-blocking", action="append", default=[], metavar="NAME=on|off",
                   help="task agent NAME blocking or not (.omp/agents/NAME.md `blocking:`)")
    p.add_argument("--session", help="the omp main session .jsonl that capture recorded; without it the snapshot is "
                   "built from the capture alone (see --subagent-agent, --digest)")
    p.add_argument("--subagent-agent", help="without --session: the agent name of the capture's subagent (e.g. sonic)")
    p.add_argument("--digest", help="split the main prompt (after --retarget) into blocks giving this "
                   "system-prompt-digest")
    p.add_argument("--omp-version", help="omp version to record (with --session: read from the session if unset)")
    p.add_argument("--retarget", help="PROVIDER/MODEL to retarget the system prompt Model line and body model")
    p.add_argument("--split-main-at", type=int, help="re-split the main prompt into 2 blocks at this \\n\\n offset")
    p.add_argument("-o", "--output")
    p.set_defaults(fn=_cmd_snapshot)
    p = sub.add_parser("golden", help="rebuild every captured request from the transcript and compare")
    p.add_argument("--capture", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--snapshot", help="use this snapshot instead of one built from the capture")
    p.set_defaults(fn=_cmd_golden)
    p = sub.add_parser("build", help="print the rebuilt request of one turn, or check all turns (--all)")
    p.add_argument("--session", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--stream", default="main", help="main or a subagent key")
    p.add_argument("--turn", type=int, default=0)
    p.add_argument("--compaction", type=int, help="compaction index to rebuild on stream")
    p.add_argument("--all", action="store_true", help="rebuild every turn of every stream, report errors")
    p.add_argument("--tool-source", action="append", default=[], metavar="POOL=TOOL",
                   help="take a subagent tool's schema from this snapshot pool (e.g. main=grep)")
    p.add_argument("--compact", action="store_true")
    p.set_defaults(fn=_cmd_build)
    p = sub.add_parser("digest", help="print the snapshot main system prompt digest")
    p.add_argument("--snapshot", required=True)
    p.set_defaults(fn=_cmd_digest)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
