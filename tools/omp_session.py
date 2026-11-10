#!/usr/bin/env python3
"""Read an omp session tree: a main session `.jsonl` plus its subagent sessions.

Layout (oh-my-pi `packages/coding-agent/src/session/`, `sub-sessions.ts`, `session-entries.ts`):

* `<dir>/<name>.jsonl` is a session file. Line 1.. are JSON records: a `title` slot record (no id), a `session`
  header, then entries that carry `id` / `parentId` / `timestamp` (`model_change`, `message`, `compaction`,
  `custom`, `custom_message`, `session_init`, `model_usage`, ...). The entries form a tree.
* Its subagents live in `<dir>/<name>/<AgentId>.jsonl`; their own children nest in `<dir>/<name>/<AgentId>/`.
  `__advisor*.jsonl` and `*.bak*` files share those directories and are not subagents.
* Active branch: omp replays the file in order; the active leaf is `activeLeafId` of the LAST entry when that key is
  present (late `model_usage` markers carry one), else the last entry's own id. If that id is not in the file, the
  newest entry that selects itself wins (`resolveSessionLeafId`). The branch is leaf -> root via `parentId`.
* Subagent link: the main `task` tool call (assistant message `toolCall` named `task`) is answered by a
  `toolResult` whose `details.results[]` lists `{index, id, agent}`; `id` is the subagent file stem. The subagent
  header also has `parentSession` (path of the spawning session file).

Stdlib only. CLI: `python3 tools/omp_session.py <main.jsonl> [--json]`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional


def parse_iso(ts: str) -> float:
    """ISO-8601 (`2026-11-06T20:25:59.011Z`) to epoch seconds."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


@dataclass
class Turn:
    """One assistant message (= one model request) on a stream's active branch."""
    stream: str                 # "main" or the subagent key ("P161CodeFalsify", "A/B" when nested)
    index: int                  # 0-based among the stream's active-branch turns
    entry_id: str
    parent_id: Optional[str]
    timestamp: str              # entry timestamp (when the response completed)
    start: float                # epoch seconds the request started (message.timestamp), ordering key
    end: float                  # epoch seconds the response completed
    prompt_tokens: Optional[int]
    non_message_tokens: Optional[int]
    tool_schema_tokens: Optional[int]
    compaction_epoch: Optional[int]
    stop_reason: Optional[str]
    tool_calls: list            # [{"id", "name"}]
    output_tokens: Optional[int] = None
    cache_read: Optional[int] = None
    duration_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    branch_pos: int = 0         # index of the entry in the stream's active branch


@dataclass
class Compaction:
    stream: str
    entry_id: str
    timestamp: str
    branch_pos: int
    turns_before: int           # active-branch turns before the compaction entry
    first_kept_entry_id: Optional[str]
    tokens_before: Optional[int]
    tokens_after: Optional[int]
    method: Optional[str]
    summary_chars: int


@dataclass
class Spawn:
    """A `task` tool call in `parent` that started `child` (a subagent stream key)."""
    parent: str
    parent_turn: Optional[int]      # index of the assistant turn holding the tool call
    parent_entry_id: Optional[str]
    tool_call_id: str
    child: str
    agent: Optional[str]
    task_index: Optional[int]
    call_timestamp: str
    result_timestamp: Optional[str]
    child_first_turn_start: Optional[float] = None
    linked_by: str = "tool-result"  # "tool-result" | "parentSession+time" (fallback)


@dataclass
class Stream:
    name: str
    path: str
    kind: str                       # "main" | "subagent"
    parent: Optional[str]           # parent stream key (None for main)
    header: Optional[dict]
    entries: list                   # all id-bearing entries in file order (dicts)
    leaf_id: Optional[str]
    branch: list                    # entries root -> leaf
    turns: list                     # [Turn]
    compactions: list               # [Compaction]
    session_init: Optional[dict]    # subagent: systemPrompt blocks, task, tools...
    abandoned_entries: int = 0      # id-bearing entries not on the active branch
    malformed_lines: int = 0

    @property
    def agent_id(self) -> str:
        return self.name.rsplit("/", 1)[-1]

    @property
    def max_prompt_tokens(self) -> int:
        return max((t.prompt_tokens or 0 for t in self.turns), default=0)


@dataclass
class Session:
    path: str
    main: Stream
    subagents: dict                 # key -> Stream
    spawns: list                    # [Spawn]
    unlinked: list = field(default_factory=list)   # subagent keys with no spawn found

    def streams(self) -> list:
        return [self.main] + list(self.subagents.values())

    def interleaved(self, include_subagents: bool = True) -> list:
        """All turns of all streams in the order the requests were started (stable; per-stream order kept)."""
        order = {s.name: i for i, s in enumerate(self.streams())}
        turns = [t for s in (self.streams() if include_subagents else [self.main]) for t in s.turns]
        return sorted(turns, key=lambda t: (t.start, order[t.stream], t.index))


# --------------------------------------------------------------------------------------------- loading

def read_records(path: str) -> tuple:
    """(records, malformed_count). Blank/invalid JSON lines are skipped like omp does."""
    out, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(rec, dict):
                out.append(rec)
            else:
                bad += 1
    return out, bad


def _entry_leaf(e: dict) -> Optional[str]:
    return e["activeLeafId"] if "activeLeafId" in e else e.get("id")


def resolve_leaf_id(entries: list) -> Optional[str]:
    """omp `resolveSessionLeafId`: replay in file order, the last entry decides."""
    if not entries:
        return None
    last = entries[-1]
    leaf = _entry_leaf(last)
    if leaf is None or leaf == last.get("id") or any(e.get("id") == leaf for e in entries):
        return leaf
    for e in reversed(entries):  # lastSelfSelectedLeafId
        if _entry_leaf(e) == e.get("id"):
            return e.get("id")
    return None


def active_branch(entries: list, leaf_id: Optional[str]) -> list:
    """Entries from root to `leaf_id`, following parentId (cycle safe)."""
    by_id = {e["id"]: e for e in entries}
    path, seen, cur = [], set(), leaf_id
    while cur is not None and cur in by_id and cur not in seen:
        seen.add(cur)
        e = by_id[cur]
        path.append(e)
        cur = e.get("parentId")
    path.reverse()
    return path


def _ms_to_s(v: Any) -> Optional[float]:
    return v / 1000.0 if isinstance(v, (int, float)) else None


def _make_turn(stream: str, index: int, pos: int, e: dict) -> Turn:
    m = e["message"]
    cs = m.get("contextSnapshot") or {}
    usage = m.get("usage") or {}
    end = _ms_to_s(m.get("completedAt"))
    if end is None:
        end = parse_iso(e["timestamp"])
    start = _ms_to_s(m.get("timestamp"))
    if start is None:
        start = end - (m.get("duration") or 0) / 1000.0
    content = m.get("content")
    calls = []
    if isinstance(content, list):
        calls = [{"id": c.get("id"), "name": c.get("name")}
                 for c in content if isinstance(c, dict) and c.get("type") == "toolCall"]
    return Turn(
        stream=stream, index=index, entry_id=e["id"], parent_id=e.get("parentId"), timestamp=e["timestamp"],
        start=start, end=end, prompt_tokens=cs.get("promptTokens"), non_message_tokens=cs.get("nonMessageTokens"),
        tool_schema_tokens=cs.get("toolSchemaTokens"), compaction_epoch=cs.get("compactionEpoch"),
        stop_reason=m.get("stopReason"), tool_calls=calls, output_tokens=usage.get("output"),
        cache_read=usage.get("cacheRead"), duration_ms=m.get("duration"), ttft_ms=m.get("ttft"),
        model=m.get("model"), provider=m.get("provider"), branch_pos=pos)


def load_stream(path: str, name: str = "main", kind: str = "main", parent: Optional[str] = None) -> Stream:
    records, bad = read_records(path)
    header = next((r for r in records if r.get("type") == "session"), None)
    entries = [r for r in records if r.get("type") not in ("session", "title") and "id" in r]
    leaf = resolve_leaf_id(entries)
    branch = active_branch(entries, leaf)
    turns, compactions = [], []
    for pos, e in enumerate(branch):
        if e.get("type") == "message" and (e.get("message") or {}).get("role") == "assistant":
            turns.append(_make_turn(name, len(turns), pos, e))
        elif e.get("type") == "compaction":
            compactions.append(Compaction(
                stream=name, entry_id=e["id"], timestamp=e["timestamp"], branch_pos=pos, turns_before=len(turns),
                first_kept_entry_id=e.get("firstKeptEntryId"), tokens_before=e.get("tokensBefore"),
                tokens_after=e.get("tokensAfter"), method=e.get("method"), summary_chars=len(e.get("summary") or "")))
    init = next((e for e in branch if e.get("type") == "session_init"), None)
    return Stream(name=name, path=path, kind=kind, parent=parent, header=header, entries=entries, leaf_id=leaf,
                  branch=branch, turns=turns, compactions=compactions, session_init=init,
                  abandoned_entries=len(entries) - len(branch), malformed_lines=bad)


def _is_subagent_file(name: str) -> bool:
    return name.endswith(".jsonl") and ".bak" not in name and not name.startswith("__advisor")


def _collect_subagents(directory: str, parent_key: Optional[str], out: dict) -> None:
    try:
        names = sorted(os.listdir(directory))
    except (FileNotFoundError, NotADirectoryError):
        return
    for fn in names:
        full = os.path.join(directory, fn)
        if not (os.path.isfile(full) and _is_subagent_file(fn)) or os.path.getsize(full) == 0:
            continue
        agent_id = fn[:-6]
        key = f"{parent_key}/{agent_id}" if parent_key else agent_id
        out[key] = load_stream(full, key, "subagent", parent_key)
        child_dir = os.path.join(directory, agent_id)
        if os.path.isdir(child_dir) and not os.path.islink(child_dir):
            _collect_subagents(child_dir, key, out)


def _tool_result_index(stream: Stream) -> dict:
    """toolCallId -> toolResult entry, over the whole file (results sit on the branch, but be lenient)."""
    out = {}
    for e in stream.branch:
        m = e.get("message") if e.get("type") == "message" else None
        if m and m.get("role") == "toolResult" and m.get("toolCallId"):
            out[m["toolCallId"]] = e
    return out


def _find_spawns(stream: Stream, subs: dict) -> list:
    """Task tool calls on `stream`'s active branch -> child streams."""
    results = _tool_result_index(stream)
    turn_by_entry = {t.entry_id: t for t in stream.turns}
    spawns = []
    prefix = "" if stream.kind == "main" else stream.name + "/"
    for e in stream.branch:
        if e.get("type") != "message" or (e.get("message") or {}).get("role") != "assistant":
            continue
        content = e["message"].get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if not (isinstance(c, dict) and c.get("type") == "toolCall" and c.get("name") == "task"):
                continue
            res = results.get(c.get("id"))
            details = ((res or {}).get("message") or {}).get("details") or {}
            turn = turn_by_entry.get(e["id"])
            for r in details.get("results") or []:
                rid = r.get("id")
                key = prefix + rid if rid else None
                if key not in subs:
                    continue
                child = subs[key]
                spawns.append(Spawn(
                    parent=stream.name, parent_turn=turn.index if turn else None, parent_entry_id=e["id"],
                    tool_call_id=c.get("id"), child=key, agent=r.get("agent"), task_index=r.get("index"),
                    call_timestamp=e["timestamp"], result_timestamp=res["timestamp"] if res else None,
                    child_first_turn_start=child.turns[0].start if child.turns else None))
    return spawns


def load_session(main_path: str) -> Session:
    main_path = os.path.abspath(main_path)
    main = load_stream(main_path)
    subs: dict = {}
    _collect_subagents(main_path[:-6] if main_path.endswith(".jsonl") else main_path, None, subs)
    spawns = _find_spawns(main, subs)
    for s in subs.values():
        spawns += _find_spawns(s, subs)
    linked = {sp.child for sp in spawns}
    unlinked = []
    # Fallback for a result that lost its details: parentSession header + a task call whose window holds the start.
    for key, s in subs.items():
        if key in linked:
            continue
        parent_stream = main if s.parent is None else subs.get(s.parent)
        hdr_parent = (s.header or {}).get("parentSession")
        if parent_stream is None or not s.turns or (hdr_parent and os.path.abspath(hdr_parent) != parent_stream.path):
            unlinked.append(key)
            continue
        t0 = s.turns[0].start
        cand = None
        for e in parent_stream.branch:
            m = e.get("message") if e.get("type") == "message" else None
            if not m or m.get("role") != "assistant" or not isinstance(m.get("content"), list):
                continue
            for c in m["content"]:
                if isinstance(c, dict) and c.get("type") == "toolCall" and c.get("name") == "task":
                    if parse_iso(e["timestamp"]) <= t0 + 1:
                        cand = (e, c)  # latest task call that started before the child's first request
        if cand is None:
            unlinked.append(key)
            continue
        e, c = cand
        turn = next((t for t in parent_stream.turns if t.entry_id == e["id"]), None)
        spawns.append(Spawn(parent=parent_stream.name, parent_turn=turn.index if turn else None,
                            parent_entry_id=e["id"], tool_call_id=c.get("id"), child=key,
                            agent=(s.session_init or {}).get("agent"), task_index=None, call_timestamp=e["timestamp"],
                            result_timestamp=None, child_first_turn_start=t0, linked_by="parentSession+time"))
    spawns.sort(key=lambda sp: (sp.call_timestamp, sp.task_index or 0))
    return Session(path=main_path, main=main, subagents=subs, spawns=spawns, unlinked=unlinked)


# --------------------------------------------------------------------------------------------- summary

def stream_summary(s: Stream) -> dict:
    pts = [t.prompt_tokens for t in s.turns if t.prompt_tokens is not None]
    return {
        "stream": s.name, "kind": s.kind, "file": os.path.basename(s.path), "turns": len(s.turns),
        "max_prompt_tokens": max(pts, default=0), "first_prompt_tokens": pts[0] if pts else None,
        "compactions": len(s.compactions), "epochs": sorted({t.compaction_epoch for t in s.turns
                                                             if t.compaction_epoch is not None}),
        "abandoned_entries": s.abandoned_entries, "malformed_lines": s.malformed_lines,
        "agent": ((s.session_init or {}).get("agent")), "start": s.turns[0].timestamp if s.turns else None,
        "end": s.turns[-1].timestamp if s.turns else None,
    }


def session_summary(sess: Session) -> dict:
    subs = [stream_summary(s) for s in sess.subagents.values()]
    return {
        "session": sess.path,
        "main": stream_summary(sess.main),
        "compactions": [asdict(c) for c in sess.main.compactions],
        "subagents": subs,
        "subagent_totals": {"count": len(subs), "turns": sum(x["turns"] for x in subs),
                            "max_prompt_tokens": max((x["max_prompt_tokens"] for x in subs), default=0)},
        "total_turns": len(sess.main.turns) + sum(x["turns"] for x in subs),
        "spawns": [asdict(sp) for sp in sess.spawns],
        "unlinked_subagents": sess.unlinked,
    }


def format_summary(sm: dict) -> str:
    m = sm["main"]
    L = [f"session: {sm['session']}",
         f"main: {m['turns']} turns, max promptTokens {m['max_prompt_tokens']}, {m['compactions']} compactions, "
         f"epochs {m['epochs']}, abandoned entries {m['abandoned_entries']}"]
    for c in sm["compactions"]:
        L.append(f"  compaction {c['entry_id']} at {c['timestamp']} after turn {c['turns_before']} "
                 f"(tokens {c['tokens_before']} -> {c['tokens_after']}, method {c['method']})")
    t = sm["subagent_totals"]
    L.append(f"subagents: {t['count']}, {t['turns']} turns, max promptTokens {t['max_prompt_tokens']}")
    spawn_of = {sp["child"]: sp for sp in sm["spawns"]}
    for x in sm["subagents"]:
        sp = spawn_of.get(x["stream"])
        where = f"spawned by {sp['parent']} turn {sp['parent_turn']} call {sp['tool_call_id']}" if sp else "UNLINKED"
        L.append(f"  {x['stream']:<24} {x['agent'] or '-':<14} {x['turns']:>4} turns, max {x['max_prompt_tokens']:>7}, "
                 f"{x['compactions']} compactions; {where}")
    L.append(f"total turns: {sm['total_turns']}")
    return "\n".join(L)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Summarise an omp session tree (main + subagents).")
    ap.add_argument("session", help="main session .jsonl (subagents are read from the sibling directory)")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    ap.add_argument("--turns", action="store_true", help="also list every turn, interleaved chronologically")
    args = ap.parse_args(argv)
    sess = load_session(args.session)
    sm = session_summary(sess)
    if args.turns:
        sm["interleaved"] = [asdict(t) for t in sess.interleaved()]
    if args.json:
        json.dump(sm, sys.stdout, indent=1)
        print()
    else:
        print(format_summary(sm))
        if args.turns:
            for t in sess.interleaved():
                print(f"{t.timestamp} {t.stream:<20} #{t.index:<4} prompt={t.prompt_tokens} epoch={t.compaction_epoch} "
                      f"stop={t.stop_reason} calls={[c['name'] for c in t.tool_calls]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
