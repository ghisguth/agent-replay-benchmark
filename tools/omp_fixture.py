#!/usr/bin/env python3
"""Build and check replay fixtures from omp sessions.

Converts recorded omp sessions (main agent, subagents, compactions) into a deduplicated,
redacted fixture JSON file (.json.gz) for long-context benchmark replays.

Subcommands:
    build-proxy: Build a fixture from proxy dumps (exact requests, recorded token counts).
    build: Load an omp session and snapshot, redact PII, optionally re-count tokens with SplashCounter,
           deduplicate messages into a pool, and save to fixtures/<name>.json.gz.
    check: Validate a fixture's structure, turn count, message reconstruction, and scan for PII.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omp_requests  # noqa: E402
import omp_session  # noqa: E402
import pii_patterns  # noqa: E402

# --------------------------------------------------------------------------------------------- Redaction rules
#
# Names, hosts and other personal strings are not in the repo. They come from a JSON file kept outside git
# (default sources/redact.json): a list of [regex, replacement] pairs applied in order with re.sub.
# Only generic rules live here.

DEFAULT_REDACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sources", "redact.json")

Rules = List[Tuple["re.Pattern[str]", str]]

# Neutral home-directory user name. It is 16 characters and tokenizes like a typical `first.last` login in the Qwen
# tokenizer (also inside `/Users/<name>/...`), so redaction barely moves prompt token counts.
HOME_USER = "standard.analyst"

# Generic rules, applied after the file rules: any other home directory becomes /Users/<HOME_USER>; the address in a
# co-author trailer (vendor tools add one to commit messages) becomes noreply@example.com.
BUILTIN_RULES: Rules = [
    (re.compile(r"/Users/(?!%s(?![A-Za-z0-9._-]))[A-Za-z0-9_][A-Za-z0-9._-]*" % re.escape(HOME_USER)),
     "/Users/" + HOME_USER),
    (re.compile(r"(?i)(co-authored-by:[^<\n]*<)[^<>\s]+@[^<>\s]+(>)"), r"\1noreply@example.com\2"),
]


def load_rules(path: str) -> Rules:
    """Load [regex, replacement] pairs from a JSON file; raise with a clear message."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"redaction rules file not found: {path} (pass --redact FILE; keep it out of git)")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of [regex, replacement] pairs")
    rules: Rules = []
    for i, pair in enumerate(data):
        if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)):
            raise ValueError(f"{path}: entry {i} is not a [regex, replacement] pair of strings")
        rules.append((re.compile(pair[0]), pair[1]))
    return rules


def redact_text(text: str, rules: Rules = ()) -> str:
    """Apply the given rules, then the built-in generic rules, to text."""
    for rx, repl in rules:
        text = rx.sub(repl, text)
    for rx, repl in BUILTIN_RULES:
        text = rx.sub(repl, text)
    return text


def redact_obj(obj: Any, rules: Rules = ()) -> Any:
    """Recursively redact strings (and dict keys) and drop 'syntax' keys from dicts."""
    if isinstance(obj, dict):
        return {
            redact_text(k, rules): redact_obj(v, rules)
            for k, v in obj.items()
            if k != "syntax"
        }
    elif isinstance(obj, list):
        return [redact_obj(v, rules) for v in obj]
    elif isinstance(obj, str):
        return redact_text(obj, rules)
    else:
        return obj


def redact_stream(stream: omp_session.Stream, rules: Rules = ()) -> None:
    """Redact a loaded stream in place upfront."""
    if stream.header:
        stream.header = redact_obj(stream.header, rules)
    if stream.session_init:
        stream.session_init = redact_obj(stream.session_init, rules)
    if stream.entries:
        stream.entries = redact_obj(stream.entries, rules)
    if stream.branch:
        stream.branch = redact_obj(stream.branch, rules)
    for t in stream.turns:
        if t.model:
            t.model = redact_text(t.model, rules)
        if t.provider:
            t.provider = redact_text(t.provider, rules)
        if t.tool_calls:
            t.tool_calls = redact_obj(t.tool_calls, rules)
        if t.timestamp:
            t.timestamp = redact_text(t.timestamp, rules)
    for c in stream.compactions:
        if c.timestamp:
            c.timestamp = redact_text(c.timestamp, rules)


# --------------------------------------------------------------------------------------------- PII check

# Values of the repo scanner's patterns that are expected in a fixture (placeholders and code false positives).
ALLOWED_VALUES = {
    "pii:email": {"dev@example.com", "noreply@example.com", "n@pytest.fixture"},
    "pii:ip": {"127.0.0.1", "0.0.0.0"},
    "id:home path": {"/Users/" + HOME_USER, "/Users/..."},
    # Case D: C# loop variables and prose about tokens, recipe words containing "mba" (checked by hand).
    "secret:assign": {"token = candidateTokens", "token: ExtractCandidateSatisfiers", "token: capers/drained",
                      "token: KeyHeadPhrase", "token: single-token"},
    "id:hostname": {"jambalaya", "chicken-jambalaya", "chicken-jambalaya-i", "SAMBAL", "cambala"},
}
INFO_ONLY = {"secret:hex40+"}  # git SHAs: counted, reported, not failed


def iter_strings(obj: Any):
    """Yield every string (keys and values) in a JSON-like object."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_strings(v)


def scan_pii(text_or_obj: Any, names: Optional[List[str]] = None, forbidden: Optional[List[str]] = None
             ) -> Tuple[List[str], Dict[str, int]]:
    """Scan with the repo scanner's patterns. Returns (issues, info counts).

    `names` are extra regexes reported as id:names (the scanner's --names format); `forbidden` are redact-file regexes.
    Both are fatal on any hit. Allowlisted values and INFO_ONLY patterns do not produce issues.
    """
    text = text_or_obj if isinstance(text_or_obj, str) else "\n".join(iter_strings(text_or_obj))
    pats = pii_patterns.patterns(list(names or []))
    for i, rx in enumerate(forbidden or []):  # one pattern each: redact-file regexes may carry inline (?i)
        pats[f"id:forbidden #{i}"] = rx
    counts, examples = pii_patterns.scan_text(text, pats)
    issues: List[str] = []
    info: Dict[str, int] = {}
    for name in pats:
        if not counts[name]:
            continue
        if name in INFO_ONLY:
            info[name] = counts[name]
            continue
        allowed = ALLOWED_VALUES.get(name, set())
        bad = {v: n for v, n in examples[name].items() if v not in allowed}
        if bad:
            sample = [v for v, _ in sorted(bad.items(), key=lambda kv: -kv[1])[:3]]
            issues.append(f"{name}: {sum(bad.values())} hit(s) e.g. {sample}")
    if re.search(r'"syntax"\s*:', text):
        issues.append("syntax field present")
    return issues, info


# --------------------------------------------------------------------------------------------- Fixture builder

FIXTURE_VERSION = 1

def build_fixture(
    session_path: str,
    snapshot_path: str,
    counter: Optional[Any] = None,
    suite: str = "conv-2",
    case: str = "case-a",
    rules: Rules = (),
) -> dict:
    """Build a complete fixture dictionary from session and snapshot.

    `rules` are the redaction rules (see load_rules). With a counter, prompt_tokens is the re-count of the redacted
    request and the compactions' recorded_prompt_tokens is the count of the unredacted request.
    """
    sess = omp_session.load_session(session_path)
    snap = omp_requests.load_snapshot(snapshot_path)

    raw_streams: Dict[str, Any] = {}
    raw_snap = None
    if counter is not None:
        # Unredacted copy: compaction requests have no recorded prompt count, so count what the original server saw.
        raw_snap = omp_requests.load_snapshot(snapshot_path)
        raw_streams = {st.name: st for st in omp_session.load_session(session_path).streams()}

    # Redact upfront for high performance
    for st in sess.streams():
        redact_stream(st, rules)
    snap = redact_obj(snap, rules)

    # Collect all items (turns and compactions) and sort chronologically
    all_items = []
    stream_order = {st.name: i for i, st in enumerate(sess.streams())}

    for st in sess.streams():
        for t in st.turns:
            all_items.append(("turn", t.start, st, t.index, t))
        for i, c in enumerate(st.compactions):
            ts = omp_session.parse_iso(c.timestamp)
            all_items.append(("compaction", ts, st, i, c))

    all_items.sort(key=lambda x: (x[1], stream_order[x[2].name], x[3]))

    messages_pool: list[dict] = []
    msg_to_index: dict[str, int] = {}

    def get_msg_idx(m: dict) -> int:
        key = json.dumps(m, sort_keys=True)
        idx = msg_to_index.get(key)
        if idx is None:
            idx = len(messages_pool)
            msg_to_index[key] = idx
            messages_pool.append(m)
        return idx

    turns_list = []
    for kind, _start_time, st, idx, item in all_items:
        if kind == "turn":
            t = item
            notes: list = []
            req = omp_requests.build_request(st, idx, snap, notes=notes)
            msg_indices = [get_msg_idx(m) for m in req.get("messages", [])]
            tools = req.get("tools")
            params = {k: v for k, v in req.items() if k not in ("messages", "tools")}

            # Re-count tokens or fallback to recorded promptTokens
            if counter is not None:
                prompt_tokens = counter.count(req)
            else:
                prompt_tokens = t.prompt_tokens or 0

            # Extract recorded answer from the active branch entry
            e = st.branch[t.branch_pos]
            m = e["message"]
            texts = [
                b for b in m.get("content") or []
                if b.get("type") == "text" and b.get("text") and not omp_requests.js_is_blank(b["text"])
            ]
            assistant_text = "".join(omp_requests.well_formed(b["text"]) for b in texts) if texts else None
            thinking = [
                b for b in m.get("content") or []
                if b.get("type") == "thinking" and b.get("thinking") and not omp_requests.js_is_blank(b["thinking"])
            ]
            reasoning_content = "\n".join(b["thinking"] for b in thinking) if thinking else None
            calls = [b for b in m.get("content") or [] if b.get("type") == "toolCall"]
            tool_calls = [
                {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name"),
                        "arguments": (
                            omp_requests.js_stringify(tc.get("arguments", {}))
                            if isinstance(tc.get("arguments"), (dict, list))
                            else (tc.get("arguments") or "")
                        ),
                    },
                }
                for tc in calls
            ] if calls else None

            if assistant_text is None and (reasoning_content or tool_calls):
                assistant_text = ""

            recorded_answer = {
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": tool_calls,
                "reasoning_content": reasoning_content,
                "stopReason": t.stop_reason or m.get("stopReason"),
            }

            epoch = t.compaction_epoch if t.compaction_epoch is not None else 0
            label = f"main/e{epoch}" if st.name == "main" else st.name
            turn_id = f"{st.name.replace('/', '_')}_turn_{idx}"

            turns_list.append({
                "id": turn_id,
                "stream": st.name,
                "label": label,
                "kind": "turn",
                "turn": idx,
                "compaction": None,
                "epoch": epoch,
                "messages": msg_indices,
                "tools": tools,
                "params": params,
                "prompt_tokens": prompt_tokens,
                "recorded_prompt_tokens": t.prompt_tokens,
                "recorded_output_tokens": t.output_tokens,
                "recorded_answer": recorded_answer,
                "time": t.timestamp,
            })
        else:
            c = item
            notes = []
            req = omp_requests.build_compaction_request(st, idx, snap, notes=notes)
            msg_indices = [get_msg_idx(m) for m in req.get("messages", [])]
            tools = req.get("tools")
            params = {k: v for k, v in req.items() if k not in ("messages", "tools")}

            if counter is not None:
                prompt_tokens = counter.count(req)
            else:
                prompt_tokens = c.tokens_before or 0

            recorded_prompt_tokens = None
            if counter is not None:
                raw_req = omp_requests.build_compaction_request(raw_streams[st.name], idx, raw_snap, notes=[])
                recorded_prompt_tokens = counter.count(raw_req)

            e = st.branch[c.branch_pos]
            summary_text = e.get("summary") or ""
            recorded_answer = {
                "role": "assistant",
                "content": summary_text,
                "summary": summary_text,
            }

            epoch = idx
            label = f"main/c{idx}"
            comp_id = f"{st.name.replace('/', '_')}_compaction_{idx}"

            turns_list.append({
                "id": comp_id,
                "stream": st.name,
                "label": label,
                "kind": "compaction",
                "turn": None,
                "compaction": idx,
                "epoch": epoch,
                "messages": msg_indices,
                "tools": tools,
                "params": params,
                "prompt_tokens": prompt_tokens,
                "recorded_prompt_tokens": recorded_prompt_tokens,
                "recorded_output_tokens": None,
                "recorded_answer": recorded_answer,
                "time": c.timestamp,
            })

    session_id = sess.main.header.get("id") if sess.main.header else os.path.basename(session_path).replace(".jsonl", "")
    model = snap.get("model", "unsloth/Qwen3.8-Flash-Next-GGUF")

    streams: Dict[str, int] = {}
    for t in turns_list:
        streams[t["stream"]] = streams.get(t["stream"], 0) + 1
    counts = {
        "turns": sum(1 for t in turns_list if t["kind"] == "turn"),
        "compactions": sum(1 for t in turns_list if t["kind"] == "compaction"),
        "streams": streams,
    }

    fixture = {
        "fixture_version": FIXTURE_VERSION,
        "suite": suite,
        "case": case,
        "session_id": session_id,
        "model": model,
        "provider": "LOCAL-8133",
        "counts": counts,
        "messages": messages_pool,
        "turns": turns_list,
    }
    return fixture


# --------------------------------------------------------------------------------------------- Proxy dumps

_DUMP_REQ = re.compile(r"^(\d+)_req\.json$")


def _sse_usage(text: str) -> Optional[dict]:
    """The last `usage` object of an SSE chat-completion stream (llama.cpp sends it in a final chunk), else None."""
    usage = None
    for line in text.splitlines():
        if not line.startswith("data:") or line.startswith("data: [DONE]"):
            continue
        try:
            chunk = json.loads(line[5:])
        except json.JSONDecodeError:
            continue
        if isinstance(chunk, dict) and isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
    return usage


def build_proxy_fixture(dump_dir: str, rules: Rules = (), suite: str = "conv-1", case: str = "f417-t3",
                        skipped: Optional[List[str]] = None) -> dict:
    """Build a fixture from tools/proxy.py dumps (<ms>_req.json, _resp_combined.json, _resp.json per request).

    The requests are exact, so there is nothing to rebuild: each chat request (it has a non-empty `messages` list)
    becomes one main-stream turn, in filename (time) order. Dumps that are not chat requests, or have no
    combined response, are skipped (their names are appended to `skipped`). The prompt token count is the
    recorded one (no tokenizer for this model is available): `prompt_tokens_recounted` is false.
    """
    names = sorted((int(m.group(1)), n) for n in os.listdir(dump_dir) for m in [_DUMP_REQ.match(n)] if m)
    pool: list = []
    index: Dict[str, int] = {}
    turns_list: list = []
    model = None
    for ms, name in names:
        stem = name[:-len("_req.json")]
        try:
            with open(os.path.join(dump_dir, name), encoding="utf-8") as f:
                req = json.load(f)
            with open(os.path.join(dump_dir, stem + "_resp_combined.json"), encoding="utf-8") as f:
                ans = json.load(f)
        except (OSError, ValueError):
            ans = None
        if not isinstance(ans, dict) or not isinstance(req, dict) or not req.get("messages"):
            if skipped is not None:
                skipped.append(name)
            continue
        req = redact_obj(req, rules)
        ans = redact_obj(ans, rules)
        model = model or req.get("model")
        usage = None
        try:
            with open(os.path.join(dump_dir, stem + "_resp.json"), encoding="utf-8") as f:
                usage = _sse_usage(f.read())
        except OSError:
            pass
        idxs = []
        for m in req["messages"]:
            key = json.dumps(m, sort_keys=True)
            if key not in index:
                index[key] = len(pool)
                pool.append(m)
            idxs.append(index[key])
        n = len(turns_list)
        prompt = (usage or {}).get("prompt_tokens")
        turns_list.append({
            "id": f"main_turn_{n}",
            "stream": "main",
            "label": "main/e0",
            "kind": "turn",
            "turn": n,
            "compaction": None,
            "epoch": 0,
            "messages": idxs,
            "tools": req.get("tools"),
            "params": {k: v for k, v in req.items() if k not in ("messages", "tools")},
            "prompt_tokens": prompt or 0,
            "recorded_prompt_tokens": prompt,
            "recorded_output_tokens": (usage or {}).get("completion_tokens"),
            "recorded_answer": {
                "role": "assistant",
                "content": ans.get("content"),
                "tool_calls": ans.get("tool_calls") or None,
                "reasoning_content": ans.get("reasoning_content"),
            },
            "time": datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        })
    return {
        "fixture_version": FIXTURE_VERSION,
        "suite": suite,
        "case": case,
        "session_id": redact_text(case, rules),
        "model": model or "unknown",
        "provider": "LOCAL-8133",
        "prompt_tokens_recounted": False,
        "counts": {"turns": len(turns_list), "compactions": 0, "streams": {"main": len(turns_list)}},
        "messages": pool,
        "turns": turns_list,
    }


def reconstruct_turn_request(fixture: dict, turn: dict) -> dict:
    """Reconstruct the full chat completion request dict for a given turn."""
    pool = fixture["messages"]
    messages = [pool[i] for i in turn["messages"]]
    req = {
        **turn["params"],
        "messages": messages,
    }
    if turn.get("tools") is not None:
        req["tools"] = turn["tools"]
    return req


def save_fixture(fixture: dict, path: str) -> None:
    """Save fixture to disk (compressed if path ends with .gz)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    raw = json.dumps(fixture, separators=(",", ":")).encode("utf-8")
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as f:
            f.write(raw)
    else:
        with open(path, "wb") as f:
            f.write(raw)


def load_fixture(path: str) -> dict:
    """Load fixture from disk (supporting .gz and uncompressed .json)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))
    else:
        with open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))


def check_fixture(
    fixture_or_path: Union[dict, str],
    redact_rules_path: Optional[str] = None,
    names: Optional[List[str]] = None,
    forbidden: Optional[List[str]] = None,
    info: Optional[Dict[str, Any]] = None,
) -> tuple[bool, list[str]]:
    """Check fixture integrity against its own counts block, reconstruct every entry, and scan for PII.

    `forbidden` are extra regexes (redact-file patterns) and `names` the scanner's --names regexes; any hit fails.
    `redact_rules_path` is a convenience: its regexes are added to `forbidden`. `info` (if given) receives notes.
    """
    errors = []
    if isinstance(fixture_or_path, str):
        if not os.path.isfile(fixture_or_path):
            return False, [f"Fixture file not found: {fixture_or_path}"]
        try:
            fixture = load_fixture(fixture_or_path)
        except Exception as exc:
            return False, [f"Failed to load fixture JSON: {exc}"]
    else:
        fixture = fixture_or_path

    required_meta = ["fixture_version", "suite", "case", "session_id", "model", "provider", "counts", "messages", "turns"]
    for k in required_meta:
        if k not in fixture:
            errors.append(f"Missing top-level key: {k}")
    if errors:
        return False, errors

    if fixture["fixture_version"] != FIXTURE_VERSION:
        errors.append(f"Unsupported fixture_version {fixture['fixture_version']} (expected {FIXTURE_VERSION})")
    if fixture.get("provider") != "LOCAL-8133":
        errors.append(f"Expected provider LOCAL-8133, got {fixture.get('provider')}")

    pool = fixture["messages"]
    if not isinstance(pool, list):
        errors.append("fixture['messages'] must be a list")
        return False, errors
    turns = fixture["turns"]
    if not isinstance(turns, list):
        errors.append("fixture['turns'] must be a list")
        return False, errors

    counts = fixture["counts"]
    turns_count = sum(1 for t in turns if t.get("kind") == "turn")
    comps_count = sum(1 for t in turns if t.get("kind") == "compaction")
    stream_counts: Dict[str, int] = {}
    for t in turns:
        stream_counts[t.get("stream")] = stream_counts.get(t.get("stream"), 0) + 1
    if turns_count != counts.get("turns"):
        errors.append(f"counts.turns = {counts.get('turns')} but fixture has {turns_count} turn entries")
    if comps_count != counts.get("compactions"):
        errors.append(f"counts.compactions = {counts.get('compactions')} but fixture has {comps_count}")
    if stream_counts != counts.get("streams"):
        errors.append(f"counts.streams {counts.get('streams')} != actual {stream_counts}")

    required_turn_keys = [
        "id", "stream", "label", "kind", "turn", "compaction", "epoch",
        "messages", "tools", "params", "prompt_tokens", "recorded_prompt_tokens", "recorded_output_tokens",
        "recorded_answer", "time",
    ]
    seen_ids = set()
    for i, t in enumerate(turns):
        tid = t.get("id")
        missing = [k for k in required_turn_keys if k not in t]
        if missing:
            errors.append(f"Turn #{i} ({tid}) missing keys: {missing}")
            continue
        if tid in seen_ids:
            errors.append(f"Duplicate turn id: {tid}")
        seen_ids.add(tid)

        if not all(isinstance(m, int) and 0 <= m < len(pool) for m in t["messages"]):
            errors.append(f"Turn {tid} references an invalid message index")
            continue

        pt = t["prompt_tokens"]
        if not (isinstance(pt, int) and pt > 0):
            errors.append(f"Turn {tid} has invalid prompt_tokens: {pt}")
        elif fixture.get("prompt_tokens_recounted") is False and pt != t["recorded_prompt_tokens"]:
            errors.append(f"Turn {tid}: prompt_tokens_recounted is false but prompt_tokens != recorded_prompt_tokens")
        if t["recorded_answer"].get("role") != "assistant":
            errors.append(f"Turn {tid} recorded_answer role != 'assistant'")

        try:
            req = reconstruct_turn_request(fixture, t)
        except Exception as exc:
            errors.append(f"Turn {tid} reconstruction failed: {exc}")
            continue
        msgs = req.get("messages")
        if not msgs:
            errors.append(f"Turn {tid} reconstructed empty messages")
        elif msgs[-1].get("role") not in ("user", "tool"):
            errors.append(f"Turn {tid} last message role is {msgs[-1].get('role')!r}, expected user or tool")
        if t["kind"] == "compaction" and t["params"].get("tool_choice") != "none":
            errors.append(f"Compaction {tid} params.tool_choice is not 'none'")

    forbidden = list(forbidden or [])
    if redact_rules_path:
        with open(redact_rules_path, encoding="utf-8") as f:
            forbidden += [pair[0] for pair in json.load(f)]
    pii_issues, pii_info = scan_pii(fixture, names=names, forbidden=forbidden)
    for issue in pii_issues:
        errors.append(f"PII hit: {issue}")
    if info is not None:
        info.update(pii_info)
        info["forbidden_patterns"] = len(forbidden) + len(names or [])

    return len(errors) == 0, errors


# --------------------------------------------------------------------------------------------- CLI

def load_names(path: Optional[str]) -> List[str]:
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build_p = sub.add_parser("build", help="Build a fixture from session and snapshot")
    build_p.add_argument("--session", required=True, help="Path to main session .jsonl")
    build_p.add_argument("--snapshot", required=True, help="Path to snapshot JSON")
    build_p.add_argument("--output", default="fixtures/conv-2.json.gz", help="Output fixture file (.json.gz)")
    build_p.add_argument("--suite", default="conv-2", help="Suite identifier")
    build_p.add_argument("--case", default="case-a", help="Case identifier")
    build_p.add_argument("--redact", default=DEFAULT_REDACT,
                         help="JSON file of [regex, replacement] redaction pairs, kept out of git (default sources/redact.json)")
    build_p.add_argument("--names", help="JSON list of regexes (the scanner's --names format) that must not appear")
    build_p.add_argument("--recount", action="store_true", help="Re-count prompt tokens with SplashCounter")
    build_p.add_argument("--splash", default=None, help="Path to splash directory")
    build_p.add_argument("--gguf", default=None, help="Path to model GGUF")
    build_p.add_argument("--family", default=None, help="Model family")
    build_p.add_argument("--effort", default=None, help="Default reasoning effort")

    proxy_p = sub.add_parser("build-proxy", help="Build a fixture from tools/proxy.py dumps (exact requests)")
    proxy_p.add_argument("--dumps", required=True, help="Directory of <ms>_req.json / _resp_combined.json / _resp.json")
    proxy_p.add_argument("--output", default="fixtures/conv-1.json.gz", help="Output fixture file (.json.gz)")
    proxy_p.add_argument("--suite", default="conv-1", help="Suite identifier")
    proxy_p.add_argument("--case", default="f417-t3", help="Case identifier")
    proxy_p.add_argument("--redact", default=DEFAULT_REDACT,
                         help="JSON file of [regex, replacement] redaction pairs, kept out of git (default sources/redact.json)")
    proxy_p.add_argument("--names", help="JSON list of regexes (the scanner's --names format) that must not appear")

    check_p = sub.add_parser("check", help="Check fixture structure and scan for PII")
    check_p.add_argument("--fixture", required=True, help="Path to fixture file (.json.gz)")
    check_p.add_argument("--redact", default=DEFAULT_REDACT,
                         help="redaction rules file whose regexes must not appear (skipped with a note if missing)")
    check_p.add_argument("--names", help="JSON list of regexes (the scanner's --names format) that must not appear")

    args = parser.parse_args()

    if args.cmd == "build":
        try:
            rules = load_rules(args.redact)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        counter = None
        if args.recount:
            try:
                import omp_tokens
                counter = omp_tokens.SplashCounter(
                    splash_dir=args.splash or omp_tokens.DEFAULT_SPLASH,
                    gguf_path=args.gguf,
                    family=args.family or omp_tokens.DEFAULT_FAMILY,
                    default_reasoning_effort=args.effort or omp_tokens.DEFAULT_EFFORT,
                )
                print(f"Initialized SplashCounter: {counter.describe()}")
            except Exception as exc:
                print(f"ERROR initializing SplashCounter: {exc}", file=sys.stderr)
                print("Run with Splash's venv python to enable re-counting:", file=sys.stderr)
                print(f"  ~/Code/splash/.venv/bin/python {sys.argv[0]} ...", file=sys.stderr)
                sys.exit(1)

        t0 = time.time()
        print(f"Building fixture from session {args.session} and snapshot {args.snapshot}...")
        fixture = build_fixture(args.session, args.snapshot, counter=counter, suite=args.suite, case=args.case, rules=rules)
        save_fixture(fixture, args.output)
        t1 = time.time()

        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"Saved fixture to {args.output} ({size_mb:.2f} MB, {len(fixture['turns'])} turns, "
              f"{len(fixture['messages'])} unique messages in pool) in {t1 - t0:.1f}s")

        # Run verification check
        notes: Dict[str, Any] = {}
        ok, errors = check_fixture(fixture, redact_rules_path=args.redact, names=load_names(args.names), info=notes)
        if ok:
            print(f"Verification check PASSED: structure valid, {len(fixture['turns'])} entries, zero PII hits "
                  f"({notes.get('forbidden_patterns')} forbidden patterns, {notes.get('secret:hex40+', 0)} hex40 info).")
        else:
            print(f"Verification check FAILED with {len(errors)} error(s):", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "build-proxy":
        try:
            rules = load_rules(args.redact)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        skipped: List[str] = []
        fixture = build_proxy_fixture(args.dumps, rules, suite=args.suite, case=args.case, skipped=skipped)
        save_fixture(fixture, args.output)
        print(f"Saved {args.output} ({os.path.getsize(args.output) / (1024 * 1024):.2f} MB, {len(fixture['turns'])} turns, "
              f"{len(fixture['messages'])} unique messages); skipped {len(skipped)}: {', '.join(skipped) or '-'}")
        notes = {}
        ok, errors = check_fixture(fixture, redact_rules_path=args.redact, names=load_names(args.names), info=notes)
        if not ok:
            print(f"Verification check FAILED with {len(errors)} error(s):", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)
        print(f"Verification check PASSED ({notes.get('forbidden_patterns')} forbidden patterns, "
              f"{notes.get('secret:hex40+', 0)} hex40 info).")

    elif args.cmd == "check":
        print(f"Checking fixture {args.fixture}...")
        t0 = time.time()
        redact_path = args.redact if os.path.isfile(args.redact) else None
        notes = {}
        ok, errors = check_fixture(args.fixture, redact_rules_path=redact_path, names=load_names(args.names), info=notes)
        t1 = time.time()
        if ok:
            fixture = load_fixture(args.fixture)
            print(f"Check PASSED in {t1 - t0:.2f}s:")
            print(f"  Suite: {fixture.get('suite')}, Case: {fixture.get('case')}")
            print(f"  Provider: {fixture.get('provider')}, Model: {fixture.get('model')}")
            c = fixture["counts"]
            print(f"  Entries: {len(fixture['turns'])} ({c['turns']} turns, {c['compactions']} compactions)")
            print(f"  Streams: {c['streams']}")
            print(f"  Message pool: {len(fixture['messages'])} unique messages")
            print(f"  PII scan: CLEAN (zero hits besides the allowlist; {notes.get('secret:hex40+', 0)} hex40 git SHAs, info only)")
            if redact_path:
                print(f"  Forbidden patterns checked: {notes.get('forbidden_patterns')}")
            else:
                print(f"  NOTE: {args.redact} not found; redaction-file patterns were not checked")
        else:
            print(f"Check FAILED with {len(errors)} error(s):", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
