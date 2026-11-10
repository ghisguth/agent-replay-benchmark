"""Replay cases (--suite coding, judge): recorded agent requests from a fixture, sent one by one.

The fixture (``fixtures/conv-3.json.gz``, built by tools/omp_fixture.py) holds a pool of chat messages and a
chronological list of requests (main agent turns, subagent turns, compaction calls). A case only points at its
fixture entry; the request body is built right before it is sent (``build_payload``) and dropped afterwards, so a
run never holds 470 prompts of up to ~240k tokens in memory.
"""

import gzip
import json
import os
import re
import sys

from .. import paths
from ..state import file_sha256 as _file_sha256

DEFAULT_FIXTURES = {"coding": "conv-3.json.gz", "judge": "judge.json.gz"}  # file names in paths.FIXTURES_DIR


def default_fixture(suite):
    """The shipped fixture of a replay suite (under paths.FIXTURES_DIR, read at call time), or None."""
    name = DEFAULT_FIXTURES.get(suite)
    return os.path.join(paths.FIXTURES_DIR, name) if name else None

# Runaway cutoffs for replay cases (the recorded run allowed 32768 completion tokens).
MAX_GEN_TOKENS = 32768
MAX_GEN_SECONDS = 1800

# Tokens kept free for the answer when deciding whether a prompt fits the target's context.
SKIP_RESERVE = 4096


_warned_fixture_sha = set()


def checks_path(fixture_path):
    """The sidecar of a fixture file: ``conv-3.json.gz`` -> ``conv-3.checks.json`` in the same folder."""
    name = os.path.basename(str(fixture_path))
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return os.path.join(os.path.dirname(str(fixture_path)), name + ".checks.json")


def load_checks(fixture_path):
    """The fixture's expectations from its .checks.json sidecar ({} when there is none)."""
    try:
        with open(checks_path(fixture_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def load_fixture(path):
    """The fixture dict from a .json or .json.gz file."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        fixture = json.loads(f.read().decode("utf-8"))
    checks = load_checks(path)
    if checks:
        expected_sha = checks.get("fixture_sha256")
        if expected_sha:
            actual_sha = _file_sha256(path)
            if expected_sha != actual_sha:
                cpath = checks_path(path)
                if cpath not in _warned_fixture_sha:
                    _warned_fixture_sha.add(cpath)
                    print(f"warning: {os.path.basename(cpath)}: fixture sha256 mismatch ({expected_sha[:8]} vs {actual_sha[:8]}); ignoring accepted steps", file=sys.stderr)
                checks.pop("accepted", None)
        fixture["checks"] = checks
    return fixture


def recorded_prompt_tokens(entry):
    """The original (pre-redaction) prompt token count when the fixture has it, else the re-counted one."""
    v = entry.get("recorded_prompt_tokens")
    return v if v is not None else entry.get("prompt_tokens", 0)


def recorded_tool_names(entry):
    """Names of the tools the recorded answer called, in order."""
    calls = (entry.get("recorded_answer") or {}).get("tool_calls") or []
    return [(c.get("function") or {}).get("name", "") for c in calls]


def case_id(pos, entry):
    """Stable, sortable id: position in the full chronological list, then the label (R001_main_e0_t000)."""
    label = re.sub(r"[^A-Za-z0-9]+", "_", str(entry["label"])).strip("_")
    if entry["kind"] == "compaction":
        return "R%03d_%s" % (pos, label)
    return "R%03d_%s_t%03d" % (pos, label, entry.get("turn") or 0)


def _case_type(entry):
    if entry["kind"] == "compaction":
        return "Replay compaction"
    return "Replay main" if entry["stream"] == "main" else "Replay subagent"


def limit_entries(entries, max_context):
    """Apply --max-context to [(pos, entry)]: drop compactions and main epochs >= 1, and end every stream at its
    first turn that does not fit a server whose context is exactly the limit (recorded prompt tokens + SKIP_RESERVE >
    limit, the same rule as skip_reason). None keeps everything."""
    if max_context is None:
        return list(entries)
    out, ended = [], set()
    for pos, e in entries:
        if e["kind"] == "compaction" or (e["stream"] == "main" and (e.get("epoch") or 0) >= 1):
            continue
        key = e["label"]
        if key in ended:
            continue
        if recorded_prompt_tokens(e) + SKIP_RESERVE > max_context:
            ended.add(key)
            continue
        out.append((pos, e))
    return out


def select_entries(fixture, max_context=None, no_subagents=False, turns=None):
    """The fixture entries to replay as [(position in the full list, entry)], in chronological order.

    Order of the filters: --no-subagents, --max-context, --turns A-B (1-based, inclusive, over what is left).
    """
    entries = list(enumerate(fixture["turns"], 1))
    if no_subagents:
        entries = [(p, e) for p, e in entries if e["stream"] == "main"]
    entries = limit_entries(entries, max_context)
    if turns:
        entries = entries[turns[0] - 1:turns[1]]
    return entries


def make_case(fixture, pos, entry):
    """A case dict that points at its fixture entry (no messages are copied or built)."""
    return {
        "id": case_id(pos, entry),
        "type": _case_type(entry),
        "fixture": fixture,
        "entry": entry,
        "stream": entry["stream"],
        "kind": entry["kind"],
        "epoch": entry.get("epoch"),
        "turn": entry.get("turn"),
        "recorded_prompt_tokens": recorded_prompt_tokens(entry),
        "recorded_tools": recorded_tool_names(entry),
        "max_gen_tokens": MAX_GEN_TOKENS,
        "max_gen_seconds": MAX_GEN_SECONDS,
    }


def build_payload(case, model_name, reasoning_effort=None, temperature=None):
    """The request body of a replay case: the recorded params and messages, model swapped for the target's.

    Nothing is added to what omp sent (no max_tokens); reasoning_effort and temperature only when given.
    """
    entry = case["entry"]
    pool = case["fixture"]["messages"]
    payload = dict(entry["params"])
    payload["model"] = model_name
    payload["messages"] = [pool[i] for i in entry["messages"]]
    if entry.get("tools") is not None:
        payload["tools"] = entry["tools"]
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def need_tokens(case):
    """The prompt token count of a replay case, known before it is sent."""
    return case["entry"].get("prompt_tokens") or case["recorded_prompt_tokens"]


def skip_reason(case, target):
    """A SKIP reason when the prompt cannot fit the target's known context window, else None."""
    n_ctx = target.get("n_ctx")
    need = need_tokens(case)
    if n_ctx is not None and need + SKIP_RESERVE > n_ctx:
        return "prompt %d + %d reserve > ctx %d" % (need, SKIP_RESERVE, n_ctx)
    return None


def summary_line(selected):
    """One line describing a selection: entries per stream kind and the largest prompt."""
    n = {"main": 0, "subagent": 0, "compaction": 0}
    for _, e in selected:
        n["compaction" if e["kind"] == "compaction" else "main" if e["stream"] == "main" else "subagent"] += 1
    top = max((recorded_prompt_tokens(e) for _, e in selected), default=0)
    return "%d entries (main %d, subagent %d, compaction %d), max prompt %d tokens (%.1fk)" % (
        len(selected), n["main"], n["subagent"], n["compaction"], top, top / 1024)


def dry_run_table(selected):
    """--dry-run as a report Table (id, stream, kind, epoch, prompt tokens, recorded tools)."""
    from ..report import Table
    rows = [[case_id(pos, e), e["label"], e["kind"], str(e.get("epoch")), str(recorded_prompt_tokens(e)),
             ",".join(recorded_tool_names(e)) or "-"] for pos, e in selected]
    return Table(["id", "stream", "kind", "epoch", "prompt", "recorded tools"], rows,
                 aligns=["l", "l", "l", "l", "r", "l"])
