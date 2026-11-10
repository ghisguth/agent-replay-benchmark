#!/usr/bin/env python3
"""Build the `judge` replay fixture from recorded match-judge data (a JSON file of cases).

One match-judge case (default row 53) becomes a list replay: each recorded assistant turn of the case's passing
track is one request (judge prefix + few-shots + the case input + the track up to that turn, the judge's tools and
request settings), as the original judge run sent it. There is no live tool loop.

Inputs: the match-judge data JSON (--data), and optionally a replay state file of the original run
(states/match-judge-*.json) for the prompt token counts each request was measured at. The fixture is
fixture v1 (pool + entries, like the conv fixtures) with `prompt_tokens_recounted: false`; every
string goes through the redaction rules (sources/redact.json) first.

    tools/judge_fixture.py build --data match_judge_data.json --state match-judge-run.json
    tools/judge_fixture.py check fixtures/judge.json.gz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omp_fixture  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "fixtures", "judge.json.gz")
DEFAULT_ROW = 53
# The original judge run capped generation at 4096 tokens.
MAX_TOKENS = 4096


def state_prompt_tokens(path: str, row: int) -> Dict[int, int]:
    """{turn (1-based): prompt tokens} of the row's MJ<row>_T<turn>::iter1 results in a replay state file."""
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    out: Dict[int, int] = {}
    prefix = "MJ%02d_T" % row
    for key, res in state["targets"][0]["results"].items():
        if key.startswith(prefix) and key.endswith("::iter1") and isinstance(res, dict) and res.get("prompt_tokens"):
            out[int(key[len(prefix):].split("::")[0])] = int(res["prompt_tokens"])
    return out


def build_judge_fixture(data: dict, row: int, tokens: Optional[Dict[int, int]] = None, rules=(),
                        built: str = "") -> dict:
    """The fixture of one match-judge case. `tokens` maps turn (1-based) to the recorded prompt tokens; a turn
    without a count gets prompt_tokens 0 and recorded_prompt_tokens None (the selection tolerates that)."""
    case = next((c for c in data["cases"] if c["row"] == row), None)
    if case is None:
        raise ValueError("no case with row %d" % row)
    tokens = tokens or {}
    prefix, tools = data["prefix"], data["tools"]
    params = {"stream": True, "stream_options": {"include_usage": True}, **data["request"], "max_tokens": MAX_TOKENS}
    head = prefix + [{"role": "user", "content": case["input"]}]
    track = case["track"]
    pool: List[Any] = []
    index: Dict[str, int] = {}

    def idx(m: dict) -> int:
        key = json.dumps(m, sort_keys=True)
        if key not in index:
            index[key] = len(pool)
            pool.append(m)
        return index[key]

    turns: List[dict] = []
    assistant_idx = [i for i, m in enumerate(track) if m["role"] == "assistant"]
    for k, i in enumerate(assistant_idx):
        messages = omp_fixture.redact_obj(head + track[:i], rules)
        answer = omp_fixture.redact_obj(track[i], rules)
        n = tokens.get(k + 1)
        final = k == len(assistant_idx) - 1
        turns.append({
            "id": "main_turn_%d" % k,
            "stream": "main",
            "label": "main/e0",
            "kind": "turn",
            "turn": k,
            "compaction": None,
            "epoch": 0,
            "messages": [idx(m) for m in messages],
            "tools": omp_fixture.redact_obj(tools, rules),
            "params": omp_fixture.redact_obj(params, rules),
            "prompt_tokens": n or 0,
            "recorded_prompt_tokens": n,
            "recorded_output_tokens": None,
            "recorded_answer": {
                "role": "assistant",
                "content": answer.get("content"),
                "tool_calls": answer.get("tool_calls") or None,
                "reasoning_content": None,
            },
            "judge_final": final,
            "time": built,
        })
    return {
        "fixture_version": omp_fixture.FIXTURE_VERSION,
        "suite": "judge",
        "case": "mj%02d" % row,
        "session_id": "mj%02d" % row,
        "model": case.get("track_model") or "unknown",
        "provider": "LOCAL-8133",
        "prompt_tokens_recounted": False,
        "counts": {"turns": len(turns), "compactions": 0, "streams": {"main": len(turns)}},
        "judge": {"row": row, "case": omp_fixture.redact_obj(case["id"], rules),
                  "expect": omp_fixture.redact_obj(case["expect"], rules)},
        "messages": pool,
        "turns": turns,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build fixtures/judge.json.gz")
    b.add_argument("--data", required=True, help="match-judge data JSON (the recorded cases)")
    b.add_argument("--state", help="replay state file of the original run, with the prompt token counts per turn")
    b.add_argument("--row", type=int, default=DEFAULT_ROW)
    b.add_argument("--redact", default=omp_fixture.DEFAULT_REDACT)
    b.add_argument("--names", help="JSON list of name regexes for the PII scan (outside git)")
    b.add_argument("--out", default=DEFAULT_OUT)
    c = sub.add_parser("check", help="check a judge fixture (structure and PII)")
    c.add_argument("fixture", nargs="?", default=DEFAULT_OUT)
    c.add_argument("--redact", default=omp_fixture.DEFAULT_REDACT)
    c.add_argument("--names")
    args = ap.parse_args(argv)

    names = omp_fixture.load_names(args.names)
    if args.cmd == "build":
        rules = omp_fixture.load_rules(args.redact)
        with open(args.data, encoding="utf-8") as f:
            data = json.load(f)
        tokens = state_prompt_tokens(args.state, args.row) if args.state else {}
        fx = build_judge_fixture(data, args.row, tokens, rules, built=data.get("meta", {}).get("built", ""))
        ok, errors = omp_fixture.check_fixture(fx, redact_rules_path=args.redact, names=names)
        if not ok:
            print("fixture check failed:\n  " + "\n  ".join(errors), file=sys.stderr)
            return 1
        omp_fixture.save_fixture(fx, args.out)
        print("wrote %s: %d turns, %d pooled messages, prompt tokens %s" % (
            args.out, len(fx["turns"]), len(fx["messages"]), [t["recorded_prompt_tokens"] for t in fx["turns"]]))
        return 0
    ok, errors = omp_fixture.check_fixture(args.fixture, redact_rules_path=args.redact, names=names)
    print("OK" if ok else "FAILED:\n  " + "\n  ".join(errors))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
