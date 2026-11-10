"""How far the answers of agent-replay runs diverge, turn by turn, across models and quants.

Usage:
  python3 tools/replay_divergence.py [OPTIONS] STATE1.jsonl STATE2.jsonl ...
  python3 tools/replay_divergence.py [OPTIONS] REF=state1.jsonl NAME=state2.jsonl ...

Options:
  --label FILE=NAME      Override label for a state file
  --format {tui,markdown,json}
                         Output format (default: tui)
  --md                   Alias for --format markdown
  --min-split N          Minimum distinct first tools to report a turn as split (default: 2)
  --stream {main,subagent,compaction}
                         Filter split turns by stream group
  --args                 Compare path arguments for read/edit/write where first tool matches
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench import fmt as F  # noqa: E402
from bench import (  # noqa: E402
    paths,  # noqa: E402
    report,
    results,
)
from bench.cases import grading, replay  # noqa: E402
from bench.report import Cell, Note, Table  # noqa: E402
from bench.resume import resolve_state  # noqa: E402
from bench.state import read_blocks, same_suite  # noqa: E402


def run_first_step(r):
    """The first tool name of an answer, or '·' when missing, 'SKIP', 'FAIL'."""
    if r is None:
        return "·"
    if r.get("eval_result") == "SKIP":
        return "SKIP"
    if r.get("eval_result") == "FAIL":
        return "FAIL"
    names = r.get("tool_names")
    if names is None:
        calls = r.get("tool_calls") or []
        names = [(c.get("function") or {}).get("name", "") for c in calls]
    return grading.first_tool(names)


def extract_path(r):
    """Extract path argument from the first tool call of a result."""
    if not r:
        return None
    calls = r.get("tool_calls") or []
    if not calls:
        return None
    fn = calls[0].get("function") or {}
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return None
    if isinstance(args, dict):
        return args.get("path")
    return None


def stream_group_of(r):
    if not r:
        return "main"
    if r.get("kind") == "compaction":
        return "compaction"
    stream = r.get("stream")
    if stream == "main" or not stream:
        return "main"
    return "subagent"


def load_runs(paths_or_args, user_labels=None):
    user_labels = dict(user_labels or {})
    raw_runs = []

    files_to_load = []
    for item in paths_or_args:
        if "=" in item and not os.path.exists(item):
            lbl, path = item.split("=", 1)
            user_labels[path] = lbl
            user_labels[os.path.basename(path)] = lbl
            files_to_load.append(path)
        else:
            files_to_load.append(item)

    for p in files_to_load:
        try:
            p = resolve_state(p)  # a path, or a name found in states/, fixtures/refs/ or fixtures/samples/
            blocks = read_blocks(p)
        except Exception as e:
            print(f"warning: cannot read {p}: {e}", file=sys.stderr)
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        for idx, b in enumerate(blocks):
            h = b.get("header") or {}
            model = h.get("model") or "unknown"
            dev = ((h.get("device") or {}) if isinstance(h.get("device"), dict) else {}).get("name") or (
                h.get("device") if isinstance(h.get("device"), str) else ""
            )
            server = h.get("server") or {}
            server_version = server.get("version") if isinstance(server, dict) else (server if isinstance(server, str) else None)
            fixture = h.get("fixture") or {}
            fixture_sha = fixture.get("sha256") if isinstance(fixture, dict) else None
            fixture_file = fixture.get("file") if isinstance(fixture, dict) else (fixture if isinstance(fixture, str) else None)

            raw_runs.append({
                "path": p,
                "stem": stem,
                "index": idx,
                "header": h,
                "results": b.get("results") or {},
                "model": model,
                "device": dev,
                "server_version": server_version,
                "suite": h.get("suite"),
                "fixture_sha": fixture_sha,
                "fixture_file": fixture_file,
            })

    if not raw_runs:
        return []

    ref_suite = next((r["suite"] for r in raw_runs if r.get("suite")), None)
    ref_sha = next((r["fixture_sha"] for r in raw_runs if r.get("fixture_sha")), None)
    runs = []
    seen_skipped = set()
    for r in raw_runs:
        file_id = r["path"]
        if ref_suite and r["suite"] and not same_suite(ref_suite, r["suite"]):
            if file_id not in seen_skipped:
                seen_skipped.add(file_id)
                print(f"skipped from divergence: {file_id}: suite {ref_suite} vs {r['suite']}", file=sys.stderr)
            continue
        if ref_sha and r["fixture_sha"] and ref_sha != r["fixture_sha"]:
            if file_id not in seen_skipped:
                seen_skipped.add(file_id)
                print(f"skipped from divergence: {file_id}: fixture {ref_sha[:8]} vs {r['fixture_sha'][:8]}", file=sys.stderr)
            continue
        runs.append(r)

    if not runs:
        return []

    # Assign initial labels
    for r in runs:
        override = user_labels.get(r["path"]) or user_labels.get(os.path.basename(r["path"])) or user_labels.get(r["stem"])
        if override:
            r["label"] = f"{override} #{r['index']+1}" if r["index"] > 0 else override
        else:
            r["label"] = f"{r['model']}@{r['device']}" if r["device"] else r["model"]

    # Disambiguate if duplicate labels
    counts = {}
    for r in runs:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    for r in runs:
        override = user_labels.get(r["path"]) or user_labels.get(os.path.basename(r["path"])) or user_labels.get(r["stem"])
        if not override and counts[r["label"]] > 1:
            r["label"] = f"{r['stem']} #{r['index']+1}" if r["index"] > 0 else r["stem"]

    seen = {}
    for r in runs:
        n = seen.get(r["label"], 0) + 1
        seen[r["label"]] = n
        if n > 1:
            r["label"] = f"{r['label']} #{n}"

    return runs


def load_checks(runs):
    for r in runs:
        ff = r.get("fixture_file")
        if not ff:
            continue
        # The sidecar sits next to the shipped fixture, wherever the tool is started from.
        checks_file = replay.checks_path(os.path.join(paths.FIXTURES_DIR, os.path.basename(ff)))
        if os.path.exists(checks_file):
            try:
                with open(checks_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def build_matrix(runs):
    all_cids = sorted(set().union(*(r["results"].keys() for r in runs)))
    header = ["Run"] + [r["label"] for r in runs] + ["In line"]
    rows = []
    matrix_data = []

    for ra in runs:
        row_cells = [Cell(ra["label"], "id")]
        row_dict = {"run": ra["label"], "matches": {}}
        for rb in runs:
            if ra is rb:
                row_cells.append(Cell(F.DASH, "dim"))
                row_dict["matches"][rb["label"]] = None
                continue
            common = [
                cid for cid in all_cids
                if cid in ra["results"] and cid in rb["results"]
                and ra["results"][cid].get("eval_result") != "SKIP"
                and rb["results"][cid].get("eval_result") != "SKIP"
            ]
            matches = sum(
                1 for cid in common
                if run_first_step(ra["results"][cid]) == run_first_step(rb["results"][cid])
            )
            share = 100 * matches / len(common) if common else None
            is_noise = (
                ra["model"] == rb["model"]
                and ra["model"] != "unknown"
                and ra["server_version"] == rb["server_version"]
                and ra["server_version"] is not None
            )
            row_dict["matches"][rb["label"]] = {
                "common": len(common),
                "matching": matches,
                "percent": share,
                "noise_floor": is_noise,
            }
            if share is None:
                row_cells.append(Cell(F.DASH))
            else:
                cell_text = f"{round(share)}% (noise floor)" if is_noise else f"{round(share)}%"
                row_cells.append(Cell(cell_text))
        con = results.in_line_share(list(ra["results"].values()))
        row_dict["in_line"] = (
            {"matching": con[0], "total": con[1], "percent": con[2]} if con else None
        )
        if con is not None:
            row_cells.append(Cell(f"{con[0]}/{con[1]} ({round(con[2])}%)"))
        else:
            row_cells.append(Cell(F.DASH))
        rows.append(row_cells)
        matrix_data.append(row_dict)

    aligns = ["l"] + ["r"] * len(runs) + ["r"]
    table = Table(header, rows, aligns=aligns)
    return table, matrix_data


def build_splits(runs, checks, min_split=2, stream_filter=None):
    all_cids = sorted(set().union(*(r["results"].keys() for r in runs)))
    accepted = checks.get("accepted") or {}

    header = ["Case", "Stream", "Prompt", "Recorded", "Accepted"] + [r["label"] for r in runs]
    rows = []
    splits_data = []

    for cid in all_cids:
        sample_r = next((r["results"][cid] for r in runs if cid in r["results"]), {})
        sg = stream_group_of(sample_r)
        if stream_filter and sg != stream_filter:
            continue

        steps = [run_first_step(r["results"].get(cid)) for r in runs]
        distinct = set(steps)
        if len(distinct) < min_split:
            continue

        pt = sample_r.get("prompt_tokens") if sample_r.get("prompt_tokens") is not None else sample_r.get("recorded_prompt_tokens")
        recorded = sample_r.get("recorded_tools")
        rec_str = ", ".join(recorded) if recorded else ("<text>" if sample_r.get("kind") == "compaction" else "·")
        ekey = grading.entry_key(sample_r)
        acc_str = "/".join(accepted[ekey]) if ekey in accepted else "·"

        stream_lbl = results.stream_label(sample_r) or sample_r.get("stream") or ""
        row_cells = [
            Cell(cid, "id"),
            Cell(stream_lbl, "code"),
            Cell(F.num(pt) if pt else F.DASH),
            Cell(rec_str),
            Cell(acc_str),
        ]
        for step in steps:
            style = "fail" if step == "FAIL" else ("skip" if step == "SKIP" else ("dim" if step == "·" else ""))
            row_cells.append(Cell(step, style))
        rows.append(row_cells)

        splits_data.append({
            "id": cid,
            "stream": stream_lbl,
            "prompt_tokens": pt,
            "recorded": rec_str,
            "accepted": acc_str,
            "tools": {r["label"]: step for r, step in zip(runs, steps)},
        })

    aligns = ["l", "l", "r", "l", "l"] + ["l"] * len(runs)
    table = Table(header, rows, aligns=aligns)
    return table, splits_data


def build_args_comparison(runs):
    all_cids = sorted(set().union(*(r["results"].keys() for r in runs)))
    PATH_TOOLS = {"read", "edit", "write"}

    total = 0
    matching = 0
    differing_rows = []
    differing_data = []

    for cid in all_cids:
        active_runs = [r for r in runs if cid in r["results"] and r["results"][cid].get("eval_result") not in ("SKIP", "FAIL")]
        if len(active_runs) < 2:
            continue
        first_tools = {run_first_step(r["results"][cid]) for r in active_runs}
        if len(first_tools) != 1:
            continue
        tool = next(iter(first_tools))
        if tool not in PATH_TOOLS:
            continue

        norm_paths = [grading.normalize_path(extract_path(r["results"][cid])) for r in active_runs]
        distinct_paths = set(norm_paths)
        total += 1
        if len(distinct_paths) == 1:
            matching += 1
        else:
            path_map = {r["label"]: (grading.normalize_path(extract_path(r["results"].get(cid))) or "·") for r in runs}
            row_cells = [Cell(cid, "id"), Cell(tool, "code")]
            for r in runs:
                row_cells.append(Cell(path_map[r["label"]]))
            differing_rows.append(row_cells)
            differing_data.append({
                "id": cid,
                "tool": tool,
                "paths": path_map,
            })

    header = ["Case", "Tool"] + [r["label"] for r in runs]
    aligns = ["l", "l"] + ["l"] * len(runs)
    table = Table(header, differing_rows, aligns=aligns)
    return {
        "total": total,
        "matching": matching,
        "differing_count": len(differing_data),
        "differing": differing_data,
        "table": table,
    }


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(
        description="Per-turn cross-model divergence comparison between agent-replay runs."
    )
    ap.add_argument("states", nargs="*", help="State files to compare (or LABEL=state.jsonl)")
    ap.add_argument("--label", action="append", default=[], help="Override label for a state file (FILE=NAME)")
    ap.add_argument("--format", choices=["tui", "markdown", "json"], default=None,
                    help="Output format (tui, markdown, json)")
    ap.add_argument("--md", action="store_true", help="Alias for --format markdown")
    ap.add_argument("--min-split", type=int, default=2,
                    help="Minimum distinct first tools to consider a turn split (default: 2)")
    ap.add_argument("--stream", choices=["main", "subagent", "compaction"], default=None,
                    help="Filter split turns to a stream group (main, subagent, compaction)")
    ap.add_argument("--args", action="store_true",
                    help="Compare path arguments for read/edit/write where first tool matches")

    args = ap.parse_args(argv)

    user_labels = {}
    for item in args.label:
        if "=" in item:
            f, l = item.split("=", 1)
            user_labels[f] = l

    runs = load_runs(args.states, user_labels)
    if len(runs) < 2:
        print("error: need at least two runs to compare divergence", file=sys.stderr)
        return 1

    checks = load_checks(runs)
    matrix_table, matrix_data = build_matrix(runs)
    splits_table, splits_data = build_splits(runs, checks, min_split=args.min_split, stream_filter=args.stream)
    args_comp = build_args_comparison(runs)

    fmt = "markdown" if args.md else (args.format or "tui")

    if fmt == "json":
        out_data = {
            "suite": runs[0]["suite"],
            "runs": [
                {"label": r["label"], "model": r["model"], "device": r["device"], "server_version": r["server_version"]}
                for r in runs
            ],
            "matrix": matrix_data,
            "splits": splits_data,
            "args": {
                "total": args_comp["total"],
                "matching": args_comp["matching"],
                "differing_count": args_comp["differing_count"],
                "differing": args_comp["differing"],
            },
        }
        print(json.dumps(out_data, indent=2))
        return 0

    blocks = ["## Summary matrix", matrix_table, "## Split turns"]
    if splits_data:
        blocks.append(splits_table)
    else:
        blocks.append(Note(f"No split turns found (min-split {args.min_split})."))

    if args.args:
        blocks.append("## Path arguments")
        tot, mat, diff = args_comp["total"], args_comp["matching"], args_comp["differing_count"]
        pct = 100 * mat / tot if tot else 0.0
        blocks.append(Note(f"Paths: {mat}/{tot} matching ({round(pct)}%), {diff} differing."))
        if diff:
            blocks.append(args_comp["table"])

    if fmt == "markdown":
        print(report.blocks_markdown(blocks))
    else:
        print(report.blocks_tui(blocks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
