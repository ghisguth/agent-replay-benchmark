"""Incremental triage of state files: shows only what is new since the last run.

Usage: triage_states.py [--all] [--reset] [--no-save] [--quiet-warn] [FILE ...]
  FILE ...      state files to look at (default: states/*.jsonl, never states/review/; a symlink into
                fixtures/refs is a reference, one into fixtures/samples a sample: listed as 'ref' / 'sample', not triaged)
  --all         ignore the memory, show everything
  --reset       forget what was shown (delete states/.triage.json) and exit
  --no-save     dry run: do not update the memory
  --quiet-warn  list FAILs only (WARN still counted)

States are loaded through bench.state (replay results are graded again with today's grade() when the fixture is in
this repo). The memory, states/.triage.json, holds per file the size, mtime and the verdict of every id already shown.
A run prints, per new file, a header, the verdict line and every FAIL / WARN; per grown file the new FAIL / WARN and
the totals; a verdict that changed because grading changed as "regraded: id old -> new (reason)"; and the count of
unchanged files. Colors only on a terminal (NO_COLOR respected).
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench import results, state  # noqa: E402
from bench.colors import Clr, configure  # noqa: E402
from bench.report import color_ok  # noqa: E402
from bench.utils import UserError  # noqa: E402

STATES_DIR = os.path.join(ROOT, "states")
MEMORY_NAME = ".triage.json"


def load_memory(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        files = data.get("files")
        return files if isinstance(files, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def save_memory(path, files):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "files": files}, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def collect(blocks):
    """({key: metrics}, header, end, fixture_missing) of the file's blocks; ids of later blocks are prefixed `N/`."""
    out = {}
    for i, b in enumerate(blocks):
        for cid, m in b["results"].items():
            out[cid if i == 0 else "%d/%s" % (i, cid)] = m
    first = blocks[0]
    header = first.get("header") or {}
    end = blocks[-1].get("end")
    missing = state.fixture_missing(header)
    if missing:
        missing = missing.split(".")[0]
    return out, header, end, missing


def run_status(end):
    """running (no end line yet), interrupted or finished."""
    return "running" if end is None else "interrupted" if end.get("interrupted") else "finished"


def head_line(header, end, n_blocks):
    req = header.get("request") or {}
    dev = header.get("device")
    dev = dev.get("name") if isinstance(dev, dict) else dev
    suite = header.get("suite") or "?"
    if header.get("preset"):
        suite += "/" + header["preset"]
    temp = req.get("temperature")
    status = run_status(end)
    parts = [header.get("backend") or "?", dev or "?", state.with_note(header.get("model") or "?", header.get("note")), suite,
             "t%s" % temp if temp is not None else "t-", header.get("started_at") or "?", status]
    if n_blocks > 1:
        parts.append("%d runs" % n_blocks)
    return "%s%s%s" % (Clr.BLD, " | ".join(parts), Clr.RST)


def verdict_line(res):
    ms = list(res.values())
    return results.counts_line(results.verdict_counts(ms), results.tools_agreement(ms),
                               results.in_line_share(ms)).tui()


def case_line(cid, m):
    v = m.get("eval_result") or "?"
    col = Clr.BRED if v == "FAIL" else Clr.BYEL
    why = results.clip(m.get("fail_reason") or m.get("error") or "", 100)
    pt = m.get("prompt_tokens")
    return "  %s%-4s%s %s %s %s %s" % (col, v, Clr.RST, cid, results.stream_label(m) or "-",
                                       results.tokens_k(pt) + " ctx" if pt is not None else "-", why)


def shown(res, ids, quiet_warn):
    keep = ("FAIL",) if quiet_warn else ("FAIL", "WARN")
    return [(c, res[c]) for c in ids if res[c].get("eval_result") in keep]


def triage_file(path, mem, args):
    """(lines, kind, memory entry) for one file; kind is new, grew or unchanged."""
    blocks = state.read_blocks(path)
    if not blocks or not blocks[0].get("header"):
        raise UserError("%s holds no run header" % path)
    res, header, end, missing = collect(blocks)
    st = os.stat(path)
    entry = {"size": st.st_size, "mtime": st.st_mtime, "ids": {c: m.get("eval_result") for c, m in res.items()}}
    seen = None if args.all else mem
    lines, kind = [], "unchanged"
    if seen is None:
        kind = "new"
        lines.append(head_line(header, end, len(blocks)))
        new_ids = list(res)
    else:
        new_ids = [c for c in res if c not in seen["ids"]]
        if new_ids:
            kind = "grew"
            lines.append("%s%s%s  +%d cases, %s" % (Clr.BLD, os.path.basename(path), Clr.RST, len(new_ids),
                                                    run_status(end)))
    regraded = []
    if seen is not None:
        for c, old in seen["ids"].items():
            m = res.get(c)
            if m and old and m.get("eval_result") != old:
                regraded.append("  regraded: %s %s -> %s (%s)" % (c, old, m.get("eval_result"), m.get("fail_reason") or "no reason"))
    if kind != "unchanged" or regraded:
        if kind == "unchanged":
            lines.append("%s%s%s" % (Clr.BLD, os.path.basename(path), Clr.RST))
        if kind == "new":
            lines.insert(0, "%s%s%s" % (Clr.DIM, os.path.basename(path), Clr.RST))
        if missing:
            lines.append("  not re-graded (fixture %s not here), stored counts: %s" % (
                missing, results.counts_line(results.verdict_counts(list(res.values()))).tui()))
        else:
            lines.append("  " + verdict_line(res))
        lines += [case_line(c, m) for c, m in shown(res, new_ids, args.quiet_warn)]
        lines += regraded
    return lines, kind, entry


def main(argv=None):
    ap = argparse.ArgumentParser(description="Incremental triage of state files (see the module docstring).")
    ap.add_argument("files", nargs="*", help="state files (default: states/*.jsonl)")
    ap.add_argument("--all", action="store_true", help="ignore the memory, show everything")
    ap.add_argument("--reset", action="store_true", help="forget what was shown and exit")
    ap.add_argument("--no-save", action="store_true", help="do not update the memory")
    ap.add_argument("--quiet-warn", action="store_true", help="list FAILs only")
    ap.add_argument("--states-dir", default=STATES_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--refs-dir", default=os.path.join(ROOT, "fixtures", "refs"), help=argparse.SUPPRESS)
    ap.add_argument("--samples-dir", default=os.path.join(ROOT, "fixtures", "samples"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    configure(color_ok(sys.stdout))
    mem_path = os.path.join(args.states_dir, MEMORY_NAME)
    if args.reset:
        try:
            os.remove(mem_path)
            print("triage memory reset")
        except FileNotFoundError:
            print("no triage memory")
        return 0
    files = [os.path.abspath(f) for f in args.files] or sorted(glob.glob(os.path.join(args.states_dir, "*.jsonl")))
    memory = load_memory(mem_path)
    new_memory = dict(memory)
    unchanged = 0
    linked = {os.path.realpath(args.refs_dir): "ref", os.path.realpath(args.samples_dir): "sample"}
    for path in files:
        kind_of_link = linked.get(os.path.dirname(os.path.realpath(path))) if os.path.islink(path) else None
        if kind_of_link:
            print("%s%s%s  %s" % (Clr.DIM, kind_of_link, Clr.RST, os.path.basename(path)))  # already triaged
            continue
        try:
            lines, kind, entry = triage_file(path, None if args.all else memory.get(path), args)
        except (UserError, OSError, ValueError) as e:
            print("%s: skipped (%s)" % (os.path.basename(path), e))
            continue
        new_memory[path] = entry
        if kind == "unchanged" and not lines:
            unchanged += 1
        for ln in lines:
            print(ln)
    if unchanged:
        print("%d files unchanged" % unchanged if unchanged != 1 else "1 file unchanged")
    if not args.no_save:
        save_memory(mem_path, new_memory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
