"""Add the accepted first steps of reference runs to a fixture's checks sidecar.

Usage: build_checks.py [--fixture FILE] [--checks FILE] [--min-runs N] [STATE ...]
  STATE ...   reference state files (JSONL; default: fixtures/refs/*.jsonl; fixtures/samples/ is never read)
  --min-runs  a first-step category is accepted at a turn only if at least N distinct MODELS took it there
              (default MIN_RUNS = 2; the recorded agent's own step is always accepted; FAILs and WARNs other than
              "off pattern" do not count as a run taking the step). A model is the model id without organisation,
              quant and format tags (model_key()), so several quants or repeated runs of one model count once.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench import state  # noqa: E402
from bench.cases import grading, replay  # noqa: E402

MIN_RUNS = 2   # distinct reference models that must have taken a step at a turn for it to be accepted there
DEFAULT_FIXTURE = os.path.join(ROOT, "fixtures", "conv-3.json.gz")
DEFAULT_REFS_GLOB = os.path.join(ROOT, "fixtures", "refs", "*.jsonl")


# Tokens at the end of a model id that name a quantisation or format, not the model.
_TAGS = {"gguf", "mlx", "4bit", "8bit", "gptq", "awq", "fp16", "bf16", "gsq", "rco"}


def model_key(model_id):
    """The model behind a model id, lower case, without organisation, quant (after ':') and format tags:
    'unsloth/Qwen3.8-27B-GGUF:Q8_0', 'mlx-community/Qwen3.8-27B-GPTQ-4bit' and 'qwen3.8-27b' all give 'qwen3.8-27b';
    '...Qwen3.8-Flash-Next-GSQ-RCO-GGUF' and 'unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ1_S' give 'qwen3.8-flash-next'."""
    name = str(model_id or "?").strip().lower().rsplit("/", 1)[-1].split(":", 1)[0]
    parts = name.split("-")
    if "mtplx" in parts:   # '-MTPLX-Optimized-Speed-FP16': everything from the engine tag on
        parts = parts[:parts.index("mtplx")]
    while len(parts) > 1 and parts[-1] in _TAGS:
        parts.pop()
    return "-".join(parts) or "?"


def broken(m):
    """True for a reference answer that must not widen "accepted": a FAIL, or a WARN for anything but being off the
    pattern (a schema problem, leaked markup, an early report): one weaker reference run would otherwise teach every
    later run that its broken step is fine."""
    if m.get("eval_result") == "FAIL":
        return True
    if m.get("eval_result") != "WARN":
        return False
    notes = [n.strip() for n in (m.get("fail_reason") or "").split(";") if n.strip()]
    return any(not n.startswith("off pattern") for n in notes)


def read_header(path):
    """(header of the first run block, its fixture sha256 or None, all blocks) of a state file."""
    blocks = state.read_blocks(path)
    header = (blocks[0].get("header") or {}) if blocks else {}
    fx = header.get("fixture")
    return header, (fx.get("sha256") if isinstance(fx, dict) else None), blocks


def first_tools(blocks, fixture):
    """{entry key: first-step category} of the usable answers in a state file's blocks (results are keyed by case id)."""
    key_of = {replay.case_id(pos, e): grading.entry_key(e) for pos, e in enumerate(fixture["turns"], 1)}
    cats = {}
    for b in blocks:
        for cid, m in b["results"].items():
            if cid in key_of and not (m.get("error") or m.get("eval_result") == "SKIP" or broken(m)):
                cats[key_of[cid]] = grading.tool_category(grading.calls_from(m))
    return cats


def build(fixture, ref_paths, min_runs=MIN_RUNS):
    """(accepted {entry_key: [category]}, refs [description]) from the fixture's recording and the reference states.
    A category taken by a reference run counts once per distinct model (model_key); it is accepted at a turn when
    min_runs models took it there. The recorded agent's category is always accepted."""
    accepted_by_key = {grading.entry_key(e): {grading.tool_category((e.get("recorded_answer") or {}).get("tool_calls"))}
                       for e in fixture["turns"] if e.get("kind") != "compaction"}
    taken = {ek: {} for ek in accepted_by_key}   # {entry key: {category: model keys that took it}}
    refs = []
    for p in ref_paths:
        header, _, blocks = read_header(p)
        mk = model_key(header.get("model"))
        for ek, cat in first_tools(blocks, fixture).items():
            if ek in accepted_by_key:
                taken[ek].setdefault(cat, set()).add(mk)
        refs.append("%s (%s, %s)" % (state.with_note(header.get("model") or "?", header.get("note")), os.path.basename(p),
                                     (header.get("started_at") or "?")[:10]))
    for ek, counts in taken.items():
        accepted_by_key[ek].update(c for c, ms in counts.items() if len(ms) >= min_runs)
    return {ek: sorted(cats) for ek, cats in accepted_by_key.items()}, refs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Add the accepted first steps of reference runs to a checks sidecar.")
    ap.add_argument("states", nargs="*", help="reference state files (default: fixtures/refs/*.jsonl)")
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE, help="fixture file path (default: conv-3.json.gz)")
    ap.add_argument("--checks", help="sidecar to update (default: next to the fixture)")
    ap.add_argument("--min-runs", type=int, default=MIN_RUNS, metavar="N",
                    help="distinct reference models that must have taken a step at a turn to accept it (default: %d)" % MIN_RUNS)
    args = ap.parse_args(argv)
    if args.min_runs < 1:
        ap.error("--min-runs must be at least 1")

    out_path = args.checks or replay.checks_path(args.fixture)
    paths = list(args.states) or sorted(glob.glob(DEFAULT_REFS_GLOB))

    if not paths:
        ap.error("no reference states found")
    if len(paths) < 2:
        ap.error("need at least 2 reference states (got %d)" % len(paths))

    fixture = replay.load_fixture(args.fixture)
    fixture_sha = state.file_sha256(args.fixture)

    ref_shas = []
    for p in paths:
        try:
            ref_shas.append((p, read_header(p)[1]))
        except Exception as e:
            ap.error("cannot read %s: %s" % (p, e))

    first_sha = ref_shas[0][1]
    for p, s in ref_shas:
        if s != first_sha:
            ap.error("reference states must all have the same fixture sha256 (%s has %s, %s has %s)"
                     % (ref_shas[0][0], str(first_sha)[:8], p, str(s)[:8]))
    if first_sha and fixture_sha and first_sha != fixture_sha:
        ap.error("reference fixture sha256 (%s) does not match %s (%s)"
                 % (str(first_sha)[:8], os.path.basename(args.fixture), str(fixture_sha)[:8]))

    accepted, refs = build(fixture, paths, args.min_runs)

    try:
        with open(out_path, encoding="utf-8") as f:
            checks = json.load(f)
    except FileNotFoundError:
        checks = {}

    checks["fixture_sha256"] = fixture_sha
    checks["accepted"] = accepted
    checks["accepted_refs"] = refs

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)
        f.write("\n")

    models = len({model_key(read_header(p)[0].get("model")) for p in paths})
    print("accepted steps for %d turns (compactions excluded) from %d reference runs of %d models, a step needs %d models -> %s"
          % (len(accepted), len(refs), models, args.min_runs, out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
