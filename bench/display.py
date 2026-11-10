"""State display, per-model finishing, final run reporting, and resume hints."""

import os
import shlex

from . import compare, perf
from . import results as R
from . import state as state_mod
from .baseline import Comparison, check_baseline, load_baseline
from .outputs import output
from .state import load_state
from .utils import UserError


def finish_target(out, target, ordered_targets, raw_results, base_run, has_stats, mdcat, interrupted=None):
    """A model's tables once its cases are done. Its baseline is the --compare-state run (`base_run`, a BaselineRun) if
    there is one, else, in a run of several models, the first model (unless this is it)."""
    entries = compare.entries_from_run(ordered_targets, raw_results)
    i = ordered_targets.index(target)
    if base_run is not None:
        base_e = compare.baseline_entry(base_run)
    else:
        base_e = entries[0] if i and entries[0]["results"] else None
    comparison = Comparison(entries[i], base_e) if base_e else None
    out.target_done(target, entries[i], comparison, has_stats, mdcat, interrupted)


def resume_hint(state_paths, argv0, total=None):
    """The last lines of an interrupted run that saved state: how to continue each file. Empty string when there is
    nothing to resume. A file holding several models' blocks cannot be resumed (--resume takes single-model files)."""
    lines = []
    script = argv0 if argv0 else "replay_bench.py"
    cwd = os.getcwd()
    for path in dict.fromkeys(state_paths):
        try:
            blocks = state_mod.read_blocks(path)
        except Exception:
            continue
        shown = os.path.relpath(path) if os.path.abspath(path).startswith(cwd + os.sep) else os.path.abspath(path)
        if len(blocks) != 1:
            lines.append("interrupted: %s holds %d models; --resume takes single-model state files, so it cannot "
                         "be resumed (run each model into its own file)" % (shown, len(blocks)))
            continue
        done = sum(1 for m in blocks[0]["results"].values() if not m.get("error"))
        count = "%d of %d cases done" % (done, total) if total else "%d cases done" % done
        launcher = "python3 " + shlex.quote(script) if script.endswith(".py") else shlex.quote(script)
        lines.append("interrupted: %s. To continue:\n    %s --resume %s" % (count, launcher, shlex.quote(shown)))
    return "\n".join(lines)


def print_final(out, ordered_targets, raw_results, base_runs, mdcat, state_paths, interrupted):
    """End of a run: the comparison of the models (several), the two perf tables (the --compare-state baselines join them
    as runs, first, so the deltas are against them), then the done line. `base_runs`: {target key: BaselineRun}."""
    entries = compare.entries_from_run(ordered_targets, raw_results)
    saved = [compare.baseline_entry(base_runs[t["key"]], name=t["name"] + " (baseline)")
             for t in ordered_targets if t["key"] in base_runs]
    compare.print_summary(entries, mdcat)
    compare.print_perf(saved + entries, mdcat)
    out.done(state_paths, mdcat, interrupted, shows_delta=bool(saved) or len(entries) > 1, spec=perf.has_spec(saved + entries),
             est=perf.has_new_estimate(saved + entries))


def display_state(load_state_path, target_test_ids=None, compare_state_path=None, mdcat=True):
    """Show a saved run again (no server): each model block of the file as a finished run, compared case by case with
    the --compare-state file when given. `target_test_ids` narrows a smoke state to those cases."""
    raw_results, targets = load_state(load_state_path)
    baseline = load_baseline(compare_state_path) if compare_state_path else None
    for t in targets:
        h = t.get("header") or {}
        if target_test_ids:
            if not (state_mod.same_suite(h.get("suite"), "smoke") or state_mod.same_suite(h.get("suite"), "sanity")):
                raise UserError("--test selects smoke or sanity cases; %s is a %s state (use --turns with a replay)"
                                % (load_state_path, h.get("suite") or "replay"))
            missing = [i for i in target_test_ids if i not in raw_results[t["key"]]]
            if missing:
                raise UserError("test ID(s) not found in %s: %s" % (load_state_path, ", ".join(missing)))
            raw_results[t["key"]] = {i: raw_results[t["key"]][i] for i in target_test_ids}
        if baseline:
            check_baseline(baseline, h.get("suite"), (h.get("fixture") or {}).get("sha256"))
    out = output()
    ids = sorted({i for results in raw_results.values() for i in results})
    infos = [R.run_info("smoke", t) for t in targets]
    out.begin(infos, targets, len(ids), mdcat)
    base_runs = {}
    for target, info in zip(targets, infos):
        key = target["key"]
        results = raw_results[key]
        has_stats = any((r.get("prompt_tokens") or 0) > 0 for r in results.values())
        shown = list(results.values())  # the run's order (a sanity state is not alphabetical)
        base_run = baseline.run_for(target, info) if baseline else None
        if base_run:
            base_runs[key] = base_run
        out.target_start(target, has_stats, shown, base_run)
        for metrics in shown:
            out.case_done(target, metrics, has_stats)
        end = target.get("end") or {}
        finish_target(out, target, targets, raw_results, base_run, has_stats, mdcat,
                      interrupted=(end.get("reason") or "interrupted") if end.get("interrupted") else None)
    print_final(out, targets, raw_results, base_runs, mdcat, [], None)
