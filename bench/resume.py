"""Resume interrupted benchmark runs."""

import argparse
import difflib
import os
import re
import shlex
import sys

from . import paths
from . import state as state_mod
from .endpoints import resolve_endpoints
from .presets import CASE_SUITES, SUITES, resolve_selection
from .utils import UserError

# --resume: every option of the parser is in exactly one class (tests/test_resume.py checks it, so a new flag cannot be
# forgotten). Keys are argparse dests.
RESUME_MATCH = {  # change what is sent or how it is graded: a given value must equal the recorded one
    "suite", "preset", "fixture", "max_context", "no_subagents", "turns", "test", "endpoint", "model", "first_model",
    "warmup_first_test", "reasoning_effort", "temperature", "no_temperature", "device",
}
RESUME_PRESENTATION = {  # how the report looks or how the server is reached: the given value wins
    "api_key", "ca_file", "insecure", "no_prompt_progress", "run_note", "report_format", "no_mdcat", "no_nerd_font", "icons", "server_version",
}
RESUME_INCOMPATIBLE = {"load_state", "state_file", "no_save_state", "compare_state", "dry_run", "list"}
RESUME_OWN = {"resume", "version"}  # plus -h/--help, which exits before any of this
_STATE_FLAGS = {"--state", "--state-file", "--compare-state"}  # take a value; dropped from the recorded command
_NOT_GIVEN = object()


def _tokens_without(argv, flags, bare=()):
    """argv without the given options (`--flag VALUE` and `--flag=VALUE` for `flags`, `--flag` for `bare`)."""
    out, skip = [], False
    for t in argv:
        if skip:
            skip = False
        elif t in flags:
            skip = True
        elif t.split("=", 1)[0] in flags and "=" in t or t in bare:
            pass
        else:
            out.append(t)
    return out


def given_options(parser, argv):
    """{dest: parsed value} of the options explicitly present in argv (defaults are not applied)."""
    ns = argparse.Namespace()
    for a in parser._actions:
        setattr(ns, a.dest, None if isinstance(a, argparse._AppendAction) else _NOT_GIVEN)
    parser.parse_args(argv, namespace=ns)
    appended = {a.dest for a in parser._actions if isinstance(a, argparse._AppendAction)}
    return {d: v for d, v in vars(ns).items() if v is not _NOT_GIVEN and not (d in appended and v is None)}


def _effective_temperature(ns):
    suite = resolve_selection(ns)[1]
    if ns.no_temperature:
        return None
    return ns.temperature if ns.temperature is not None else (None if suite not in CASE_SUITES else 0.0)


def canonical(ns, dest):
    """A comparable form of one result-affecting option of a parsed namespace (64k == 65536, aliases resolved)."""
    if dest in ("temperature", "no_temperature"):
        return _effective_temperature(ns)
    v = getattr(ns, dest)
    if dest in ("suite", "preset", "max_context", "no_subagents", "turns"):  # preset values laid under the explicit ones
        sel = resolve_selection(ns)
        if dest == "suite":
            return SUITES[sel[1]]
        if dest == "preset":
            return sel[0]
        v = {"max_context": sel[2], "no_subagents": sel[3], "turns": sel[4]}[dest]
        return list(v) if dest == "turns" and v else v
    if dest == "fixture":
        return os.path.realpath(v) if v else None
    if dest == "endpoint":
        return sorted((e["base_url"] for e in resolve_endpoints(v)))
    if dest == "model":
        return sorted(m.lower() for m in v) if v else None
    if dest == "test":
        return sorted(v) if v else None
    if dest == "reasoning_effort":
        return v or None
    return (v or None) if dest == "device" else v


def _show(v):
    return "none" if v is None or v == [] else ("[%s]" % ", ".join(map(str, v)) if isinstance(v, list) else str(v))


def resolve_state(path, states_dir=None, refs_dir=None, samples_dir=None):
    """A state file given by name: as is when it exists, else looked up in states/, fixtures/refs/ and then
    fixtures/samples/ (a bare file name, or a path relative to any); else a unique substring (so also prefix) of one
    name among states/*.jsonl, fixtures/refs/*.jsonl and fixtures/samples/*.jsonl (a symlink and its target count
    once). Several matches: a UserError listing them; none: a UserError that points at the closest names. Symlinks are followed."""
    if os.path.exists(path):
        return path
    if states_dir is None:
        states_dir = paths.default_states_dir()
    if refs_dir is None:
        refs_dir = paths.REFS_DIR
    if samples_dir is None:
        samples_dir = paths.SAMPLES_DIR
    dirs = (states_dir, refs_dir, samples_dir)
    for d in dirs:
        cand = os.path.join(d, path)
        if os.path.isfile(cand):
            return cand
    found = {}  # realpath -> path, states/ first
    for d in dirs:
        for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            full = os.path.join(d, n)
            if n.endswith(".jsonl") and path in n and os.path.isfile(full):
                found.setdefault(os.path.realpath(full), full)
    if len(found) == 1:
        return next(iter(found.values()))
    if found:
        raise UserError("state name %r is ambiguous, it matches:\n  %s" % (path, "\n  ".join(
            os.path.relpath(p, paths.ROOT) if p.startswith(paths.ROOT + os.sep) else p for p in found.values())))
    names = sorted({n for d in dirs if os.path.isdir(d) for n in os.listdir(d)})
    near = suggest_names(os.path.basename(path), names)
    hint = " (did you mean %s?)" % " or ".join(near) if near else ""
    raise UserError("state file not found: %s%s" % (path, hint))


def _tokens(text):
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def suggest_names(query, names, n=3):
    """The closest file names to a misspelled state name: the share of the query's words (split at - _ . and spaces)
    that appear, possibly misspelled, among a name's words; whole-name similarity breaks ties and catches the rest."""
    q = _tokens(query)
    scored = []
    for name in names:
        words = _tokens(name)
        hits = sum(1 for t in q if any(t == w or difflib.SequenceMatcher(None, t, w).ratio() >= 0.8 for w in words))
        share = hits / len(q) if q else 0.0
        whole = difflib.SequenceMatcher(None, query.lower(), name.lower()).ratio()
        if share >= 0.6 or whole >= 0.5:
            scored.append((-share, -whole, name))
    return [name for _, _, name in sorted(scored)[:n]]


def apply_resume(parser, args, argv, resolve_state=resolve_state):
    """--resume FILE: the namespace to run (the recorded command with the given presentation options laid over it)
    and the state path. A UserError when an incompatible option is given or a result-affecting one differs."""
    given = given_options(parser, sys.argv[1:] if argv is None else argv)
    flag = lambda d: next(a.option_strings[0] for a in parser._actions if a.dest == d)
    bad = sorted(flag(d) for d in given if d in RESUME_INCOMPATIBLE)
    if bad:
        raise UserError("--resume cannot be combined with %s" % ", ".join(bad))
    path = resolve_state(args.resume)
    header, _, _ = state_mod.load_resume(path)
    recorded_argv = header.get("argv")
    if recorded_argv is None:
        raise UserError("%s records no command line (its header has no argv); give the options and start a new run" % path)
    recorded_argv = _tokens_without(recorded_argv, _STATE_FLAGS, bare={"--no-save-state"})
    rec = parser.parse_args(recorded_argv)
    # Compare what was given with the recorded run, option by option.
    merged = argparse.Namespace(**vars(rec))
    for d, v in given.items():
        if d in RESUME_MATCH or d in RESUME_PRESENTATION:
            setattr(merged, d, v)
    seen, diffs = set(), []
    for d in sorted(d for d in given if d in RESUME_MATCH):
        key = "temperature" if d in ("temperature", "no_temperature") else d
        if key in seen:
            continue
        seen.add(key)
        before, now = canonical(rec, d), canonical(merged, d)
        if before != now:
            names = [flag(x) for x in sorted(given) if x in RESUME_MATCH and
                     ("temperature" if x in ("temperature", "no_temperature") else x) == key]
            diffs.append("%s: recorded %s, given %s" % (" / ".join(names), _show(before), _show(now)))
    if diffs:
        raise UserError("--resume: options differ from the recorded run in %s:\n  %s\n(start a new run to change them)"
                        % (path, "\n  ".join(diffs)))
    for d in ("state_file", "no_save_state", "compare_state", "load_state", "resume"):
        setattr(merged, d, None if d != "no_save_state" else False)
    given_tokens = _tokens_without(sys.argv[1:] if argv is None else argv, {"--resume"})
    shown = state_mod.sanitize_argv(recorded_argv + given_tokens)  # never echo an API key
    prog = os.path.basename(sys.argv[0] or "")
    prog = "replay-bench" if not prog or prog.endswith(".py") else prog
    print("resuming with: %s " % prog + " ".join(shlex.quote(t) for t in shown),
          file=sys.stderr)
    return merged, path
