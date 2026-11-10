"""Named benchmark presets and selection resolution."""

import argparse

from .cases import replay as replay_cases
from .cli_args import parse_max_context
from .utils import UserError

SUITES = {
    "smoke": None,
    "sanity": None,
    "coding": "coding",
    "judge": "judge",
}

# Suites of generic cases that need no replay fixture (their key is also the run's mode).
CASE_SUITES = ("smoke", "sanity")

DEFAULT_PRESET = "large"
PRESET_ALIASES = {"128k": "large", "236k": "xlarge"}
UNSET = object()

# Presets: named sets of --suite/--max-context/--no-subagents/--turns.
PRESETS = {
    "quickstart": {
        "suite": "coding",
        "max_context": "full",
        "no_subagents": True,
        "turns": (1, 5),
        "what": "coding, main agent, requests 1-5",
        "use": "first try of a server or model",
    },
    "small": {
        "suite": "coding",
        "max_context": "32k",
        "no_subagents": True,
        "turns": None,
        "what": "coding, main agent up to 32k",
        "use": "quick checks, slow machines",
    },
    "medium": {
        "suite": "coding",
        "max_context": "64k",
        "no_subagents": True,
        "turns": None,
        "what": "coding, main agent up to 64k",
        "use": "quick checks, slow machines",
    },
    "large": {
        "suite": "coding",
        "max_context": "128k",
        "no_subagents": True,
        "turns": None,
        "what": "coding, main agent up to 128k",
        "use": "everyday model comparison (the default)",
    },
    "xlarge": {
        "suite": "coding",
        "max_context": "236k",
        "no_subagents": True,
        "turns": None,
        "what": "coding, main agent up to the compaction",
        "use": "long-context quality",
    },
    "compaction": {
        "suite": "coding",
        "max_context": "full",
        "no_subagents": True,
        "turns": (1, 217),
        "what": "coding, main agent, requests 1-217",
        "use": "the compaction call and the restart after it",
    },
    "main": {
        "suite": "coding",
        "max_context": "full",
        "no_subagents": True,
        "turns": None,
        "what": "coding, whole main session",
        "use": "the whole main session",
    },
    "full": {
        "suite": "coding",
        "max_context": "full",
        "no_subagents": False,
        "turns": None,
        "what": "coding, everything incl. subagents",
        "use": "everything, many hours",
    },
    "judge": {
        "suite": "judge",
        "max_context": "full",
        "no_subagents": False,
        "turns": None,
        "what": "one recorded match-judge case",
        "use": "the match-judge case",
    },
    "smoke": {
        "suite": "smoke",
        "max_context": "full",
        "no_subagents": False,
        "turns": None,
        "what": "4 generic cases",
        "use": "quick server check",
    },
    "sanity": {
        "suite": "sanity",
        "max_context": "full",
        "no_subagents": False,
        "turns": None,
        "what": "10 generic cases (format, JSON, tools, math, recall)",
        "use": "does the model work at all",
    },
}

_FIXTURE_CACHE = {}


def _get_fixture(suite):
    path = replay_cases.default_fixture(suite)
    if not path:
        return None
    if path not in _FIXTURE_CACHE:
        try:
            _FIXTURE_CACHE[path] = replay_cases.load_fixture(path)
        except Exception:
            return None
    return _FIXTURE_CACHE[path]


def preset_size(name):
    """(n, top) request count and largest prompt (formatted string, or None) for preset `name`."""
    p = PRESETS[name]
    if p["suite"] in CASE_SUITES:
        from .cases import select_cases
        return len(select_cases(suite=p["suite"])), None
    fx = _get_fixture(p["suite"])
    if fx is not None:
        mc = parse_max_context(p["max_context"])
        sel = replay_cases.select_entries(
            fx, max_context=mc, no_subagents=p["no_subagents"], turns=p["turns"]
        )
        top_tok = max((replay_cases.recorded_prompt_tokens(e) for _, e in sel), default=0)
        top = ("%.1fk" % (top_tok / 1024)) if top_tok else None
        return len(sel), top
    return 0, None


def parse_preset(value):
    v = str(value).strip().lower()
    v = PRESET_ALIASES.get(v, v)
    if v not in PRESETS:
        raise argparse.ArgumentTypeError("invalid preset %r (choose from %s)" % (value, ", ".join(PRESETS)))
    return v


def preset_line(name):
    """The line printed at the start of a run: 'preset large: coding, <=128k, main agent only, 58 requests'."""
    p = PRESETS[name]
    parts = [p["suite"]]
    if p["suite"] not in CASE_SUITES:
        if p["max_context"] != "full":
            parts.append("\u2264" + p["max_context"])
        if p["turns"]:
            parts.append("requests %d-%d" % p["turns"])
        if p["suite"] == "coding":
            parts.append("main agent only" if p["no_subagents"] else "all streams")
    n, _ = preset_size(name)
    parts.append("%d requests" % n if p["suite"] not in CASE_SUITES else "%d cases" % n)
    return "preset %s: %s" % (name, ", ".join(parts))


def presets_help():
    rows = [("name", "replays", "requests", "max prompt", "use")]
    for n, p in PRESETS.items():
        alias = [a for a, t in PRESET_ALIASES.items() if t == n]
        count, top = preset_size(n)
        rows.append(
            (
                n + (" (alias %s)" % alias[0] if alias else ""),
                p["what"],
                str(count),
                top or "-",
                p["use"],
            )
        )
    w = [max(len(r[i]) for r in rows) for i in range(4)]
    return "\n".join("  " + "  ".join(r[i].ljust(w[i]) for i in range(4)) + "  " + r[4] for r in rows)


def resolve_selection(ns, default=True):
    """(preset, suite key, max_context, no_subagents, turns) a parsed namespace stands for, preset values laid under
    the explicit options. A UserError when an explicit option differs from what the preset sets. No suite, preset
    or fixture and no --test: the default preset, or the coding suite when selection options are given."""
    suite, preset = ns.suite, ns.preset
    mc_given = ns.max_context is not UNSET
    mc = None if ns.max_context is UNSET else ns.max_context
    sel_given = mc_given or ns.no_subagents or ns.turns is not None or ns.fixture is not None
    if preset is None and suite is None and not ns.test and default:
        if sel_given:
            suite = "coding"
        else:
            preset = DEFAULT_PRESET
    if preset is None:
        return None, suite or "smoke", mc, ns.no_subagents, ns.turns
    p = PRESETS[preset]
    pmc = parse_max_context(p["max_context"])
    clashes = []
    if suite is not None and (SUITES[suite] or suite) != p["suite"]:
        clashes.append("--suite %s" % suite)
    if mc_given and mc != pmc:
        clashes.append("--max-context %s" % (("%dk" % (mc // 1024) if mc % 1024 == 0 else str(mc)) if mc else "full"))
    if ns.no_subagents and not p["no_subagents"]:
        clashes.append("--no-subagents")
    if ns.turns is not None and ns.turns != p["turns"]:
        clashes.append("--turns %d-%d" % ns.turns)
    if clashes:
        if p["suite"] in CASE_SUITES:
            raise UserError("%s conflicts with --preset %s (%s); drop one of them (%s takes no selection options)"
                            % (", ".join(clashes), preset, preset_line(preset)[len("preset %s: " % preset):], p["suite"]))
        raise UserError("%s conflicts with --preset %s (%s); drop one of them, or use --suite %s with the options"
                        % (", ".join(clashes), preset, preset_line(preset)[len("preset %s: " % preset):], p["suite"]))
    return preset, p["suite"], pmc, p["no_subagents"], p["turns"]
