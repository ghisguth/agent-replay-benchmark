"""--reference: grade a run against one other run (the reference) instead of against the recording.

The reference is a saved state of the same suite and fixture (for a quant study: the highest quant at temperature 0).
Per request the answer is compared with the reference's answer at that request; judge() gives the result copy with the
verdict of that comparison and a `ref` dict of the per-turn agreement, stats() sums those into the numbers of the
`vs reference` line and the Metrics rows. Stored verdicts are never changed: the reference is a view, like --compare-state.

--match picks what counts as the same answer: `step` (the first action category), `targets` (the same tool calls in the
same order, each on the same target: file, set of programs, path), `calls` (the same tool calls in the same order with
the same arguments, the same text), `exact` (also the intent argument and the exact text).
"""

import difflib
import json
import os
import re
import shlex

from . import fmt as F
from .baseline import check_baseline, load_baseline
from .cases import grading
from .state import file_sha256, model_label, with_note
from .utils import UserError

MATCHES = ("step", "targets", "calls", "exact")
INTENT = "i"  # omp's free-text intent argument, ignored unless --match exact
# Notes of the normal grading that compare with the recording or the reference models: --reference replaces them.
_RECORDING_NOTES = ("off pattern", "stopped:", "text where the recording", "reported early")


class ReferenceRun:
    """The reference of one run target: `results` {case id: metrics}, the state `header`, the `path` and the `label`."""

    def __init__(self, results, header, path, label):
        self.results, self.header, self.path, self.label = results, header or {}, path, label

    @property
    def temperature(self):
        return (self.header.get("request") or {}).get("temperature")


class Reference:
    """A loaded --reference file and the --match rule: run_for() picks the reference block of a run, judge_all() applies it."""

    def __init__(self, baseline, match):
        self.baseline, self.match = baseline, match
        self.path = baseline.path

    def info(self):
        """What the state header records: the file, its sha256 and the match rule."""
        return {"file": os.path.basename(self.path), "sha256": file_sha256(self.path), "match": self.match}

    def run_for(self, backend_label, model):
        """The ReferenceRun for a run of `model` on `backend_label` (same pick rule as --compare-state), None when the
        file holds no run."""
        block = self.baseline.pick(backend_label, model)
        if block is None:
            return None
        h = block.header or {}
        label = with_note(model_label(h.get("model"), h.get("model_info")), block.note)
        return ReferenceRun(block, h, self.path, label)

    def judge_all(self, results, backend_label, model):
        """A copy of `results` ({case id: metrics}) judged against the reference of that run."""
        ref = self.run_for(backend_label, model)
        if ref is None:
            return dict(results)
        return {cid: judge(m, ref.results.get(cid), self.match, ref.label) for cid, m in results.items()}


def load_reference(path, suite, sha, match):
    """The Reference of a state file, after checking that it is the same suite and fixture as the run; a UserError for
    the smoke and sanity suites (they have no recorded turns to compare) and for a file without a run."""
    if suite in ("smoke", "sanity"):
        raise UserError("--reference compares replay answers; the %s suite has none (use --suite coding or judge)" % suite)
    baseline = load_baseline(path)
    check_baseline(baseline, suite, sha, flag="--reference")
    if not baseline.models:
        raise UserError("--reference: %s holds no run" % path)
    return Reference(baseline, match)


def temperature_warning(reference, run_header, match):
    """One line when `--match calls|exact` compares a run, or the reference, that was not sent at temperature 0 (sampling
    noise then dominates the differences); None otherwise. `run_header`: the run's state header."""
    if match == "step":
        return None
    temps = {"this run": (run_header.get("request") or {}).get("temperature"),
             "the reference": (reference.baseline.blocks()[0].header.get("request") or {}).get("temperature")}
    bad = [who for who, t in temps.items() if t != 0]
    if not bad:
        return None
    return ("--match %s: %s %s not sent at temperature 0, so sampling noise dominates the differences "
            "(use --temperature 0 for both)" % (match, " and ".join(bad), "were" if len(bad) > 1 else "was"))


# -- one answer against another ------------------------------------------------------------------------------

def _text(m):
    return grading._content(m)


def _flat(text):
    return " ".join((text or "").split())


def _name(call):
    return (call.get("function") or {}).get("name") or "?"


def _args(call, exact):
    """The comparable arguments of a call: the parsed dict (intent left out unless `exact`), else the raw string."""
    _, args, _ = grading._call_args(call)
    if args is None:
        return (call.get("function") or {}).get("arguments")
    return args if exact else {k: v for k, v in args.items() if k != INTENT}


_FILE_TOOLS = ("read", "write", "edit")
_PATH_TOOLS = ("grep", "find", "glob", "ls", "search", "lsp")
_SEGMENT = re.compile(r";|&&|\|\||\|")
_HEREDOC = re.compile(r"<<-?\s*(?:'([^'\n]+)'|\"([^\"\n]+)\"|\\?([A-Za-z_][A-Za-z0-9_]*))")
_QUOTED = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"")
_WRAPPERS = ("sudo", "time", "env", "nohup", "command", "exec")
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _strip_heredocs(command):
    """The command without heredoc bodies (the lines after `<<WORD` up to the closing WORD line; all the rest when
    the terminator is missing)."""
    lines = command.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        tags = [m.group(1) or m.group(2) or m.group(3) for m in _HEREDOC.finditer(line)]
        for tag in tags:
            while i < len(lines) and lines[i].strip() != tag:
                i += 1
            i += 1  # the closing line (or past the end)
    return "\n".join(out)


def _newlines_to_semicolons(command):
    """Unquoted newlines become `;` (shlex would treat them as plain whitespace); a trailing backslash joins lines."""
    out, quote, i = [], None, 0
    while i < len(command):
        c = command[i]
        if quote:
            if c == "\\" and quote == '"' and i + 1 < len(command):
                out.append(c)
                i += 1
                c = command[i]
            elif c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c == "\\" and i + 1 < len(command):
            if command[i + 1] == "\n":
                i += 2
                continue
            out.append(c)
            i += 1
            c = command[i]
        elif c == "\n":
            c = ";"
        out.append(c)
        i += 1
    return "".join(out)


def _first_words(segments):
    words = set()
    for parts in segments:
        i = 0
        while i < len(parts):
            w = parts[i]
            if _ASSIGN.match(w) or w in _WRAPPERS or w in ("(", "{", ")", "}", "!"):
                i += 1
                continue
            break
        if i < len(parts):
            w = parts[i].lstrip("({")
            base = os.path.basename(w.rstrip("/")) if w else ""
            if base and base not in (")", "}"):
                words.add(base)
    return words


def programs(command):
    """The sorted set of programs a shell command runs: the first word of each command, split on unquoted ; && || |
    & and newlines, heredoc bodies dropped, leading VAR=value and sudo/time/env/... skipped, basename only."""
    if not isinstance(command, str):
        return ()
    command = _strip_heredocs(command)
    try:
        command = _newlines_to_semicolons(command)
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        lex.commenters = ""
        tokens = list(lex)
        segments, cur = [], []
        for tok in tokens:
            if tok and set(tok) <= set(";&|"):
                segments.append(cur)
                cur = []
            else:
                cur.append(tok)
        segments.append(cur)
    except ValueError:
        plain = _QUOTED.sub(" ", command)
        segments = [seg.split() for seg in re.split(r"\n|;|&&|\|\||\||&", plain)]
    return tuple(sorted(_first_words(segments)))


def target(call):
    """What a call acts on, for --match targets: (kind, value). read/write/edit: the file (no `:N-M` range, no
    whitespace); bash: the programs run; grep/find/glob/ls: the `path` argument, else nothing; others: nothing."""
    name = _name(call)
    _, args, _ = grading._call_args(call)
    args = args or {}
    if name in _FILE_TOOLS:
        return "file", grading.normalize_path(args.get("path"))
    if name == "bash":
        return "programs", programs(args.get("command"))
    if name in _PATH_TOOLS and isinstance(args.get("path"), str) and args["path"].strip():
        return "path", args["path"].strip()
    return "none", None


def _show_target(kind, value):
    return ", ".join(value) if kind == "programs" else str(value)


def targets_difference(m, ref):
    """Why the answer `m` is not on the same targets as `ref`'s (same tool names in order, same target per call; text
    answers compare as under `calls`), else None."""
    calls, rcalls = grading.calls_from(m), grading.calls_from(ref)
    if not calls and not rcalls:
        return None if _flat(_text(m)) == _flat(_text(ref)) else "text differs"
    if not calls or not rcalls:
        return "reference: %s, this: %s" % (_summary(rcalls), _summary(calls))
    for i, (c, rc) in enumerate(zip(calls, rcalls), 1):
        if _name(c) != _name(rc):
            return "call %d: %s, reference %s" % (i, _name(c), _name(rc))
        (kind, a), (_, b) = target(c), target(rc)
        if a != b:
            what = {"file": "other file", "programs": "other programs", "path": "other path"}.get(kind, "other target")
            return "call %d %s: %s (%s vs %s)" % (i, _name(c), what, _show_target(kind, a), _show_target(kind, b))
    if len(calls) != len(rcalls):
        return "%d calls vs %d" % (len(calls), len(rcalls))
    return None


def _summary(calls):
    return ",".join(_name(c) for c in calls) if calls else "text"


def difference(m, ref, match):
    """Why the answer `m` is not the same as `ref`'s under `match` ("calls" or "exact"), else None."""
    exact = match == "exact"
    calls, rcalls = grading.calls_from(m), grading.calls_from(ref)
    if not calls and not rcalls:
        a, b = _text(m), _text(ref)
        same = a.strip() == b.strip() if exact else _flat(a) == _flat(b)
        return None if same else "text differs"
    if not calls or not rcalls:
        return "reference: %s, this: %s" % (_summary(rcalls), _summary(calls))
    for i, (c, rc) in enumerate(zip(calls, rcalls), 1):
        if _name(c) != _name(rc):
            return "call %d: %s, reference %s" % (i, _name(c), _name(rc))
        a, b = _args(c, exact), _args(rc, exact)
        if a != b:
            if isinstance(a, dict) and isinstance(b, dict):
                keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
                return "call %d %s: %s differ%s" % (i, _name(c), ", ".join(keys[:3]), "" if len(keys) != 1 else "s")
            return "call %d %s: arguments differ" % (i, _name(c))
    if len(calls) != len(rcalls):
        return "%d calls vs %d" % (len(calls), len(rcalls))
    if exact and _text(m).strip() != _text(ref).strip():
        return "text before the calls differs"
    return None


def _canonical(call):
    args = _args(call, False)
    return args if isinstance(args, str) else json.dumps(args, sort_keys=True, ensure_ascii=False)


def args_similarity(calls, rcalls):
    """Mean similarity (1.0 identical) of the arguments of two call lists with the same tool names in the same order."""
    ratios = []
    for c, rc in zip(calls, rcalls):
        a, b = _canonical(c), _canonical(rc)
        ratios.append(1.0 if a == b else difflib.SequenceMatcher(None, a, b).ratio())
    return sum(ratios) / len(ratios) if ratios else None


def usable(ref):
    """Whether a reference result is an answer to compare with (not missing, an error, a SKIP or a FAIL)."""
    return bool(ref) and not ref.get("error") and ref.get("eval_result") in ("PASS", "WARN")


def _kept_notes(reason):
    """The notes of a normal WARN reason that do not compare with the recording (schema, markup, path, write notes)."""
    return [p for p in (reason or "").split("; ") if p and not p.startswith(_RECORDING_NOTES)]


def judge(m, ref, match, label=""):
    """A copy of the result `m` with its verdict set by the comparison with the reference result `ref` and a `ref` dict
    (`ok`, `step`, `tools`, `calls`, `text`, `sim`, `tokens`). A broken answer stays a FAIL, a compaction and a request
    the reference has no answer to keep the normal verdict (the latter marked `(no reference answer)`)."""
    out = dict(m)
    if m.get("eval_result") == "SKIP" or m.get("error"):
        return out
    if m.get("kind") == "compaction":
        return out
    if not usable(ref):
        out["ref"] = {"ok": False, "label": label}
        if m.get("eval_result") != "SKIP":
            out["fail_reason"] = ((m.get("fail_reason") or "") + " (no reference answer)").strip()
        return out
    calls, rcalls = grading.calls_from(m), grading.calls_from(ref)
    names, rnames = [_name(c) for c in calls], [_name(c) for c in rcalls]
    cat, rcat = grading.tool_category(calls), grading.tool_category(rcalls)
    calls_diff = difference(m, ref, "calls")
    targets_diff = targets_difference(m, ref)
    text = None
    if not rcalls:
        text = not calls and _flat(_text(m)) == _flat(_text(ref))
    out["ref"] = {"ok": True, "label": label, "step": cat == rcat, "tools": names == rnames, "targets": targets_diff is None,
                  "calls": calls_diff is None, "first": rnames[0] if rnames else grading.TEXT_ONLY,
                  "first_calls": calls_diff is None, "first_targets": targets_diff is None,
                  "text": text, "tokens": ref.get("tokens"),
                  "sim": args_similarity(calls, rcalls) if calls and names == rnames else None}
    if m.get("eval_result") == "FAIL":
        return out
    if match == "step":
        why = None if cat == rcat else "reference: %s, this: %s" % (rcat, cat)
    elif match == "targets":
        why = targets_diff
    else:
        why = difference(m, ref, match)
    notes = _kept_notes(m.get("fail_reason")) if m.get("eval_result") == "WARN" else []
    if why:
        notes.insert(0, why)
    if notes:
        out["eval_result"], out["fail_reason"] = "WARN", "; ".join(dict.fromkeys(notes))
    else:
        out["eval_result"] = "PASS"
        out["fail_reason"] = ("same step as reference: %s" % cat if match == "step"
                              else "same targets as reference: %s" % _summary(calls) if match == "targets"
                              else "same calls as reference: %s" % _summary(calls))
    return out


# -- the numbers -----------------------------------------------------------------------------------------------

def by_tool(ok, key):
    """[(first tool of the reference, same, turns)] of judged results, most turns first (`<text>` for text answers)."""
    groups = {}
    for r in ok:
        g = groups.setdefault(r["ref"].get("first") or grading.TEXT_ONLY, [0, 0])
        g[1] += 1
        g[0] += bool(r["ref"][key])
    return [(t, g[0], g[1]) for t, g in sorted(groups.items(), key=lambda kv: (-kv[1][1], kv[0]))]


def by_tool_text(groups, limit=6):
    """`read 8/12 · bash 8/27 · ...` of by_tool() (the `limit` most frequent tools, then `+N more`); a dash when empty."""
    if not groups:
        return F.DASH
    shown = ["%s %d/%d" % g for g in groups[:limit]]
    if len(groups) > limit:
        shown.append("+%d more" % (len(groups) - limit))
    return " \u00b7 ".join(shown)


def stats(results):
    """The agreement of a list of judged results with their reference, or None when none was judged: `turns` (answers
    compared, compactions and turns without a reference answer left out), `missing` (turns without one), `step`, `tools`,
    `targets`, `calls`, `text` (percent), `sim` (mean arguments similarity), `tokens` (output tokens as a ratio of the reference's),
    `broken` (FAIL answers) and `loops` (FAILs for repeated calls or text)."""
    judged = [r for r in results if r.get("ref") is not None]
    if not judged:
        return None
    ok = [r for r in judged if r["ref"].get("ok")]
    share = lambda key, rs: 100 * sum(1 for r in rs if r["ref"][key]) / len(rs) if rs else None
    texts = [r for r in ok if r["ref"].get("text") is not None]
    sims = [r["ref"]["sim"] for r in ok if r["ref"].get("sim") is not None]
    ref_tokens = sum(r["ref"].get("tokens") or 0 for r in ok)
    fails = [r for r in results if r.get("eval_result") == "FAIL" and r.get("kind") != "compaction"]
    return {"label": judged[0]["ref"].get("label") or "reference", "turns": len(ok), "missing": len(judged) - len(ok),
            "step": share("step", ok), "tools": share("tools", ok), "targets": share("targets", ok),
            "calls": share("calls", ok), "text": share("text", texts),
            "calls_by_tool": by_tool(ok, "calls"), "targets_by_tool": by_tool(ok, "targets"),
            "sim": sum(sims) / len(sims) if sims else None,
            "tokens": sum(r.get("tokens") or 0 for r in ok) / ref_tokens if ref_tokens else None,
            "broken": len(fails),
            "loops": sum(1 for r in fails if "repeat" in (r.get("fail_reason") or ""))}


BY_TOOL_ROWS = (("Same calls by first tool", "calls_by_tool"), ("Same targets by first tool", "targets_by_tool"))
ROWS = (("Same first step %", "step", "pct"), ("Same tools %", "tools", "pct"), ("Same targets %", "targets", "pct"),
        ("Same calls %", "calls", "pct"),
        ("Same text %", "text", "pct"), ("Args similarity", "sim", "ratio"), ("Output tokens vs reference", "tokens", "ratio"),
        ("Broken calls", "broken", "count"), ("Loops", "loops", "count"))


def value_text(v, kind):
    if v is None:
        return F.DASH
    return F.pct(v) if kind == "pct" else "%.2f" % v if kind == "ratio" else str(v)


def delta_text(t, b, kind):
    if t is None or b is None:
        return ""
    return F.signed_pt(t, b) if kind == "pct" else "%+.2f" % (t - b) if kind == "ratio" else "%+d" % (t - b)


def line(st):
    """The `vs reference <label>: ...` summary line of stats()."""
    parts = ["%s %s" % (name.lower().replace(" %", "").replace("output tokens vs reference", "output tokens"),
                        value_text(st[key], kind) if kind != "ratio" or key == "sim" else "x" + value_text(st[key], kind))
             for name, key, kind in ROWS]
    tail = " (%d turns%s)" % (st["turns"], ", %d without a reference answer" % st["missing"] if st["missing"] else "")
    return "vs reference %s: %s%s" % (st["label"], " · ".join(parts), tail)
