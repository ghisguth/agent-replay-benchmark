"""Grading for replay requests: PASS/WARN/FAIL verdicts, tool schema checks, repetition,
compaction requirements, step-category validation and notes.
"""

import ast
import collections
import json
import os
import re
import zlib

from .judge import judge_verdict_check

# Compaction summaries: at least this many tokens (chars / CHARS_PER_TOKEN) and at most this multiple of the recorded one.
COMPACTION_MIN_TOKENS = 300
COMPACTION_MAX_RATIO = 3.0
CHARS_PER_TOKEN = 3.5
# Degenerate repetition in the answer text (checked on content; thinking is stored as counts only).
REPEAT_CALL_FAIL = 3
REPEAT_MIN_CHARS = 2000      # shorter texts are not checked with zlib
REPEAT_ZLIB_RATIO = 0.15     # compressed / raw size below this: mostly a repeated chunk (good answers: 0.41+)
REPEAT_LINE_COUNT = 8        # one line (>= 12 chars) occurring this often (good answers: 2 at most)
REPEAT_LINE_SHARE = 0.3      # ... and making up this share of the text
MAX_SCHEMA_NOTES = 3

_MARKUP_RE = re.compile(r'<tool_call>|<function=|<\|im_start\|>|\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:')


def calls_from(metrics):
    """The tool calls of a stored or live answer (the OpenAI ``tool_calls`` list), [] when it made none."""
    return metrics.get("tool_calls") or []


def _content(metrics):
    """The answer's text without the tool-call JSON that the runner appends to ``text``."""
    if metrics.get("content") is not None:
        return metrics["content"]
    return metrics.get("text") or ""


def entry_key(entry):
    """Stable key of a fixture entry in the checks sidecar: <stream>/<epoch>/<kind>/<turn>
    (e.g. main/0/turn/3, main/0/compaction/-)."""
    turn = entry.get("turn")
    return "%s/%s/%s/%s" % (entry.get("stream", ""), entry.get("epoch") or 0, entry.get("kind", "turn"),
                            "-" if turn is None else turn)


def _call_args(call):
    """(name, args or None, problem or None): the arguments parsed, or why they are unusable."""
    fn = call.get("function") or {}
    name = fn.get("name") or ""
    try:
        args = json.loads(fn.get("arguments") or "")
    except (json.JSONDecodeError, TypeError):
        return name, None, "tool call %s: arguments are not valid JSON" % (name or "?")
    if not isinstance(args, dict):
        return name, None, "tool call %s: arguments are not a JSON object" % (name or "?")
    return name, args, None


_JSON_TYPES = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer()),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _json_type(v):
    return ("null" if v is None else "boolean" if isinstance(v, bool) else "string" if isinstance(v, str)
            else "array" if isinstance(v, list) else "object" if isinstance(v, dict) else "number")


def schema_problems(schema, value, path=""):
    """Plain-words problems of a JSON value against a (simple) JSON schema: missing required keys, unknown keys of
    closed objects, wrong types, values outside an enum. Unknown schema keywords are ignored."""
    out = []
    if not isinstance(schema, dict):
        return out
    for key in ("anyOf", "oneOf"):
        if schema.get(key):
            if not any(not schema_problems(s, value, path) for s in schema[key]):
                out.append("%s: matches none of the allowed shapes" % (path or "arguments"))
            return out
    want = schema.get("type")
    if want is not None:
        wants = want if isinstance(want, list) else [want]
        if not any(_JSON_TYPES.get(w, lambda v: True)(value) for w in wants):
            what = "%s: %s, expected %s" % (path or "arguments", _json_type(value), "/".join(wants))
            if isinstance(value, str) and "array" in wants:
                what = "%s: JSON string, expected array" % (path or "arguments")
            return [what]
    if "enum" in schema and value not in schema["enum"]:
        out.append("%s: %r not in the allowed values" % (path or "arguments", value))
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                out.append("%smissing required %r" % (path + ": " if path else "", key))
        for key, v in value.items():
            sub = "%s.%s" % (path, key) if path else key
            if key in props:
                if v is None and key not in (schema.get("required") or []):
                    continue  # omp strips null optionals
                out += schema_problems(props[key], v, sub)
            elif schema.get("additionalProperties") is False:
                out.append("%sunknown key %r (omp drops it)" % (path + ": " if path else "", key))
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value[:20]):
            out += schema_problems(schema["items"], v, "%s[%d]" % (path or "arguments", i))
    return out


def _repetition(content):
    """A reason when a long text is mostly one repeated chunk, else None."""
    if len(content) < REPEAT_MIN_CHARS:
        return None
    raw = content.encode("utf-8")
    ratio = len(zlib.compress(raw, 6)) / len(raw)
    lines = [ln.strip() for ln in content.splitlines() if len(ln.strip()) >= 12]
    top, n = collections.Counter(lines).most_common(1)[0] if lines else ("", 0)
    share = n * len(top) / len(content)
    if ratio < REPEAT_ZLIB_RATIO:
        return "text is mostly repeated (compresses to %.1f%%)" % (100 * ratio)
    if n >= REPEAT_LINE_COUNT and share >= REPEAT_LINE_SHARE:
        return "text repeats one line %d times" % n
    return None


def _heading_present(content, name):
    return re.search(r"^##(?!#)[ \t]*%s[ \t:]*$" % re.escape(name).replace(r"\ ", r"[ \t]+"), content,
                     re.M | re.I) is not None


def _grade_compaction(case, content, checks):
    """(verdict, reason) problems of a compaction summary, or None when it is fine."""
    est = len(content) / CHARS_PER_TOKEN
    if est < COMPACTION_MIN_TOKENS:
        return "FAIL", "compaction summary too short (~%d tokens)" % est
    heads = (checks.get("compaction") or {}).get("headings") or []
    notes = []
    if heads:
        missing = [h for h in heads if not _heading_present(content, h)]
        need = (checks.get("compaction") or {}).get("min_headings", len(heads) - 2)
        if len(heads) - len(missing) < need:
            return "FAIL", "compaction lacks %d of %d headings: %s" % (len(missing), len(heads), ", ".join(missing))
        if missing:
            notes.append("compaction lacks heading %s" % ", ".join(missing))
        m = re.search(r"^##(?!#)[ \t]*%s" % re.escape(heads[0]), content, re.M | re.I)
        if m and content[:m.start()].strip():
            notes.append("text before '## %s'" % heads[0])
    rec = len((case["entry"].get("recorded_answer") or {}).get("content") or "")
    if rec and len(content) > COMPACTION_MAX_RATIO * rec:
        notes.append("summary %.1fx the recorded length" % (len(content) / rec))
    return ("WARN", "; ".join(notes)) if notes else None


def _report_spec(case, checks):
    for spec in checks.get("reports") or []:
        pre = spec.get("stream_prefix") or ""
        if str(case["entry"].get("label", "")).startswith(pre) or str(case.get("stream", "")).startswith(pre):
            return spec
    return None


def _report_missing(spec, content):
    """Names of the required parts a final report lacks."""
    out = []
    for req in spec.get("require") or []:
        if not any(re.search(rx, content) for rx in req.get("any") or []):
            out.append(req["name"])
    return out


TEXT_ONLY = "<text>"   # the "first tool" of an answer without tool calls


def first_tool(names):
    """The first tool name of an answer, or "<text>" when it called none."""
    return names[0] if names else TEXT_ONLY


_LOOK_TOOLS = {"read", "grep", "find", "eval", "lsp", "glob", "ls", "bash"}
_CHANGE_TOOLS = {"write", "edit"}
_CATEGORY_OF = {"wait": "wait", "task": "delegate", "todo": "plan"}


_HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1([^\n]*)\n.*?^[ \t]*\2[ \t]*$", re.S | re.M)
_QUOTED = re.compile(r"'[^']*'|\"(?:[^\"\\\\]|\\\\.)*\"")
_SEGMENTS = re.compile(r"&&|\|\||[;|(\n]")
_REDIRECT = re.compile(r"(?<![\d&>])>>?(?!&)\s*(\S+)")
_PATH_TOOLS = {"mkdir", "touch", "rm", "mv", "cp", "tee"}
_GIT_MUTATING = {"add", "commit", "rm", "mv", "checkout", "reset", "apply", "stash", "restore", "switch",
                 "cherry-pick", "revert"}


def _unquote(m):
    """A quoted string without spaces or shell syntax loses its quotes (a path); any other becomes `Q`."""
    inner = m.group(0)[1:-1]
    return "Q" if re.search(r"[\s;&|<>()]", inner) else inner


def _scratch(path, in_tmp=False):
    """True for a path under /tmp or $TMPDIR (or /dev/null); a relative path counts when the last `cd` went there."""
    if path.startswith(("/tmp/", "$TMPDIR/", "${TMPDIR}/", "/dev/null")) or path in ("/tmp", "$TMPDIR", "${TMPDIR}"):
        return True
    return in_tmp and not path.startswith(("/", "~", "$"))


def bash_changes_files(command):
    """True when a bash command line modifies the workspace (conservative: scripts that write files stay False).
    Counts: `dotnet new|add|sln add`, `git add|commit|...`, `sed|perl -i`, mkdir/touch/rm/mv/cp/tee unless every path
    argument is under /tmp, and `cat|printf|echo ... > FILE` (hand-made content) to a non-/tmp file. Output captured
    from other commands (`dotnet run > out.tsv`) is inspection, so it stays False."""
    cmd = _HEREDOC.sub(r"\3", command)
    cmd = _QUOTED.sub(_unquote, cmd)
    in_tmp = False   # the last `cd` went under /tmp
    for seg in _SEGMENTS.split(cmd):
        words = seg.split()
        if not words:
            continue
        w, rest = words[0], words[1:]
        if w == "cd":
            target = rest[0] if rest else "~"
            if target.startswith(("/", "~", "$")) or not rest:
                in_tmp = _scratch(target)
            continue
        if w == "dotnet" and rest[:1] in (["new"], ["add"]):
            return True
        if w == "dotnet" and rest[:1] == ["sln"] and "add" in rest:
            return True
        if w == "git" and rest[:1] and rest[0] in _GIT_MUTATING:
            return True
        if w in ("sed", "perl") and any(re.fullmatch(r"-\w*i\w*", x) for x in rest):
            return True
        if w in _PATH_TOOLS:
            paths = [x for x in rest if not x.startswith("-")]
            if not (paths and all(_scratch(x, in_tmp) for x in paths)):
                return True
        if w in ("cat", "printf", "echo"):
            if any(not _scratch(t, in_tmp) for t in _REDIRECT.findall(seg)):
                return True
    return False


def tool_category(calls):
    """The kind of step an answer takes, from its FIRST tool call (a raw tool_calls list): `look` (read, grep, find,
    eval, lsp, glob, ls, bash), `wait` (wait, or a read of a proc:// path), `change` (write, edit, or a bash command that modifies files; a write or edit under /tmp is scratch: `look`), `delegate` (task),
    `plan` (todo), "<text>" without calls, any other tool name as is. Good models differ in which look tool they
    pick, so the pattern check compares categories, not tool names."""
    if not calls:
        return TEXT_ONLY
    fn = calls[0].get("function") or {}
    name = fn.get("name") or ""
    if name == "read":
        try:
            path = json.loads(fn.get("arguments") or "").get("path")
        except (ValueError, TypeError, AttributeError):
            path = None
        return "wait" if isinstance(path, str) and path.startswith("proc://") else "look"
    if name == "bash":
        try:
            command = json.loads(fn.get("arguments") or "").get("command")
        except (ValueError, TypeError, AttributeError):
            command = None
        return "change" if isinstance(command, str) and bash_changes_files(command) else "look"
    if name in _LOOK_TOOLS:
        return "look"
    if name in _CHANGE_TOOLS:
        try:
            path = json.loads(fn.get("arguments") or "").get("path")
        except (ValueError, TypeError, AttributeError):
            path = None
        return "look" if isinstance(path, str) and _scratch(path) else "change"   # a scratch script is not a change
    return _CATEGORY_OF.get(name, name)


def _strip_intent(tool_schema, args, checks):
    """(schema, args, problem or None) without omp's intent field (checks "intent_field", e.g. "i"). omp injects the
    field into every tool schema and strips it from the arguments before validating, so a missing or extra one is
    harmless; one longer than "intent_max" makes omp refuse to run the tool. Without "intent_field" in the sidecar
    nothing is changed."""
    field = checks.get("intent_field")
    schema = tool_schema or {}
    if not field:
        return schema, args, None
    if isinstance(schema.get("properties"), dict):
        schema = dict(schema, properties={k: v for k, v in schema["properties"].items() if k != field},
                      required=[k for k in schema.get("required") or [] if k != field])
    rest = {k: v for k, v in args.items() if k != field}
    limit = checks.get("intent_max")
    v = args.get(field)
    if limit and isinstance(v, str) and len(v) > limit:
        return schema, rest, "intent %s over %d chars: tool not run" % (field, limit)
    return schema, rest, None


_GLOB_RE = re.compile(r"[*?\[]")
_LEAK_RE = re.compile(r"<parameter=|<function=|</parameter|</function|</?tool_call>")
_XML_EXT = (".xml", ".csproj", ".slnx", ".props", ".targets")


def normalize_path(path):
    """Normalize a file path for comparison: strip :N-M line-range suffix and whitespace."""
    if not path or not isinstance(path, str):
        return ""
    return re.sub(r":\d+(?:-\d+)?$", "", path.strip())


def _path_notes(tools, name, args):
    """WARN reasons for the path argument of a read/write/edit call (tools whose schema has a "path" property): a
    directory given to write/edit, or glob characters. omp's "path:58-60" line-range suffix on read is fine."""
    props = (tools.get(name) or {}).get("properties")
    if name not in ("read", "write", "edit") or not isinstance(props, dict) or "path" not in props:
        return []
    path = args.get("path")
    if not isinstance(path, str):
        return []
    if name != "read" and path.endswith("/"):
        return ["%s to a directory" % name]
    if _GLOB_RE.search(path):
        return ["glob in path (%s)" % name]
    return []


def _write_notes(name, args):
    """WARN reason when the content of a write is plainly wrong for the file's extension (.json, .py, XML files)."""
    path, content = args.get("path"), args.get("content")
    if name != "write" or not isinstance(path, str) or not isinstance(content, str):
        return []
    base = path.rsplit("/", 1)[-1]
    ext = os.path.splitext(base.split(":")[0])[1].lower()
    if ext == ".json":
        try:
            json.loads(content)
        except ValueError:
            return ["write %s: not JSON" % base]
    elif ext == ".py":
        try:
            ast.parse(content)
        except SyntaxError as e:
            return ["write %s: SyntaxError line %s" % (base, e.lineno)]
        except ValueError:
            return ["write %s: not Python" % base]
    elif ext in _XML_EXT:
        if not content.lstrip("\ufeff \t\r\n").startswith("<"):
            return ["write %s: not XML" % base]
    return []


def _leak_clean(value):
    """(value without the keys that carry leaked tool-call markup, whether anything leaked), at any depth. A key leaks
    when it contains markup or a newline; a string value leaks when it matches the markup (the value is kept)."""
    if isinstance(value, str):
        return value, bool(_LEAK_RE.search(value))
    if isinstance(value, list):
        out = [_leak_clean(v) for v in value]
        return [v for v, _ in out], any(f for _, f in out)
    if isinstance(value, dict):
        kept, leaked = {}, False
        for k, v in value.items():
            if "\n" in k or _LEAK_RE.search(k):
                leaked = True
                continue
            kept[k], f = _leak_clean(v)
            leaked = leaked or f
        return kept, leaked
    return value, False


def _tools_phrase(names):
    return ",".join(names) if names else "text"


def schema_tool_names(entry):
    return [(t.get("function") or {}).get("name", "") for t in entry.get("tools") or []]


def grade(case, metrics):
    """(verdict, reason) of a finished replay request. Sets metrics["tool_names"] and metrics["matches_recorded"].

    FAIL (clearly broken): cut at the token cap (finish_reason "length"), empty answer, tool-call arguments that are
    not a JSON object, a tool name outside the request's schemas, tool-call markup written as text, the same call 3+
    times or mostly repeated text, a compaction with tool calls / too short / missing most of the required headings.
    WARN (real but recoverable): a write/edit to a directory path, a glob in a read/write/edit path, a write whose
    content is not valid for its extension (.json, .py, XML files), a first step (look / wait / change / delegate / plan / text) that no reference model took there,
    schema problems in the arguments, tool-call markup leaked into the arguments (parser), the same call twice, text where the recording
    called a tool unless a reference model also answered in text there (main: stopped; subagent: reported early), a final report without its required parts, a compaction
    missing a heading or wrapped in preamble or 3x the recorded length. Otherwise PASS; the reason only says how it
    compares with the recording. The fixture's sidecar (``case["fixture"]["checks"]``) holds the expectations that
    belong to one conversation (headings, report parts); the judge suite's final request is graded on its verdict.
    """
    entry = case["entry"]
    calls = calls_from(metrics)
    names = [(c.get("function") or {}).get("name", "") for c in calls]
    recorded = case["recorded_tools"]
    metrics["tool_names"] = names
    metrics["matches_recorded"] = sorted(names) == sorted(recorded)
    content = _content(metrics)
    checks = (case["fixture"] or {}).get("checks") or {}
    category = tool_category(calls)
    accepted = (checks.get("accepted") or {}).get(entry_key(entry))
    metrics["in_line"] = (category == "look" or category in accepted) if accepted is not None else None
    if metrics.get("error"):
        return "FAIL", str(metrics["error"])
    if metrics.get("finish_reason") == "length":
        cap = (metrics.get("usage") or {}).get("completion_tokens") or metrics.get("tokens") \
            or entry["params"].get("max_completion_tokens")
        what = "a tool call" if calls else "text" if content.strip() else "thinking"
        return "FAIL", "cut at %s tokens (%s)" % (cap, what)
    if not content.strip() and not calls:
        extra = ", only reasoning" if (metrics.get("reasoning_chars") or 0) > 0 else ""
        return "FAIL", "Empty response (no content and no tool calls%s)" % extra
    allowed = set(schema_tool_names(entry))
    parsed = []
    for c in calls:
        name, args, problem = _call_args(c)
        if problem:
            return "FAIL", problem
        if name not in allowed:
            return "FAIL", "tool %r is not in the request's tool schemas" % name
        parsed.append((name, args))
    if not calls and _MARKUP_RE.search(content):
        return "FAIL", "tool call written as text"
    seen = collections.Counter((n, json.dumps(a, sort_keys=True)) for n, a in parsed)
    (top_key, top_n), = seen.most_common(1) if seen else [(("", ""), 0)]
    if top_n >= REPEAT_CALL_FAIL:
        return "FAIL", "same call %s repeated %d times" % (top_key[0], top_n)
    rep = _repetition(content)
    if rep:
        return "FAIL", rep
    if case["kind"] == "compaction":
        if calls:
            return "FAIL", "tool calls in a compaction: %s" % ",".join(names)
        return _grade_compaction(case, content, checks) or ("PASS", "")
    expect = (case["fixture"].get("judge") or {}).get("expect")
    if entry.get("judge_final") and expect and not calls:
        # the judge's final request: the verdict is graded, not just the (empty) tool list
        ok, why = judge_verdict_check(expect, content)
        return ("PASS", "") if ok else ("FAIL", why)

    notes = []
    tools = {(t.get("function") or {}).get("name"): (t.get("function") or {}).get("parameters")
             for t in entry.get("tools") or []}
    problems = []
    intent = []
    leaked = []
    for name, args in parsed:
        args, leak = _leak_clean(args)
        if leak and name not in leaked:
            leaked.append(name)
        schema, rest, bad = _strip_intent(tools.get(name), args, checks)
        if bad:
            intent.append(bad)
        problems += ["%s %s" % (name, p) for p in schema_problems(schema, rest)]
    notes += intent[:1]
    notes += ["markup: tool-call markup leaked into %s arguments (parser)" % n for n in leaked]
    for name, args in parsed:
        notes += _path_notes(tools, name, args) + _write_notes(name, args)
    if problems:
        notes.append("schema: " + "; ".join(problems[:MAX_SCHEMA_NOTES])
                     + (" (+%d more)" % (len(problems) - MAX_SCHEMA_NOTES) if len(problems) > MAX_SCHEMA_NOTES else ""))
    if top_n == 2:
        notes.append("same call twice: %s" % top_key[0])
    spec = _report_spec(case, checks) if case["stream"] != "main" else None
    if not calls:
        if recorded and accepted is not None and TEXT_ONLY in accepted:
            pass   # a reference model also answered in text here
        elif recorded:
            if spec is not None and re.search(spec.get("detect") or "$^", content):
                notes.append("reported early (recorded %s)" % _tools_phrase(recorded))
            elif case["stream"] == "main":
                notes.append("stopped: text where the agent continued (recorded %s)" % _tools_phrase(recorded))
            else:
                notes.append("text where the recording called %s" % _tools_phrase(recorded))
        elif spec is not None:
            missing = _report_missing(spec, content)
            if missing:
                notes.append("final report lacks: %s" % ", ".join(missing))
    if metrics["in_line"] is False and not (recorded and not calls):
        notes.append("off pattern: %s (reference models: %s)" % (category, "/".join(accepted)))
    if notes:
        return "WARN", "; ".join(dict.fromkeys(notes))
    if metrics["matches_recorded"]:
        return "PASS", "same as recorded: %s" % _tools_phrase(sorted(names))
    return "PASS", "%s (recorded %s)" % ("other tools: " + ",".join(names) if names else "text", _tools_phrase(recorded))
