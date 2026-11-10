"""Offline tests for replay grading (bench/cases/grading.py): one small synthetic case per rule."""

import contextlib
import io
import json
import os
import random
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_checks  # noqa: E402
import isolation  # noqa: E402,F401

from bench import results, state  # noqa: E402
from bench.cases import (  # noqa: E402
    grading,
    replay,  # noqa: E402
)

SCHEMA_BASH = {"type": "object", "additionalProperties": False, "required": ["command", "i"],
               "properties": {"i": {"type": "string"}, "command": {"type": "string"}, "timeout": {"type": "number"},
                              "skip": {"type": ["number", "null"]}}}
SCHEMA_TASK = {"type": "object", "additionalProperties": False, "required": ["tasks", "i"],
               "properties": {"i": {"type": "string"},
                              "tasks": {"type": "array", "items": {"type": "object", "required": ["id"],
                                                                   "properties": {"id": {"type": "string"}}}}}}
SCHEMA_PATH = {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}
TOOLS = [{"type": "function", "function": {"name": n, "description": n, "parameters": s}}
         for n, s in (("bash", SCHEMA_BASH), ("task", SCHEMA_TASK), ("edit", SCHEMA_PATH),
                      ("read", SCHEMA_PATH), ("write", SCHEMA_PATH))]

HEADINGS = ["Goal", "Constraints & Preferences", "Progress", "Key Decisions", "Critical Context", "Next Steps"]
CHECKS = {
    "compaction": {"headings": HEADINGS, "min_headings": 4},
    "reports": [{"stream_prefix": "Fals", "detect": r"\*\*Experiment",
                 "require": [{"name": "Experiment line", "any": [r"\*\*Experiment[^\n]*:\*\*"]},
                             {"name": "GO / NO-GO verdict", "any": [r"\bGO\b"]},
                             {"name": "NO-GO class", "any": [r"(?<!NO-)\bGO\b", r"NO-GO class:?\*{0,2}:?\s*\w+"]}]}],
}
REPORT = "**Experiment:** *I did x; I expected y; got z.*\n\n**Falsifier verdict:** GO"


def prose(n, seed=1):
    """n words of varied text (does not compress like a loop)."""
    rnd = random.Random(seed)
    return " ".join("".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rnd.randint(3, 9))) for _ in range(n))


def summary(headings):
    return "\n".join("## %s\n%s" % (h, prose(60, i)) for i, h in enumerate(headings))


BODY = summary(HEADINGS)


def call(name, args=None, raw=None):
    return {"id": "c_" + name, "type": "function",
            "function": {"name": name, "arguments": raw if raw is not None else json.dumps(args if args is not None else {})}}


def bash(cmd="ls", **extra):
    return call("bash", dict({"command": cmd, "i": "look"}, **extra))


def entry(label, kind, recorded_calls=(), content="ok"):
    stream = label.split("/")[0]
    return {"id": label, "stream": stream, "label": label, "kind": kind, "turn": 1, "epoch": 0, "messages": [0],
            "tools": TOOLS, "prompt_tokens": 100, "params": {"max_completion_tokens": 1000},
            "recorded_answer": {"content": content, "tool_calls": list(recorded_calls) or None}}


def make_case(label="main/e0", kind="turn", recorded=("bash",), checks=CHECKS, content="ok"):
    e = entry(label, kind, [call(n) for n in recorded], content)
    fx = {"turns": [e], "messages": [{}], "checks": checks}
    return replay.make_case(fx, 1, e)


def grade(case, content="", calls=None, **extra):
    m = {"content": content, "text": content, "tool_calls": calls or []}
    m.update(extra)
    return grading.grade(case, m), m


class Defects(unittest.TestCase):
    """WARNs for plainly wrong path arguments and file contents (generic, no sidecar needed)."""
    def setUp(self):
        self.case = make_case(recorded=("write",))

    def reason(self, name, **args):
        (v, r), _ = grade(self.case, "", [call(name, args)])
        return v, r

    def test_directory_and_glob_paths(self):
        self.assertEqual(self.reason("write", path="src/", content="x"), ("WARN", "write to a directory"))
        self.assertEqual(self.reason("edit", path="src/"), ("WARN", "edit to a directory"))
        self.assertEqual(self.reason("read", path="src/*.py"), ("WARN", "glob in path (read)"))
        self.assertEqual(self.reason("write", path="a?.txt", content="x"), ("WARN", "glob in path (write)"))
        self.assertEqual(self.reason("read", path="src/")[0], "PASS")   # reading a directory lists it
        self.assertEqual(self.reason("read", path="a.py:58-60")[0], "PASS")   # omp line-range suffix
        self.assertNotIn("glob", self.reason("bash", command="ls *")[1])   # only tools with a path property

    def test_write_content_by_extension(self):
        self.assertEqual(self.reason("write", path="a.json", content='{"a": 1}')[0], "PASS")
        self.assertEqual(self.reason("write", path="a.json", content="{a: 1}"), ("WARN", "write a.json: not JSON"))
        self.assertEqual(self.reason("write", path="d/a.py", content="x = 1\n")[0], "PASS")
        self.assertEqual(self.reason("write", path="d/a.py", content="x = 1\ndef f(:\n"),
                         ("WARN", "write a.py: SyntaxError line 2"))
        self.assertEqual(self.reason("write", path="X.csproj", content="\ufeff \n<Project/>")[0], "PASS")
        self.assertEqual(self.reason("write", path="X.csproj", content="// c\n<Project/>"),
                         ("WARN", "write X.csproj: not XML"))
        self.assertEqual(self.reason("write", path="a.yml", content="{{ not yaml")[0], "PASS")

    def test_reasons_combine(self):
        (v, r), _ = grade(self.case, "", [call("write", {"path": "d/", "content": "x"}),
                                         call("write", {"path": "a.json", "content": "no"})])
        self.assertEqual((v, r), ("WARN", "write to a directory; write a.json: not JSON"))


class Fail(unittest.TestCase):
    def setUp(self):
        self.case = make_case()

    def fails(self, *args, **kw):
        (v, r), _ = grade(self.case, *args, **kw)
        self.assertEqual(v, "FAIL", r)
        return r

    def test_empty_and_reasoning_only(self):
        self.assertIn("Empty", self.fails("  "))
        self.assertIn("only reasoning", self.fails("", reasoning_chars=5000))

    def test_cut_at_token_cap_says_what_was_cut(self):
        u = {"completion_tokens": 1000}
        self.assertEqual(self.fails("", finish_reason="length", usage=u, reasoning_chars=9000),
                         "cut at 1000 tokens (thinking)")
        self.assertEqual(self.fails("partial text", finish_reason="length", usage=u), "cut at 1000 tokens (text)")
        self.assertEqual(self.fails("", [call("bash", raw='{"command": "ls')], finish_reason="length", usage=u),
                         "cut at 1000 tokens (a tool call)")
        self.assertIn("cut at 1000 tokens", self.fails("", [bash()], finish_reason="length"))  # cap from the request

    def test_request_error(self):
        self.assertEqual(self.fails("", error="timeout after 1800s"), "timeout after 1800s")

    def test_bad_json_not_object_unknown_tool(self):
        self.assertIn("not valid JSON", self.fails("", [call("bash", raw='{"x": ')]))
        self.assertIn("not valid JSON", self.fails("", [call("bash", raw="")]))
        self.assertIn("JSON object", self.fails("", [call("bash", raw="[1]")]))
        self.assertIn("rm_rf", self.fails("", [call("rm_rf", {})]))

    def test_tool_call_written_as_text(self):
        for text in ('<tool_call>{"name": "bash"}</tool_call>', "<function=bash><parameter=command>ls", "<|im_start|>x",
                     'I will run {"name": "bash", "arguments": {"command": "ls"}}'):
            self.assertEqual(self.fails(text), "tool call written as text")
        (v, _), _ = grade(self.case, "<tool_call> is how calls look", [bash()])  # parsed calls present: not this rule
        self.assertEqual(v, "PASS")

    def test_same_call_three_times(self):
        self.assertIn("repeated 3 times", self.fails("", [bash(), bash(), bash()]))
        (v, _), _ = grade(self.case, "", [bash("a"), bash("b"), bash("c")])
        self.assertEqual(v, "PASS")

    def test_repeated_text(self):
        self.assertIn("mostly repeated", self.fails("All work and no play makes Jack a dull boy. " * 200))
        lines = "\n".join(["checking the same file again"] * 60 + [prose(8, i) for i in range(40)])
        self.assertIn("repeats one line", self.fails(lines))

    def test_compaction(self):
        case = make_case("main/c0", "compaction", (), content=BODY)
        (v, r), _ = grade(case, "short")
        self.assertEqual((v, "too short" in r), ("FAIL", True))
        (v, r), _ = grade(case, BODY, [bash()])
        self.assertEqual((v, r), ("FAIL", "tool calls in a compaction: bash"))
        three = summary(HEADINGS[:3])
        (v, r), _ = grade(case, three)
        self.assertEqual(v, "FAIL")
        self.assertIn("lacks 3 of 6 headings", r)


class Warn(unittest.TestCase):
    def warns(self, case, *args, **kw):
        (v, r), _ = grade(case, *args, **kw)
        self.assertEqual(v, "WARN", r)
        return r

    def test_schema_problems_are_named(self):
        case = make_case()
        self.assertEqual(self.warns(case, "", [call("bash", {"command": "ls"})]), "schema: bash missing required 'i'")
        self.assertIn("unknown key 'description'", self.warns(case, "", [bash(description="x")]))
        self.assertIn("timeout: string, expected number", self.warns(case, "", [bash(timeout="5")]))
        self.assertIn("tasks: JSON string, expected array", self.warns(case, "", [call("task", {"tasks": "[]", "i": "x"})]))
        self.assertIn("tasks[0]: missing required 'id'",
                      self.warns(case, "", [call("task", {"tasks": [{}], "i": "x"})]))
        (v, _), _ = grade(case, "", [bash(skip=None)])  # null optional: omp strips it
        self.assertEqual(v, "PASS")

    def test_leaked_tool_call_markup(self):
        case = make_case()
        leak = call("bash", None, raw=json.dumps({
            "command": "cat tools/x.sh; echo ===", "i": "look",
            "command\nbash tools/f.sh\n</parameter": "<parameter=i>\nRunning step 0"}))
        self.assertEqual(self.warns(case, "", [leak]), "markup: tool-call markup leaked into bash arguments (parser)")
        xml = call("write", {"path": "a.txt", "content": '<config><parameter name="a"/></config>'})
        self.assertEqual(grade(make_case(), "", [xml])[0][0], "PASS")

    def test_same_call_twice(self):
        self.assertEqual(self.warns(make_case(), "", [bash(), bash()]), "same call twice: bash")

    def test_text_where_the_recording_called_a_tool(self):
        self.assertEqual(self.warns(make_case(), "I am done."), "stopped: text where the agent continued (recorded bash)")
        sub = make_case("Fals1", "turn", ("bash",))
        self.assertEqual(self.warns(sub, REPORT), "reported early (recorded bash)")
        self.assertEqual(self.warns(sub, "Looking at it."), "text where the recording called bash")

    def test_final_report_markers(self):
        case = make_case("Fals1", "turn", (), content=REPORT)
        self.assertEqual(grade(case, REPORT)[0][0], "PASS")
        self.assertEqual(self.warns(case, "Everything is fine."),
                         "final report lacks: Experiment line, GO / NO-GO verdict, NO-GO class")
        self.assertEqual(self.warns(case, "**Experiment:** *x*\n**Falsifier verdict:** NO-GO"),
                         "final report lacks: NO-GO class")
        self.assertEqual(grade(case, "**Experiment:** *x*\n> **NO-GO class:** work")[0][0], "PASS")

    def test_compaction_problems(self):
        case = make_case("main/c0", "compaction", (), content=BODY)
        two = summary(HEADINGS[:4] + HEADINGS[5:])
        self.assertEqual(self.warns(case, two), "compaction lacks heading Critical Context")
        self.assertIn("text before '## Goal'", self.warns(case, "Here is the summary:\n" + BODY))
        self.assertIn("x the recorded length", self.warns(case, summary(HEADINGS * 4)))
        (v, _), _ = grade(case, BODY)
        self.assertEqual(v, "PASS")


class Pass(unittest.TestCase):
    def test_reasons_compare_with_the_recording(self):
        case = make_case(recorded=("edit",))
        (v, r), m = grade(case, "", [call("edit", {})])
        self.assertEqual((v, r, m["matches_recorded"]), ("PASS", "same as recorded: edit", True))
        (v, r), m = grade(case, "", [bash()])
        self.assertEqual((v, r, m["matches_recorded"], m["tool_names"]), ("PASS", "other tools: bash (recorded edit)", False, ["bash"]))
        case = make_case(recorded=())
        (v, r), m = grade(case, "Waiting on the background job.")
        self.assertEqual((v, r, m["matches_recorded"]), ("PASS", "same as recorded: text", True))
        (v, r), _ = grade(case, "", [bash()])
        self.assertEqual((v, r), ("PASS", "other tools: bash (recorded text)"))

    def test_without_sidecar_only_generic_checks(self):
        case = make_case("Fals1", "turn", (), checks={})
        self.assertEqual(grade(case, "no markers needed")[0][0], "PASS")
        comp = make_case("main/c0", "compaction", (), checks={}, content="x" * 2000)
        self.assertEqual(grade(comp, prose(300))[0][0], "PASS")


class Sidecar(unittest.TestCase):
    def test_path_and_loading(self):
        self.assertEqual(replay.checks_path("fixtures/conv-3.json.gz"), "fixtures/conv-3.checks.json")
        self.assertEqual(replay.checks_path("/a/b/x.json"), "/a/b/x.checks.json")
        with tempfile.TemporaryDirectory() as d:
            fx = {"turns": [], "messages": []}
            path = os.path.join(d, "t.json")
            with open(path, "w") as f:
                json.dump(fx, f)
            self.assertNotIn("checks", replay.load_fixture(path))  # absent: generic checks only
            with open(os.path.join(d, "t.checks.json"), "w") as f:
                json.dump(CHECKS, f)
            self.assertEqual(replay.load_fixture(path)["checks"], CHECKS)

    def test_shipped_sidecar_is_valid(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures")
        checks = replay.load_checks(os.path.join(root, "conv-3.json.gz"))
        self.assertEqual(checks["compaction"]["headings"], HEADINGS)
        self.assertTrue(checks["reports"])
        rule = [r for r in checks["reports"][0]["require"] if r["name"] == "NO-GO class"][0]
        for text in ("**NO-GO class:** `tests`", "**NO-GO class:** tests", "NO-GO class: *work*"):
            self.assertTrue(any(re.search(rx, text) for rx in rule["any"]), text)


class Judge(unittest.TestCase):
    def test_final_verdict_still_graded(self):
        e = entry("main/e0", "turn", [])
        e["judge_final"] = True
        fx = {"turns": [e], "messages": [{}], "judge": {"expect": {"status": "ACCEPT", "fdcIds": [42]}}}
        case = replay.make_case(fx, 1, e)
        self.assertEqual(grade(case, "STATUS: ACCEPT\nID: 42\nDESCRIPTION: x")[0][0], "PASS")
        self.assertEqual(grade(case, "STATUS: ACCEPT\nID: 43\nDESCRIPTION: x")[0][0], "FAIL")


INTENT = dict(CHECKS, intent_field="i", intent_max=200)


class Intent(unittest.TestCase):
    def setUp(self):
        self.case = make_case(checks=INTENT)

    def test_missing_or_extra_i_is_harmless(self):
        (v, r), _ = grade(self.case, "", [call("bash", {"command": "ls"})])
        self.assertEqual((v, r), ("PASS", "same as recorded: bash"))
        (v, _), _ = grade(self.case, "", [call("edit", {"i": "why"})])   # edit's schema has no i
        self.assertEqual(v, "PASS")

    def test_over_200_chars_is_warn(self):
        (v, r), _ = grade(self.case, "", [bash(i="x" * 200)])
        self.assertEqual(v, "PASS")
        (v, r), _ = grade(self.case, "", [call("bash", {"command": "ls", "i": "x" * 201})])
        self.assertEqual((v, r), ("WARN", "intent i over 200 chars: tool not run"))

    def test_without_sidecar_keys_nothing_is_stripped(self):
        (v, r), _ = grade(make_case(), "", [call("bash", {"command": "ls"})])
        self.assertEqual((v, r), ("WARN", "schema: bash missing required 'i'"))

    def test_unknown_key_wording(self):
        (v, r), _ = grade(self.case, "", [call("bash", {"command": "ls", "command2": "x"})])
        self.assertEqual((v, r), ("WARN", "schema: bash unknown key 'command2' (omp drops it)"))


class Category(unittest.TestCase):
    def test_tool_category(self):
        cat = lambda name, **a: grading.tool_category([call(name, a)])
        for name in ("read", "grep", "find", "eval", "lsp", "glob", "ls", "bash"):
            self.assertEqual(cat(name, path="src/a.py"), "look")
        self.assertEqual(cat("read", path="proc://bg_6"), "wait")
        self.assertEqual(cat("wait"), "wait")
        self.assertEqual((cat("write"), cat("edit")), ("change", "change"))
        self.assertEqual((cat("write", path="/tmp/p1213f2/check.py"), cat("edit", path="/tmp/x/a.cs")), ("look", "look"))
        self.assertEqual(cat("write", path="tmp/a.py"), "change")   # relative: in the repo
        self.assertEqual((cat("task"), cat("todo"), cat("browser")), ("delegate", "plan", "browser"))
        self.assertEqual(grading.tool_category([]), "<text>")
        self.assertEqual(grading.tool_category([call("read", {}), call("write", {})]), "look")   # the first call counts


class BashCategory(unittest.TestCase):
    def cat(self, command):
        return grading.tool_category([call("bash", {"command": command})])

    def test_changes(self):
        for c in ("dotnet new console -n tools/X; dotnet new xunit -n T", "cd /w && dotnet sln add a.csproj",
                  "dotnet add pkg", "mkdir -p x", "ls && touch a", "rm -f a", "git add -A", "sed -i 's/a/b/' f",
                  "cat > x.cs <<'EOF'\nhi\nEOF", "cat <<EOF > x.cs\nhi\nEOF", "echo foo >> file",
                  "printf 'a b' > out.txt", "ls | tee out.txt", "git checkout -- a", "mkdir /tmp/a src",
                  "cd /tmp/x && mv a /Users/me/repo/b", "cd /tmp/x && cd /w && mkdir y"):
            self.assertEqual(self.cat(c), "change", c)

    def test_looks(self):
        for c in ("python3 - <<'EOF'\nif a > b:\n    print(1)\nEOF", "dotnet build 2>&1", "ls 2>/dev/null",
                  "ls >/dev/null", "grep x > /dev/null", "dotnet test", "dotnet run", "git log | head",
                  "bash tools/f428-base-traces.sh a", "python3 tools/f428.py found x", "grep 'a>b' f",
                  "dotnet run -- census a.jsonl > out.tsv 2> out.summary", "ls > listing.txt",
                  "cat > /tmp/x.jsonl <<'EOF'\n{}\nEOF", "echo hi >> /tmp/log", "ls | tee /tmp/o.txt",
                  "cat a > $TMPDIR/b", "mkdir -p /tmp/a /tmp/b", "rm -rf /tmp/x", "cp /tmp/a /tmp/b",
                  "cd /tmp/x && mkdir synthetic && cat > synthetic/a.jsonl <<EOF\n{}\nEOF"):
            self.assertEqual(self.cat(c), "look", c)


class AcceptedSteps(unittest.TestCase):
    def case(self, accepted):
        return make_case(checks=dict(INTENT, accepted={"main/0/turn/1": accepted, "other": ["change"]}))

    def test_matches_accepted_categories(self):
        case = self.case(["look"])
        self.assertEqual(case["id"], "R001_main_e0_t001")
        (v, _), m = grade(case, "", [call("grep", {"pattern": "x", "i": "i"})])   # another look tool is fine
        self.assertIs(m["in_line"], True)
        self.assertNotEqual(v, "WARN")
        (v, r), m = grade(case, "", [call("edit", {})])
        self.assertEqual((v, r, m["in_line"]), ("WARN", "off pattern: change (reference models: look)", False))
        (v, _), m = grade(self.case(["change"]), "", [call("grep", {"pattern": "x", "i": "i"})])   # look always fits
        self.assertIs(m["in_line"], True)
        self.assertNotEqual(v, "WARN")
        (_, _), m = grade(make_case(checks=INTENT), "", [bash()])
        self.assertIsNone(m["in_line"])

    def test_text_where_no_reference_model_wrote_text(self):
        (v, r), m = grade(self.case(["look", "wait"]), "thinking out loud")
        self.assertEqual((v, m["in_line"]), ("WARN", False))
        self.assertIn("stopped: text where the agent continued", r)
        self.assertNotIn("off pattern", r)

    def test_text_accepted(self):
        (v, r), m = grade(self.case(["<text>", "look"]), "waiting on bg_6")
        self.assertEqual((v, r, m["in_line"]), ("PASS", "text (recorded bash)", True))

    def test_summary_text(self):
        rs = [{"eval_result": "PASS", "in_line": True}, {"eval_result": "PASS", "in_line": False},
              {"eval_result": "PASS", "in_line": True, "kind": "compaction"},
              {"eval_result": "PASS"}, {"eval_result": "SKIP", "in_line": True}]
        share = results.in_line_share(rs)
        self.assertEqual(share, (1, 2, 50.0))
        self.assertIsNone(results.in_line_share([{"eval_result": "PASS"}]))

        line = results.counts_line(results.verdict_counts(rs), 80.0, share)
        self.assertTrue(line.endswith("same tools as recorded 80% · in line with reference models 1/2 (50%)"))
        self.assertTrue(line.markdown().endswith("in line with reference models 1/2 (50%)"))
        self.assertNotIn("reference models", results.counts_line(results.verdict_counts(rs), 80.0))


def state_file(d, name, firsts):
    """A state file whose result for each case id calls the given tool (None = text only)."""
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(json.dumps({"type": "run", "schema": 1, "suite": "conv-3", "model": name, "started_at": "2026-11-07T1"}) + "\n")
        for cid, tool in firsts.items():
            m = {"type": "result", "id": cid, "content": "", "tool_calls": [call(tool, {})] if tool else [],
                 "eval_result": "PASS"}
            if tool == "ERR":
                m = {"type": "result", "id": cid, "error": "boom", "eval_result": "FAIL"}
            f.write(json.dumps(m) + "\n")
    return p


class BuildChecks(unittest.TestCase):
    def test_broken_reference_answers_do_not_widen_accepted(self):
        b = build_checks.broken
        self.assertFalse(b({"eval_result": "PASS"}))
        self.assertFalse(b({"eval_result": "WARN", "fail_reason": "off pattern: change (reference models: look)"}))
        self.assertTrue(b({"eval_result": "FAIL", "fail_reason": "cut at 32768 tokens (text)"}))
        self.assertTrue(b({"eval_result": "WARN", "fail_reason": "schema: edit missing required 'new_string'"}))
        self.assertTrue(b({"eval_result": "WARN",
                           "fail_reason": "schema: edit missing required 'new_string'; off pattern: change (reference models: <text>)"}))

    def test_accepted_of_refs_and_recording(self):
        def e(label, tools, turn=1, kind="turn"):
            return {"label": label, "stream": "main", "kind": kind, "turn": turn,
                    "recorded_answer": {"tool_calls": [call(t) for t in tools] or None}}
        fx = {"turns": [e("a", ["bash", "read"], turn=0), e("b", ["edit"], turn=1), e("c", [], turn=2),
                        e("d", ["bash"], turn=3), e("f", ["bash"], turn=4),
                        e("g", [], turn=None, kind="compaction")]}
        ids = [replay.case_id(i, t) for i, t in enumerate(fx["turns"], 1)]
        keys = [grading.entry_key(t) for t in fx["turns"]]
        with tempfile.TemporaryDirectory() as d:
            r1 = state_file(d, "m1", dict(zip(ids, ["grep", "edit", None, "bash", "bash", None])))
            r2 = state_file(d, "m2", dict(zip(ids, ["bash", "write", None, "task", "ERR", None])))
            r3 = state_file(d, "m3", dict(zip(ids, ["bash", "bash", None, "task", "task", None])))
            accepted, refs = build_checks.build(fx, [r1, r2])
            # one run taking a step is not enough: "delegate" (m2 only) and "look" at turn 1 (m2: write) stay out
            self.assertEqual(accepted, {keys[0]: ["look"], keys[1]: ["change"], keys[2]: ["<text>"],
                                        keys[3]: ["look"], keys[4]: ["look"]})
            self.assertEqual(len(refs), 2)
            self.assertIn("m1", refs[0])
            # two runs taking it accepts it (turn d: m2 and m3 delegate; turn f: only m3 delegates, m2 errored)
            accepted3, _ = build_checks.build(fx, [r1, r2, r3])
            self.assertEqual(accepted3[keys[3]], ["delegate", "look"])
            self.assertEqual(accepted3[keys[4]], ["look"])
            self.assertEqual(accepted3[keys[1]], ["change"])
            # the recorded step stays accepted even when no run took it (turn b: recorded edit, runs said look)
            accepted_none, _ = build_checks.build(fx, [r1, r3])
            self.assertEqual(accepted_none[keys[1]], ["change"])
            self.assertEqual(accepted_none[keys[2]], ["<text>"])
            # the same model twice (two runs, or two quants) is one model: it does not make a step accepted
            twice, _ = build_checks.build(fx, [r3, r3])
            self.assertEqual(twice[keys[1]], ["change"])
            self.assertEqual(twice[keys[3]], ["look"])
            q4 = state_file(d, "other.jsonl", dict(zip(ids, ["bash", "bash", None, "task", "task", None])))
            with open(q4) as f:
                lines = f.read().split("\n", 1)
            with open(q4, "w") as f:   # the same model as m3, another quant
                f.write(json.dumps(dict(json.loads(lines[0]), model="org/M3-GGUF:Q4_0")) + "\n" + lines[1])
            r3q = state_file(d, "M3", dict(zip(ids, ["bash", "bash", None, "task", "task", None])))
            self.assertEqual(build_checks.build(fx, [r3q, q4])[0][keys[3]], ["look"])
            # --min-runs 1 keeps the old rule
            accepted1, _ = build_checks.build(fx, [r1, r2], min_runs=1)
            self.assertEqual(accepted1[keys[3]], ["delegate", "look"])


class ModelKey(unittest.TestCase):
    def test_quant_org_and_tags_are_dropped(self):
        k = build_checks.model_key
        self.assertEqual(k("unsloth/Qwen3.8-27B-GGUF:Q8_0"), "qwen3.8-27b")
        self.assertEqual(k("unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"), "qwen3.8-27b")
        self.assertEqual(k("mlx-community/Qwen3.8-27B-4bit"), "qwen3.8-27b")
        self.assertEqual(k("mlx-community/Qwen3.8-27B-GPTQ-4bit"), "qwen3.8-27b")
        self.assertEqual(k("Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16"), "qwen3.8-27b")
        self.assertEqual(k("qwen3.8-27b"), "qwen3.8-27b")
        self.assertEqual(k("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"), "qwen3.8-flash-next")
        self.assertEqual(k("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0"), "qwen3.8-flash-next")
        self.assertEqual(k("unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ1_S"), "qwen3.8-flash-next")
        self.assertEqual(k("unsloth/Muse-Glimmer-30B-GGUF"), "muse-glimmer-30b")
        self.assertEqual(k("mlx-community/Qwen3.6-35B-A3B-4bit"), "qwen3.6-35b-a3b")
        self.assertNotEqual(k("Qwen3.8-27B"), k("Qwen3.8-Flash-Next"))
        self.assertEqual(k(None), "?")


class EntryKeysAndShaTests(unittest.TestCase):
    def test_entry_key_turn_and_compaction(self):
        self.assertEqual(grading.entry_key({"stream": "main", "epoch": 0, "kind": "turn", "turn": 3}), "main/0/turn/3")
        self.assertEqual(grading.entry_key({"stream": "main", "epoch": 0, "kind": "compaction", "turn": None}), "main/0/compaction/-")
        self.assertEqual(grading.entry_key({"stream": "S1", "epoch": 1, "kind": "turn", "turn": 0}), "S1/1/turn/0")

    def test_sha_mismatch_ignores_accepted_steps_with_warning(self):
        with tempfile.TemporaryDirectory() as d:
            fx = {
                "turns": [
                    {"label": "a", "stream": "main", "epoch": 0, "kind": "turn", "turn": 0, "tools": []},
                ],
                "messages": [],
            }
            fx_path = os.path.join(d, "fx.json")
            with open(fx_path, "w", encoding="utf-8") as f:
                json.dump(fx, f)
            checks_path = os.path.join(d, "fx.checks.json")
            with open(checks_path, "w", encoding="utf-8") as f:
                json.dump({
                    "fixture_sha256": "bad_sha_12345678",
                    "accepted": {"main/0/turn/0": ["look"]},
                    "compaction": {"headings": ["Goal"]},
                }, f)

            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                loaded = replay.load_fixture(fx_path)

            self.assertIn("fixture sha256 mismatch", err.getvalue())
            self.assertIn("ignoring accepted steps", err.getvalue())
            # compaction stays, accepted is removed
            self.assertIn("compaction", loaded["checks"])
            self.assertNotIn("accepted", loaded["checks"])

            # Grade case: in_line is None
            case = replay.make_case(loaded, 1, loaded["turns"][0])
            m = {"content": "", "tool_calls": [call("read", {})]}
            grading.grade(case, m)
            self.assertIsNone(m["in_line"])

    def test_build_checks_cli_writes_stable_keys_and_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            fx = {
                "turns": [
                    {"label": "a", "stream": "main", "epoch": 0, "kind": "turn", "turn": 0,
                     "recorded_answer": {"tool_calls": [call("read")]}},
                    {"label": "b", "stream": "main", "epoch": 0, "kind": "turn", "turn": 1,
                     "recorded_answer": {"tool_calls": [call("bash")]}},
                ],
                "messages": [],
            }
            fx_path = os.path.join(d, "fx.json")
            with open(fx_path, "w", encoding="utf-8") as f:
                json.dump(fx, f)
            fx_sha = state.file_sha256(fx_path)

            def write_ref(name, t0, t1):
                p = os.path.join(d, name)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "type": "run", "schema": 1, "suite": "conv-3", "model": name,
                        "fixture": {"file": "fx.json", "sha256": fx_sha},
                    }) + "\n")
                    f.write(json.dumps({"type": "result", "id": "R001_a_t000", "content": "", "eval_result": "PASS",
                        "tool_calls": [call(t0)]}) + "\n")
                    f.write(json.dumps({"type": "result", "id": "R002_b_t001", "content": "", "eval_result": "PASS",
                        "tool_calls": [call(t1)]}) + "\n")
                return p

            r1 = write_ref("ref1.jsonl", "read", "bash")
            r2 = write_ref("ref2.jsonl", "read", "bash")
            checks_path = os.path.join(d, "fx.checks.json")

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = build_checks.main(["--fixture", fx_path, "--checks", checks_path, r1, r2])
            self.assertEqual(rc, 0)
            self.assertIn("accepted steps for 2 turns (compactions excluded) from 2 reference runs of 2 models, a step needs 2 models", out.getvalue())

            with open(checks_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["fixture_sha256"], fx_sha)
            self.assertEqual(data["accepted"], {"main/0/turn/0": ["look"], "main/0/turn/1": ["look"]})
            self.assertEqual(len(data["accepted_refs"]), 2)

            # Test default reference set (mocked glob to r1, r2)
            out2 = io.StringIO()
            with contextlib.redirect_stdout(out2), mock.patch("build_checks.glob.glob", return_value=[r1, r2]):
                rc2 = build_checks.main(["--fixture", fx_path, "--checks", checks_path])
            self.assertEqual(rc2, 0)
            self.assertIn("accepted steps for 2 turns (compactions excluded) from 2 reference runs of 2 models, a step needs 2 models", out2.getvalue())


if __name__ == "__main__":
    unittest.main()
