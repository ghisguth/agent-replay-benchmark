"""--reference: one answer against another (identical, other arguments, other order, text, intent), the verdicts that
follow, and the agreement numbers. Canned results, no server.

    python3 -m unittest tests.test_reference
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from bench import reference as ref  # noqa: E402


def call(name, **args):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def res(calls=(), content="", verdict="PASS", reason="", tokens=10, kind="turn"):
    return {"id": "x", "kind": kind, "content": content, "tool_calls": list(calls), "eval_result": verdict,
            "fail_reason": reason, "tokens": tokens}


class Difference(unittest.TestCase):
    def d(self, a, b, match="calls"):
        return ref.difference(a, b, match)

    def test_identical(self):
        a = res([call("bash", command="ls"), call("read", path="a")])
        self.assertIsNone(self.d(a, dict(a)))

    def test_same_tools_other_args(self):
        why = self.d(res([call("bash", command="ls")]), res([call("bash", command="pwd")]))
        self.assertEqual(why, "call 1 bash: command differs")

    def test_key_order_and_whitespace_of_the_json_do_not_count(self):
        a = res([{"function": {"name": "edit", "arguments": '{"path": "a", "x": 1}'}}])
        b = res([{"function": {"name": "edit", "arguments": '{"x":1,"path":"a"}'}}])
        self.assertIsNone(self.d(a, b))

    def test_other_order(self):
        a = res([call("read", path="a"), call("bash", command="ls")])
        b = res([call("bash", command="ls"), call("read", path="a")])
        self.assertEqual(self.d(a, b), "call 1: read, reference bash")

    def test_extra_call(self):
        a = res([call("read", path="a"), call("read", path="b"), call("read", path="c")])
        b = res([call("read", path="a"), call("read", path="b")])
        self.assertEqual(self.d(a, b), "3 calls vs 2")

    def test_text_same_different_and_whitespace(self):
        self.assertIsNone(self.d(res(content="done"), res(content="done")))
        self.assertIsNone(self.d(res(content="a  b\n c"), res(content="a b c")))
        self.assertEqual(self.d(res(content="done"), res(content="finished")), "text differs")
        self.assertEqual(self.d(res(content="a  b"), res(content="a b"), "exact"), "text differs")
        self.assertIsNone(self.d(res(content=" a b\n"), res(content="a b"), "exact"))

    def test_text_against_a_call(self):
        self.assertEqual(self.d(res(content="done"), res([call("bash", command="ls")])), "reference: bash, this: text")
        self.assertEqual(self.d(res([call("bash", command="ls")]), res(content="done")), "reference: text, this: bash")

    def test_intent_counts_only_for_exact(self):
        a, b = res([call("bash", command="ls", i="list files")]), res([call("bash", command="ls", i="look around")])
        self.assertIsNone(self.d(a, b, "calls"))
        self.assertEqual(self.d(a, b, "exact"), "call 1 bash: i differs")

    def test_exact_compares_the_text_before_the_calls(self):
        a = res([call("bash", command="ls")], content="Let me look.")
        b = res([call("bash", command="ls")], content="I will look.")
        self.assertIsNone(self.d(a, b, "calls"))
        self.assertEqual(self.d(a, b, "exact"), "text before the calls differs")


class Targets(unittest.TestCase):
    def d(self, a, b):
        return ref.targets_difference(a, b)

    def test_read_line_ranges_and_whitespace_are_ignored(self):
        a, b = res([call("read", path="a.cs:10-20")]), res([call("read", path=" a.cs ")])
        self.assertIsNone(self.d(a, b))
        self.assertIsNone(self.d(res([call("edit", path="a.cs:5", old="x")]), res([call("edit", path="a.cs", old="y")])))

    def test_other_file(self):
        self.assertEqual(self.d(res([call("read", path="a.cs"), call("read", path="a.cs")]),
                                res([call("read", path="a.cs"), call("read", path="b.cs")])),
                         "call 2 read: other file (a.cs vs b.cs)")

    def test_bash_program_sets(self):
        same = lambda x, y: self.assertIsNone(self.d(res([call("bash", command=x)]), res([call("bash", command=y)])))
        same("ls -la src | grep cs", "grep -r x . | ls")
        same("cd a && make; make", "cd b || make")
        why = self.d(res([call("bash", command="grep x . | ls")]), res([call("bash", command="cat f")]))
        self.assertEqual(why, "call 1 bash: other programs (grep, ls vs cat)")

    def test_path_tools_and_other_tools(self):
        self.assertIsNone(self.d(res([call("grep", pattern="a", path="src")]), res([call("grep", pattern="b", path="src")])))
        self.assertIsNone(self.d(res([call("grep", pattern="a")]), res([call("grep", pattern="b")])))
        self.assertEqual(self.d(res([call("find", path="a")]), res([call("find", path="b")])), "call 1 find: other path (a vs b)")
        self.assertIsNone(self.d(res([call("todo", ops=[1])]), res([call("todo", ops=[2])])))

    def test_extra_call_order_and_text(self):
        a, b = res([call("read", path="a"), call("read", path="b")]), res([call("read", path="a")])
        self.assertEqual(self.d(a, b), "2 calls vs 1")
        self.assertEqual(self.d(res([call("read", path="a"), call("bash", command="ls")]),
                                res([call("bash", command="ls"), call("read", path="a")])), "call 1: read, reference bash")
        self.assertIsNone(self.d(res(content="a  b"), res(content="a b")))
        self.assertEqual(self.d(res(content="a"), res(content="b")), "text differs")
        self.assertEqual(self.d(res(content="a"), res([call("read", path="a")])), "reference: read, this: text")

    def test_judge_wording_and_verdicts(self):
        j = lambda a, b: ref.judge(a, b, "targets", "REF")
        out = j(res([call("bash", command="ls x")]), res([call("bash", command="ls y")]))
        self.assertEqual((out["eval_result"], out["fail_reason"].startswith("same targets")), ("PASS", True))
        self.assertTrue(out["ref"]["targets"] and not out["ref"]["calls"])
        out = j(res([call("read", path="a")]), res([call("read", path="b")]))
        self.assertEqual((out["eval_result"], out["fail_reason"]), ("WARN", "call 1 read: other file (a vs b)"))

    def test_the_cli_choice_is_between_step_and_calls(self):
        self.assertEqual(ref.MATCHES, ("step", "targets", "calls", "exact"))
        import replay_bench
        with self.assertRaises(SystemExit):
            replay_bench.main(["--state", "x", "--reference", "y", "--match", "bogus"])


class PerTool(unittest.TestCase):
    def test_strings(self):
        pairs = ([("read", "a", "a")] * 8 + [("read", "a", "b")] * 4 + [("bash", "ls", "ls")] * 8
                 + [("bash", "ls", "cat")] * 19 + [("write", "a", "b")] * 13)
        judged = [ref.judge(res([call(n, **({"command": x} if n == "bash" else {"path": x}))]),
                            res([call(n, **({"command": y} if n == "bash" else {"path": y}))]), "targets", "R")
                  for n, x, y in pairs]
        judged.append(ref.judge(res(content="done"), res(content="done"), "targets", "R"))
        st = ref.stats(judged)
        self.assertEqual(ref.by_tool_text(st["targets_by_tool"]), "bash 8/27 · write 0/13 · read 8/12 · <text> 1/1")
        self.assertEqual(ref.by_tool_text(st["calls_by_tool"]), "bash 8/27 · write 0/13 · read 8/12 · <text> 1/1")
        self.assertEqual(ref.by_tool_text([]), ref.F.DASH)
        many = [("t%d" % i, 1, 2) for i in range(8)]
        self.assertTrue(ref.by_tool_text(many).endswith("+2 more"))
        self.assertIn("same targets", ref.line(st))


class Judge(unittest.TestCase):
    def j(self, a, b, match="calls"):
        return ref.judge(a, b, match, "REF")

    def test_same_calls_pass(self):
        a = res([call("bash", command="ls")], reason="same as recorded: bash")
        out = self.j(a, dict(a))
        self.assertEqual(out["eval_result"], "PASS")
        self.assertTrue(out["ref"]["calls"] and out["ref"]["step"] and out["ref"]["tools"])
        self.assertEqual(out["ref"]["sim"], 1.0)
        self.assertEqual(a["eval_result"], "PASS")  # a copy: the stored result is not changed

    def test_other_args_warn_with_the_difference(self):
        out = self.j(res([call("bash", command="ls")]), res([call("bash", command="pwd")]))
        self.assertEqual((out["eval_result"], out["fail_reason"]), ("WARN", "call 1 bash: command differs"))
        self.assertTrue(out["ref"]["tools"] and not out["ref"]["calls"])
        self.assertTrue(0 < out["ref"]["sim"] < 1)

    def test_step_match_compares_the_first_action_category(self):
        a, b = res([call("read", path="a")]), res([call("grep", pattern="x")])
        self.assertEqual(self.j(a, b, "step")["eval_result"], "PASS")  # both look
        out = self.j(res([call("write", path="/src/a.py", content="")]), res([call("read", path="a")]), "step")
        self.assertEqual((out["eval_result"], out["fail_reason"]), ("WARN", "reference: look, this: change"))

    def test_schema_notes_are_kept_and_recording_notes_dropped(self):
        a = res([call("read", path="a")], verdict="WARN",
                reason="off pattern: look (reference models: change); schema: read missing required 'path'")
        out = self.j(a, dict(a, eval_result="PASS"), "step")
        self.assertEqual((out["eval_result"], out["fail_reason"]), ("WARN", "schema: read missing required 'path'"))

    def test_a_fail_stays_a_fail(self):
        a = res(verdict="FAIL", reason="Empty response")
        out = self.j(a, res([call("bash", command="ls")]))
        self.assertEqual((out["eval_result"], out["fail_reason"]), ("FAIL", "Empty response"))
        self.assertFalse(out["ref"]["step"])

    def test_no_usable_reference_answer_keeps_the_normal_verdict(self):
        a = res([call("bash", command="ls")], reason="same as recorded: bash")
        for b in (None, res(verdict="FAIL"), res(verdict="SKIP"), dict(res(), error="boom", eval_result="FAIL")):
            out = self.j(a, b)
            self.assertEqual(out["eval_result"], "PASS")
            self.assertTrue(out["fail_reason"].endswith("(no reference answer)"))
            self.assertFalse(out["ref"]["ok"])

    def test_compaction_and_skip_are_left_alone(self):
        c = res(content="summary", kind="compaction")
        self.assertNotIn("ref", self.j(c, res(content="other", kind="compaction")))
        s = res(verdict="SKIP")
        self.assertNotIn("ref", self.j(s, res()))


class Stats(unittest.TestCase):
    def test_numbers(self):
        r1 = res([call("bash", command="ls")], tokens=10)
        r2 = res([call("read", path="a")], tokens=10)
        r3 = res(content="done", tokens=10)
        r4 = res([call("bash", command="ls")], tokens=10)
        refs = [r1, r2, r3, r4, res([call("bash", command="x")]), res(verdict="FAIL")]
        runs = [r1, res([call("read", path="b")], tokens=20), res(content="finished", tokens=20),
                res([call("grep", pattern="x")], tokens=20),
                res(verdict="FAIL", reason="same call bash repeated 3 times", tokens=5), res(content="x")]
        judged = [ref.judge(a, b, "calls", "REF") for a, b in zip(runs, refs)]
        st = ref.stats(judged)
        self.assertEqual((st["turns"], st["missing"]), (5, 1))
        self.assertEqual(st["step"], 80.0)     # bash, read, text and grep-for-bash are all the same kind of step
        self.assertEqual(st["tools"], 60.0)    # bash, read, text
        self.assertEqual(st["calls"], 20.0)
        self.assertEqual(st["text"], 0.0)      # the one text turn differs
        self.assertAlmostEqual(st["sim"], (1.0 + judged[1]["ref"]["sim"]) / 2)
        self.assertAlmostEqual(st["tokens"], (10 + 20 + 20 + 20 + 5) / 50)
        self.assertEqual((st["broken"], st["loops"]), (1, 1))
        self.assertIn("vs reference REF:", ref.line(st))
        self.assertIsNone(ref.stats([r1]))

    def test_self_comparison_is_complete(self):
        rs = [res([call("bash", command="ls")]), res(content="done"), res([call("read", path="a")])]
        st = ref.stats([ref.judge(r, r, "exact") for r in rs])
        self.assertEqual([st[k] for k in ("step", "tools", "calls", "text", "sim", "tokens")], [100.0, 100.0, 100.0, 100.0, 1.0, 1.0])
        self.assertEqual((st["broken"], st["loops"]), (0, 0))


class Programs(unittest.TestCase):
    def test_cases(self):
        P = ref.programs
        self.assertEqual(P("grep -n 'ComputeSlug\\|class' f | head"), ("grep", "head"))
        self.assertEqual(P('echo "a | b; c" && ls'), ("echo", "ls"))
        self.assertEqual(P("cat > f <<'EOF'\nfoo | bar\nvar x = 1;\nEOF"), ("cat",))
        self.assertEqual(P("cat <<-EOF > f\n  x\n  EOF\nls"), ("cat", "ls"))
        self.assertEqual(P("cat <<EOF\nnever closed\nrm -rf x"), ("cat",))
        self.assertEqual(P("cd x && make test | tail -5"), ("cd", "make", "tail"))
        self.assertEqual(P("FOO=1 ./run.sh"), ("run.sh",))
        self.assertEqual(P("sudo env A=1 ./.venv/bin/python3 x.py"), ("python3",))
        self.assertEqual(P("ls\ncd y\n(cd z; pwd) & { echo hi; }"), ("cd", "echo", "ls", "pwd"))
        self.assertEqual(P(None), ())

    def test_unbalanced_quotes_do_not_crash(self):
        self.assertIn("echo", P2("echo 'oops | grep x; ls"))
        self.assertEqual(P2("echo 'a | b' \"c; d\" 'oops | ls"), ("echo", "ls"))
        self.assertEqual(P2('cd a && echo "x'), ("cd", "echo"))

    def test_warn_text_uses_fixed_sets(self):
        why = ref.targets_difference(res([call("bash", command="grep -n 'a\\|b' f | head")]),
                                     res([call("bash", command="cat > f <<'EOF'\nx; y\nEOF")]))
        self.assertEqual(why, "call 1 bash: other programs (grep, head vs cat)")


P2 = ref.programs


if __name__ == "__main__":
    unittest.main()
