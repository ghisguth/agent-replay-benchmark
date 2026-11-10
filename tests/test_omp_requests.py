"""Tests for tools/omp_requests.py: synthetic sessions offline; golden + case A checks only where the raw inputs exist.

The golden check needs a snapshot capture (a proxy dump directory) and its omp session; case A and D need their omp
sessions. They are read in place (never copied into the repo) and located only by environment variables:
OMP_GOLDEN_CAPTURE / OMP_GOLDEN_SESSION / OMP_CASE_A_SESSION / OMP_CASE_D_SESSION / OMP_CASE_D_REPO. Unset: skipped.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401
import omp_requests as r  # noqa: E402
import omp_session as o  # noqa: E402

GOLDEN_CAPTURE = os.environ.get("OMP_GOLDEN_CAPTURE", "")
GOLDEN_SESSION = os.environ.get("OMP_GOLDEN_SESSION", "")
CASE_A_SESSION = os.environ.get("OMP_CASE_A_SESSION", "")
CASE_D_SESSION = os.environ.get("OMP_CASE_D_SESSION", "")
# Case D's checkout (read with `git show`, never written): the session ran on a branch from this commit.
CASE_D_REPO = os.environ.get("OMP_CASE_D_REPO", "")
CASE_D_REV = "a1b7a4d03"

T0 = 1_700_000_000_000  # ms
REM = "<system-reminder>\nToday: 2000-01-01; current working directory: '/w'. Do not repeat this information in " \
      "your reply.\n</system-reminder>"


def iso(ms):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class Builder:
    """Append-only synthetic omp session: each entry's parent is the previous entry."""

    def __init__(self):
        self.recs = [{"type": "session", "version": 3, "id": "s", "timestamp": iso(T0), "cwd": "/w"}]
        self.last = None
        self.n = 0

    def add(self, rec, ms):
        self.n += 1
        rec = dict(rec, id=f"e{self.n}", parentId=self.last, timestamp=iso(ms))
        self.recs.append(rec)
        self.last = rec["id"]
        return rec

    def reminder(self, ms, conversation, turn=None, placement="prefix"):
        return self.add({"type": "custom", "customType": "date-cwd-reminder",
                         "data": {"conversation": conversation, "turn": turn or conversation, "reminder": REM,
                                  "placement": placement}}, ms)

    def skill(self, ms, text, attribution="user"):
        return self.add({"type": "custom_message", "customType": "skill-prompt", "content": text, "display": True,
                         "attribution": attribution}, ms)

    def custom(self, ms, ctype, text, attribution="agent"):
        return self.add({"type": "custom_message", "customType": ctype, "content": text, "display": False,
                         "attribution": attribution}, ms)

    def user(self, ms, text, attribution="user"):
        return self.add({"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": text}],
                                                        "attribution": attribution, "timestamp": ms}}, ms)

    def assistant(self, ms, thinking="think", text=None, calls=(), stop="toolUse", prompt=100):
        content = []
        if thinking is not None:
            content.append({"type": "thinking", "thinking": thinking, "thinkingSignature": "reasoning_content"})
        if text is not None:
            content.append({"type": "text", "text": text})
        for cid, name, args in calls:
            content.append({"type": "toolCall", "id": cid, "name": name, "arguments": args})
        return self.add({"type": "message", "message": {
            "role": "assistant", "content": content, "api": "openai-completions", "provider": "P", "model": "M",
            "stopReason": stop, "timestamp": ms, "completedAt": ms + 50,
            "contextSnapshot": {"promptTokens": prompt, "nonMessageTokens": 10, "toolSchemaTokens": 5,
                                "compactionEpoch": sum(1 for x in self.recs if x["type"] == "compaction")}}},
            ms + 50)

    def result(self, ms, cid, name, text, **extra):
        m = {"role": "toolResult", "toolCallId": cid, "toolName": name, "content": [{"type": "text", "text": text}],
             "isError": False, "timestamp": ms}
        m.update(extra)
        return self.add({"type": "message", "message": m}, ms)

    def compaction(self, ms, summary="summary", method="handoff", first_kept=None, tokens_before=100, tokens_after=50):
        return self.add({"type": "compaction", "summary": summary, "method": method,
                         "firstKeptEntryId": first_kept, "tokensBefore": tokens_before, "tokensAfter": tokens_after}, ms)

    def write(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for rec in self.recs:
                f.write(json.dumps(rec) + "\n")


def tool(name, desc="d"):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": {}}}}


SNAP = {
    "format": r.SNAPSHOT_FORMAT, "omp_version": "test", "source": "synthetic", "provider": "P", "model": "M",
    "body_keys": ["model", "messages", "stream", "stream_options", "tools", "max_completion_tokens",
                  "chat_template_kwargs"],
    "params": {"stream": True, "stream_options": {"include_usage": True}, "max_completion_tokens": 32768,
               "chat_template_kwargs": {"preserve_thinking": True, "enable_thinking": True}},
    "system_prompt": {"main": ["SYS one", "SYS two\n- Model: P/M"]},
    "tools": {"main": ["read", "bash"], "helper": ["read", "grep"]},
    "tool_pool": {"main": {"read": tool("read"), "bash": tool("bash"), "grep": tool("grep", "main grep")},
                  "helper": {"read": tool("read"), "grep": tool("grep", "helper grep")}},
    "compat": dict(r.DEFAULT_COMPAT), "notes": [],
}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "s.jsonl")

    def load(self, b):
        b.write(self.path)
        return o.load_session(self.path)


# --------------------------------------------------------------------------------------------- helpers

class HashTests(unittest.TestCase):
    # Values from `Bun.hash.wyhash(s, 0n)` (bun 1.x), which Bun.hash(s) equals.
    VECTORS = {"": "409638ee2bde459", "a": "28d2053309d28531", "abc": "2a4f1d7cb516c72",
               "abcd": "48dfe2b09ab52113", "hello world!!": "f47a273a52237f52", "x" * 17: "7ba3186dc79f5203",
               "y" * 48: "89641857d86ae3fa", "z" * 49: "76aa47dc56d0bc26",
               "é漢字\U0001f600": "e579b697324c689e"}

    def test_wyhash_matches_bun(self):
        for s, want in self.VECTORS.items():
            self.assertEqual(r.bun_hash_hex(s), want, repr(s))

    def test_digest_split_roundtrip(self):
        blocks = ["first block\n\nwith a paragraph", "second", "third block"]
        digest = r.system_prompt_digest(blocks)
        self.assertEqual(r.split_blocks_for_digest("\n\n".join(blocks), digest), blocks)
        self.assertIsNone(r.split_blocks_for_digest("\n\n".join(blocks), "0", max_blocks=2))

    def test_digest_depends_on_block_count(self):
        self.assertNotEqual(r.system_prompt_digest(["a\n\nb"]), r.system_prompt_digest(["a", "b"]))


class JsTests(unittest.TestCase):
    def test_stringify_like_json_stringify(self):
        self.assertEqual(r.js_stringify({"b": 1, "a": [True, None, 1.5, 2.0], "s": "é\n\" "}),
                         '{"b":1,"a":[true,null,1.5,2],"s":"é\\n\\" "}')

    def test_stringify_integer_keys_first(self):
        self.assertEqual(r.js_stringify({"x": 1, "2": 2, "1": 3, "01": 4}), '{"1":3,"2":2,"x":1,"01":4}')

    def test_iso_to_ms(self):
        self.assertEqual(r.iso_to_ms("2026-11-07T03:30:00.113Z"), 1794022200113)
        self.assertEqual(r.iso_to_ms(iso(T0 + 7)), T0 + 7)

    def test_well_formed(self):
        self.assertEqual(r.well_formed("a\ud800b"), "a�b")


class FormatTests(unittest.TestCase):
    def test_blank_runs_and_trailing_space(self):
        self.assertEqual(r.prompt_format("a  \n\nb \n\n"), "a\n\nb")
        self.assertEqual(r.prompt_format("a\n\n\n\nb"), "a\nb")  # a run of 2+ blank lines goes entirely

    def test_tables_compacted_outside_code(self):
        src = "| a  |  b |\n| :-- | --: |\n```\n| keep  |  me |\n```"
        self.assertEqual(r.prompt_format(src), "|a|b|\n|:---|---:|\n```\n| keep  |  me |\n```")

    def test_blank_before_closing_tag_dropped(self):
        self.assertEqual(r.prompt_format("<handoff>\nx\n\n</handoff>"), "<handoff>\nx\n</handoff>")

    def test_handoff_template(self):
        out = r.render_summary_template(r.HANDOFF_SUMMARY_TEMPLATE, "## Goal\nDo it.")
        self.assertTrue(out.startswith("Context replaced. The <handoff>"))
        self.assertTrue(out.endswith("<handoff>\n## Goal\nDo it.\n</handoff>"))

    def test_format_duration(self):
        self.assertEqual(r.format_duration(-10), "0ms")
        self.assertEqual(r.format_duration(0), "0ms")
        self.assertEqual(r.format_duration(500), "500ms")
        self.assertEqual(r.format_duration(5000), "5.0s")
        self.assertEqual(r.format_duration(15000), "15.0s")
        self.assertEqual(r.format_duration(59940), "59.9s")
        self.assertEqual(r.format_duration(59960), "1m")
        self.assertEqual(r.format_duration(90000), "1m30s")
        self.assertEqual(r.format_duration(3600000), "1h")
        self.assertEqual(r.format_duration(3660000), "1h1m")
        self.assertEqual(r.format_duration(86400000), "1d")
        self.assertEqual(r.format_duration(90000000), "1d1h")


# --------------------------------------------------------------------------------------------- rebuild

def main_session(b):
    """Skill prompt turn, two tool turns, a nudge, a final answer."""
    t = T0
    b.add({"type": "model_change", "model": "P/M"}, t)
    b.reminder(t + 5, f"user:{t + 1}:user")
    b.skill(t + 1, "[skill] do the thing")
    b.add({"type": "custom", "customType": "system-prompt-digest", "data": "x"}, t + 900)
    b.assistant(t + 10, "plan", None, [("c1", "read", {"path": "a"}), ("c2", "bash", {"command": "ls", "i": "x"})])
    b.result(t + 100, "c1", "read", "A")
    b.result(t + 101, "c2", "bash", "B")
    b.custom(t + 102, "mid-run-todo-nudge", "<system-reminder>\n2 todo items still open.\n</system-reminder>")
    b.assistant(t + 200, "more", "Looking.", [("c3", "read", {"path": "b"})])
    b.add({"type": "custom", "customType": "tool_execution_start", "data": {}}, t + 290)
    b.result(t + 300, "c3", "read", "line1")
    b.assistant(t + 400, "done", "All done.", (), stop="stop")


class RebuildTests(Base):
    def setUp(self):
        super().setUp()
        b = Builder()
        main_session(b)
        self.sess = self.load(b)

    def build(self, k, stream=None, **kw):
        return r.build_request(stream or self.sess.main, k, SNAP, **kw)

    def test_first_turn(self):
        body = self.build(0)
        self.assertEqual(list(body), SNAP["body_keys"])
        self.assertEqual(body["model"], "M")
        self.assertEqual(body["messages"], [
            {"role": "system", "content": "SYS one\n\nSYS two\n- Model: P/M"},
            {"role": "user", "content": [{"type": "text", "text": REM}, {"type": "text", "text": "[skill] do the thing"}]},
        ])
        self.assertEqual([t["function"]["name"] for t in body["tools"]], ["read", "bash"])
        self.assertEqual(body["max_completion_tokens"], 32768)

    def test_tool_turns_thinking_and_custom(self):
        msgs = self.build(2)["messages"]
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "tool", "tool", "user",
                                                     "assistant", "tool"])
        a = msgs[2]
        self.assertEqual(list(a), ["role", "content", "reasoning_content", "tool_calls"])
        self.assertEqual(a["content"], "")
        self.assertEqual(a["reasoning_content"], "plan")
        self.assertEqual(a["tool_calls"][1], {"id": "c2", "type": "function",
                                              "function": {"name": "bash", "arguments": '{"command":"ls","i":"x"}'}})
        self.assertEqual(msgs[3], {"role": "tool", "content": "A", "tool_call_id": "c1"})
        # custom messages become developer turns, sent as user with text parts on a local backend
        self.assertEqual(msgs[5], {"role": "user", "content": [
            {"type": "text", "text": "<system-reminder>\n2 todo items still open.\n</system-reminder>"}]})
        self.assertEqual(msgs[6]["content"], "Looking.")

    def test_turn_count_and_prefix_property(self):
        prev = None
        for t in self.sess.main.turns:
            msgs = self.build(t.index)["messages"]
            if prev is not None:
                self.assertEqual(msgs[:len(prev)], prev)  # append-only between turns without compaction
            prev = msgs

    def test_snapshot_param_order_kept(self):
        body = self.build(1)
        self.assertEqual(list(body["chat_template_kwargs"]), ["preserve_thinking", "enable_thinking"])


class CleanupTests(Base):
    def test_error_turn_and_dangling_calls(self):
        b = Builder()
        t = T0
        b.reminder(t + 5, f"user:{t + 1}:user")
        b.user(t + 1, "hi")
        b.assistant(t + 10, "x", None, [("c1", "read", {})], stop="error")  # errored with a call
        b.result(t + 20, "c1", "read", "aborted")
        b.assistant(t + 30, "y", "partial", [("c2", "read", {}), ("c3", "bash", {})])
        b.result(t + 40, "c2", "read", "R")  # c3 never answered
        b.assistant(t + 50, "z", "end", (), stop="stop")
        sess = self.load(b)
        body = r.build_request(sess.main, 2, SNAP)
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])
        self.assertEqual([c["id"] for c in body["messages"][2]["tool_calls"]], ["c2"])

    def test_empty_error_turn_skipped(self):
        b = Builder()
        b.reminder(T0 + 5, f"user:{T0 + 1}:user")
        b.user(T0 + 1, "hi")
        b.assistant(T0 + 10, None, None, (), stop="error")
        b.assistant(T0 + 20, "ok", "fine", (), stop="stop")
        sess = self.load(b)
        self.assertEqual(len(sess.main.turns), 2)  # the empty error turn is a transcript turn, not context
        self.assertEqual([m["role"] for m in r.build_request(sess.main, 1, SNAP)["messages"]], ["system", "user"])


class CompactionTests(Base):
    def test_handoff_epoch(self):
        b = Builder()
        t = T0
        b.reminder(t + 5, f"user:{t + 1}:user")
        b.user(t + 1, "task")
        b.assistant(t + 10, "a", None, [("c1", "read", {})])
        b.result(t + 20, "c1", "read", "old")
        kept = b.assistant(t + 30, "b", None, [("c2", "read", {})])
        b.result(t + 40, "c2", "read", "kept")
        comp_ms = t + 50
        b.add({"type": "compaction", "summary": "## Goal\nfinish", "firstKeptEntryId": kept["id"],
               "tokensBefore": 999, "tokensAfter": 99, "method": "handoff"}, comp_ms)
        b.reminder(comp_ms + 1, f"user:{comp_ms}:agent")
        b.assistant(t + 60, "c", None, [("c3", "bash", {})])
        b.result(t + 70, "c3", "bash", "new")
        b.assistant(t + 80, "d", "done", (), stop="stop")
        sess = self.load(b)
        self.assertEqual([tt.compaction_epoch for tt in sess.main.turns], [0, 0, 1, 1])
        msgs = r.build_request(sess.main, 2, SNAP)["messages"]
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "tool"])
        self.assertEqual(msgs[1]["content"][0]["text"], REM)
        self.assertTrue(msgs[1]["content"][1]["text"].endswith("<handoff>\n## Goal\nfinish\n</handoff>"))
        self.assertEqual(msgs[3]["content"], "kept")
        msgs = r.build_request(sess.main, 3, SNAP)["messages"]
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "tool", "assistant", "tool"])
        # pre-compaction turn still sees the full history
        self.assertEqual(len(r.build_request(sess.main, 1, SNAP)["messages"]), 4)

    def test_missing_reminder_record_is_an_error(self):
        b = Builder()
        b.user(T0 + 1, "hi")
        b.assistant(T0 + 10, "x", "y", (), stop="stop")
        sess = self.load(b)
        with self.assertRaises(r.RebuildError):
            r.build_request(sess.main, 0, SNAP)


class ReminderTests(Base):
    def test_control_placement_and_later_prefix(self):
        b = Builder()
        t = T0
        b.reminder(t + 5, f"user:{t + 1}:user")
        b.user(t + 1, "hi")
        a = b.assistant(t + 10, "x", "first", (), stop="stop")
        b.user(t + 100, "again")
        b.reminder(t + 101, f"user:{t + 1}:user", f"user:{t + 100}:user")  # date changed: prefix on the new turn
        b.assistant(t + 110, "y", None, [("c1", "read", {})])
        b.result(t + 120, "c1", "read", "R")
        b.reminder(t + 121, f"user:{t + 1}:user", "toolResult:%d" % (t + 120), placement="control")
        b.assistant(t + 130, "z", "end", (), stop="stop")
        sess = self.load(b)
        msgs = r.build_request(sess.main, 1, SNAP)["messages"]
        self.assertEqual(msgs[3]["content"][0]["text"], REM)
        msgs = r.build_request(sess.main, 2, SNAP)["messages"]
        self.assertEqual(msgs[-1], {"role": "user", "content": REM})  # developer string turn -> user
        self.assertEqual(msgs[-2]["role"], "tool")
        del a

    def test_first_turn_hash(self):
        m = {"role": "user", "content": [{"type": "text", "text": "x"}], "attribution": "user"}
        self.assertEqual(r.first_turn_hash(m), r.bun_hash_hex('["user",[{"type":"text","text":"x"}]]'))


class PruneTests(Base):
    def test_wait_result_before_and_after_prune(self):
        b = Builder()
        t = T0
        b.reminder(t + 5, f"user:{t + 1}:user")
        b.user(t + 1, "go")
        b.assistant(t + 10, "w", None, [("c1", "wait", {})])
        details = {"op": "wait", "jobs": [{"id": "bg_1", "type": "bash", "status": "running", "label": "make all",
                                           "intent": "Building", "durationMs": 5}]}
        b.result(t + 20, "c1", "wait", "[Uneventful result elided]", details=details, useless=True,
                 prunedAt=t + 200)
        b.assistant(t + 30, "x", "waiting", (), stop="stop")
        b.custom(t + 300, "async-result", "<system-notice>\nBackground job bg_1 done\n</system-notice>")
        b.assistant(t + 310, "y", "ok", (), stop="stop")
        sess = self.load(b)
        notes = []
        before = r.build_request(sess.main, 1, SNAP, notes=notes)["messages"]
        self.assertEqual(before[3]["content"], "## Still Running (1)\n\n- `bg_1` [bash] — Building — make all")
        self.assertTrue(any("re-rendered" in n for n in notes))
        after = r.build_request(sess.main, 2, SNAP)["messages"]
        self.assertEqual(after[3]["content"], "[Uneventful result elided]")

    def test_wait_idle_notice_streak_and_reset(self):
        b = Builder()
        t = T0
        b.reminder(t + 5, f"user:{t + 1}:user")
        b.user(t + 1, "go")
        # 1st wait: idle, rung 0 (no notice)
        b.assistant(t + 10, "w1", None, [("c1", "wait", {})])
        details = {"op": "wait", "jobs": [{"id": "bg_1", "type": "bash", "status": "running", "label": "make all",
                                           "intent": "Building", "durationMs": 5}]}
        b.result(t + 5_000, "c1", "wait", "[Uneventful result elided]", details=details, useless=True,
                 prunedAt=t + 500_000)
        # 2nd wait: 10s later (< 60s), idle, rung 1 -> notice appended
        b.assistant(t + 5_050, "w2", None, [("c2", "wait", {})])
        b.result(t + 15_000, "c2", "wait", "[Uneventful result elided]", details=details, useless=True,
                 prunedAt=t + 500_000)
        # Turn right after 2nd wait (req_ms = t + 15_050 < prunedAt)
        b.assistant(t + 15_050, "w3", None, [("c3", "wait", {})])
        # 3rd wait: 80s later (> 60s), idle streak resets -> rung 0 (no notice)
        b.result(t + 95_000, "c3", "wait", "[Uneventful result elided]", details=details, useless=True,
                 prunedAt=t + 500_000)
        b.assistant(t + 95_050, "end", "done", (), stop="stop")
        sess = self.load(b)

        # In request for turn 2 (pos of w3): c1 has no notice, c2 has notice
        msgs = r.build_request(sess.main, 2, SNAP)["messages"]
        c1_msg = next(m for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") == "c1")
        c2_msg = next(m for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") == "c2")
        self.assertEqual(c1_msg["content"], "## Still Running (1)\n\n- `bg_1` [bash] — Building — make all")
        self.assertIn("<system-notice>\nNothing new after 15.0s of back-to-back waits.", c2_msg["content"])
        self.assertTrue(c2_msg["content"].startswith("## Still Running (1)\n\n- `bg_1` [bash] — Building — make all\n<system-notice>"))

        # In request for turn 3 (pos of end): c3 has reset streak to rung 0 (no notice)
        msgs3 = r.build_request(sess.main, 3, SNAP)["messages"]
        c3_msg = next(m for m in msgs3 if m.get("role") == "tool" and m.get("tool_call_id") == "c3")
        self.assertEqual(c3_msg["content"], "## Still Running (1)\n\n- `bg_1` [bash] — Building — make all")


class SubagentTests(Base):
    def make(self):
        b = Builder()
        main_session(b)
        b.write(self.path)
        sb = Builder()
        t = T0 + 1000
        sb.add({"type": "session_init", "systemPrompt": ["SUB a", "SUB b"], "task": "x",
                "tools": ["read", "grep", "debug"], "mountedTools": ["debug"], "agent": "helper2", "spawns": ""}, t)
        sb.custom(t + 1, "skill-prompt", "falsifier skill", attribution="agent")
        sb.reminder(t + 3, f"user:{t + 2}:agent")
        sb.user(t + 2, "task text", attribution="agent")
        sb.assistant(t + 10, "s", "done", (), stop="stop")
        sb.write(os.path.join(self.path[:-6], "Sub.jsonl"))
        return o.load_session(self.path)

    def test_subagent_request(self):
        sess = self.make()
        sub = sess.subagents["Sub"]
        body = r.build_request(sub, 0, SNAP)
        self.assertEqual(body["messages"][0], {"role": "system", "content": "SUB a\n\nSUB b"})
        # agent-attributed skill prompt is a developer turn (no reminder); the reminder goes on the user turn
        self.assertEqual(body["messages"][1], {"role": "user", "content": [{"type": "text", "text": "falsifier skill"}]})
        self.assertEqual(body["messages"][2]["content"][0]["text"], REM)
        # mounted tools are not sent; unknown agent -> default subagent pool (first non-main pool)
        self.assertEqual([t["function"]["description"] for t in body["tools"]], ["d", "helper grep"])
        body = r.build_request(sub, 0, SNAP, options={"tool_sources": {"grep": "main"}})
        self.assertEqual(body["tools"][1]["function"]["description"], "main grep")

    def test_bash_description_follows_eval(self):
        on = "Persistent shell.\nScripts/heredocs/`$(…)`/complex pipelines → `eval`.\nMore."
        off = "Persistent shell.\nScripts/heredocs/`$(…)`/complex flow → dedicated tool or checked-in script.\nMore."
        tools = [tool("bash", on), tool("read")]
        out = r.adapt_tool_descriptions(tools, ["bash", "read"])
        self.assertEqual(out[0]["function"]["description"], off)
        self.assertEqual(tools[0]["function"]["description"], on)  # pool untouched
        self.assertIs(out[1], tools[1])
        self.assertEqual(r.adapt_tool_descriptions(tools, ["bash", "eval"])[0]["function"]["description"], on)
        self.assertEqual(r.adapt_tool_descriptions([tool("bash", off)], ["eval"])[0]["function"]["description"], on)
        # a subagent without eval gets the eval-less bash line
        sess = self.make()
        snap = json.loads(json.dumps(SNAP))
        snap["tool_pool"]["main"]["read"] = tool("read")
        snap["tool_pool"]["helper"]["read"] = tool("read", on.replace("bash", "read"))
        sub = sess.subagents["Sub"]
        sub.branch[0]["tools"] = ["read", "bash"]
        snap["tool_pool"]["main"]["bash"] = tool("bash", on)
        body = r.build_request(sub, 0, snap)
        self.assertEqual(body["tools"][1]["function"]["description"], off)

    def test_unknown_tool_raises(self):
        sess = self.make()
        snap = json.loads(json.dumps(SNAP))
        del snap["tool_pool"]["helper"]["grep"], snap["tool_pool"]["main"]["grep"]
        with self.assertRaises(KeyError):
            r.build_request(sess.subagents["Sub"], 0, snap)


class SnapshotTests(Base):
    def test_retarget(self):
        snap = r.retarget_snapshot(SNAP, "Q", "N")
        self.assertEqual(snap["system_prompt"]["main"][1], "SYS two\n- Model: Q/N")
        self.assertEqual((snap["provider"], snap["model"]), ("Q", "N"))
        self.assertEqual(snap["system_prompt_digest"], r.system_prompt_digest(snap["system_prompt"]["main"]))
        with self.assertRaises(ValueError):
            r.retarget_snapshot(dict(SNAP, provider="nope"), "Q", "N")

    def test_capture_roundtrip(self):
        """snapshot_from_capture + golden_check plumbing on a synthetic capture written from the rebuild."""
        b = Builder()
        main_session(b)
        sess = self.load(b)
        cap = os.path.join(self.tmp.name, "cap")
        os.makedirs(cap)
        for t in sess.main.turns:
            body = r.build_request(sess.main, t.index, SNAP)
            ms = r.turn_request_ms(sess.main, t.index) + 3
            with open(os.path.join(cap, f"{ms}_req.json"), "w") as f:
                json.dump(body, f)
            with open(os.path.join(cap, f"{ms}_resp.json"), "w") as f:
                f.write('data: {"choices":[],"usage":{"prompt_tokens":%d}}\n\ndata: [DONE]\n' % t.prompt_tokens)
        with open(os.path.join(cap, f"{T0 + 150}_req.json"), "w") as f:  # a side request (title call)
            json.dump({"model": "M", "messages": []}, f)
        res = r.golden_check(self.path, cap)
        self.assertEqual((res["equal"], res["matched"]), (3, 3))
        self.assertEqual(len(res["unmatched"]), 1)
        self.assertEqual(res["snapshot"]["system_prompt"]["main"], ["SYS one\n\nSYS two\n- Model: P/M"])
        self.assertEqual(res["snapshot"]["tools"], {"main": ["read", "bash"]})

    def test_capture_only_equals_session_snapshot(self):
        b = Builder()
        main_session(b)
        sess = self.load(b)
        cap = os.path.join(self.tmp.name, "cap")
        os.makedirs(cap)
        for t in sess.main.turns:
            ms = r.turn_request_ms(sess.main, t.index)
            with open(os.path.join(cap, f"{ms}_req.json"), "w") as f:
                json.dump(r.build_request(sess.main, t.index, SNAP), f)
        with_session = r.snapshot_from_capture(cap, self.path)
        alone = r.snapshot_from_capture_only(cap, omp_version=with_session["omp_version"],
                                             digest=with_session["system_prompt_digest"])
        for k in with_session:
            if k != "notes":
                self.assertEqual(alone[k], with_session[k], k)
        with open(os.path.join(cap, f"{T0 + 1}_req.json"), "w") as f:  # a second tool group needs a name
            json.dump({"model": "M", "messages": [{"role": "system", "content": "- Model: P/M"}]}, f)
        with self.assertRaises(ValueError):
            r.snapshot_from_capture_only(cap)
        self.assertEqual(r.snapshot_from_capture_only(cap, "sub")["default_subagent_pool"], "sub")

    def test_capture_split_by_digest(self):
        b = Builder()
        main_session(b)
        b.recs[4]["data"] = r.system_prompt_digest(["SYS one", "SYS two\n- Model: P/M"])
        sess = self.load(b)
        cap = os.path.join(self.tmp.name, "cap")
        os.makedirs(cap)
        ms = r.turn_request_ms(sess.main, 0)
        with open(os.path.join(cap, f"{ms}_req.json"), "w") as f:
            json.dump(r.build_request(sess.main, 0, SNAP), f)
        snap = r.snapshot_from_capture(cap, self.path)
        self.assertEqual(snap["system_prompt"]["main"], ["SYS one", "SYS two\n- Model: P/M"])


class SnapshotEditTests(unittest.TestCase):
    """adapt_snapshot and its parts: a session whose checkout differed from the capture's."""
    PROMPT = ("Head.\n\n# Tool Inventory\n- Read: `read`\n- Grep: `grep`\n- Task: `task`\n\n# Specialized Tools\n"
              "- Regex/target search: `grep`, NEVER shell.\n\n# Exploration\nNEVER open guessed files. Use `read`."
              "\n\n<project-context>\n<repo-rules>\n<file path=\"/w/AGENTS.md\">\n# Old\n| a | b |\n</file>\n"
              "</repo-rules>\n</project-context>")
    GREP = ("Regex. A `!glob` entry excludes (`src;!**/*.test.ts`).\n\nMulti-round search MUST use Task, "
            "not chained calls.")
    GLOB = "Glob; dirs end `/`.\n\nMulti-round discovery → Task."
    EVAL = "One cell.\nSession tool names: `read`, `eval`, `grep`, `task`, `lsp`.\n</prelude>"
    TASK = ("Spawn `tasks[]` concurrently; IDs return immediately. BLOCKING agents return inline.\n\n"
            "# Available Agents\n- `fals` (BLOCKING; inline result): Falsifier.\n"
            "- `scout` (READ-ONLY; investigation only, no edits): Scout.\n- `task`: General.")

    def snap(self):
        snap = json.loads(json.dumps(SNAP))
        snap["system_prompt"]["main"] = self.PROMPT.split("\n\n<project-context>")
        snap["system_prompt"]["main"][1] = "<project-context>" + snap["system_prompt"]["main"][1]
        snap["tools"]["main"] = ["read", "eval", "grep", "task"]
        snap["tool_pool"]["main"] = {"read": tool("read"), "eval": tool("eval", self.EVAL),
                                     "grep": tool("grep", self.GREP), "task": tool("task", self.TASK)}
        return snap

    def test_find_lines_in_grep_glob_and_eval(self):
        tools = [tool("grep", self.GREP), tool("glob", self.GLOB), tool("eval", self.EVAL)]
        on = r.adapt_tool_descriptions(tools, ["read", "eval", "grep", "find", "task"],
                                       ["read", "eval", "grep", "task", "find"])
        self.assertEqual(on[0]["function"]["description"], self.GREP.replace(
            ").\n\nMulti", ").\nBehavior/unknown symbol → `find`; literals/regex → `grep`.\nMulti"))
        self.assertEqual(on[1]["function"]["description"], self.GLOB.replace(
            "`/`.\n\nMulti", "`/`.\nBehavior search → `find`.\nMulti"))
        # active tools in order, then the captured agent's non-tools (xd:// devices) kept at the end
        self.assertIn("names: `read`, `eval`, `grep`, `find`, `task`, `lsp`.", on[2]["function"]["description"])
        back = r.adapt_tool_descriptions(on, ["read", "eval", "grep", "task"], ["read", "eval", "grep", "task", "find"])
        self.assertEqual([t["function"]["description"] for t in back], [self.GREP, self.GLOB, self.EVAL])
        self.assertEqual(r.adapt_tool_descriptions(tools, ["read"])[2], tools[2])  # no captured list: eval kept

    def test_prompt_tool_lines_and_inventory(self):
        text = r.adapt_prompt_tools(self.PROMPT, ["read", "grep", "find", "task"])
        self.assertIn("- Grep: `grep`\n- Find: `find`\n- Task: `task`\n", text)
        self.assertIn("- Unknown behavior/location: descriptive `find` FIRST; NEVER guess `grep`/`glob` targets.\n"
                      "- Regex/literal/known-symbol search: `grep`, NEVER shell.", text)
        self.assertIn("NEVER open guessed files. Read `find` hits only. Use `read`.", text)
        self.assertEqual(r.adapt_prompt_tools(text, ["read", "grep", "task"]), self.PROMPT)
        self.assertIn("# Tool Inventory\n- Read: `read`\n- Task: `task`\n\n",
                      r.adapt_prompt_tools(self.PROMPT, ["read", "task"]))

    def test_replace_context_file(self):
        text = r.replace_context_file(self.PROMPT, "/w/AGENTS.md", "# New\n| x  |  y |\n\n\n")
        self.assertIn('<file path="/w/AGENTS.md">\n# New\n|x|y|\n</file>', text)  # omp's format pass
        with self.assertRaises(ValueError):
            r.replace_context_file(self.PROMPT, "/w/CLAUDE.md", "x")

    def test_agent_blocking(self):
        off = r.set_agent_blocking(self.TASK, "fals", False)
        self.assertNotIn("BLOCKING", off)
        self.assertIn("- `fals`: Falsifier.", off)
        self.assertEqual(r.set_agent_blocking(off, "fals", True), self.TASK)
        ro = r.set_agent_blocking(off, "scout", True)
        self.assertIn("- `scout` (READ-ONLY; investigation only, no edits) (BLOCKING; inline result): Scout.", ro)
        self.assertTrue(ro.startswith("Spawn `tasks[]` concurrently; IDs return immediately. BLOCKING agents"))
        with self.assertRaises(ValueError):
            r.set_agent_blocking(self.TASK, "nope", False)

    def test_adapt_snapshot(self):
        snap = self.snap()
        out = r.adapt_snapshot(snap, {"/w/AGENTS.md": "# New"}, ("find",), ("eval",), {"fals": False})
        self.assertEqual(out["tools"]["main"], ["read", "grep", "find", "task"])  # builtin order
        self.assertEqual(snap["tools"]["main"], ["read", "eval", "grep", "task"])  # input untouched
        self.assertEqual(out["tool_pool"]["main"]["find"], r.EXTRA_TOOLS["find"][1])
        self.assertEqual(len(out["system_prompt"]["main"]), 2)
        self.assertTrue(out["system_prompt"]["main"][1].startswith("<project-context>"))
        joined = "\n\n".join(out["system_prompt"]["main"])
        self.assertIn("- Find: `find`", joined)
        self.assertIn("# New\n</file>", joined)
        self.assertNotIn("BLOCKING", out["tool_pool"]["main"]["task"]["function"]["description"])
        self.assertEqual(out["system_prompt_digest"], r.system_prompt_digest(out["system_prompt"]["main"]))
        d = r.system_prompt_digest(["\n\n".join(out["system_prompt"]["main"])])
        self.assertEqual(r.adapt_snapshot(snap, {"/w/AGENTS.md": "# New"}, ("find",), ("eval",),
                                          {"fals": False}, digest=d)["system_prompt"]["main"], [joined])
        with self.assertRaises(ValueError):
            r.adapt_snapshot(snap, enable_tools=("nope",))
        with self.assertRaises(ValueError):
            r.adapt_snapshot(snap, digest="0")

    def test_main_request_gets_find_lines(self):
        snap = self.snap()
        snap["tool_pool"]["main"]["glob"] = tool("glob", self.GLOB)
        snap["tools"]["main"] = ["read", "glob", "grep", "task"]
        out = r.adapt_snapshot(snap, enable_tools=("find",))
        main = type("Stream", (), {"kind": "main"})()
        tools = r.stream_tools(main, out)
        self.assertEqual([t["function"]["name"] for t in tools], ["read", "glob", "grep", "find", "task"])
        self.assertIn("Behavior search → `find`.", tools[1]["function"]["description"])
        self.assertIn("Behavior/unknown symbol → `find`", tools[2]["function"]["description"])
        self.assertIs(r.stream_tools(main, snap)[1], snap["tool_pool"]["main"]["glob"])  # unchanged tool set: as captured


class CompactionRequestTests(Base):
    def test_build_compaction_request_handoff(self):
        b = Builder()
        b.reminder(T0 + 5, f"user:{T0}:user")
        b.user(T0, "hello")
        b.assistant(T0 + 10, "thinking", "doing work", calls=[("c1", "read", {"path": "a.txt"})])
        b.result(T0 + 20, "c1", "read", "content")
        b.compaction(T0 + 30, summary="handoff doc", method="handoff")
        sess = self.load(b)
        self.assertEqual(len(sess.main.compactions), 1)

        notes = []
        req = r.build_compaction_request(sess.main, 0, SNAP, notes=notes)
        # System prompt preserved
        self.assertEqual(req["messages"][0]["role"], "system")
        self.assertEqual(req["messages"][0]["content"], "SYS one\n\nSYS two\n- Model: P/M")
        # Appended user message with handoff prompt
        last_msg = req["messages"][-1]
        self.assertEqual(last_msg["role"], "user")
        self.assertIn("<critical>", last_msg["content"][0]["text"])
        self.assertIn("Additional focus: Threshold-triggered maintenance", last_msg["content"][0]["text"])
        # Tools preserved and tool_choice is "none"
        self.assertIn("tools", req)
        self.assertEqual(len(req["tools"]), len(SNAP["tools"]["main"]))
        self.assertEqual(req.get("tool_choice"), "none")

    def test_build_compaction_request_unsupported_method(self):
        b = Builder()
        b.reminder(T0 + 5, f"user:{T0}:user")
        b.user(T0, "hello")
        b.compaction(T0 + 10, summary="s", method="unknown")
        sess = self.load(b)
        with self.assertRaises(r.RebuildError):
            r.build_compaction_request(sess.main, 0, SNAP)


# --------------------------------------------------------------------------------------------- real inputs

@unittest.skipUnless(os.path.isdir(GOLDEN_CAPTURE) and os.path.isfile(GOLDEN_SESSION), "golden capture not present")
class GoldenTests(unittest.TestCase):
    def test_every_captured_request_rebuilt_exactly(self):
        res = r.golden_check(GOLDEN_SESSION, GOLDEN_CAPTURE)
        bad = [(x["capture"], x.get("diffs") or x.get("error")) for x in res["rows"] if x["status"] in ("diff", "error")]
        self.assertEqual(bad, [])
        self.assertEqual((res["equal"], res["matched"]), (13, 13))
        self.assertEqual(len(res["unmatched"]), 1)  # the session-title side request
        self.assertEqual(res["snapshot"]["system_prompt_digest"], "73b077bfd8cddc10")


def case_a_snapshot():
    """The case A snapshot: from the golden capture + its session, or (where that session is not synced) from the
    capture alone, split by case A's digest."""
    if os.path.isfile(GOLDEN_SESSION):
        base = r.snapshot_from_capture(GOLDEN_CAPTURE, GOLDEN_SESSION)
        return r.retarget_snapshot(base, "MBP-8133", "unsloth/Qwen3.8-Flash-Next-GGUF")
    base = r.snapshot_from_capture_only(GOLDEN_CAPTURE, "sonic", omp_version="18.17.0")
    snap = r.retarget_snapshot(base, "MBP-8133", "unsloth/Qwen3.8-Flash-Next-GGUF")
    snap["system_prompt"]["main"] = r.split_blocks_for_digest("\n\n".join(snap["system_prompt"]["main"]),
                                                              CASE_A_DIGEST)
    return snap


CASE_A_DIGEST = "d607e65777c1cb9b"


@unittest.skipUnless(os.path.isdir(GOLDEN_CAPTURE) and os.path.isfile(CASE_A_SESSION), "case A inputs not present")
class CaseATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snap = case_a_snapshot()
        cls.sess = o.load_session(CASE_A_SESSION)

    def test_main_prompt_digest(self):
        self.assertEqual(r.system_prompt_digest(self.snap["system_prompt"]["main"]), "d607e65777c1cb9b")
        self.assertEqual(len(self.snap["system_prompt"]["main"][0]), 14240)

    def test_subagent_prompt_digests(self):
        for s in self.sess.subagents.values():
            pos = s.turns[0].branch_pos
            self.assertEqual(r.system_prompt_digest(r.stream_system_blocks(s, self.snap, pos)),
                             r._digest_entry_before(s, pos), s.name)

    def test_all_turns_rebuild(self):
        n = 0
        for s in self.sess.streams():
            for t in s.turns:
                body = r.build_request(s, t.index, self.snap)
                self.assertEqual(body["messages"][0]["role"], "system")
                n += 1
        self.assertEqual(n, 473)



def _git_show(repo, rev, path):
    import subprocess
    try:
        res = subprocess.run(["git", "-C", repo, "show", f"{rev}:{path}"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout if res.returncode == 0 else None


def case_d_snapshot(case_a):
    """Case D = case A's snapshot with the checkout at CASE_D_REV: its AGENTS.md, `find` on (its `.omp/config.yml`
    does not turn it off yet) and `f428-falsify` not blocking (its `.omp/agents/f428-falsify.md`)."""
    import re
    agents = _git_show(CASE_D_REPO, CASE_D_REV, "AGENTS.md")
    path = re.search(r'<file path="([^"]+/AGENTS\.md)">', "\n\n".join(case_a["system_prompt"]["main"])).group(1)
    return r.adapt_snapshot(case_a, {path: agents}, ("find",), (), {"f428-falsify": False}, CASE_D_DIGEST)


CASE_D_DIGEST = "b1a8f92cbf4f8a2b"


@unittest.skipUnless(os.path.isdir(GOLDEN_CAPTURE) and os.path.isfile(CASE_D_SESSION)
                     and _git_show(CASE_D_REPO, CASE_D_REV, "AGENTS.md"), "case D inputs not present")
class CaseDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snap = case_d_snapshot(case_a_snapshot())
        cls.sess = o.load_session(CASE_D_SESSION)

    def test_main_prompt_digest(self):
        self.assertEqual(r.system_prompt_digest(self.snap["system_prompt"]["main"]), CASE_D_DIGEST)
        self.assertEqual(r._digest_entry_before(self.sess.main, self.sess.main.turns[0].branch_pos), CASE_D_DIGEST)

    def test_subagent_prompt_digests(self):
        for s in self.sess.subagents.values():
            pos = s.turns[0].branch_pos
            self.assertEqual(r.system_prompt_digest(r.stream_system_blocks(s, self.snap, pos)),
                             r._digest_entry_before(s, pos), s.name)

    def test_all_turns_and_compaction_rebuild(self):
        res = r.rebuild_all(CASE_D_SESSION, self.snap)
        self.assertEqual(res["errors"], [])
        self.assertEqual(sum(v["rebuilt"] for v in res["streams"].values()), 469)
        self.assertEqual(res["streams"]["main"].get("compactions_rebuilt"), 1)
        body = r.build_request(self.sess.main, 0, self.snap)
        self.assertEqual([t["function"]["name"] for t in body["tools"]][5:8], ["glob", "grep", "find"])


if __name__ == "__main__":
    unittest.main()
