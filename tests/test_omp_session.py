"""Offline tests for tools/omp_session.py on small synthetic sessions (no real transcripts)."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401
import omp_session as o  # noqa: E402

T0 = 1_700_000_000.0  # epoch seconds


def iso(sec):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(sec, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def assistant(eid, parent, start, dur, prompt, epoch=0, calls=(), stop="toolUse"):
    content = [{"type": "text", "text": "x"}] + [{"type": "toolCall", "id": cid, "name": n, "arguments": {}}
                                                 for cid, n in calls]
    return {"type": "message", "id": eid, "parentId": parent, "timestamp": iso(start + dur),
            "message": {"role": "assistant", "content": content, "stopReason": stop,
                        "timestamp": int(start * 1000), "completedAt": int((start + dur) * 1000), "duration": dur * 1000,
                        "usage": {"input": 1, "output": 2, "cacheRead": 3},
                        "contextSnapshot": {"promptTokens": prompt, "nonMessageTokens": 10, "toolSchemaTokens": 5,
                                            "compactionEpoch": epoch}}}


def tool_result(eid, parent, ts, call_id, name, details=None):
    m = {"role": "toolResult", "toolCallId": call_id, "toolName": name, "content": [{"type": "text", "text": "ok"}]}
    if details:
        m["details"] = details
    return {"type": "message", "id": eid, "parentId": parent, "timestamp": iso(ts), "message": m}


def write(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def header(sid, parent=None):
    h = {"type": "session", "version": 3, "id": sid, "timestamp": iso(T0), "cwd": "/x"}
    if parent:
        h["parentSession"] = parent
    return h


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.main = os.path.join(self.tmp.name, "s1.jsonl")
        self.dir = os.path.join(self.tmp.name, "s1")


class BranchTests(Base):
    def test_abandoned_branch_excluded(self):
        # a1 -> a2 -> a3 (abandoned) ; a1 -> b2 -> b3 (active, last in file)
        recs = [{"type": "title", "v": 1, "title": ""}, header("s1"),
                {"type": "model_change", "id": "m0", "parentId": None, "timestamp": iso(T0)},
                assistant("a1", "m0", T0 + 1, 1, 100),
                assistant("a2", "a1", T0 + 3, 1, 200),
                assistant("a3", "a2", T0 + 5, 1, 300),
                assistant("b2", "a1", T0 + 7, 1, 250),
                assistant("b3", "b2", T0 + 9, 1, 350)]
        write(self.main, recs)
        s = o.load_session(self.main).main
        self.assertEqual([t.entry_id for t in s.turns], ["a1", "b2", "b3"])
        self.assertEqual([t.prompt_tokens for t in s.turns], [100, 250, 350])
        self.assertEqual(s.abandoned_entries, 2)

    def test_model_usage_active_leaf_selects_earlier_leaf(self):
        # the last record is a late usage marker pointing back at a2: a3 is off the active branch
        recs = [header("s1"), assistant("a1", None, T0 + 1, 1, 100), assistant("a2", "a1", T0 + 3, 1, 200),
                assistant("a3", "a2", T0 + 5, 1, 300),
                {"type": "model_usage", "id": "u1", "parentId": "a3", "timestamp": iso(T0 + 6), "activeLeafId": "a2"}]
        write(self.main, recs)
        s = o.load_session(self.main).main
        self.assertEqual(s.leaf_id, "a2")
        self.assertEqual([t.entry_id for t in s.turns], ["a1", "a2"])

    def test_dangling_active_leaf_falls_back(self):
        recs = [header("s1"), assistant("a1", None, T0 + 1, 1, 100), assistant("a2", "a1", T0 + 3, 1, 200),
                {"type": "model_usage", "id": "u1", "parentId": "a2", "timestamp": iso(T0 + 6), "activeLeafId": "gone"}]
        write(self.main, recs)
        self.assertEqual(o.load_session(self.main).main.leaf_id, "a2")

    def test_malformed_and_blank_lines_skipped(self):
        write(self.main, [header("s1"), assistant("a1", None, T0 + 1, 1, 100)])
        with open(self.main, "a") as f:
            f.write("\n{not json\n")
        s = o.load_session(self.main).main
        self.assertEqual(len(s.turns), 1)
        self.assertEqual(s.malformed_lines, 1)


class CompactionTests(Base):
    def test_compaction_position_and_epochs(self):
        recs = [header("s1"),
                assistant("a1", None, T0 + 1, 1, 100, 0), assistant("a2", "a1", T0 + 3, 1, 900, 0),
                {"type": "compaction", "id": "c1", "parentId": "a2", "timestamp": iso(T0 + 5), "summary": "abc",
                 "firstKeptEntryId": "a2", "tokensBefore": 900, "tokensAfter": 50, "method": "handoff"},
                assistant("a3", "c1", T0 + 7, 1, 60, 1)]
        write(self.main, recs)
        s = o.load_session(self.main).main
        self.assertEqual(len(s.compactions), 1)
        c = s.compactions[0]
        self.assertEqual((c.turns_before, c.branch_pos, c.tokens_before, c.tokens_after, c.summary_chars),
                         (2, 2, 900, 50, 3))
        self.assertEqual([t.compaction_epoch for t in s.turns], [0, 0, 1])
        self.assertEqual(s.turns[2].prompt_tokens, 60)


class SubagentTests(Base):
    def build(self, link_details=True):
        call = "call_1"
        recs = [header("s1"),
                assistant("a1", None, T0 + 1, 1, 100, calls=[(call, "task")]),
                tool_result("r1", "a1", T0 + 40, call, "task",
                            {"results": [{"index": 0, "id": "Sub1", "agent": "falsify"}]} if link_details else None),
                assistant("a2", "r1", T0 + 41, 1, 120, stop="stop")]
        write(self.main, recs)
        sub = lambda sid, t, n: [header(sid, self.main), {"type": "session_init", "id": "i" + sid, "parentId": None,
                                 "timestamp": iso(t), "systemPrompt": ["p1", "p2"], "task": "do", "agent": "falsify",
                                 "tools": ["read"]}] + [assistant(f"{sid}{k}", "i" + sid if k == 0 else f"{sid}{k-1}",
                                 t + 1 + 2 * k, 1, 50 + k) for k in range(n)]
        write(os.path.join(self.dir, "Sub1.jsonl"), sub("S1", T0 + 2, 3))
        return o.load_session(self.main)

    def test_link_by_task_result(self):
        sess = self.build()
        self.assertEqual(list(sess.subagents), ["Sub1"])
        self.assertEqual(len(sess.spawns), 1)
        sp = sess.spawns[0]
        self.assertEqual((sp.parent, sp.parent_turn, sp.tool_call_id, sp.child, sp.linked_by),
                         ("main", 0, "call_1", "Sub1", "tool-result"))
        self.assertEqual(sess.subagents["Sub1"].session_init["systemPrompt"], ["p1", "p2"])
        self.assertEqual(sess.unlinked, [])

    def test_link_fallback_by_parent_session_and_time(self):
        sess = self.build(link_details=False)
        self.assertEqual(len(sess.spawns), 1)
        self.assertEqual(sess.spawns[0].linked_by, "parentSession+time")
        self.assertEqual((sess.spawns[0].parent_turn, sess.spawns[0].child), (0, "Sub1"))

    def test_non_subagent_files_ignored(self):
        self.build()
        for n in ("__advisor.jsonl", "X.bak.jsonl", "notes.md"):
            write(os.path.join(self.dir, n), [header("z")])
        self.assertEqual(list(o.load_session(self.main).subagents), ["Sub1"])

    def test_nested_subagent(self):
        sess = self.build()
        # Sub1 spawns Child1 (file under s1/Sub1/)
        sub1 = os.path.join(self.dir, "Sub1.jsonl")
        recs, _ = o.read_records(sub1)
        recs.append(assistant("S1x", "S12", T0 + 20, 1, 70, calls=[("call_n", "task")]))
        recs.append(tool_result("S1y", "S1x", T0 + 30, "call_n", "task", {"results": [{"id": "Child1", "agent": "q"}]}))
        write(sub1, recs)
        write(os.path.join(self.dir, "Sub1", "Child1.jsonl"),
              [header("C", sub1), assistant("C0", None, T0 + 21, 1, 10)])
        sess = o.load_session(self.main)
        self.assertIn("Sub1/Child1", sess.subagents)
        sp = [x for x in sess.spawns if x.child == "Sub1/Child1"][0]
        self.assertEqual((sp.parent, sp.parent_turn), ("Sub1", 3))


class OrderingTests(SubagentTests):
    def test_interleaved_by_request_start(self):
        sess = self.build()
        order = [(t.stream, t.index) for t in sess.interleaved()]
        # main a1 starts T0+1; Sub1 turns start T0+3, T0+5, T0+7; main a2 starts T0+41
        self.assertEqual(order, [("main", 0), ("Sub1", 0), ("Sub1", 1), ("Sub1", 2), ("main", 1)])
        self.assertEqual(len(sess.interleaved(include_subagents=False)), 2)

    def test_summary_numbers(self):
        sm = o.session_summary(self.build())
        self.assertEqual(sm["main"]["turns"], 2)
        self.assertEqual(sm["main"]["max_prompt_tokens"], 120)
        self.assertEqual(sm["subagent_totals"], {"count": 1, "turns": 3, "max_prompt_tokens": 52})
        self.assertEqual(sm["total_turns"], 5)
        json.dumps(sm)
        self.assertIn("total turns: 5", o.format_summary(sm))


if __name__ == "__main__":
    unittest.main()
