#!/usr/bin/env python3
"""Unit tests for tools.omp_fixture (redaction, build, check, PII scan). Synthetic people only; no server needed."""
import copy
import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from tools.omp_fixture import (  # noqa: E402
    FIXTURE_VERSION,
    build_fixture,
    check_fixture,
    load_fixture,
    load_rules,
    reconstruct_turn_request,
    redact_obj,
    redact_text,
    scan_pii,
)

# Synthetic redaction rules, in the sources/redact.json format: [regex, replacement], applied in order.
RULE_PAIRS = [
    ["jroe@example\\.org", "dev@example.com"],
    ["Janes-MBP(?:\\.localdomain|\\.local)?", "host-a"],
    ["(?i)Jane Roe", "Dev User"],
    ["(?i)jroe", "dev"],
    ["(?i)(?<![\\w-])roe(?![\\w-])", "dev"],
    ["MBP-8133", "LOCAL-8133"],
]
RULES = [(re.compile(p), r) for p, r in RULE_PAIRS]


class TestRedaction(unittest.TestCase):
    def test_rules_paths_emails_names(self):
        sample = (
            "Path: /Users/jroe/Code/app\nEmail: jroe@example.org\nName: Jane Roe, jane roe\n"
            "Host: Janes-MBP.local, Janes-MBP\nProvider: MBP-8133\n"
        )
        out = redact_text(sample, RULES)
        self.assertIn("/Users/standard.analyst/Code/app", out)
        self.assertIn("dev@example.com", out)
        self.assertIn("Dev User", out)
        self.assertIn("host-a", out)
        self.assertIn("LOCAL-8133", out)
        for bad in ("jroe", "Janes", "MBP-8133", "Roe"):
            self.assertNotIn(bad, out)

    def test_builtin_rule_hides_any_home_dir_without_rules(self):
        self.assertEqual(redact_text("/Users/somebody/x and /Users/standard.analyst/y and /Users/..."),
                         "/Users/standard.analyst/x and /Users/standard.analyst/y and /Users/...")
        self.assertEqual(redact_text("Co-authored-by: Bot <bot@vendor.example>\nx"), "Co-authored-by: Bot <noreply@example.com>\nx")
        self.assertEqual(redact_text("mail bot@vendor.example"), "mail bot@vendor.example")

    def test_preserves_legitimate_words(self):
        out = redact_text("a heroes roster, throes of change, roe deer", RULES)
        self.assertIn("heroes", out)
        self.assertIn("throes", out)

    def test_drops_syntax_fields_and_redacts_keys_and_values(self):
        obj = {"name": "bash", "syntax": "AQ...", "details": {"units": [{"cmd": "echo 1", "syntax": "AQ"}]}, "user": "jroe"}
        cleaned = redact_obj(obj, RULES)
        self.assertNotIn("syntax", cleaned)
        self.assertNotIn("syntax", cleaned["details"]["units"][0])
        self.assertEqual(cleaned["user"], "dev")
        self.assertEqual(cleaned["details"]["units"][0]["cmd"], "echo 1")

    def test_load_rules(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                load_rules(os.path.join(d, "missing.json"))
            bad = os.path.join(d, "bad.json")
            with open(bad, "w") as f:
                json.dump({"a": "b"}, f)
            with self.assertRaises(ValueError):
                load_rules(bad)
            good = os.path.join(d, "good.json")
            with open(good, "w") as f:
                json.dump(RULE_PAIRS, f)
            rules = load_rules(good)
            self.assertEqual(redact_text("Jane Roe", rules), "Dev User")


class TestPiiScan(unittest.TestCase):
    def test_detects_secrets_and_identifiers(self):
        cases = [
            "/Users/jroe/repo",
            "Connecting to Janes-MBP.local",
            "Model on MBP-8133",
            '{"syntax": "AQAAAA..."}',
            "sk-proj-123456789012345678901234567890",
            "sk-ant-api03-123456789012345678901234567890",
            "ghp_123456789012345678901234567890123456",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIE...",
            "mail me at jroe@example.org",
            "server at 192.168.1.20",
        ]
        for text in cases:
            issues, _ = scan_pii(text)
            self.assertTrue(issues, f"expected a hit for {text!r}")

    def test_names_and_forbidden_are_fatal(self):
        issues, _ = scan_pii("hello Jane", names=["jane"])
        self.assertTrue(issues)
        issues, _ = scan_pii("hello Jane", forbidden=["(?i)jane"])
        self.assertTrue(issues)
        issues, _ = scan_pii({"k": "hello Jane"}, forbidden=["zzz", "(?i)jane"])
        self.assertTrue(issues)

    def test_allowlist_and_info(self):
        sha = "a" * 40
        issues, info = scan_pii({"a": f"/Users/standard.analyst/x dev@example.com 127.0.0.1 0.0.0.0 commit {sha}"})
        self.assertEqual(issues, [])
        self.assertEqual(info, {"secret:hex40+": 1})
        issues, _ = scan_pii("/Users/standard.analyst2/x")
        self.assertTrue(issues)


class TestSyntheticSessionFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snap_path = os.path.join(self.tmp.name, "snap.json")
        snap = {
            "format": "omp-request-snapshot/1",
            "omp_version": "18.17.0",
            "provider": "LOCAL-8133",
            "model": "unsloth/Qwen3.8-Flash-Next-GGUF",
            "body_keys": ["model", "messages", "stream", "stream_options", "tools", "max_completion_tokens", "chat_template_kwargs"],
            "params": {"stream": True, "max_completion_tokens": 32768,
                       "chat_template_kwargs": {"preserve_thinking": True, "enable_thinking": True}},
            "system_prompt": {"main": ["You are a test assistant.", "Model: MBP-8133/test"]},
            "tools": {"main": ["read"]},
            "tool_pool": {"main": {"read": {"type": "function", "function": {"name": "read", "parameters": {}}}}},
            "compat": {"replay_reasoning_content": True},
        }
        with open(self.snap_path, "w") as f:
            json.dump(snap, f)
        self.sess_path = os.path.join(self.tmp.name, "sess.jsonl")
        entries = [
            {"type": "session", "version": 3, "id": "test-session-001", "timestamp": "2026-11-06T12:00:00.000Z",
             "cwd": "/Users/jroe/Code/test"},
            {"type": "message", "id": "m1", "parentId": None, "timestamp": "2026-11-06T12:00:01.000Z",
             "message": {"role": "user", "attribution": "user", "timestamp": 1794000001000,
                         "content": [{"type": "text", "text": "hello from Jane Roe"}]}},
            {"type": "custom", "id": "r1", "parentId": "m1", "customType": "date-cwd-reminder",
             "timestamp": "2026-11-06T12:00:02.000Z",
             "data": {"conversation": "user:1794000001000:user", "turn": "user:1794000001000:user",
                      "reminder": "Current date: 2026-11-06\nCurrent directory: /Users/jroe/Code/test",
                      "placement": "prefix"}},
            {"type": "message", "id": "m2", "parentId": "r1", "timestamp": "2026-11-06T12:00:05.000Z",
             "message": {"role": "assistant", "timestamp": 1794000005000, "content": [{"type": "text", "text": "hi"}],
                         "provider": "MBP-8133", "stopReason": "endTurn",
                         "usage": {"input": 90, "output": 12}, "contextSnapshot": {"promptTokens": 100}}},
        ]
        with open(self.sess_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    def test_build_reconstruct_and_check(self):
        fixture = build_fixture(self.sess_path, self.snap_path, counter=None, rules=RULES)
        self.assertEqual(fixture["fixture_version"], FIXTURE_VERSION)
        self.assertEqual(fixture["counts"], {"turns": 1, "compactions": 0, "streams": {"main": 1}})
        self.assertEqual(len(fixture["turns"]), 1)
        turn = fixture["turns"][0]
        self.assertEqual(turn["recorded_prompt_tokens"], 100)
        self.assertEqual(turn["recorded_output_tokens"], 12)
        self.assertEqual(turn["prompt_tokens"], 100)  # no counter: falls back to the recorded count
        req = reconstruct_turn_request(fixture, turn)
        self.assertEqual(len(req["messages"]), len(turn["messages"]))
        self.assertIn("LOCAL-8133", req["messages"][0]["content"])
        self.assertNotIn("MBP-8133", req["messages"][0]["content"])
        text = json.dumps(fixture)
        for bad in ("jroe", "Jane", "Roe"):
            self.assertNotIn(bad, text)
        ok, errors = check_fixture(fixture, forbidden=["(?i)jane", "(?i)jroe"])
        self.assertTrue(ok, errors)

    def test_check_catches_problems(self):
        fixture = build_fixture(self.sess_path, self.snap_path, counter=None, rules=RULES)
        # unredacted build leaks the synthetic name: the forbidden pattern catches it
        leaky = build_fixture(self.sess_path, self.snap_path, counter=None, rules=[])
        ok, errors = check_fixture(leaky, forbidden=["(?i)jane roe"])
        self.assertFalse(ok)
        self.assertTrue(any("PII" in e for e in errors))

        bad = copy.deepcopy(fixture)
        bad["counts"]["turns"] = 2
        self.assertFalse(check_fixture(bad)[0])

        bad = copy.deepcopy(fixture)
        bad["turns"][0]["messages"].append(10**6)
        self.assertFalse(check_fixture(bad)[0])

        bad = copy.deepcopy(fixture)
        bad["turns"][0]["prompt_tokens"] = 0
        self.assertFalse(check_fixture(bad)[0])

        bad = copy.deepcopy(fixture)
        bad["messages"][bad["turns"][0]["messages"][-1]] = {"role": "assistant", "content": "x"}
        self.assertFalse(check_fixture(bad)[0])

        bad = copy.deepcopy(fixture)
        bad["turns"][0]["kind"] = "compaction"
        bad["counts"] = {"turns": 0, "compactions": 1, "streams": {"main": 1}}
        ok, errors = check_fixture(bad)
        self.assertTrue(any("tool_choice" in e for e in errors))


class TestCodingFixture(unittest.TestCase):
    """The committed fixture must pass its own check (no personal data, no redact file needed)."""

    FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "..", "fixtures", "conv-3.json.gz")

    def test_coding_fixture(self):
        info = {}
        ok, errors = check_fixture(self.FIXTURE_PATH, info=info)
        self.assertTrue(ok, errors)
        fixture = load_fixture(self.FIXTURE_PATH)
        c = fixture["counts"]
        self.assertEqual(len(fixture["turns"]), c["turns"] + c["compactions"])
        self.assertEqual(sum(c["streams"].values()), len(fixture["turns"]))
        for t in fixture["turns"]:
            self.assertIn("recorded_prompt_tokens", t)


if __name__ == "__main__":
    unittest.main()


class TestProxyDumps(unittest.TestCase):
    """build_proxy_fixture on fake proxy dumps (synthetic names only)."""

    def _dump(self, d, ms, messages, answer, usage=None, chat=True):
        req = {"model": "m", "temperature": 0.3, "stream": True, "tools": [{"type": "function", "function": {"name": "t"}}]}
        if chat:
            req["messages"] = messages
        with open(os.path.join(d, f"{ms}_req.json"), "w") as f:
            json.dump(req, f)
        with open(os.path.join(d, f"{ms}_resp_combined.json"), "w") as f:
            json.dump(answer, f)
        sse = 'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
        if usage:
            sse += "data: " + json.dumps({"choices": [], "usage": usage}) + "\n\ndata: [DONE]\n"
        with open(os.path.join(d, f"{ms}_resp.json"), "w") as f:
            f.write(sse)

    def test_build_and_check(self):
        from tools.omp_fixture import build_proxy_fixture
        sysm = {"role": "system", "content": "hi"}
        u = {"role": "user", "content": "task in /Users/jroe/app"}
        a = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                                  "function": {"name": "t", "arguments": "{}"}}]}
        tr = {"role": "tool", "tool_call_id": "c1", "content": "ok"}
        with tempfile.TemporaryDirectory() as d:
            self._dump(d, 2000, [sysm, u, a, tr], {"content": "done", "reasoning_content": None, "tool_calls": None},
                       {"prompt_tokens": 50, "completion_tokens": 7})
            self._dump(d, 1000, [sysm, u], {"content": "", "reasoning_content": "r",
                                            "tool_calls": [{"id": "c1", "type": "function",
                                                            "function": {"name": "t", "arguments": "{}"}}]},
                       {"prompt_tokens": 20, "completion_tokens": 3})
            self._dump(d, 3000, None, {"content": "title"}, chat=False)
            skipped = []
            fx = build_proxy_fixture(d, RULES, skipped=skipped)
        self.assertEqual(skipped, ["3000_req.json"])
        self.assertEqual([t["recorded_prompt_tokens"] for t in fx["turns"]], [20, 50])
        self.assertEqual([len(t["messages"]) for t in fx["turns"]], [2, 4])
        self.assertEqual(len(fx["messages"]), 4)  # shared pool: system and user stored once
        self.assertEqual(fx["turns"][0]["params"], {"model": "m", "temperature": 0.3, "stream": True})
        self.assertEqual(fx["turns"][0]["recorded_answer"]["tool_calls"][0]["id"], "c1")
        self.assertEqual(fx["turns"][1]["recorded_output_tokens"], 7)
        self.assertEqual(fx["turns"][1]["time"], "1970-01-01T00:00:02.000Z")
        self.assertFalse(fx["prompt_tokens_recounted"])
        self.assertIn("/Users/standard.analyst/app", json.dumps(fx))
        self.assertNotIn("jroe", json.dumps(fx))
        ok, errors = check_fixture(fx)
        self.assertTrue(ok, errors)
        fx["turns"][0]["prompt_tokens"] = 21
        ok, errors = check_fixture(fx)
        self.assertFalse(ok)
