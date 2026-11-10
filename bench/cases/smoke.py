"""Smoke-test cases (--suite smoke): a quick check that a server answers and can call a tool, with their validators and
the one grading routine (evaluate_response) that live runs and saved states both use."""

import copy
import json
import re

from ..utils import safe_json, strip_code_fence, verdict


def validate_tc01_git_commit(text):
    """Validates TC01 text summarization against Conventional Commits spec."""
    t = text.strip()

    if t.startswith("```"):
        return False, "Failed: Output contains forbidden markdown blocks."

    if len(t) >= 100:
        return False, f"Failed: Exceeds 100 character limit (Length: {len(t)})."

    # Enforce standard Conventional Commits structure
    cc_pattern = r"^(feat|fix|docs|style|refactor|perf|test|ci|build|chore)(\([^)]+\))?: .+"
    if not re.match(cc_pattern, t, flags=re.IGNORECASE):
        return False, f"Failed: Does not match Conventional Commits format. Output: '{t}'"

    if not re.search(r"async", t, re.IGNORECASE):
        return False, "Failed: Missing required semantic keyword 'async'."

    if not re.search(r"save", t, re.IGNORECASE):
        return False, "Failed: Missing required semantic keyword 'save'."

    return True, "Valid (Conventional Commit)"


SMOKE_CASES = [
    {
        "id": "TC01_GIT_COMMIT",
        "type": "Text Summarization",
        "system": "You are a git automation assistant. Output ONLY a single line summary of the diff. Do not include markdown blocks, greetings, explanations, or quotes.",
        "prompt": "Summarize this diff into a standard conventional commit message:\n--- a/src/Core/Storage.cs\n+++ b/src/Core/Storage.cs\n@@ -12,4 +12,6 @@ public class Storage {\n-    public void Save(string data) => File.WriteAllText(_path, data);\n+    public async Task SaveAsync(string data) {\n+        if (string.IsNullOrEmpty(data)) throw new ArgumentNullException(nameof(data));\n+        await File.WriteAllTextAsync(_path, data);\n+    }",
        "is_json": False,
        "validator": validate_tc01_git_commit,
    },
    {
        "id": "TC02_INTENT_ROUTING",
        "type": "JSON Routing",
        "system": "You are a routing agent. Respond ONLY with a valid JSON object matching this schema: {\"route\": \"string\", \"confidence\": float, \"args\": {}}. No markdown wrapping.",
        "prompt": "Route this user command: 'Drop the local postgres container, rebuild the docker image from scratch, and spin up the cluster on port 5432.'",
        "is_json": True,
        "validator": lambda t: safe_json(t) is not None and "route" in safe_json(t),
    },
    {
        "id": "TT01_WEATHER_TOOL",
        "type": "Tool Call",
        "system": "You are a helpful assistant.",
        "prompt": "Fetch the current weather metrics for Seattle.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Retrieves real-time weather metrics for a given city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city name, e.g., Seattle, WA"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
        "is_json": False,
        "validator": lambda t: "get_weather" in t and "Seattle" in t,
    },
    {
        "id": "TT02_TOOL_CONV_BASIC",
        "type": "Tool Conv",
        "messages": [
            {"role": "system", "content": "You are an agent testing multi-turn tool logic."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "get_capital", "arguments": "{\"country\":\"France\"}"}}]},
            {"role": "tool", "tool_call_id": "call_abc123", "content": "Paris"},
            {"role": "user", "content": "Return the capital as a JSON object, schema: {\"capital\": string}."},
        ],
        "is_json": True,
        "validator": lambda t: safe_json(t) is not None and isinstance(safe_json(t), dict) and safe_json(t).get("capital") == "Paris",
    },
]


def build_messages(tc):
    """Build the message list for a smoke case."""
    if tc.get("messages") is None:
        return [{"role": "system", "content": tc["system"]}, {"role": "user", "content": tc["prompt"]}]
    return copy.deepcopy(tc["messages"])


def evaluate_response(tc, text):
    """Grade an answer to a smoke case: (PASS | WARN | FAIL, reason). JSON cases recover a markdown-fenced answer as WARN."""

    def run_val(t):
        try:
            return verdict(tc["validator"](t))
        except Exception as e:
            return "FAIL", f"Validator crash: {str(e)}"

    if not text or not text.strip():
        return "FAIL", "Empty response (no content and no tool calls)"

    if tc.get("is_json", False):
        raw_parsed = safe_json(text)
        if raw_parsed is not None:
            return run_val(text)

        stripped = strip_code_fence(text)
        strip_parsed = safe_json(stripped)
        if strip_parsed is not None:
            status, msg = run_val(stripped)
            if status == "PASS":
                return "WARN", "Recovered from MarkDown Fence"
            if status == "WARN":
                return "WARN", f"Recovered from MarkDown Fence; {msg}"
            return "FAIL", f"Failed on Strip: {msg}"

        try:
            json.loads(text)
            err_msg = "Unknown structural issue"
        except json.JSONDecodeError as e:
            err_msg = str(e)
        preview = text.strip()[:60].replace('\n', ' ')
        return "FAIL", f"Invalid JSON ({err_msg}). Preview: '{preview}...'"
    return run_val(text)


def regrade(results):
    """Grade the stored results of a smoke state again with evaluate_response, as the live run did (results of other
    cases, errors and SKIPs keep their verdicts). `results` {case id: metrics} is changed in place."""
    from .sanity import SANITY_CASES  # smoke cases are the first of them
    cases = {tc["id"]: tc for tc in SANITY_CASES}
    for cid, m in results.items():
        if cid in cases and not m.get("error") and m.get("eval_result") != "SKIP":
            m["eval_result"], m["fail_reason"] = evaluate_response(cases[cid], m.get("text") or "")
