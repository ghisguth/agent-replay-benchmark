"""Sanity cases (--suite sanity): ten generic, deterministic checks graded by validators alone (no judge model).
The first four are the smoke cases (smoke.SMOKE_CASES, defined once in smoke.py); the rest are defined here.
Grading is smoke.evaluate_response, as for smoke."""

import json
import re

from ..utils import safe_json
from .smoke import SMOKE_CASES


def tool_calls_in(text):
    """[(name, arguments dict)] of the tool calls serialised at the end of an answer's text (transport appends the
    assembled calls as a JSON list). Arguments that are not a JSON object come back as {}."""
    dec = json.JSONDecoder()
    found, i = [], text.find("[")
    while i >= 0:
        try:
            val, end = dec.raw_decode(text, i)
        except ValueError:
            i = text.find("[", i + 1)
            continue
        calls = [c for c in val if isinstance(c, dict) and isinstance(c.get("function"), dict)] if isinstance(val, list) else []
        if calls:
            found = []
            for c in calls:
                args = safe_json(c["function"].get("arguments") or "{}")
                found.append((c["function"].get("name"), args if isinstance(args, dict) else {}))
        i = text.find("[", end)
    return found


def validate_count_stop(text):
    """Exactly '1, 2, 3, 4, 5' (a trailing full stop or newline is fine, nothing else, no counting past 5)."""
    t = text.strip().rstrip(".")
    if t == "1, 2, 3, 4, 5":
        return True, ""
    nums = re.findall(r"\d+", text)
    if nums[:5] == ["1", "2", "3", "4", "5"] and len(nums) > 5:
        return False, "Failed: did not stop after 5 (%d numbers)." % len(nums)
    return False, "Failed: expected '1, 2, 3, 4, 5', got '%s'." % text.strip()[:60]


def validate_two_calls(text):
    calls = tool_calls_in(text)
    cities = sorted(str(a.get("location", "")).split(",")[0].strip().lower() for n, a in calls if n == "get_weather")
    if len(calls) != 2:
        return False, "Failed: expected 2 tool calls in one turn, got %d." % len(calls)
    if cities != ["paris", "tokyo"]:
        return False, "Failed: expected get_weather for Paris and Tokyo, got %s." % cities
    return True, ""


def validate_tool_choice(text):
    calls = tool_calls_in(text)
    if len(calls) != 1:
        return False, "Failed: expected exactly 1 tool call, got %d." % len(calls)
    name, args = calls[0]
    if name != "convert_currency":
        return False, "Failed: called %s instead of convert_currency." % name
    amount, src, dst = args.get("amount"), str(args.get("from", "")).upper(), str(args.get("to", "")).upper()
    try:
        ok = float(amount) == 250 and src == "EUR" and dst == "JPY"
    except (TypeError, ValueError):
        ok = False
    return (True, "") if ok else (False, "Failed: wrong arguments %s." % json.dumps(args)[:80])


def validate_arithmetic(text):
    # 3 crates of 24 apples = 72, 18 rotten -> 54, packed in bags of 6 -> 9 bags, at 4 each = 36
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", " "))
    if not nums:
        return False, "Failed: no number in the answer."
    last = nums[-1].replace(",", "").rstrip(".")
    return (True, "") if last == "36" else (False, "Failed: expected 36, got '%s'." % last)


_FILLER = [
    "The quarterly review covered staffing, budgets and the schedule for the next release.",
    "Maintenance windows are announced on the internal wiki two weeks in advance.",
    "Shipping labels are printed at the warehouse and checked before the truck leaves.",
    "The cafeteria menu rotates weekly and always lists the allergens next to each dish.",
    "Meeting rooms on the third floor are booked through the shared calendar.",
    "Travel requests above the standard limit need a second approval from the department head.",
]
NEEDLE_CODE = "AMBER-4417-HERON"


def _haystack(paragraphs=150):
    out = []
    for i in range(paragraphs):
        out.append("Note %d. %s %s %s" % (i + 1, _FILLER[i % 6], _FILLER[(i * 5 + 2) % 6], _FILLER[(i * 7 + 4) % 6]))
        if i == paragraphs * 2 // 3:
            out.append("Note %d. The vault access code for the archive room is %s; keep it private." % (i + 1, NEEDLE_CODE))
    return "\n".join(out)


def validate_needle(text):
    if NEEDLE_CODE in text:
        return True, ""
    return False, "Failed: the hidden code was not found in the answer."


UNICODE_TEXT = "naïve café — Zürich → 東京 ✓ 🙂 ñ"


def validate_unicode(text):
    t = text.strip().strip('"')
    if t == UNICODE_TEXT:
        return True, ""
    return False, "Failed: not echoed exactly. Got '%s'." % text.strip()[:60]


_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Retrieves real-time weather metrics for a given city.",
        "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}},
                       "required": ["location"]},
    },
}
_TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Returns the current local time in a given IANA timezone.",
        "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]},
    },
}
_CONVERT = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": "Converts an amount of money from one currency to another at the current rate.",
        "parameters": {"type": "object",
                       "properties": {"amount": {"type": "number"}, "from": {"type": "string", "description": "ISO code"},
                                      "to": {"type": "string", "description": "ISO code"}},
                       "required": ["amount", "from", "to"]},
    },
}

EXTRA_CASES = [
    {
        "id": "SC05_COUNT_AND_STOP",
        "type": "Stop Condition",
        "system": "You follow output instructions literally.",
        "prompt": "Count from 1 to 5 as numbers separated by a comma and a space, then stop. Output nothing else.",
        "is_json": False,
        "validator": validate_count_stop,
    },
    {
        "id": "SC06_TWO_TOOL_CALLS",
        "type": "Tool Call",
        "system": "You are a helpful assistant. Call every tool you need in the same turn.",
        "prompt": "What is the weather in Paris and in Tokyo right now?",
        "tools": [_WEATHER],
        "is_json": False,
        "validator": validate_two_calls,
    },
    {
        "id": "SC07_TOOL_CHOICE",
        "type": "Tool Call",
        "system": "You are a helpful assistant.",
        "prompt": "How many Japanese yen is 250 euros?",
        "tools": [_WEATHER, _TIME, _CONVERT],
        "is_json": False,
        "validator": validate_tool_choice,
    },
    {
        "id": "SC08_ARITHMETIC",
        "type": "Reasoning",
        "system": "Answer with the final number only, on the last line.",
        "prompt": ("A shop receives 3 crates of 24 apples. 18 apples are rotten and thrown away. The rest are packed "
                   "in bags of 6, and each bag sells for 4 dollars. How many dollars do all the bags sell for?"),
        "is_json": False,
        "validator": validate_arithmetic,
    },
    {
        "id": "SC09_NEEDLE_8K",
        "type": "Long Context",
        "system": "Answer from the notes only. Reply with the code and nothing else.",
        "prompt": _haystack() + "\n\nWhat is the vault access code for the archive room?",
        "is_json": False,
        "validator": validate_needle,
    },
    {
        "id": "SC10_UNICODE_ECHO",
        "type": "Unicode",
        "system": "You copy text exactly.",
        "prompt": "Repeat the following line exactly, character for character, with no quotes and nothing else:\n" + UNICODE_TEXT,
        "is_json": False,
        "validator": validate_unicode,
    },
]

SANITY_CASES = SMOKE_CASES + EXTRA_CASES
