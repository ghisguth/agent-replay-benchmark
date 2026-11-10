"""Verdict parsing and expectations for the judge suite."""

import re

_LEADING_MARKUP_RE = re.compile(r"^[\s*#_`]+")
_INLINE_MARKUP_RE = re.compile(r"[*_`]")
_FIRST_INT_RE = re.compile(r"\b(\d+)\b")


def parse_verdict(text):
    """The judge's STATUS/ID/DESCRIPTION lines as {status, fdcId, description}."""
    v = {"status": "REJECT", "fdcId": None, "description": ""}
    for raw_line in (text or "").split("\n"):
        line = _INLINE_MARKUP_RE.sub("", _LEADING_MARKUP_RE.sub("", raw_line).strip()).strip()
        if line.startswith("STATUS:"):
            val = line[7:].strip().upper()
            if val in ("ACCEPT", "REJECT", "NONE"):
                v["status"] = val
        elif line.startswith("ID:"):
            m = _FIRST_INT_RE.search(line[3:].strip())
            if m:
                v["fdcId"] = int(m.group(1))
        elif line.startswith("DESCRIPTION:"):
            v["description"] = line[12:].strip()
    return v


def judge_verdict_check(expect, text):
    """(ok, reason) of a judge verdict against the case's expectation (the verdict only, no tool assertion)."""
    v = parse_verdict(text)
    got = "%s/%s" % (v["status"], v["fdcId"])
    if "STATUS:" not in (text or ""):
        return False, "final turn: no STATUS verdict"
    if v["status"] != expect["status"]:
        return False, "got %s, expected %s" % (got, expect["status"])
    if expect["status"] == "ACCEPT" and v["fdcId"] not in expect["fdcIds"]:
        return False, "got %s, expected ACCEPT %s" % (got, "/".join(map(str, expect["fdcIds"])))
    want = expect.get("descriptionContains")
    if want and want.lower() not in (v["description"] or "").lower():
        return False, "description %r lacks %r" % (v["description"], want)
    return True, ""
