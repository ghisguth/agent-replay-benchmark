"""Add runs to the reference set (or, with --sample, to the shipped samples): scrub host names, scan for personal
data, copy to fixtures/refs, rebuild checks.

Usage: add_ref.py [--sample] [--dry-run] [--force] [--hosts FILE] STATE ...
  STATE ...   state files (JSONL, from states/)
  --sample    a sample, not a reference: the same scrub and scan, the copy goes to fixtures/samples/ and the checks are
              not rebuilt (samples are never graded against or used for grading); same states/ symlink convention

Per file: host names (the header's endpoint, backend and argv URLs other than localhost, plus any NAME.local / .lan /
.home / .localdomain token) are replaced everywhere, with or without a trailing dot, by a placeholder. The alias table
(host -> placeholder) is read from sources/hosts.json (gitignored: host names are personal data and are never
committed); without it a placeholder is host-<first 8 hex of sha1(host)>. With a table, a new host gets host-<n>, the
next free number, and is added to it. The scrubbed text is scanned with tools/pii_patterns.py; only ALLOWED hits pass,
anything else is printed and the file is not added (exit 1). Then the file is copied to fixtures/refs/ (refused when it
is already there, unless --force) and fixtures/conv-3.checks.json is rebuilt (tools/build_checks.py). --dry-run stops
after the scan and prints what would be replaced.

Afterwards a regular source file under states/ is replaced by a relative symlink to its fixtures/refs (or samples) copy and the
original is moved to states/review/refs-originals/ (not with --dry-run, for a symlink, or for a file outside states/).
"""
import argparse
import hashlib
import json
import os
import re
import sys
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_checks  # noqa: E402
import pii_patterns  # noqa: E402

REFS_DIR = os.path.join(ROOT, "fixtures", "refs")
SAMPLES_DIR = os.path.join(ROOT, "fixtures", "samples")
STATES_DIR = os.path.join(ROOT, "states")
HOSTS_FILE = os.path.join(ROOT, "sources", "hosts.json")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
LOCAL_NAME_RX = re.compile(r"\b[A-Za-z0-9-]+\.(?:local|lan|home|localdomain)\b\.?")
PLACEHOLDER_RX = re.compile(r"^host-")

# Allowed scan hits, by pattern name: a predicate on the matched value.
FOOD_WORDS = {"sambal"}
ALLOWED = {
    "secret:hex40+": lambda v: True,  # the fixture sha256, build ids
    "id:home path": lambda v: v == "/Users/standard.analyst",  # the fixture's redacted home
    "id:hostname": lambda v: bool(PLACEHOLDER_RX.match(v)) or v.lower() == "macbook" or v.lower() in FOOD_WORDS,
}


def header_hosts(text):
    """Host names of the run header (first line): endpoint, backend, argv URLs; localhost and placeholders left out."""
    try:
        header = json.loads(text.split("\n", 1)[0])
    except ValueError:
        return set()
    urls = [header.get("endpoint"), header.get("backend")]
    urls += [a for a in header.get("argv") or [] if isinstance(a, str) and "://" in a]
    hosts = set()
    for u in urls:
        if not isinstance(u, str) or not u:
            continue
        try:
            h = urlparse(u if "://" in u else "//" + u).hostname
        except ValueError:
            continue
        if h:
            hosts.add(h.lower().rstrip("."))
    return hosts


def collect_hosts(text):
    hosts = header_hosts(text)
    hosts |= {m.group(0).rstrip(".").lower() for m in LOCAL_NAME_RX.finditer(text)}
    return {h for h in hosts if h not in LOCAL_HOSTS and not PLACEHOLDER_RX.match(h)}


def load_aliases(path):
    """(alias table or None when the file is missing)."""
    try:
        with open(path, encoding="utf-8") as f:
            return {k.lower(): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        return None


def save_aliases(path, aliases):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(aliases, f, indent=2, sort_keys=True)
        f.write("\n")


def assign(host, aliases):
    """Placeholder of host; with a table (not None) a new host takes the next free host-<n> and is added to it."""
    if aliases is None:
        return "host-" + hashlib.sha1(host.encode()).hexdigest()[:8]
    if host not in aliases:
        used = set(aliases.values())
        n = 1
        while "host-%d" % n in used:
            n += 1
        aliases[host] = "host-%d" % n
    return aliases[host]


def scrub(text, aliases):
    """(scrubbed text, {host: placeholder}); aliases (a dict, or None) gets new hosts added."""
    found = {}
    for host in sorted(collect_hosts(text), key=len, reverse=True):
        placeholder = assign(host, aliases)
        rx = re.compile(r"(?<![A-Za-z0-9.-])" + re.escape(host) + r"\.?(?![A-Za-z0-9-])", re.I)
        text, n = rx.subn(placeholder, text)
        if n:
            found[host] = placeholder
    return text, found


def blocked_hits(text):
    """[(pattern name, value, count)] of scan hits that are not allowed."""
    _, examples = pii_patterns.scan_text(text, pii_patterns.patterns())
    out = []
    for name, values in examples.items():
        ok = ALLOWED.get(name)
        out += [(name, v, n) for v, n in values.items() if not (ok and ok(v))]
    return out


def accepted_of(checks_path):
    try:
        with open(checks_path, encoding="utf-8") as f:
            return json.load(f).get("accepted") or {}
    except (FileNotFoundError, ValueError):
        return {}


def link_source(src, dest, states_dir):
    """Replace the regular file src (under states_dir) by a relative symlink to dest; the original goes to
    states_dir/review/refs-originals/. Returns the new location of the original, or None when src is skipped."""
    real_states = os.path.realpath(states_dir)
    if os.path.islink(src) or not os.path.isfile(src) or os.path.dirname(os.path.realpath(src)) != real_states:
        return None
    keep_dir = os.path.join(states_dir, "review", "refs-originals")
    os.makedirs(keep_dir, exist_ok=True)
    kept = os.path.join(keep_dir, os.path.basename(src))
    os.replace(src, kept)
    os.symlink(os.path.relpath(dest, os.path.dirname(os.path.abspath(src))), src)
    return kept


def main(argv=None):
    ap = argparse.ArgumentParser(description="Add runs to the reference set (fixtures/refs) or, with --sample, to fixtures/samples.")
    ap.add_argument("states", nargs="+", help="state files")
    ap.add_argument("--dry-run", action="store_true", help="scrub and scan only; print what would be replaced")
    ap.add_argument("--force", action="store_true", help="overwrite a file already in the target directory")
    ap.add_argument("--hosts", default=HOSTS_FILE, help="host alias table (default: sources/hosts.json, gitignored)")
    ap.add_argument("--sample", action="store_true",
                    help="add to fixtures/samples/ instead of fixtures/refs/; the checks are not rebuilt")
    ap.add_argument("--refs-dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--states-dir", default=STATES_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--fixture", default=build_checks.DEFAULT_FIXTURE)
    ap.add_argument("--checks", help="sidecar to rebuild (default: next to the fixture)")
    args = ap.parse_args(argv)
    if args.refs_dir is None:   # --refs-dir names the target directory, whichever kind it is
        args.refs_dir = SAMPLES_DIR if args.sample else REFS_DIR

    aliases = load_aliases(args.hosts)
    plans = []
    for path in args.states:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        text, found = scrub(text, aliases)
        if found:
            for host, ph in sorted(found.items()):
                print("%s: %s -> %s" % (os.path.basename(path), host, ph))
        else:
            print("%s: no host names to replace" % os.path.basename(path))
        try:
            argv_in_header = json.loads(text.split("\n", 1)[0]).get("argv") or []
        except ValueError:
            argv_in_header = []
        if any(isinstance(a, str) and (a == "--api-key" or a.startswith("--api-key=")) for a in argv_in_header):
            print("%s: BLOCKED header argv contains --api-key (remove the key from the header first)"
                  % os.path.basename(path), file=sys.stderr)
            print("not added: %s" % path, file=sys.stderr)
            return 1
        bad = blocked_hits(text)
        if bad:
            for name, v, n in bad:
                print("%s: BLOCKED %s %r x%d" % (os.path.basename(path), name, v, n), file=sys.stderr)
            print("not added: %s" % path, file=sys.stderr)
            return 1
        dest = os.path.join(args.refs_dir, os.path.basename(path))
        if not args.dry_run and os.path.exists(dest) and not args.force:
            print("%s already exists (use --force to overwrite)" % dest, file=sys.stderr)
            return 1
        plans.append((dest, text, path))
    if args.dry_run:
        return 0

    from bench.cases import replay  # noqa: E402
    checks_path = args.checks or replay.checks_path(args.fixture)
    before = accepted_of(checks_path)
    os.makedirs(args.refs_dir, exist_ok=True)
    for dest, text, _ in plans:
        with open(dest, "w", encoding="utf-8") as f:
            f.write(text)
        try:
            note = json.loads(text.split("\n", 1)[0]).get("note")  # the run note stays in the header, as written
        except ValueError:
            note = None
        print("added %s%s" % (dest, " (note: %s)" % note if note else ""))
    if aliases is not None:
        save_aliases(args.hosts, aliases)
    rc = 0 if args.sample else build_checks.main(["--fixture", args.fixture, "--checks", checks_path] + sorted(
        os.path.join(args.refs_dir, f) for f in os.listdir(args.refs_dir) if f.endswith(".jsonl")))
    if rc == 0:
        for dest, _, src in plans:
            kept = link_source(os.path.abspath(src), dest, args.states_dir)
            if kept:
                print("linked %s -> %s (original moved to %s)" % (src, os.path.relpath(dest, os.path.dirname(os.path.abspath(src))), kept))
    if args.sample:
        print("sample added, checks not rebuilt")
        return rc
    after = accepted_of(checks_path)
    changed = sum(1 for k in set(before) | set(after) if before.get(k) != after.get(k))
    print("%d accepted entries changed (of %d)" % (changed, len(after)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
