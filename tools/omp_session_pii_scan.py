#!/usr/bin/env python3
"""Scan omp session files for secrets and personal data before anything is turned into a fixture.

    tools/omp_session_pii_scan.py SESSION [SESSION ...] [--names names.json]

SESSION is a session .jsonl file, or a session directory (its .jsonl plus the files of the sibling directory of
the same name, e.g. subagent transcripts). --names is an optional JSON file with a list of regexes (personal
names, user ids, project paths) that are reported as 'id:names'; keep that file out of the repo.
"""
import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import pii_patterns

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('sessions', nargs='+', metavar='SESSION')
ap.add_argument('--names', metavar='FILE', help='JSON list of regexes for personal names / paths')
args = ap.parse_args()
rxs = []
if args.names:
    with open(args.names) as f:
        rxs = json.load(f)
P = pii_patterns.patterns(rxs)

for k in args.sessions:
    base = k[:-6] if k.endswith('.jsonl') else k
    files = [base + '.jsonl'] + glob.glob(base + '/*')
    c = collections.Counter()
    ex = {}
    for f in files:
        if os.path.isdir(f) or not os.path.exists(f):
            continue
        t = open(f, errors='replace').read()
        c1, ex1 = pii_patterns.scan_text(t, P)
        c.update(c1)
        for name, vals in ex1.items():
            ex.setdefault(name, collections.Counter()).update(vals)
    print(f'\n### {k}  files={len(files)}')
    for name in P:
        if c[name]:
            print(f'  {name:26} {c[name]:6}  e.g. {[x for x, _ in ex[name].most_common(4)]}')
