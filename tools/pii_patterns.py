"""Secret / personal-data patterns shared by tools/omp_session_pii_scan.py and tools/omp_fixture.py."""
from __future__ import annotations

import collections
import re
from typing import Dict, List, Optional, Tuple

PATTERNS: Dict[str, str] = {
    'secret:openai/anthropic key': r'\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}',
    'secret:github token': r'\bgh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}',
    'secret:aws key': r'\bAKIA[0-9A-Z]{16}\b',
    'secret:slack': r'\bxox[abpr]-[A-Za-z0-9-]{10,}',
    'secret:google key': r'\bAIza[0-9A-Za-z_-]{35}\b',
    'secret:private key': r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
    'secret:jwt': r'\beyJ[A-Za-z0-9_-]{15,}\.eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}',
    'secret:bearer': r'(?i)bearer\s+[A-Za-z0-9._-]{20,}',
    'secret:assign': r'(?i)\b(?:api[_-]?key|secret|password|passwd|token|client_secret|connectionstring)\b["\']?\s*[:=]\s*["\']?(?!\$|<|\{|null|none|true|false|"")[A-Za-z0-9/+_.=-]{12,}',
    'secret:conn string': r'(?i)(?:postgres|mysql|mongodb|redis|amqp)(?:\+\w+)?://[^\s"\']*:[^\s"\'@]+@',
    'secret:hex40+': r'\b[0-9a-f]{40,64}\b',
    'pii:email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    'pii:phone': r'(?<![\d.])\+?1?[ (-]?\d{3}[) -]\d{3}-\d{4}\b',
    'pii:ip': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'pii:ipv6 link': r'fe80::[0-9a-f:]+',
    'id:home path': r'/Users/[A-Za-z0-9._-]+',
    'id:hostname': r'(?i)\b[a-z0-9-]*(?:macbook|mbp|mba|imac)[a-z0-9-]*(?:\.local)?\b',
}


def patterns(names: Optional[List[str]] = None) -> Dict[str, str]:
    """PATTERNS plus an 'id:names' entry when a list of name regexes is given."""
    p = dict(PATTERNS)
    if names:
        p['id:names'] = '(?i)(?:' + '|'.join(names) + ')'
    return p


def scan_text(text: str, pats: Dict[str, str]) -> Tuple[collections.Counter, Dict[str, collections.Counter]]:
    """Return (hit count per pattern name, matched value counts per pattern name)."""
    c: collections.Counter = collections.Counter()
    ex: Dict[str, collections.Counter] = {}
    for name, rx in pats.items():
        for m in re.finditer(rx, text):
            v = m.group(0)
            c[name] += 1
            ex.setdefault(name, collections.Counter())[v[:60]] += 1
    return c, ex
