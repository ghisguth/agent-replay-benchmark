"""Test-case registry: assembles ALL_TEST_CASES and selects subsets by mode."""

from .sanity import SANITY_CASES
from .smoke import SMOKE_CASES

# SMOKE_CASES are the first four of SANITY_CASES: one definition of each case.
ALL_TEST_CASES = list(SANITY_CASES)
SUITE_CASES = {"smoke": SMOKE_CASES, "sanity": SANITY_CASES}

# Default warmup case: small and generic on purpose. Warming up with the real
# first scored test would prime its exact prompt into the KV cache, so that
# test would then run as a spurious ~100% cache hit with no measurable (cold)
# prefill. TC01 is short and standalone.
DEFAULT_WARMUP_ID = "TC01_GIT_COMMIT"

_BY_ID = {t["id"]: t for t in ALL_TEST_CASES}


def get_warmup_case(test_id=None):
    """Return the case used to warm up a target before the scored run (None if the id is unknown)."""
    return _BY_ID.get(test_id or DEFAULT_WARMUP_ID)


def select_cases(test_ids=None, suite="smoke"):
    """The cases of `suite` (smoke or sanity) in order, or only the given ids (in that order, from any suite); unknown
    ids raise ValueError naming them."""
    if test_ids:
        missing = [tid for tid in test_ids if tid not in _BY_ID]
        if missing:
            raise ValueError(", ".join(missing))
        return [_BY_ID[tid] for tid in test_ids]
    return list(SUITE_CASES[suite])
