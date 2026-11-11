"""--compare-state: the baseline file of a run, loaded once and picked once per run target.

load_baseline() reads the file; Baseline.run_for(target, info) picks the block that is the baseline of one run and gives
a BaselineRun (results, header, path, label), which is what the outputs and the display take. A Comparison holds one run
and its baseline over the requests both have; every number of the comparison is computed from it.
"""

import sys

from . import backend
from .state import model_label, read_blocks, same_suite
from .utils import UserError


class ModelResults(dict):
    """The results of one baseline block ({case id: metrics}); `header` is the block's run header, `path` the file it
    was read from, `note` its run note, `complete` whether it ended without being interrupted."""
    header = None
    path = None
    note = None
    complete = False


class BaselineRun:
    """The baseline of one run target: `results` {case id: metrics}, the state `header`, the `path` it was read from and
    the `label` that says what differs from the run (baseline_label)."""

    def __init__(self, results, header, path, label):
        self.results, self.header, self.path, self.label = results, header or {}, path, label

    def get(self, case_id):
        return self.results.get(case_id)

    def common(self, ids):
        """How many of the request ids this baseline has a result for."""
        return sum(1 for i in ids if i in self.results)


class Baseline:
    """A loaded --compare-state file: its blocks by (backend, model), the identity (suite, fixture sha256) of the first,
    and pick(): which block is the baseline of a given run."""

    def __init__(self, path=None):
        self.path = path
        self.models = {}  # (backend, model) -> ModelResults
        self.identity = (None, None)
        self.noted = set()

    def blocks(self):
        return list(self.models.values())

    def pick(self, backend_label, model, quiet=False):
        """The baseline block for a run of `model` on `backend_label`: the block with that backend and model, else one
        of the same model (another server), else the only block of the file (two different runs), else, when the file
        holds several and none matches, the first with one stderr line naming it and the others. None for an empty
        file."""
        exact = self.models.get((backend_label, model))
        if exact is not None:
            return exact
        blocks = self.blocks()
        same_model = [m for m in blocks if (m.header or {}).get("model") == model]
        if same_model:
            return same_model[0]
        if len(blocks) <= 1:
            return blocks[0] if blocks else None
        if not quiet and (backend_label, model) not in self.noted:
            self.noted.add((backend_label, model))
            names = ["%s on %s" % ((m.header or {}).get("model"), (m.header or {}).get("backend")) for m in blocks]
            print("--compare-state %s holds %d runs and none is %s: comparing with the first, %s (others: %s)"
                  % (self.path, len(blocks), model, names[0], "; ".join(names[1:])), file=sys.stderr)
        return blocks[0]

    def run_for(self, target, info):
        """The BaselineRun of `target` (`info`: results.run_info of this run, for the label); None for an empty file."""
        bm = self.pick(target["backend"], target["name"])
        if bm is None:
            return None
        return BaselineRun(bm, bm.header, bm.path, baseline_label(info, bm.header, note=bm.note))


def load_baseline(path):
    """Read a state file as a baseline. A (backend, model) block repeated in the file counts once: the last complete one
    (an interrupted block only stands when no complete one follows or precedes it)."""
    baseline = Baseline(path)
    for i, b in enumerate(read_blocks(path)):
        h = b["header"]
        if i == 0:
            baseline.identity = (h.get("suite"), (h.get("fixture") or {}).get("sha256"))
        m = ModelResults(b["results"])
        m.header, m.path, m.note = h, path, h.get("note")
        m.complete = bool(b["end"]) and not b["end"].get("interrupted")
        prev = baseline.models.get((h["backend"], h["model"]))
        if prev is None or m.complete or not prev.complete:
            baseline.models[(h["backend"], h["model"])] = m
    return baseline


def baseline_identity(baseline):
    """(suite, fixture sha256) of the first block of a loaded baseline; None where the file does not say."""
    return baseline.identity


def check_baseline(baseline, suite, sha, flag="--compare-state"):
    """Raise UserError unless the baseline is the same suite (and fixture) as this run. Case ids alone do not make two
    runs comparable. A baseline that records no suite is accepted."""
    b_suite, b_sha = baseline_identity(baseline)
    if b_suite and suite and not same_suite(b_suite, suite):
        raise UserError("%s: the baseline %s is suite %s, this run is %s" % (flag, baseline.path, b_suite, suite))
    if b_sha and sha and b_sha != sha:
        raise UserError("%s: the baseline %s was run on another fixture (%s, this run %s)"
                        % (flag, baseline.path, b_sha[:8], sha[:8]))


def _start(h):
    return (h.get("started_at") or "")[:16].replace("T", " ")


def baseline_label(info, h, name=None, note=None):
    """What identifies a baseline run next to this one, only by what differs: `Splash 1.6.0-m1 · mlx-community/Qwen3.8-27B-4bit`
    (server kind and version, model, run note); same server: just the model; same server and model: the note, else the
    host when that is the only difference, else its start time. `info`: run_info of this run; `h`: the baseline's header."""
    h = h or {}
    server = backend.label(h["server"]) if h.get("server") else None
    model = name or model_label(h.get("model"), h.get("model_info"))
    this_model = model_label(info.get("model"), info.get("model_info"))
    note = note if note is not None else h.get("note")
    parts = []
    if server and server != info.get("server"):
        parts.append(server)
    if model and model != this_model:
        parts.append(model)
    if note and note != info.get("note"):
        parts.append(note)
    if not parts:
        if h.get("backend") and h["backend"] != info.get("backend"):
            parts.append(h["backend"])  # the host is the only difference
        else:
            parts.append(("started " + _start(h)).strip() if _start(h) else "baseline")
    return " · ".join(parts)


class Comparison:
    """One run against its baseline over the requests both have a result for. `this` and `base` are comparison entries
    (compare.entry: "label", "results" {case id: metrics}); `ids` the common request ids in the run's order, `this_results`
    and `base_results` their results; `n_this` / `n_base` the sizes before narrowing."""

    def __init__(self, this, base):
        self.this, self.base = this, base
        this_dict = this["results"] if isinstance(this["results"], dict) else {r["id"]: r for r in this["results"] if isinstance(r, dict) and "id" in r}
        base_dict = base["results"] if isinstance(base["results"], dict) else {r["id"]: r for r in base["results"] if isinstance(r, dict) and "id" in r}
        self.ids = [i for i in this_dict if i in base_dict]
        self.this_results = [this_dict[i] for i in self.ids]
        self.base_results = [base_dict[i] for i in self.ids]
        self.n_this, self.n_base = len(this_dict), len(base_dict)

    def __bool__(self):
        return bool(self.ids)

    @property
    def partial(self):
        """True when the run or its baseline has requests the other lacks."""
        return len(self.ids) != self.n_this or len(self.ids) != self.n_base
