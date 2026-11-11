"""State persistence for benchmark runs: JSONL, one file per model, appended after every case.

File layout (one JSON object per line):
  {"type": "run", ...}      header, first line of a model's block: schema, suite, fixture {file, sha256}, started_at,
                            endpoint, backend (host:port label), model, n_ctx, server (bench/backend.py's
                            {kind, version, build, details}), request {temperature, reasoning_effort, thinking}, git_commit, argv, and note (--run-note, e.g. "MTP": what the
                            server's settings were when the header cannot tell, shown next to the model everywhere)
  {"type": "result", ...}   one per case, written and flushed as soon as the case is graded: id, the runner's metrics,
                            content, tool_calls, finish_reason, reasoning_chars / reasoning_tokens (never the thinking
                            text), and the server's own telemetry objects raw: usage, timings (llama.cpp style),
                            metrics (Splash: speculation, request_latency, cache), kapitsa
  {"type": "resume", ...}   written by --resume when it continues the block: at, git_commit, server, argv of the new
                            invocation. More result lines follow, then a new end. The block stays one block: a
                            result for an id seen before replaces the earlier one (the last wins) and a resume clears
                            the earlier end (an interrupted run that was continued is no longer interrupted)
  {"type": "route", ...}    endpoint: the chat route changed after the header was written (LM Studio /api/v0 to /v1
                            fallback); the loader takes it as the header's endpoint
  {"type": "end", ...}      optional last line of a block: finished_at, cases, counts per verdict, interrupted (true when
                            the run was stopped by Ctrl+C or an error) and reason; a state without it (killed run)
                            loads as well

A crash keeps every line written so far; a torn last line is skipped on load. A live run writes one file per model;
a file with several blocks (older runs) still loads, each block starting with its own header.

Only this JSONL format (schema 1) is read; any other file is refused with an error.
Loaded results keep the runner's in-memory shape: "text" is rebuilt as content plus the tool calls as a JSON array.

  read_blocks()  : the blocks of a file, results re-graded with today's grading
  load_state()   : (raw_results, ordered_model_keys), the structures the runner builds
  (bench/baseline.py loads a file as a --compare-state baseline.)
"""

import atexit
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone

from . import device, paths, telemetry
from . import results as R
from .utils import UserError

SCHEMA = 1

REPO = paths.ROOT


# ---------------------------------------------------------------------------
# Header, record shapes and file naming
# ---------------------------------------------------------------------------

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_SHA_CACHE = {}


def file_sha256(path):
    """The sha256 hex digest of a file, cached by (resolved path, mtime, size)."""
    try:
        ap = os.path.abspath(path)
        st = os.stat(ap)
        key = (ap, st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key and key in _SHA_CACHE:
        return _SHA_CACHE[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    if key:
        _SHA_CACHE[key] = digest
    return digest


def git_commit(repo=REPO):
    """Short commit of this checkout, or None (no git, not a checkout, slow)."""
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=2)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def sanitize_argv(argv, cwd=None):
    """The command line as recorded in a state file: `--api-key VALUE` / `--api-key=VALUE` dropped, and the
    `--state-file` path made relative to the current directory when under it, else just its basename."""
    if argv is None:
        return None
    cwd = os.path.realpath(cwd or os.getcwd())

    def clean(path):
        full = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
        if full == cwd or full.startswith(cwd + os.sep):
            return os.path.relpath(full, cwd)
        return os.path.basename(path)

    out, skip = [], None
    for t in argv:
        if skip == "drop":
            skip = None
        elif skip == "path":
            out.append(clean(t))
            skip = None
        elif t == "--api-key":
            skip = "drop"
        elif t.startswith("--api-key="):
            pass
        elif t == "--state-file":
            out.append(t)
            skip = "path"
        elif t.startswith("--state-file="):
            out.append("--state-file=" + clean(t.split("=", 1)[1]))
        else:
            out.append(t)
    return out


def make_header(suite, target, request=None, fixture_path=None, argv=None, selection=None, started_at=None,
                preset=None, note=None, reference=None):
    """The run header of one (endpoint, model) target."""
    fixture = None
    if fixture_path:
        try:
            fixture = {"file": os.path.basename(fixture_path), "sha256": file_sha256(fixture_path)}
        except OSError:
            fixture = {"file": os.path.basename(fixture_path), "sha256": None}
    url = target.get("url", "")
    header = {
        "type": "run",
        "schema": SCHEMA,
        "suite": suite,
        "fixture": fixture,
        "started_at": started_at or now_iso(),
        "endpoint": url[:-len("/chat/completions")] if url.endswith("/chat/completions") else url,
        "backend": target["backend"],
        "model": target["name"],
        "n_ctx": target.get("n_ctx"),
        "server": target.get("backend_info"),
        "device": target.get("device"),
        "client_device": device.local(),
        "request": {"temperature": None, "reasoning_effort": None, "thinking": None, **(request or {})},
        "git_commit": git_commit(),
        "argv": sanitize_argv(argv),
    }
    if selection:
        header["selection"] = selection
    if preset:
        header["preset"] = preset
    if note:
        header["note"] = note
    if reference:
        header["reference"] = reference
    if target.get("model_info"):
        header["model_info"] = target["model_info"]
    return header


_SLUG_RE = re.compile(r"[^A-Za-z0-9._]+")


def slug(text, limit=80):
    """Filesystem-safe form of a name: letters, digits, '.', '_', other runs become '-'."""
    s = _SLUG_RE.sub("-", str(text or "")).strip("-.")
    return s[:limit].rstrip("-.") or "x"


def with_note(label, note):
    """`label · note`, or the label alone without a note."""
    return "%s \u00b7 %s" % (label, note) if note else label


def _decoded(model, model_info):
    """The model id, or model_info["label"] (ORG/NAME of an MLX server's hash or path id, never a local path)."""
    label = model_info.get("label") if isinstance(model_info, dict) else None
    return label if isinstance(label, str) and label else model


def model_label(model, model_info=None):
    """The model's display label: appends the quant (`qwen3.8:27b-256k · Q4_K_M`) when not already in the model id."""
    if not model:
        return model
    model = _decoded(model, model_info)
    quant = (model_info or {}).get("quant") if isinstance(model_info, dict) else None
    if quant and isinstance(quant, str) and quant.lower() not in model.lower():
        return "%s \u00b7 %s" % (model, quant)
    return model


def model_file_slug(model, model_info=None):
    """Filesystem slug for the model: appends quant when not already in model id."""
    model = _decoded(model, model_info)
    quant = (model_info or {}).get("quant") if isinstance(model_info, dict) else None
    if quant and isinstance(quant, str) and model and quant.lower() not in model.lower():
        return slug("%s-%s" % (model, quant))
    return slug(model)


def default_path(states_dir, info, model, suite, when=None, request=None, dev=None, preset=None, note=None, model_info=None):
    """states/<backend>-<version>[_<device>]_<model-slug>_<suite>[_t<temperature>][_e-<effort>][_think-on|off]_<YYYYmmdd-HHMMSS>.jsonl (never an
    existing file; `_<note>` after the sampling parts, slugified and lower-cased: ..._t0.6_mtp_...). A run with a preset is named <suite>-<preset> (coding-large) instead of <suite>, unless the preset has the suite's name (judge, smoke). The sampling parts appear only when --temperature / --reasoning-effort / --thinking were given."""
    info = info or {}
    note = note or (model_info.get("mode") if isinstance(model_info, dict) else None)  # mlx-dspark: _dflash_ / _dspark_
    prefix = slug(info.get("kind") or "generic") + ("-" + slug(info["version"]) if info.get("version") else "")
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    request = request or {}
    sampling = ""
    if request.get("temperature") is not None:
        sampling += "_t%s" % slug("%g" % request["temperature"])
    if request.get("reasoning_effort") is not None:
        sampling += "_e-%s" % slug(request["reasoning_effort"])
    if request.get("thinking") is not None:
        sampling += "_think-%s" % slug(request["thinking"])
    if note:
        sampling += "_%s" % slug(note).lower()
    short = device.short(dev.get("name")) if dev else None
    if short:
        prefix += "_" + short
    m_slug = model_file_slug(model, model_info)
    base = os.path.join(states_dir, "%s_%s_%s%s_%s" % (prefix, m_slug, slug("%s-%s" % (suite, preset) if preset and preset != suite else suite),
                                          sampling, stamp))
    path, n = base + ".jsonl", 1
    while os.path.exists(path):
        n += 1
        path = "%s-%d.jsonl" % (base, n)
    return path


def _strip_calls(text, calls):
    """The content of a combined text (content + '\\n' + tool calls as a JSON array)."""
    tail = json.dumps(calls)
    if text.endswith(tail):
        return text[:-len(tail)].rstrip("\n")
    return text


def result_record(metrics):
    """The state line of one finished case: the runner's metrics with content next to the calls, no combined text."""
    rec = {"type": "result", "id": metrics.get("id")}
    for k, v in metrics.items():
        if k not in ("type", "id", "text"):
            rec[k] = v
    if "content" not in rec:
        text = metrics.get("text") or ""
        rec["content"] = _strip_calls(text, metrics["tool_calls"]) if metrics.get("tool_calls") else text
    return rec


def resume_record(target, argv=None, note=None):
    """The line --resume appends before the new results: when, with which checkout and server, and the new command.
    `note` (the run note of this invocation) replaces the header's when it differs: a presentation option."""
    rec = {"type": "resume", "at": now_iso(), "git_commit": git_commit(), "server": target.get("backend_info"),
           "argv": sanitize_argv(argv)}
    if note:
        rec["note"] = note
    return rec


def end_record(raw_results, interrupted=None):
    """The last line of a block. `interrupted` is the reason a run stopped early (Ctrl+C, an error), else None."""
    counts = {}
    for m in raw_results.values():
        counts[m.get("eval_result")] = counts.get(m.get("eval_result"), 0) + 1
    rec = {"type": "end", "finished_at": now_iso(), "cases": len(raw_results), "counts": counts,
           "interrupted": bool(interrupted)}
    if interrupted:
        rec["reason"] = str(interrupted)
    return rec


_ACTIVE_WRITERS = set()
_WRITERS_LOCK = threading.Lock()


def _register_writer(w):
    with _WRITERS_LOCK:
        _ACTIVE_WRITERS.add(w)


def _unregister_writer(w):
    with _WRITERS_LOCK:
        _ACTIVE_WRITERS.discard(w)


def flush_all():
    """Flush and wait for all active StateWriter instances."""
    with _WRITERS_LOCK:
        writers = list(_ACTIVE_WRITERS)
    for w in writers:
        try:
            w.close()
        except Exception:
            pass


atexit.register(flush_all)


class StateWriter:
    """Writes JSON lines to a state file, appended, flushed and fsynced after each record."""

    def __init__(self, path):
        self.path = path
        dirname = os.path.dirname(self.path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        _register_writer(self)

    def write(self, record):
        if isinstance(record, (dict, list)):
            line = json.dumps(record, ensure_ascii=False) + "\n"
        elif isinstance(record, str):
            line = record if record.endswith("\n") else record + "\n"
        else:
            line = json.dumps(record, ensure_ascii=False) + "\n"

        with self._lock:
            if self._file is None:
                raise RuntimeError("write to closed StateWriter")
            self._file.write(line)
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except OSError:
                pass

    def flush(self, timeout=None):
        with self._lock:
            if self._file is not None:
                self._file.flush()

    def close(self):
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
        _unregister_writer(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _metrics_from_record(rec, header=None):
    """A result line back into the runner's metrics dict (rebuilds "text")."""
    m = {k: v for k, v in rec.items() if k != "type"}
    if "text" not in m:
        content = m.get("content") or ""
        calls = m.get("tool_calls")
        if calls:
            tail = json.dumps(calls)
            m["text"] = tail if not content.strip() else content + "\n" + tail
        else:
            m["text"] = content
    # llama.cpp / Splash prefill speed is recomputed from the stored timings (both servers send cache_n; MTPLX does not
    # and keeps its own figure). All-zero native timings count as missing (no 0.0 speed, no fake pause), and a stream
    # the server ended with finish_reason "error" is a server error, not an empty answer.
    t = m.get("timings")
    if isinstance(t, dict) and "cache_n" in t and t.get("prompt_per_second") is not None:
        m["native_prefill_tps"] = telemetry.timings_prefill_tps(t, m.get("prompt_tokens") or 0, m.get("cached_tokens") or 0)
    native = [m.get(k) for k in ("native_ttft_ms", "native_prefill_tps", "native_decode_tps")]
    telemetry.drop_nonpositive_native(m)
    if any(v is not None for v in native) and m.get("native_ttft_ms") is None:
        m.pop("server_paused_ms", None)
    if m.get("finish_reason") == "error" and not m.get("error"):
        m["error"] = "server: stream ended with finish_reason error"
        m["eval_result"], m["fail_reason"] = "FAIL", m["error"]
    return m


def _target(backend, name, url, n_ctx, key, header=None):
    t = {"key": key, "backend": backend, "name": name, "url": url, "n_ctx": n_ctx}
    if header is not None:
        t["header"] = header
        if header.get("model_info"):
            t["model_info"] = header["model_info"]
        if header.get("server"):
            t["backend_info"] = header["server"]
        if header.get("device"):
            t["device"] = header["device"]
    return t


FIXTURES_DIR = paths.FIXTURES_DIR
_fixture_cache = {}  # path -> (sha256, fixture, {case id: (position, entry)}); the last one only


def _fixture_lookup(header):
    """Look up the fixture recorded in header against FIXTURES_DIR.
    Returns (status, name, found):
      - status: 'none' (no fixture in header, e.g. smoke), 'ok' (fixture found and sha matched),
        or 'mismatch' (fixture recorded but file missing or sha differs).
      - name: the fixture file name recorded in header, or None.
      - found: (fixture, cases) if status == 'ok', else None.
    """
    fx = (header or {}).get("fixture")
    if not isinstance(fx, dict) or not fx.get("file"):
        return "none", None, None
    name = fx.get("file")
    sha = fx.get("sha256")
    path = os.path.join(FIXTURES_DIR, name)
    if not sha or not os.path.isfile(path):
        return "mismatch", name, None
    hit = _fixture_cache.get(path)
    if hit is not None and hit[0] == sha:
        return "ok", name, (hit[1], hit[2])
    if file_sha256(path) != sha:
        return "mismatch", name, None
    from .cases import replay as replay_cases
    try:
        fixture = replay_cases.load_fixture(path)
    except (OSError, ValueError):
        return "mismatch", name, None
    cases = {replay_cases.case_id(pos, e): (pos, e) for pos, e in enumerate(fixture["turns"], 1)}
    _fixture_cache.clear()
    hit = _fixture_cache[path] = (sha, fixture, cases)
    return "ok", name, (hit[1], hit[2])


def fixture_missing(header):
    """The file name of the fixture a state header records when it is not among the shipped ones (FIXTURES_DIR), else
    None (also for smoke states and headers that record no fixture). Does not check the sha256."""
    name = ((header or {}).get("fixture") or {}).get("file")
    if name and not same_suite((header or {}).get("suite"), "smoke") and not os.path.isfile(os.path.join(FIXTURES_DIR, name)):
        return name
    return None


def regrade(block):
    """Grade the results of a loaded block again with today's grading, so a state written under older grading shows
    current verdicts everywhere (report, --compare-state, comparison). A replay block needs the shipped fixture it was run
    on (same name and sha256), a smoke or sanity block their cases; SKIP and error results keep their stored verdicts.
    Returns (failed, first_error, changed_fixture): how many results could not be graded and why, and the name of the
    fixture when it changed since the run (the stored verdicts stay)."""
    header = block.get("header") or {}
    results = block.get("results", {})
    if same_suite(header.get("suite"), "smoke") or same_suite(header.get("suite"), "sanity"):
        from .cases import smoke
        smoke.regrade(results)
        return 0, None, None
    status, name, found = _fixture_lookup(header)
    if status == "mismatch":
        has_regradeable = any(not m.get("error") and m.get("eval_result") != "SKIP" for m in results.values())
        return 0, None, name if has_regradeable else None
    if not found:
        return 0, None, None
    from .cases import grading
    from .cases import replay as replay_cases
    fixture, cases = found
    failed_count = 0
    first_error = None
    for cid, m in results.items():
        if cid not in cases or m.get("error") or m.get("eval_result") == "SKIP":
            continue
        try:
            m["eval_result"], m["fail_reason"] = grading.grade(replay_cases.make_case(fixture, *cases[cid]), m)
        except Exception as e:
            failed_count += 1
            if first_error is None:
                first_error = str(e).strip().replace("\n", " ") or type(e).__name__
    return failed_count, first_error, None


def read_blocks(path):
    """The blocks of a state file as [{"header": dict, "results": {id: metrics}, "end": dict or None, "resumes": [...]}],
    results re-graded (regrade). One block per header line; a torn or unparsable line is skipped. Raises UserError for a
    file that cannot be read or is not a schema 1 JSONL state."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise UserError("cannot read state file %s: %s" % (path, e.strerror or e))
    except UnicodeDecodeError:
        raise UserError("%s is not a state file (not text)" % path)
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    try:
        head = json.loads(first)
    except ValueError:
        head = None
    if not (isinstance(head, dict) and head.get("type") == "run"):
        raise UserError("%s is not a state file: expected JSONL starting with a run header line "
                        "({\"type\": \"run\", \"schema\": 1, ...}); re-run to produce a new one" % path)
    blocks = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except ValueError:
            continue  # a line torn by a crash
        if not isinstance(rec, dict):
            continue
        kind = rec.get("type")
        if kind == "run":
            blocks.append({"header": rec, "results": {}, "end": None, "resumes": []})
        elif kind == "resume" and blocks:
            blocks[-1]["resumes"].append(rec)
            if rec.get("note"):  # the last note given wins
                blocks[-1]["header"] = dict(blocks[-1]["header"], note=rec["note"])
            blocks[-1]["end"] = None  # continued: the block's end is the one written after this line
        elif kind == "route" and blocks and rec.get("endpoint"):  # the chat route changed after the header was written
            blocks[-1]["header"] = dict(blocks[-1]["header"], endpoint=rec["endpoint"])
        elif kind == "result" and blocks and rec.get("id") is not None:
            blocks[-1]["results"][rec["id"]] = _metrics_from_record(rec, header=blocks[-1]["header"])
        elif kind == "end" and blocks:
            blocks[-1]["end"] = rec
    changed, total_failed, first_error = set(), 0, None
    for b in blocks:
        R.fill_prev_prompts(b["results"].values())
        failed, err, fixture_changed = regrade(b)
        total_failed += failed
        first_error = first_error or err
        if fixture_changed and fixture_changed not in changed:
            changed.add(fixture_changed)
            print(f"warning: {path}: fixture {fixture_changed} changed since this run; showing stored verdicts", file=sys.stderr)
    if total_failed > 0:
        print(f"warning: {path}: {total_failed} results could not be re-graded ({first_error})", file=sys.stderr)
    return blocks


def load_resume(path):
    """(header, results, resumes) of the single-model state file `path`, for --resume. A UserError for a file that
    holds several run blocks or no recorded command (no endpoint in its header)."""
    blocks = read_blocks(path)
    if len(blocks) != 1:
        raise UserError("resume works on single-model state files: %s holds %d run blocks" % (path, len(blocks)))
    b = blocks[0]
    if not b["header"].get("endpoint"):
        raise UserError("%s records no endpoint and cannot be resumed" % path)
    return b["header"], b["results"], b["resumes"]


def load_state(state_path: str):
    """Load a state file and return (raw_results, ordered_model_keys), as run_benchmark() builds them.

    A (backend, model) block repeated in one file gets a "#2" key suffix, so nothing is overwritten.
    """
    raw_results, ordered_keys = {}, []
    for b in read_blocks(state_path):
        h = b["header"]
        key = "%s::%s" % (h["backend"], h["model"])
        n = 1
        while key in raw_results:
            n += 1
            key = "%s::%s#%d" % (h["backend"], h["model"], n)
        url = (h["endpoint"] + "/chat/completions") if h.get("endpoint") else ""
        t = _target(h["backend"], h["model"], url, h.get("n_ctx"), key, header=h)
        t["end"] = b["end"]
        ordered_keys.append(t)
        raw_results[key] = b["results"]
    return raw_results, ordered_keys


# Suite names that were renamed: the header of an older state says the old one. "conv-3" (the recorded omp coding
# session) is now "coding"; both are the same suite and the same fixture, so states of either compare and resume.
SUITE_ALIASES = {"conv-3": "coding", "single": "smoke", "3": "coding", "4": "judge"}


def same_suite(a, b):
    """Whether two header suite names are the same suite (renames and the smoke/single pair included)."""
    return SUITE_ALIASES.get(a, a) == SUITE_ALIASES.get(b, b)
