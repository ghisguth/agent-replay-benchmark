"""Benchmark execution: querying models, grading, and orchestration of a live run."""

import contextlib
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

from . import backend, perf, report
from . import fmt as F
from . import reference as reference_mod
from . import results as R
from . import state as state_mod
from .baseline import check_baseline, load_baseline
from .cases import get_warmup_case, grading, select_cases, smoke
from .cases import replay as replay_cases
from .colors import Clr
from .discovery import discover_and_report
from .display import display_state, finish_target, print_final, resume_hint
from .endpoints import resolve_endpoints
from .outputs import _chatter, output
from .progress import ProgressAnimation
from .transport import PromptProgress, post_with_fallback, query_with_retry, send_progress, send_with_progress_fallback
from .utils import UserError

# What a case that sent nothing, or failed, has instead of measurements.
NO_METRICS = {"ttft_ms": 0, "total_ms": 0, "tokens": 0, "native_ttft_ms": None, "native_prefill_tps": None,
              "native_decode_tps": None}


def live_state(tc, hist, policy, target):
    """Initial spinner state of a live case: {"pp_expected": True} when the server will send real prefill progress (the
    line shows 0% until the first chunk, never an estimate), else the prefill estimate (possibly {}). MTPLX ("snapshot")
    keeps the estimate until its first chunk-phase reading arrives."""
    return {"pp_expected": True} if policy.sends(target) in (True, "header") else prefill_estimate(tc, hist)


# Estimated prefill for the live line when the server sends no progress. Uncached tokens = the request's prompt tokens
# (known from the fixture, as for the skip check) minus the expected cache: the previous finished turn of the same
# stream and epoch is its cached prefix (a turn extends it), a compaction or a first turn expects none. Speed = the
# median native prefill tok/s of the last 5 finished requests of this run; without any, there is no estimate.
ESTIMATE_WINDOW = 5


def prefill_estimate(tc, hist):
    """{"est_tokens", "est_tps"} for the live state of a replay case, or {} when it cannot be estimated."""
    if tc.get("fixture") is None or not hist["tps"]:
        return {}
    need = replay_cases.need_tokens(tc)
    cached = 0
    if tc.get("kind") != "compaction":
        cached = min(need, hist["prev"].get((tc.get("stream"), tc.get("epoch")), 0))
    median = F.p50(hist["tps"][-ESTIMATE_WINDOW:])
    return {"est_tokens": max(0, need - cached), "est_tps": median}


def note_finished(tc, metrics, hist):
    """Feed a finished case into the estimator's history (errors and SKIPs add nothing)."""
    if "error" in metrics or metrics.get("eval_result") == "SKIP" or tc.get("fixture") is None:
        return
    key = (tc.get("stream"), tc.get("epoch"))
    tokens = metrics.get("prompt_tokens") or replay_cases.need_tokens(tc)
    tps = metrics.get("native_prefill_tps")
    if not (tps and tps > 0):
        # The server reports no prefill speed (LM Studio, Ollama, vLLM): the client clock, over the tokens prefill_estimate
        # would expect to be new (cached as reported, else the previous turn's prompt).
        cached = metrics.get("cached_tokens")
        if cached is None:
            cached = 0 if tc.get("kind") == "compaction" else min(tokens, hist["prev"].get(key, 0))
        ttft = (metrics.get("ttft_ms") or 0) / 1000
        tps = (tokens - cached) / ttft if ttft > 0 and tokens - cached >= 10 else None
    if tps and tps > 0:
        hist["tps"].append(tps)
    if tc.get("kind") != "compaction":
        hist.setdefault("last", {})[key] = tokens
        hist["prev"][(tc.get("stream"), tc.get("epoch"))] = max(hist["prev"].get((tc.get("stream"), tc.get("epoch")), 0), tokens)


def _skip_metrics(tc, reason):
    """Build a SKIP result placeholder (no query issued, excluded from scoring)."""
    m = {"id": tc["id"], "eval_result": "SKIP", "fail_reason": reason, "text": "", "prompt_tokens": 0, "cached_tokens": None,
         **NO_METRICS}
    m.update(_replay_fields(tc))
    return m


def _replay_fields(tc):
    """Extra result fields of a replay case (empty for the other suites)."""
    if tc.get("fixture") is not None:
        return {k: tc[k] for k in ("stream", "kind", "epoch", "turn", "recorded_prompt_tokens", "recorded_tools")}
    return {}


def _thinking_flag(thinking):
    """--thinking on/off (or a bool) as the enable_thinking boolean; None stays None."""
    return None if thinking is None else thinking in (True, "on")


def query_model(endpoint_url, model_name, tc, max_tokens=None, reasoning_effort=None, state=None, temperature=None,
                prompt_progress=False, thinking=None):
    """prompt_progress: ask the server for prefill progress on streaming requests: True adds return_progress to the
    body, "header" sends the Kapitsa opt-in header instead, "snapshot" (MTPLX) sends x-mtplx-request-id and polls the
    snapshot endpoint meanwhile."""
    if tc.get("fixture") is not None:
        # Replay case: the recorded request, built now and dropped when this call returns.
        payload = replay_cases.build_payload(tc, model_name, reasoning_effort=reasoning_effort, temperature=temperature,
                                             thinking=_thinking_flag(thinking))
        if max_tokens is not None:  # only the --warmup-first-test warmup caps a replay request
            payload["max_tokens"] = max_tokens
        return send_progress(endpoint_url, payload, tc, state, prompt_progress)

    messages = smoke.build_messages(tc)

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True}
    }

    # Warmup caps generation: the point is to prefill the (large) prompt cache for
    # the first real test, not to wait out a full completion.
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    if thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": _thinking_flag(thinking)}

    # Sampling temperature: sent only when given (--temperature), else the server's default applies.
    if temperature is not None:
        payload["temperature"] = temperature

    if "tools" in tc:
        payload["tools"] = tc["tools"]

    return send_progress(endpoint_url, payload, tc, state, prompt_progress)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _hard_exit(signum, frame):
    """The second Ctrl+C: leave at once (the state file is complete up to the last finished case)."""
    try:
        sys.stdout.flush()
        sys.stderr.write("\nsecond interrupt: exiting now\n")
        sys.stderr.flush()
    finally:
        os._exit(130)


@contextlib.contextmanager
def _interrupt_guard():
    """First Ctrl+C: KeyboardInterrupt in the main thread (aborts the request in flight: its connection is closed on the
    way out); from then on a second Ctrl+C exits immediately."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)

    def first(signum, frame):
        signal.signal(signal.SIGINT, _hard_exit)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, first)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _resume_plan(path, mode, replay, targets, test_cases):
    """Checks of --resume against the live target, before anything is sent. {"path", "loaded": {id: metrics} of the
    graded results of the selection, "todo": ids still to send (no result, or an error)}. A UserError when the run
    cannot continue: several targets, another model, suite or fixture. Differences in server, commit or context
    size are one warning line each on stderr."""
    header, results, resumes = state_mod.load_resume(path)
    if len(targets) != 1:
        raise UserError("--resume continues one model: the selection resolved to %d (check --endpoint / --model)"
                        % len(targets))
    target = targets[0]
    if target["name"] != header["model"]:
        raise UserError("--resume: %s was run on model %s, the server now offers %s"
                        % (path, header["model"], target["name"]))
    if not state_mod.same_suite(header.get("suite") or mode, mode):
        raise UserError("--resume: %s is suite %s, this run is %s" % (path, header.get("suite"), mode))
    sha = (header.get("fixture") or {}).get("sha256")
    if replay is not None and sha:
        now = state_mod.file_sha256(replay["fixture"])
        if now != sha:
            raise UserError("--resume: the fixture changed since %s was started (sha256 %s, now %s)"
                            % (path, sha[:8], now[:8]))
    last = resumes[-1] if resumes else header  # the most recent invocation that wrote to the file
    old_server, new_server = last.get("server") or {}, target.get("backend_info") or {}
    for k in ("kind", "version", "build"):
        if old_server.get(k) != new_server.get(k):
            report.warn("--resume: server %s differs: %s before, %s now" % (k, old_server.get(k), new_server.get(k)))
    commit = state_mod.git_commit()
    if last.get("git_commit") != commit:
        report.warn("--resume: git commit differs: %s before, %s now" % (last.get("git_commit"), commit))
    if header.get("n_ctx") != target.get("n_ctx"):
        report.warn("--resume: n_ctx differs: %s before, %s now" % (header.get("n_ctx"), target.get("n_ctx")))
    if header.get("backend") != target["backend"]:
        report.warn("--resume: endpoint differs: %s before, %s now" % (header.get("backend"), target["backend"]))
    ids = {tc["id"] for tc in test_cases}
    loaded = {i: m for i, m in results.items() if i in ids and "error" not in m}
    todo = {tc["id"] for tc in test_cases if tc["id"] not in loaded}
    print("resuming %s: %d of %d cases done, %d to run" % (report.rel_path(path), len(loaded), len(test_cases), len(todo)),
          file=sys.stderr, flush=True)
    return {"path": path, "loaded": loaded, "todo": todo}


def _case_state_path(path, model, taken):
    """The state file of one model of a run with several models and one --state-file: the given name with `_<model>`
    before its extension (run.jsonl -> run_<model>.jsonl); `-2`, `-3` ... when two models share a slug."""
    stem, ext = os.path.splitext(path)
    base = "%s_%s" % (stem, state_mod.slug(model))
    out, n = base + (ext or ".jsonl"), 1
    while out in taken:
        n += 1
        out = "%s-%d%s" % (base, n, ext or ".jsonl")
    return out


def _grade(tc, metrics):
    """Give a finished request its verdict (a failed one keeps its error as the reason) and flag a server pause."""
    if "error" in metrics:
        metrics.update(NO_METRICS, eval_result="FAIL", fail_reason=metrics["error"])
    elif tc.get("fixture") is not None:
        metrics["eval_result"], metrics["fail_reason"] = grading.grade(tc, metrics)
    else:
        metrics["eval_result"], metrics["fail_reason"] = smoke.evaluate_response(tc, metrics["text"])
    paused = perf.server_paused_ms(metrics)  # the server froze mid-request: flagged, not a verdict
    if paused is not None:
        metrics["server_paused_ms"] = round(paused)


def _warm_up(ctx, tr):
    """Warm the model up (weights loaded) with a small generic case, not the first real one, so that case does not come
    back as a spurious ~100% cache hit with no cold prefill speed; --warmup-first-test warms with the first case. Returns
    whether the server reports token counts, or None when the warmup failed (one warning line; the first real result
    decides)."""
    if ctx.resume and not ctx.resume["todo"]:
        warmup_tc = None  # nothing to send
    elif ctx.warmup_first_test:
        warmup_tc = ctx.test_cases[0]
    else:
        warmup_tc = get_warmup_case() or ctx.test_cases[0]
    if warmup_tc is None:
        return any(m.get("prompt_tokens") for m in ctx.resume["loaded"].values())
    t = tr.target
    ctx.spinner.start(f"{Clr.YEL}⚡ WARMUP{Clr.RST} | {Clr.WHT}{t['backend']}{Clr.RST} :: {Clr.DIM}{t['name']}{Clr.RST}")
    try:
        res = post_with_fallback(t, lambda url: query_model(
            url, t["name"], warmup_tc, max_tokens=256, reasoning_effort=ctx.reasoning_effort,
            temperature=ctx.temperature, thinking=ctx.thinking))
    finally:
        ctx.spinner.stop()
    if res.get("error"):
        report.warn("warmup of %s failed (%s); the table columns follow the first real result" % (t["name"], res["error"]))
        return None
    return res.get("prompt_tokens", 0) > 0


def _open_state(ctx, tr):
    """The state writer of a target (None: nothing is saved, or a resumed file is already complete) and its path."""
    target = tr.target
    hdr = state_mod.make_header(
        ctx.mode, target, request=ctx.request_settings, fixture_path=ctx.replay["fixture"] if ctx.replay else None,
        argv=ctx.cmdline, selection=ctx.selection, preset=ctx.preset, note=ctx.run_note,
        reference=ctx.reference.info() if ctx.reference else None,
        started_at=ctx.run_started.astimezone(timezone.utc).isoformat(timespec="seconds"))
    target["suite"] = ctx.mode
    target["header"] = hdr
    if ctx.resume:
        if not ctx.resume["todo"]:
            return None, ctx.resume["path"]
        writer = state_mod.StateWriter(ctx.resume["path"])
        writer.write(state_mod.resume_record(target, ctx.cmdline, ctx.run_note))
        return writer, ctx.resume["path"]
    if not ctx.save_state:
        return None, None
    if ctx.state_file:
        path = ctx.state_file if ctx.n_targets == 1 else _case_state_path(ctx.state_file, target["name"], ctx.state_paths)
    else:
        path = state_mod.default_path(ctx.states_dir, target.get("backend_info"), target["name"], ctx.mode,
                                      when=ctx.run_started, request=ctx.request_settings, dev=target.get("device"),
                                      preset=ctx.preset, note=ctx.run_note, model_info=target.get("model_info"))
    writer = state_mod.StateWriter(path)
    writer.write(hdr)
    return writer, path


def _send_case(ctx, target, tc, label, live, prog, hist):
    """Send one request (retrying transient failures) behind the live spinner; returns its metrics. `live`: the dict the
    spinner reads and query_model writes."""
    spinner, policy = ctx.spinner, ctx.progress_policy
    label, formatter = ctx.out.live(tc, label, prog)

    def start_spinner():
        if not policy.sends(target) and live.get("pp_expected"):
            live.pop("pp_expected")  # progress fell back off (HTTP 400): show the estimate
            live.update(prefill_estimate(tc, hist))
        spinner.start(label, state=live, formatter=formatter)

    start_spinner()
    try:
        return query_with_retry(
            lambda: send_with_progress_fallback(
                lambda with_progress: post_with_fallback(target, lambda url: query_model(
                    url, target["name"], tc, reasoning_effort=ctx.reasoning_effort,
                    temperature=ctx.temperature, state=live, prompt_progress=with_progress,
                    thinking=ctx.thinking)),
                policy, target, spinner, start_spinner),
            target["url"], spinner, start_spinner)
    finally:
        spinner.stop()


def _run_target(ctx, tr):
    """Run every case on one model: warmup, state file, the live table row by row, then its tables and end record."""
    target, out = tr.target, ctx.out
    key = target["key"]
    results = ctx.raw_results[key] = {}
    ctx.targets.append(target)
    warm = _warm_up(ctx, tr)
    loaded = ctx.resume["loaded"] if ctx.resume else {}
    if warm is None and loaded:  # a resumed run: its loaded results say
        warm = any(m.get("prompt_tokens") for m in loaded.values())
    tr.has_stats = bool(warm)
    deferred = warm is None and out.layout_needs_stats  # the columns wait for the first real result
    tr.writer, tr.path = _open_state(ctx, tr)
    tr.target["state_writer"] = tr.writer
    if tr.path and tr.path not in ctx.state_paths:
        ctx.state_paths.append(tr.path)
    tr.base_run = ctx.baseline.run_for(target, ctx.infos[key]) if ctx.baseline else None
    if tr.base_run:
        if ctx.reference:  # the baseline is judged against the reference too
            bh = tr.base_run.header
            tr.base_run.results = ctx.reference.judge_all(tr.base_run.results, bh.get("backend"), bh.get("model"))
        ctx.base_runs[key] = tr.base_run
    ref_run = ctx.reference.run_for(target["backend"], target["name"]) if ctx.reference else None
    if not deferred:
        out.target_start(target, tr.has_stats, ctx.test_cases, tr.base_run)
        tr.started = True

    total = len(ctx.test_cases)
    prog = {"done": len(ctx.resume["loaded"]) if ctx.resume else 0, "total": total, "t0": time.time(),
            "t_last": time.time(), "cache": None, "n_ctx": target.get("n_ctx")}
    hist = {"tps": [], "prev": {}, "last": {}}  # for the prefill estimate of the live line
    order = {tc["id"]: i for i, tc in enumerate(ctx.test_cases)}
    if ctx.resume:  # the loaded results are done: shown, and they feed the prefill estimate
        for tc in ctx.test_cases:
            m = ctx.resume["loaded"].get(tc["id"])
            if m is not None:
                results[tc["id"]] = m
                note_finished(tc, m, hist)
                out.case_loaded(target, m, tr.has_stats)
    for idx, tc in enumerate(ctx.test_cases, 1):
        if ctx.resume and tc["id"] not in ctx.resume["todo"]:
            continue
        skip = replay_cases.skip_reason(tc, target) if tc.get("fixture") is not None else None
        if skip:
            metrics = _skip_metrics(tc, skip)
        else:
            live = live_state(tc, hist, ctx.progress_policy, target)
            prog["done"] = len(results)
            metrics = _send_case(ctx, target, tc, f"[{idx:02d}/{total:02d}] {tc['id']}...", live, prog, hist)
            metrics["id"] = tc["id"]
            metrics.update(_replay_fields(tc))
            _grade(tc, metrics)
            prev = None if tc.get("kind") == "compaction" else hist.get("last", {}).get((tc.get("stream"), tc.get("epoch")))
            if prev:  # new tokens are estimated from it when the server reports no cache (results.new_tokens)
                metrics["prev_prompt_tokens"] = prev
        results[tc["id"]] = metrics
        if ctx.resume:  # fixture order, whatever was loaded and what is new
            done = sorted(results.items(), key=lambda kv: order[kv[0]])
            results.clear()
            results.update(done)
        if tr.writer:
            tr.writer.write(state_mod.result_record(metrics))  # the normal verdict is stored: a reference is a view
        if ref_run:
            metrics.update(reference_mod.judge(metrics, ref_run.results.get(tc["id"]), ctx.reference.match, ref_run.label))
        prog["done"], prog["t_last"] = len(results), time.time()
        note_finished(tc, metrics, hist)
        if metrics.get("prompt_tokens"):
            prog["cache"] = (100.0 * metrics["cached_tokens"] / metrics["prompt_tokens"]) if metrics.get("cached_tokens") is not None else None
        if deferred and not tr.started and (metrics.get("eval_result") != "SKIP" and not metrics.get("error")):
            tr.has_stats = (metrics.get("prompt_tokens") or 0) > 0
            out.target_start(target, tr.has_stats, ctx.test_cases, tr.base_run)
            tr.started = True
            for shown in list(results.values())[:-1]:  # skipped or failed cases before it
                out.case_done(target, shown, tr.has_stats)
        if not deferred or tr.started:
            out.case_done(target, metrics, tr.has_stats)

    if deferred and not tr.started:  # no case produced a result: plain columns
        out.target_start(target, False, ctx.test_cases, tr.base_run)
        tr.started = True
        for shown in results.values():
            out.case_done(target, shown, False)
    finish_target(out, target, ctx.targets, ctx.raw_results, tr.base_run, tr.has_stats, ctx.mdcat)
    if tr.writer:  # a complete resumed file has no writer: it stays as it is
        tr.writer.write(state_mod.end_record(results))
        tr.writer.close()


def _select_cases(mode, target_test_ids, replay, out):
    """(test cases, one-line summary of a replay selection or None) of the run; a UserError for an empty or unknown one."""
    if replay is None:
        try:
            return select_cases(target_test_ids, suite=mode), None
        except ValueError as missing:
            raise UserError("test ID(s) not found: %s" % missing)
    try:
        fixture = replay_cases.load_fixture(replay["fixture"])
    except (OSError, ValueError) as e:
        raise UserError("cannot read fixture %s: %s" % (replay["fixture"], getattr(e, "strerror", None) or e))
    selected = replay_cases.select_entries(
        fixture, max_context=replay.get("max_context"), no_subagents=replay.get("no_subagents", False),
        turns=replay.get("turns"))
    summary = replay_cases.summary_line(selected)
    if report.current_format() != "markdown":  # markdown: the `## Run` section carries it
        report.info(f"{Clr.DIM}{mode} selection: {summary}{Clr.RST}")
    test_cases = [replay_cases.make_case(fixture, p, e) for p, e in selected]
    out.set_fixture(fixture)
    if not test_cases:
        raise UserError("the selection is empty (check --max-context, --turns and --no-subagents)")
    return test_cases, summary


def run_benchmark(mode, target_test_ids=None,
                  endpoints=None, model_filters=None, first_model=False,
                  state_file=None, save_state=True, states_dir="states", cmdline=None,
                  compare_state_path=None, load_state_path=None,
                  warmup_first_test=False, reasoning_effort=None, temperature=None, thinking=None,
                  mdcat=True, replay=None, report_format="tui", prompt_progress=True, resume=None,
                  preset=None, run_note=None, server_version=None, reference_path=None, match="step"):
    """Run a suite. ``replay`` ({"fixture": path, "max_context", "no_subagents", "turns"}) selects the
    replay suite named by ``mode`` (e.g. "coding") instead of the smoke cases.

    Every live run saves one JSONL state per model (bench/state.py) under ``states_dir``; ``state_file`` names the file
    (with several models: one file per model, `_<model>` added to the name), ``save_state=False`` saves nothing.
    ``reference_path`` (--reference) judges every answer against that saved run by ``match`` (bench/reference.py).
    ``cmdline`` is the argument list recorded in the header. ``compare_state_path`` (or, with ``load_state_path``, the
    saved run shown) is compared case by case with that state file.

    The first Ctrl+C, or any exception in the run loop, stops the run: the request in flight is dropped, the state's
    `end` record says `interrupted` and why, every report is printed for the finished cases, and the process exits
    with 130 (Ctrl+C) or 1 (error). A second Ctrl+C exits immediately.

    ``resume`` (a single-model state file, --resume) continues that run: the cases without a result, or whose last
    result is an error, are sent and appended to the same file after a `resume` record; the graded ones are not
    sent again. The report covers the loaded and the new results."""

    report.configure(report_format)
    if load_state_path:  # no discovery, no requests: show the saved run again
        display_state(load_state_path, target_test_ids=target_test_ids, compare_state_path=compare_state_path, mdcat=mdcat,
                      reference_path=reference_path, match=match)
        return

    out = output()
    baseline = None
    if compare_state_path:
        baseline = load_baseline(compare_state_path)
        check_baseline(baseline, mode, state_mod.file_sha256(replay["fixture"]) if replay and os.path.exists(replay["fixture"]) else None)
    reference = None
    if reference_path:
        reference = reference_mod.load_reference(
            reference_path, mode, state_mod.file_sha256(replay["fixture"]) if replay and os.path.exists(replay["fixture"]) else None, match)
        warning = reference_mod.temperature_warning(reference, {"request": {"temperature": temperature}}, match)
        if warning:
            report.warn(warning)
    test_cases, selection_summary = _select_cases(mode, target_test_ids, replay, out)

    spinner = ProgressAnimation()
    chatter = _chatter()
    with chatter:
        targets = discover_and_report(endpoints or resolve_endpoints(None), model_filters, spinner=spinner,
                                      first_model=first_model)
    if not targets:
        if chatter.text:
            print(chatter.text, file=sys.stderr, end="")
        raise UserError("no models to benchmark: check --endpoint / --model and that the servers are up")
    for t in targets:
        t["backend_info"] = backend.apply_version_override(t.get("backend_info"), server_version)

    request_settings = {"temperature": temperature, "reasoning_effort": reasoning_effort, "thinking": thinking}
    selection = None
    if replay is not None:
        selection = {"max_context": replay.get("max_context"), "no_subagents": replay.get("no_subagents", False),
                     "turns": list(replay["turns"]) if replay.get("turns") else None}
    infos = [R.run_info(mode, t, request_settings, selection, selection_summary, run_note, preset) for t in targets]
    out.begin(infos, targets, len(test_cases), mdcat)
    out.live_run = True

    ctx = SimpleNamespace(
        mode=mode, out=out, spinner=spinner, progress_policy=PromptProgress(prompt_progress), test_cases=test_cases,
        resume=_resume_plan(resume, mode, replay, targets, test_cases) if resume else None, baseline=baseline,
        reference=reference, infos={t["key"]: i for t, i in zip(targets, infos)}, replay=replay, preset=preset, run_note=run_note,
        selection=selection, request_settings=request_settings, cmdline=cmdline, run_started=datetime.now(),
        save_state=save_state, state_file=state_file, states_dir=states_dir, reasoning_effort=reasoning_effort, thinking=thinking,
        temperature=temperature, warmup_first_test=warmup_first_test, mdcat=mdcat,
        targets=[], raw_results={}, state_paths=[], base_runs={}, n_targets=len(targets))

    interrupted = None  # why the run stopped early: "Ctrl+C" or "error: ..."
    exit_code = 0
    tr = None  # the model being run
    with _interrupt_guard():
        try:
            for target in targets:
                tr = SimpleNamespace(target=target, writer=None, path=None, base_run=None, has_stats=False, started=False)
                _run_target(ctx, tr)
                tr = None
        except KeyboardInterrupt:
            interrupted, exit_code = "Ctrl+C", 130
        except Exception as e:
            interrupted, exit_code = "error: %s: %s" % (type(e).__name__, e), 1
            traceback.print_exc()
        spinner.stop()

        if interrupted:
            # Stop cleanly: record why in the state, then report everything that finished.
            print(f"\n{Clr.BYEL}⚠ {interrupted}: stopping, reporting the finished cases{Clr.RST}", file=sys.stderr, flush=True)
            if tr:
                key = tr.target["key"]
                try:
                    if tr.writer:
                        tr.writer.write(state_mod.end_record(ctx.raw_results[key], interrupted=interrupted))
                        tr.writer.close()
                    if tr.started:
                        finish_target(out, tr.target, ctx.targets, ctx.raw_results, tr.base_run, tr.has_stats, mdcat,
                                      interrupted=interrupted)
                    else:  # nothing shown for it yet
                        ctx.targets.remove(tr.target)
                        del ctx.raw_results[key]
                except Exception:
                    traceback.print_exc()

        state_mod.flush_all()
        print_final(out, ctx.targets, ctx.raw_results, ctx.base_runs, mdcat, ctx.state_paths, interrupted)
        if interrupted and tr and tr.path:  # only the model that was interrupted can be continued
            hint = resume_hint([tr.path], sys.argv[0], len(test_cases))
            if hint:
                print(hint, file=sys.stderr, flush=True)
    if exit_code:
        sys.exit(exit_code)
