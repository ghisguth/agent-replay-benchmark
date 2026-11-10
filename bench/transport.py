"""HTTP transport, SSE reading, timeout and stall handling, retry and progress fallback."""

import http.client
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

from . import auth
from .colors import Clr
from .discovery import discover_models
from .telemetry import finalize, init_acc, normalize
from .tls import ssl_context


# Seconds a streaming response may send nothing, once it has sent its first line, before the
# request counts as stalled. Splash writes an SSE keepalive every 2 s while it works (prefill
# included), so silence means its handler is gone; a server that sends nothing until its first
# token still gets the full 3600 s for prefill. REPLAY_BENCH_STALL_SECONDS overrides it.
def _stall_seconds():
    v = os.environ.get("REPLAY_BENCH_STALL_SECONDS")
    if v is not None:
        return float(v)
    return 300.0


STALL_SECONDS = _stall_seconds()

# A server that goes away mid-run (idle sleep, a restart) is not a model failure: these errors of query_model are
# retried after the server answers again. REPLAY_BENCH_SERVER_WAIT = seconds to wait for it per retry.
SERVER_WAIT_S = float(os.environ.get("REPLAY_BENCH_SERVER_WAIT", "900"))
SERVER_POLL_S = 10.0
MAX_ATTEMPTS = 3

# Prefill progress (--no-prompt-progress turns it off). Splash and llama.cpp answer a streaming request that carries
# "return_progress": true with chunks of an empty delta and a "prompt_progress" object before the first token.
# Kapitsa instead opts in by a request header and sends `kapitsa_progress` events (data lines without choices).
PROGRESS_BACKENDS = ("splash", "llamacpp")
KAPITSA_HEADER = {"X-Kapitsa-Bench-Telemetry": "1"}
# MTPLX sends only ": keep-alive" comments during prefill, but publishes its chunked prefill on GET /v1/mtplx/snapshot
# (in_flight[].prefill_state). The request carries its own id (x-mtplx-request-id: rb-<hex>, answered as
# "chatcmpl-rb-<hex>") and a daemon thread polls the snapshot until the first token.
MTPLX_POLL_S = 0.5
MTPLX_POLL_TIMEOUT = 2.0


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class TransportError(str):
    """An error message that also carries its kind ("unreachable", "stalled", "timeout", "http", "runaway", "server",
    "exception") and, for "http", the status code. It is a plain str everywhere else (compares, serialises)."""

    kind = None
    status = None


def _error(message, kind, status=None):
    err = TransportError(message)
    err.kind, err.status = kind, status
    return {"error": err}


_TRANSIENT_KINDS = ("unreachable", "stalled", "timeout")


def is_transient(error):
    """True for the errors of an unreachable or frozen server (not HTTP 4xx/500, not a runaway generation). Uses the
    kind a TransportError carries; a plain string (an error read back from a saved state) is matched by its prefix."""
    kind = getattr(error, "kind", None)
    if kind is not None:
        return kind in _TRANSIENT_KINDS or (kind == "http" and error.status in (502, 503, 504))
    e = str(error)
    return (e.startswith(("Inference failure:", "stalled:")) or e == "timeout"
            or e.startswith(("HTTP 502", "HTTP 503", "HTTP 504")))


def chunk_error(chunk):
    """The "server: ..." message of an SSE chunk that reports a failure (a top-level `error` object or string, as
    MTPLX and llama.cpp send, or finish_reason "error"), else None."""
    err = chunk.get("error")
    if err:
        if isinstance(err, dict):
            code, msg = err.get("code"), err.get("message") or err.get("type")
            parts = [str(x) for x in (code, msg) if x]
            return "server: " + (": ".join(parts) if parts else "stream error")
        return "server: " + " ".join(str(err).split())
    choices = chunk.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict) \
            and choices[0].get("finish_reason") == "error":
        return "server: stream ended with finish_reason error"
    return None


def wait_for_server(chat_url, limit=None, discover_fn=None):
    """Poll GET <base>/models until the server answers; False after `limit` seconds (default SERVER_WAIT_S)."""
    if discover_fn is None:
        discover_fn = discover_models
    base = chat_url.rsplit("/chat/completions", 1)[0]
    deadline = time.monotonic() + (SERVER_WAIT_S if limit is None else limit)
    while True:
        if discover_fn(base) is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(SERVER_POLL_S)


def _route_rejected(metrics):
    """True when the answer says the route does not exist or does not speak chat completions: HTTP 404/400/405, or a
    200 with nothing in it (no text, no tool calls, no token counts: an unparseable body)."""
    err = str(metrics.get("error") or "")
    if err:
        return err.startswith(("HTTP 404", "HTTP 400", "HTTP 405"))
    return not (metrics.get("text") or metrics.get("tool_calls") or metrics.get("prompt_tokens")
                or metrics.get("tokens"))


def post_with_fallback(target, send):
    """send(url) -> metrics, to the target's chat URL. A target with a `fallback_url` (LM Studio: /api/v0/chat/completions,
    with /v1 behind it) tries the first request on its own route; when that is rejected (_route_rejected) the target
    switches to the fallback for the rest of the run (target["url"], and the endpoint of its state header, one dim
    note on stderr) and the request is sent again. After the first answer that is not a rejection the route is kept."""
    metrics = send(target["url"])
    fallback = target.get("fallback_url")
    if not fallback:
        return metrics
    if not _route_rejected(metrics):
        target.pop("fallback_url")
        return metrics
    old, target["url"] = target["url"], target.pop("fallback_url")
    hdr = target.get("header")
    if isinstance(hdr, dict):
        hdr["endpoint"] = target["url"][:-len("/chat/completions")]
    writer = target.get("state_writer")
    if writer:  # the header on disk was written before the switch: record it
        writer.write({"type": "route", "endpoint": target["url"][:-len("/chat/completions")]})
    print(f"{Clr.DIM}{old} did not take the request ({metrics.get('error') or 'empty answer'}): "
          f"using {target['url']} for this run{Clr.RST}", file=sys.stderr, flush=True)
    return send(target["url"])


def query_with_retry(send, chat_url, spinner=None, restart=None, wait_fn=None):
    """send() -> metrics (query_model's result). A transient server error waits for the server and resends, up to
    MAX_ATTEMPTS attempts; `attempts` and `retry_errors` are recorded when it took more than one, and a final failure
    reads "server: <error>". Ctrl+C during the wait propagates as usual. restart() re-starts the spinner."""
    if wait_fn is None:
        wait_fn = wait_for_server
    errors, sends = [], 0
    while True:
        metrics = send()
        sends += 1
        err = metrics.get("error")
        if not err or not is_transient(err):
            break
        errors.append(err)
        if sends >= MAX_ATTEMPTS:
            metrics["error"] = "server: " + err
            break
        if spinner:
            spinner.stop()
        print(f"server unreachable ({err}), waiting for it… attempt {sends + 1}/{MAX_ATTEMPTS}",
              file=sys.stderr, flush=True)
        up = wait_fn(chat_url)
        if restart:
            restart()
        if not up:
            metrics["error"] = "server: %s (no answer for %.0fs)" % (err, SERVER_WAIT_S)
            break
    if sends > 1:
        metrics["attempts"] = sends
        # the errors of the attempts before the last one (the last is the case's own result when it failed)
        metrics["retry_errors"] = errors[:sends - 1]
    return metrics


class PromptProgress:
    """Whether requests ask the server for prefill progress: on by default, off with --no-prompt-progress or after a
    server rejected the field (send_with_progress_fallback)."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def sends(self, target):
        """True: add the return_progress body field; "header": the Kapitsa request header; "snapshot": poll the MTPLX
        snapshot endpoint (no request field, so never rejected with 400); False: nothing."""
        kind = (target.get("backend_info") or {}).get("kind")
        if not self.enabled:
            return False
        if kind == "mtplx":
            return "snapshot"
        return True if kind in PROGRESS_BACKENDS else ("header" if kind == "kapitsa" else False)


def send_with_progress_fallback(send, policy, target, spinner=None, restart=None):
    """send(with_progress) -> metrics. When the server answers HTTP 400 to a request that carried return_progress, the
    request is sent once more without it; if that one is not rejected, the field is off for the rest of the run (one
    stderr line). A 400 that stays is the case's own error, and the field stays on."""
    flag = policy.sends(target)
    metrics = send(flag)
    if flag and flag != "snapshot" and str(metrics.get("error", "")).startswith("HTTP 400"):
        again = send(False)
        if not str(again.get("error", "")).startswith("HTTP 400"):
            policy.enabled = False
            if spinner:
                spinner.stop()
            print("server rejected return_progress (HTTP 400): sent without it, prompt progress off for this run",
                  file=sys.stderr, flush=True)
            if restart:
                restart()
        return again
    return metrics


def _set_read_timeout(response, seconds):
    """Bound each further read of an open urllib response (http.client keeps the socket private)."""
    try:
        response.fp.raw._sock.settimeout(seconds)
    except AttributeError:
        pass


def snapshot_url(endpoint_url):
    """`http://h:p/v1/chat/completions` -> `http://h:p/v1/mtplx/snapshot`."""
    return endpoint_url.rsplit("/chat/completions", 1)[0].rstrip("/") + "/mtplx/snapshot"


def mtplx_pp(snapshot, response_id, last=None):
    """Map a /v1/mtplx/snapshot body to the prompt_progress shape {total, cache, processed, time_ms}, or None.
    Only a chunk-phase reading of this request counts ("started" has no tokens done yet, "completed" clears the state),
    a reading with no new token yet (processed <= cache: MTPLX publishes one as chunk work starts) does not count, so the
    estimate stays, and a reading not ahead of `last` (a stale or repeated poll) is ignored. The reading also carries `chunk` (MTPLX prefills
    in steps of that many tokens, default 2048) and `at`, the time.time() it was taken, for the display to interpolate."""
    flights = snapshot.get("in_flight") if isinstance(snapshot, dict) else None
    for f in flights if isinstance(flights, list) else ():
        if not (isinstance(f, dict) and f.get("request_id") == response_id):
            continue
        ps = f.get("prefill_state")
        if not (isinstance(ps, dict) and ps.get("phase") == "chunk"):
            return None
        try:
            total, done = int(ps["tokens_total"]), int(ps["tokens_done"])
            cache, secs = int(ps.get("cached_tokens") or 0), float(ps.get("elapsed_s") or 0.0)
        except (KeyError, TypeError, ValueError):
            return None
        try:
            chunk = int(ps.get("chunk_size") or 2048)
        except (TypeError, ValueError):
            chunk = 2048
        if total <= 0 or done <= cache or (last and done <= last["processed"]):
            return None  # an unchanged reading keeps `last` (its "at" and "shown"), so the interpolation goes on
        return {"total": total, "cache": min(cache, total), "processed": min(done, total), "time_ms": int(secs * 1000),
                "chunk": chunk if chunk > 0 else 2048, "at": time.time()}
    return None


def _poll_snapshot(url, headers, response_id, state, stop):
    """Daemon thread body: every MTPLX_POLL_S seconds read the snapshot and set state["pp"] until the first token or
    `stop`. Any 4xx/5xx (an older MTPLX without the endpoint, auth) ends the polling for this request, which then keeps
    the estimate; a timeout or a dropped connection only skips that poll."""
    while not stop.wait(MTPLX_POLL_S):
        if state.get("first"):
            return
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=MTPLX_POLL_TIMEOUT, context=ssl_context()) as r:
                body = json.loads(r.read(1 << 20).decode("utf-8"))
        except urllib.error.HTTPError as e:
            e.close()
            return
        except (OSError, http.client.HTTPException, ValueError):
            continue
        pp = mtplx_pp(body, response_id, state.get("pp"))
        if pp and not stop.is_set() and not state.get("first"):
            state["pp"] = pp


def _send_snapshot(endpoint_url, payload, tc, state, send_fn):
    """Send with an x-mtplx-request-id and poll the snapshot endpoint meanwhile (see MTPLX_POLL_S)."""
    rid = "rb-" + uuid.uuid4().hex
    if state is None:
        state = {}
    state.pop("first", None)  # a resend reuses the dict
    state.pop("pp", None)
    stop = threading.Event()
    poller = threading.Thread(
        target=_poll_snapshot, daemon=True,
        args=(snapshot_url(endpoint_url), {**auth.auth_header(), "Accept": "application/json"},
              "chatcmpl-" + rid, state, stop))
    poller.start()
    try:
        return send_fn(endpoint_url, payload, tc, state, headers={"x-mtplx-request-id": rid})
    finally:
        stop.set()


def send_progress(endpoint_url, payload, tc, state, prompt_progress, send_fn=None):
    if send_fn is None:
        send_fn = _send
    if prompt_progress == "snapshot" and payload.get("stream"):
        return _send_snapshot(endpoint_url, payload, tc, state, send_fn)
    if prompt_progress == "header" and payload.get("stream"):
        return send_fn(endpoint_url, payload, tc, state, headers=KAPITSA_HEADER)
    if prompt_progress is True and payload.get("stream"):
        payload["return_progress"] = True
    return send_fn(endpoint_url, payload, tc, state)


class _StreamAcc:
    """Accumulates one streamed chat-completions answer: text, reasoning, tool-call deltas and server telemetry.
    `feed` takes each decoded SSE chunk; `finish` builds the metrics dict."""

    def __init__(self, start_time, max_gen, max_secs, state):
        self.start, self.max_gen, self.max_secs, self.state = start_time, max_gen, max_secs, state
        self.text = ""
        self.reasoning = ""
        self.tool_calls = []  # one {"id", "type", "function": {"name", "arguments"}} per tool-call index
        self.finish_reason = None
        self.stream_error = None  # an error the server reported inside the stream
        self.ttft = None
        self.chars = 0  # streamed characters incl. tool-call arguments: the live "down" estimate
        self.tel = init_acc()

    def _tool_deltas(self, deltas):
        """Merge the tool-call deltas into self.tool_calls; return (characters added, whether any carried content)."""
        added, has = 0, False
        for d in deltas:
            idx = d.get("index", 0)
            while len(self.tool_calls) <= idx:
                self.tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            call = self.tool_calls[idx]
            fn = d.get("function") or {}
            name, args = fn.get("name") or "", fn.get("arguments") or ""
            if d.get("id"):
                call["id"] = d["id"]
            call["function"]["name"] += name
            call["function"]["arguments"] += args
            added += len(name) + len(args)
            has = has or bool(d.get("id") or name or args)
        return added, has

    def feed(self, chunk, now):
        """Take one SSE chunk (a dict). Returns a "runaway: ..." message when a generation cap is hit, else None."""
        state = self.state
        self.stream_error = self.stream_error or chunk_error(chunk)
        prev_pp = self.tel.get("last_progress")
        normalize(chunk, self.tel, current_time=now, start_time=self.start, ttft=self.ttft)
        if self.tel.get("last_progress") is not None and self.tel.get("last_progress") is not prev_pp:
            state["pp"] = self.tel["last_progress"]

        choices = chunk.get("choices")
        if not choices:
            return None
        choice = choices[0]
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        delta = choice.get("delta", {})
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thought") or ""
        tc_chars, has_tc = self._tool_deltas(delta.get("tool_calls") or [])

        self.chars += len(content) + len(reasoning) + tc_chars
        mp = chunk.get("mtplx_progress")
        mp_tokens = mp.get("completion_tokens") if isinstance(mp, dict) else None
        state["down"] = int(mp_tokens) if mp_tokens is not None else int(self.chars / 3.5)

        if content or reasoning or has_tc or (mp_tokens or 0) > 0:
            if self.ttft is None:
                self.ttft = now - self.start
                state["first"] = now  # the live line shows TTFT and decode speed
            self.text += content
            self.reasoning += reasoning

        if self.tel["exact_tokens"] > self.max_gen or len(self.text + self.reasoning) / 3.5 > self.max_gen:
            return "runaway: over %d tokens: %s" % (self.max_gen, self._tail())
        if now - self.start > self.max_secs:
            return "runaway: over %s s: %s" % (self.max_secs, self._tail())
        return None

    def _tail(self):
        return (self.text + self.reasoning)[-50:].replace("\n", " ")

    def finish(self, total_duration):
        tel = self.tel
        token_count = tel["exact_tokens"] if tel["exact_tokens"] > 0 else int(len(self.text + self.reasoning) / 3.5)
        ttft = finalize(tel, total_duration, ttft=self.ttft, token_count=token_count)

        # The answer's own text (no thinking, no tool calls) and how much was thought, as counts only.
        answer = _THINK.sub("", self.text)
        content = answer.strip()
        reasoning_chars = len(self.reasoning) + len(self.text) - len(answer)

        text = self.text
        if self.tool_calls:  # the fully assembled tool calls, serialised after the text
            calls = json.dumps(self.tool_calls)
            text = calls if not text.strip() else text + "\n" + calls
        rnd = lambda v: round(v, 2) if v is not None else None  # noqa: E731
        ret = {
            "text": _THINK.sub("", text).strip(),
            "ttft_ms": int(ttft * 1000) if ttft else 0,
            "total_ms": int(total_duration * 1000),
            "tokens": token_count,
            "prompt_tokens": tel["prompt_tokens"],
            "cached_tokens": tel["cached_tokens"],
            "native_ttft_ms": rnd(tel["native_ttft_ms"]),
            "native_prefill_tps": rnd(tel["native_prefill_tps"]),
            "native_decode_tps": rnd(tel["native_decode_tps"]),
            "content": content,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "reasoning_tokens": tel["reasoning_tokens"],
            "reasoning_chars": reasoning_chars,
        }
        if tel["last_progress"]:
            ret["prompt_progress"] = tel["last_progress"]
        for key, raw in (("usage", tel["raw_usage"]), ("timings", tel["raw_timings"]), ("metrics", tel["raw_metrics"]),
                         ("kapitsa", tel["kapitsa_meta"]), ("stats", tel["raw_stats"])):
            if raw is not None:
                ret[key] = raw
        return ret


def _debug(message):
    """REPLAY_BENCH_DEBUG=1: one line of what the transport saw, on stderr."""
    if os.environ.get("REPLAY_BENCH_DEBUG") == "1":
        print(f"[debug] {message}", file=sys.stderr, flush=True)


def _send(endpoint_url, payload, tc, state, headers=None):
    """POST a chat-completions payload, stream the answer and collect text, tool calls and server telemetry."""
    _debug(f"POST {endpoint_url} (stream={payload.get('stream')}, stream_options={payload.get('stream_options')})")
    req = urllib.request.Request(
        endpoint_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **auth.auth_header(), **(headers or {})}
    )
    start_time = time.time()
    # Live progress for the spinner (see ProgressAnimation): every line received from the server counts.
    if state is None:
        state = {}
    state.update(start=start_time, last=start_time, lines=0, down=0)
    state.pop("first", None)  # a resend reuses the dict: no first token or progress yet
    state.pop("pp", None)
    # Runaway-generation cutoffs (tokens, seconds); cases that reason at length raise them.
    acc = _StreamAcc(start_time, tc.get("max_gen_tokens", 8192), tc.get("max_gen_seconds", 600), state)
    last_chunk = None
    saw_done = False

    try:
        with urllib.request.urlopen(req, timeout=3600, context=ssl_context()) as response:
            for line in response:
                now = time.time()
                if now - start_time > 3600:
                    return _error("timeout (3600s max duration reached)", "timeout")
                state["last"] = now
                state["lines"] += 1
                if state["lines"] == 1:
                    _set_read_timeout(response, STALL_SECONDS)
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data:"):
                    continue
                data = line_str[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                last_chunk = chunk
                runaway = acc.feed(chunk, now)
                if runaway:
                    return _error(runaway, "runaway")
    except urllib.error.HTTPError as e:  # the server answered with an error status: keep it and a short body
        try:
            body = " ".join(e.read(400).decode("utf-8", "replace").split())
        except (OSError, http.client.HTTPException):
            body = ""
        e.close()
        return _error("HTTP %s %s%s" % (e.code, e.reason or "", ": " + body[:200] if body else ""), "http", e.code)
    except urllib.error.URLError as e:
        return _error(f"Inference failure: {e.reason}", "unreachable")
    except (TimeoutError, socket.timeout):
        if state["lines"]:
            return _error(f"stalled: no data for {STALL_SECONDS:.0f}s after {state['lines']} lines", "stalled")
        return _error("timeout", "timeout")
    except (OSError, http.client.HTTPException, ValueError) as e:  # a dropped connection, a truncated or undecodable body
        return _error(f"Exception: {e}", "exception")

    _debug(f"{endpoint_url}: {state['lines']} lines, last chunk keys "
           f"{sorted(last_chunk) if last_chunk else None}, stats {acc.tel.get('raw_stats')}")
    if acc.stream_error:
        # Not transient (is_transient): the server answered and refused, e.g. out of memory; waiting will not help,
        # so no retry. Already prefixed "server: ", which query_with_retry only adds to errors it retried.
        return _error(acc.stream_error, "server")
    if not saw_done and acc.finish_reason is None:
        # The connection closed cleanly in the middle of an answer: not a result to grade.
        return _error("server: stream ended without finish_reason", "server")
    return acc.finish(time.time() - start_time)
