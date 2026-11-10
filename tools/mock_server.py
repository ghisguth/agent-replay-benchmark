#!/usr/bin/env python3
"""Mock OpenAI-compatible server for offline end-to-end runs of replay_bench.py (127.0.0.1 only).

    python3 tools/mock_server.py [--port N] [--flavor llamacpp|mtplx|ollama|lmstudio|mlx-dspark|mlx_lm|generic] [--delay SECONDS] [--log FILE]
                                [--cached-ratio R] [--chunk TOKENS] [--paths-log FILE] [--v0-chat-status CODE]

Prints `listening on http://127.0.0.1:PORT/v1` and serves until killed. Answers are deterministic: a tool call
(the first tool offered, empty arguments) when the request has tools, else a short text. Streaming responses carry
usage and llama.cpp-style timings; flavor llamacpp also answers /props (so the CLI detects llama.cpp) and honours
"return_progress": true with prompt_progress chunks before the first token. Flavor mtplx answers /health and
/v1/models (owned_by mtplx) so the CLI detects MTPLX, sends only ": keep-alive" comments during --delay while it
prefills like MTPLX, publishes it on GET /v1/mtplx/snapshot (in_flight[].prefill_state for the id of the
request's x-mtplx-request-id header, answered as "chatcmpl-<id>"), and adds mtplx_progress / mtplx_stats to the decode
chunks. Flavor mlx-dspark answers `Server: mlx-dspark Python/3.11.18`, /v1/models (a snapshot-hash id, owned_by
mlx-dspark, x_mlx_dspark with HF cache paths for target and drafter) and /doctor, and ends a stream with x_mlx_dspark.
Flavor mlx_lm answers like mlx_lm.server (`Server: BaseHTTP/0.6 Python/3.11.18`, /health {"status": "ok"}, models
without owned_by); every other flavor sends a `mock-server/1.0` Server header. The MTPLX prefill: a share --cached-ratio (default 0.9; a request field "mock_cached_ratio" overrides it) of the
prompt is cached, the rest is prefilled in --chunk token steps (default 2048) taking --delay seconds in all: "started", a
short setup, a chunk reading with no new token (tokens_done == cached_tokens), then one reading per finished chunk, and the
entry is gone before the first token. --log appends one JSON line per POST
(the request body), flushed at once.
"""

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "mock-model"
DSPARK_ID = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
DSPARK_TARGET = "/Users/USER/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/" + DSPARK_ID
DSPARK_DRAFTER = ("/Users/USER/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/"
                  "015e795645c74b1a0eeef3b570031fb62e769bc5")
MLX_LM_ID = "mlx-community/Mock-4bit"
N_CTX = 65536
PROGRESS = (0.25, 0.5, 1.0)
CACHED_RATIO = 0.9  # flavor mtplx: the share of the prompt that is cached
CHUNK = 2048  # flavor mtplx: the prefill chunk size in tokens


def _models(flavor):
    owner = {"llamacpp": "llamacpp", "mtplx": "mtplx", "ollama": "library", "lmstudio": "organization_owner",
             "mlx-dspark": "mlx-dspark"}.get(flavor, "mock")
    entry = {"id": MODEL, "object": "model", "created": 0, "owned_by": owner}
    if flavor == "mlx-dspark":
        entry.update(id=DSPARK_ID, display_name=DSPARK_ID + " (mlx-dspark dflash)",
                     x_mlx_dspark={"mode": "dflash", "target": DSPARK_TARGET, "drafter": DSPARK_DRAFTER})
    elif flavor == "mlx_lm":
        entry = {"id": MLX_LM_ID, "object": "model", "created": 0}
    elif flavor == "llamacpp":
        entry["meta"] = {"n_ctx": N_CTX}
    elif flavor == "ollama":
        pass
    elif flavor == "lmstudio":
        entry.update({
            "type": "llm",
            "publisher": "mock",
            "arch": "mock_arch",
            "compatibility_type": "mlx",
            "quantization": "4bit",
            "state": "loaded",
            "max_context_length": N_CTX,
            "loaded_context_length": N_CTX,
        })
    else:
        entry["max_model_len"] = N_CTX
    return {"object": "list", "data": [entry]}


def _routes(flavor):
    routes = {"/v1/models": _models(flavor), "/health": {"status": "ok"}}
    if flavor == "mtplx":
        routes["/health"] = {"ok": True, "model": MODEL, "chip": "Mock", "profile": {"name": "mock"},
                             "mlx_runtime": {"version": "0.0.0-mock"}}
    if flavor == "mlx-dspark":
        routes["/doctor"] = {"ok": True, "problems": [], "environment": {
            "version": "0.20.4", "platform": "Darwin", "machine": "arm64", "python": "3.11.18", "device": "Apple M2 Max",
            "chip": {"name": "Apple M2 Max", "generation": "M2", "family": "M2 Max", "gpu_cores": 38}},
            "models": [{"id": "qwen3.8-27b", "target": "mlx-community/Qwen3.8-27B-4bit", "mode": "dflash", "ready": True}]}
    if flavor == "llamacpp":
        routes["/props"] = {"default_generation_settings": {"n_ctx": N_CTX}, "total_slots": 1,
                            "model_path": "/models/mock-Q4.gguf", "build_info": "b9999-mock000"}
    if flavor == "ollama":
        routes["/api/version"] = {"version": "0.40.15"}
        routes["/api/ps"] = {
            "models": [{
                "name": MODEL, "model": MODEL, "size": 26123268259,
                "digest": "4010bd202712ea6bee89a163733011c461085c2bd1154caa2a49bcabb7c5eaa6",
                "details": {"parent_model": "", "format": "gguf", "family": "qwen35", "families": ["qwen35"],
                            "parameter_size": "27.3B", "quantization_level": "Q4_K_M"},
                "expires_at": "2026-11-09T11:43:08.938437-08:00", "size_vram": 26123268259,
                "context_length": 262144, "runner": "llamacpp",
            }],
        }
    if flavor == "lmstudio":
        routes["/api/v0/models"] = {
            "data": [{
                "id": MODEL,
                "object": "model",
                "type": "llm",
                "publisher": "mock",
                "arch": "mock_arch",
                "compatibility_type": "mlx",
                "quantization": "4bit",
                "state": "loaded",
                "max_context_length": N_CTX,
                "loaded_context_length": N_CTX,
            }]
        }
    return routes


def _prompt_tokens(payload):
    return max(1, len(json.dumps(payload.get("messages", []))) // 4)


def _call(i, name, args):
    return {"index": i, "id": "call_mock%d" % i, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _sanity_answer(payload, tools):
    """The right answer to a recognised sanity-suite prompt (bench/cases/sanity.py), else None."""
    msgs = payload.get("messages") or []
    user = next((m.get("content") for m in reversed(msgs) if m.get("role") == "user"), None)
    if not isinstance(user, str):
        return None
    names = {(t.get("function") or {}).get("name") for t in tools}
    if user.startswith("Summarize this diff into a standard conventional commit"):
        return "feat(storage): add async save with empty-data check", []
    if user.startswith("Route this user command"):
        return '{"route": "docker_rebuild", "confidence": 0.9, "args": {}}', []
    if "weather metrics for Seattle" in user and "get_weather" in names:
        return "", [_call(0, "get_weather", {"location": "Seattle, WA"})]
    if user.startswith("Return the capital as a JSON object"):
        return '{"capital": "Paris"}', []
    if user.startswith("Count from 1 to 5"):
        return "1, 2, 3, 4, 5", []
    if "weather in Paris and in Tokyo" in user and "get_weather" in names:
        return "", [_call(0, "get_weather", {"location": "Paris"}), _call(1, "get_weather", {"location": "Tokyo"})]
    if "250 euros" in user and "convert_currency" in names:
        return "", [_call(0, "convert_currency", {"amount": 250, "from": "EUR", "to": "JPY"})]
    if "bags sell for" in user:
        return "36", []
    if user.rstrip().endswith("What is the vault access code for the archive room?"):
        return "AMBER-4417-HERON", []
    marker = "character for character, with no quotes and nothing else:\n"
    if marker in user:
        return user.split(marker, 1)[1], []
    return None


def _answer(payload):
    """(content, tool_calls) for a request."""
    tools = payload.get("tools") or []
    canned = _sanity_answer(payload, tools)
    if canned is not None:
        return canned
    if tools:
        name = (tools[0].get("function") or {}).get("name") or "tool"
        return "", [{"index": 0, "id": "call_mock0", "type": "function",
                     "function": {"name": name, "arguments": "{}"}}]
    return "Mock answer.", []


def make_server(flavor="generic", delay=0.0, log=None, host="127.0.0.1", port=0, cached_ratio=CACHED_RATIO, chunk=CHUNK,
                paths_log=None, v0_chat_status=None, drop_stream_end=False):
    routes = _routes(flavor)
    in_flight = {}  # flavor mtplx: response id -> prefill_state (or None), what /v1/mtplx/snapshot lists
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        server_version = {"mlx-dspark": "mlx-dspark", "mlx_lm": "BaseHTTP/0.6"}.get(flavor, "mock-server/1.0")
        sys_version = "Python/3.11.18" if flavor in ("mlx-dspark", "mlx_lm") else ""

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _nap(self, secs):
            while secs > 0:  # keep-alives about every 0.25 s while "computing"
                step = min(secs, 0.25)
                time.sleep(step)
                secs -= step
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()

        def _mtplx_prefill(self, rid, p_tok, cached):
            """Prefill as MTPLX does it: ": keep-alive" comments on the stream and the state published on the snapshot,
            `delay` seconds in all: "started", a setup, a chunk reading with no new token, one reading per finished
            `chunk` tokens of the uncached suffix; the entry is gone before the first token."""
            new = p_tok - cached
            with lock:
                in_flight[rid] = {"phase": "started", "tokens_total": p_tok}
            try:
                self._nap(delay * 0.1)
                t0 = time.monotonic()
                state = {"phase": "chunk", "tokens_total": p_tok, "cached_tokens": cached, "new_prefill_tokens": new,
                         "chunk_size": chunk}
                with lock:
                    in_flight[rid] = dict(state, tokens_done=cached, elapsed_s=0.0)
                done = 0
                while done < new:
                    n = min(chunk, new - done)
                    self._nap(delay * 0.9 * n / new)
                    done += n
                    with lock:
                        in_flight[rid] = dict(state, tokens_done=cached + done, elapsed_s=time.monotonic() - t0)
            finally:
                with lock:
                    in_flight.pop(rid, None)

        def _note_path(self):
            if paths_log:
                with open(paths_log, "a") as f:
                    f.write("%s %s\n" % (self.command, self.path.split("?")[0]))

        def do_GET(self):
            path = self.path.split("?")[0]
            self._note_path()
            if flavor == "mtplx" and path == "/v1/mtplx/snapshot":
                with lock:
                    flights = [{"request_id": rid, "prefill_state": dict(ps) if ps else None}
                               for rid, ps in in_flight.items()]
                self._json(200, {"active_requests": len(flights), "in_flight": flights})
            elif path in routes:
                self._json(200, routes[path])
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            self._note_path()
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if log:
                with open(log, "a") as f:
                    f.write(json.dumps(payload) + "\n")
            if v0_chat_status and self.path.split("?")[0] == "/api/v0/chat/completions":
                self._json(v0_chat_status, {"error": "mock: /api/v0/chat/completions is refused"})
                return
            if not self.path.split("?")[0].endswith("/chat/completions"):
                self._json(404, {"error": "not found"})
                return
            mtplx_stream = flavor == "mtplx" and payload.get("stream")
            if delay and not mtplx_stream:
                time.sleep(delay)
            content, tool_calls = _answer(payload)
            p_tok = _prompt_tokens(payload)
            c_tok = 4 if tool_calls else 3
            usage = {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok}
            finish = "tool_calls" if tool_calls else "stop"
            timings = {"prompt_n": p_tok, "prompt_ms": 10.0, "prompt_per_second": p_tok / 0.01,
                       "predicted_n": c_tok, "predicted_ms": 20.0, "predicted_per_second": c_tok / 0.02}
            if not payload.get("stream"):
                msg = {"role": "assistant", "content": content or None}
                if tool_calls:
                    msg["tool_calls"] = [{k: v for k, v in tc.items() if k != "index"} for tc in tool_calls]
                res = {"id": "mock", "object": "chat.completion", "model": MODEL,
                       "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                       "usage": usage, "timings": timings}
                if flavor == "lmstudio":
                    res["stats"] = {"tokens_per_second": 25.0, "time_to_first_token": 0.04,
                                    "generation_time": 0.12, "stop_reason": "eosFound"}
                    res["model_info"] = {"arch": "mock_arch", "quant": "4bit"}
                self._json(200, res)
                return
            chunks = []
            if flavor == "llamacpp" and payload.get("return_progress"):
                chunks += [{"choices": [{"index": 0, "delta": {}}],
                            "prompt_progress": {"total": p_tok, "cache": 0, "processed": int(p_tok * f),
                                                "time_ms": int(10 * f)}} for f in PROGRESS]
            if content:
                chunks.append({"choices": [{"index": 0, "delta": {"role": "assistant", "content": content}}]})
            if tool_calls:
                chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": tool_calls}}]})
            if mtplx_stream:  # decode progress rides on the content chunks, the stats on the last one
                for c in chunks:
                    c["mtplx_progress"] = {"completion_tokens": c_tok}
            chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            last = {"choices": [], "usage": usage}
            if flavor == "llamacpp":
                last["timings"] = timings
            if mtplx_stream:
                last["mtplx_stats"] = {"ttft_s": max(delay, 0.01), "prefill_tok_s": p_tok / max(delay, 0.01),
                                       "decode_tok_s": 150.0, "cached_tokens": 0, "prompt_tokens": p_tok,
                                       "completion_tokens": c_tok}
            if flavor == "lmstudio":  # a stream carries only the draft counters (no speeds, no cache count)
                last["stats"] = {"total_draft_tokens_count": 49, "accepted_draft_tokens_count": 21,
                                 "rejected_draft_tokens_count": 28}
            if flavor == "mlx-dspark":
                last["x_mlx_dspark"] = {"mode": "dflash", "accept_len": 5.714, "tokens_per_sec": 19.6,
                                        "decode_tokens_per_sec": 50.9, "target_forwards": 15, "cap": 7,
                                        "prompt_tokens": p_tok, "cached_tokens": 0, "completion_tokens": c_tok,
                                        "ttft_seconds": 0.5, "prefill_tokens_per_sec": p_tok / 0.5}
            chunks.append(last)
            if drop_stream_end:  # the server dies mid-answer: no finish chunk, no usage, no [DONE]
                chunks = chunks[:-2]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                if mtplx_stream:
                    hint = self.headers.get("x-mtplx-request-id") or uuid.uuid4().hex
                    rid = hint if hint.startswith("chatcmpl-") else "chatcmpl-" + hint
                    ratio = payload.get("mock_cached_ratio", cached_ratio)
                    self._mtplx_prefill(rid, p_tok, min(p_tok - 1, int(p_tok * ratio)) if p_tok > 1 else 0)
                for c in chunks:
                    c.update(id="mock", object="chat.completion.chunk", model=MODEL)
                    self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                    self.wfile.flush()
                if not drop_stream_end:
                    self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    httpd = ThreadingHTTPServer((host, port), H)
    httpd.daemon_threads = True
    return httpd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: a free one)")
    ap.add_argument("--flavor", "--flavour", dest="flavor",
                    choices=("llamacpp", "mtplx", "ollama", "lmstudio", "mlx-dspark", "mlx_lm", "generic"),
                    default="generic")
    ap.add_argument("--delay", type=float, default=0.0, metavar="SECONDS", help="sleep before every chat answer")
    ap.add_argument("--log", metavar="FILE", help="append every POST body as one JSON line")
    ap.add_argument("--cached-ratio", type=float, default=CACHED_RATIO, metavar="R",
                    help="flavor mtplx: share of the prompt that is cached (default %(default)s)")
    ap.add_argument("--chunk", type=int, default=CHUNK, metavar="TOKENS",
                    help="flavor mtplx: prefill chunk size (default %(default)s)")
    ap.add_argument("--paths-log", metavar="FILE", help="append `METHOD /path` of every request (GET and POST)")
    ap.add_argument("--v0-chat-status", type=int, metavar="CODE",
                    help="answer POST /api/v0/chat/completions with this HTTP status (e.g. 404: an LM Studio without it)")
    ap.add_argument("--drop-stream-end", action="store_true",
                    help="end every stream after the content, without finish_reason, usage or [DONE]")
    args = ap.parse_args(argv)
    httpd = make_server(args.flavor, args.delay, args.log, port=args.port, cached_ratio=args.cached_ratio, chunk=args.chunk,
                        paths_log=args.paths_log, v0_chat_status=args.v0_chat_status,
                        drop_stream_end=args.drop_stream_end)
    print("listening on http://127.0.0.1:%d/v1" % httpd.server_address[1], flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
