"""Server telemetry extraction and normalization across backends.

Which server sends which field (SSE chunk key -> what is read from it):

    source            chunk key                   fields used
    ----------------  --------------------------  ---------------------------------------------------------
    OpenAI generic    usage                       completion/prompt tokens, cached and reasoning token details
    llama.cpp         timings                     prompt_ms, predicted_n, prompt_n, *_per_second
                      prompt_progress             prefill progress (request field return_progress)
    Splash            metrics                     request_latency, decode, prefill, cache
                      prompt_progress             prefill progress (request field return_progress)
    Kapitsa           usage.kapitsa               prefill_ms, decode_ms
                      kapitsa_progress            prefill progress (a request header opts in)
    MTPLX             mtplx_stats                 prefill/decode tok/s, token counts
                      mtplx_progress              running completion_tokens, heartbeats
    LM Studio         stats                       streaming: only the speculative-decoding counters
                                                  (total_draft_tokens_count, accepted_draft_tokens_count; also read
                                                  from usage); no speeds, no cached tokens in a stream.
                                                  Non-streaming: tokens_per_second (decode), time_to_first_token
    mlx-dspark        x_mlx_dspark (VERIFIED, non-streaming shape; a streamed chunk is assumed to have the same keys)
                                                  decode_tokens_per_sec, prefill_tokens_per_sec, ttft_seconds,
                                                  accept_len (tokens per cycle), target_forwards (cycles),
                                                  prompt/cached/completion_tokens. tokens_per_sec is END-TO-END
                                                  (prefill included) and is never read as decode speed.

A value no source reports stays None ("not measured"), never 0.
"""

# Field names that a recorded response (fixtures/refs, fixtures/samples) carries: OMLX sends them in `usage`.
DECODE_FIELDS = ("generation_tokens_per_second",)
PREFILL_FIELDS = ("prompt_tokens_per_second",)
TTFT_FIELDS = (("time_to_first_token", 1000.0),)


def init_acc():
    """Fresh accumulator dict for telemetry collected across streaming chunks."""
    return {
        "exact_tokens": 0,
        "prompt_tokens": 0,
        "cached_tokens": None,
        "reasoning_tokens": None,
        "native_ttft_ms": None,
        "native_prefill_tps": None,
        "native_decode_tps": None,
        "is_kapitsa": False,
        "kapitsa_meta": None,
        "raw_usage": None,
        "raw_timings": None,
        "raw_metrics": None,
        "raw_stats": None,
        "last_progress": None,
    }


def _first_field(obj, fields):
    if not isinstance(obj, dict):
        return None
    for k in fields:
        v = obj.get(k)
        if v is not None:
            return v
    return None


def timings_prefill_tps(timings, prompt_tokens, cached_tokens):
    """New-token prefill rate from a `timings` object. Servers disagree on what prompt_n and prompt_per_second count:
    llama.cpp reports only the new tokens (prompt_n < prompt), Splash reports the whole prompt as prompt_n but its rate
    over the new tokens, and MTPLX rates the whole prompt. The raw rate is right unless it implies more tokens than were
    prefilled (rate x prompt_ms counts cached tokens); then it is recomputed as new tokens / prompt_ms."""
    raw = timings.get("prompt_per_second")
    if raw is None:
        return None
    raw = float(raw)
    try:
        ms = float(timings.get("prompt_ms") or 0.0)
    except (TypeError, ValueError):
        ms = 0.0
    prompt_tokens = prompt_tokens or int(timings.get("prompt_n") or 0)
    uncached = max(0, prompt_tokens - (cached_tokens or 0))
    if cached_tokens and ms > 0 and uncached > 0 and raw * ms / 1000.0 > 1.05 * uncached + 16:
        return uncached * 1000.0 / ms
    return raw


def _mtplx_stats(mtplx, acc):
    """MTPLX (mtplx.com, an MLX server): the final chunk of a stream, or a non-stream response, carries `mtplx_stats`.
    Seen in MTPLX 2.12.x: {"decode_tok_s": 42.0, "mtp_depth": 3, "request_elapsed_s": 1.25, ...}; `ttft_s`,
    `prefill_tok_s` and `cached_tokens`/`prompt_tokens`/`completion_tokens` appear when the server measured them.
    The speculative-decoding counters of the same request come as `timings` (draft_n, draft_n_accepted)."""
    if not isinstance(mtplx, dict) or not mtplx:
        return
    ttft_s = mtplx.get("ttft_s")
    if acc.get("native_ttft_ms") is None and ttft_s is not None:
        acc["native_ttft_ms"] = float(ttft_s) * 1000.0
    if mtplx.get("prefill_tok_s") is not None:
        acc["native_prefill_tps"] = float(mtplx["prefill_tok_s"])
    if mtplx.get("decode_tok_s") is not None:
        acc["native_decode_tps"] = float(mtplx["decode_tok_s"])
    if (acc.get("cached_tokens") is None or acc.get("cached_tokens") == 0) and mtplx.get("cached_tokens") is not None:
        acc["cached_tokens"] = int(mtplx["cached_tokens"])
    if acc.get("prompt_tokens", 0) == 0 and mtplx.get("prompt_tokens") is not None:
        acc["prompt_tokens"] = int(mtplx["prompt_tokens"])
    if acc.get("exact_tokens", 0) == 0 and mtplx.get("completion_tokens") is not None:
        acc["exact_tokens"] = int(mtplx["completion_tokens"])


def _mtplx_progress(mp, acc):
    """MTPLX streams `mtplx_progress` chunks while it works, also as keep-alive heartbeats before the first token:
    {"phase": "generating", "completion_tokens": 0, "heartbeat": true}. The token count only grows."""
    if isinstance(mp, dict) and mp.get("completion_tokens") is not None:
        acc["exact_tokens"] = max(acc.get("exact_tokens", 0), int(mp["completion_tokens"]))


def _lmstudio_stats(stats, acc):
    """LM Studio (lmstudio.ai): `stats` object at root of response or chunk.
    Seen in LM Studio 0.3.x:
      {"tokens_per_second": 22.07, "time_to_first_token": 0.786, "generation_time": 2.22, "stop_reason": "eosFound"}
    `tokens_per_second` is decode tok/s; `time_to_first_token` is in seconds (native TTFT = s * 1000.0,
    no native prefill speed: prompt tokens / TTFT would only be an estimate, which the client-clock fallback shows
    (marked `~`)). Those exist in a non-streaming response only: the last chunk of a stream (choices: []) carries
    just the speculative-decoding counters {"total_draft_tokens_count", "accepted_draft_tokens_count",
    "rejected_draft_tokens_count"} (a non-streaming response has them in `usage` too)."""
    if not isinstance(stats, dict) or not stats:
        return
    acc["raw_stats"] = stats
    _draft_counts(stats, acc)
    tps = stats.get("tokens_per_second")
    if tps is not None and acc.get("native_decode_tps") is None:
        try:
            acc["native_decode_tps"] = float(tps)
        except (TypeError, ValueError):
            pass
    ttft_s = stats.get("time_to_first_token")
    if ttft_s is not None:
        try:
            ttft_val = float(ttft_s)
            if ttft_val > 0:
                if acc.get("native_ttft_ms") is None:
                    acc["native_ttft_ms"] = ttft_val * 1000.0
        except (TypeError, ValueError):
            pass


def _draft_counts(obj, acc):
    """LM Studio's speculative-decoding counters (total_draft_tokens_count, accepted_draft_tokens_count) as
    metrics.speculation {drafted_tokens, accepted_draft_tokens}, the fields Splash's speculation uses (acceptance =
    accepted / drafted). No `cycles`: LM Studio does not report them."""
    if not isinstance(obj, dict):
        return
    total, accepted = obj.get("total_draft_tokens_count"), obj.get("accepted_draft_tokens_count")
    if isinstance(total, int) and isinstance(accepted, int) and total > 0:
        metrics = acc.get("raw_metrics") if isinstance(acc.get("raw_metrics"), dict) else {}
        sp = dict(metrics.get("speculation") or {})
        sp.update(drafted_tokens=total, accepted_draft_tokens=accepted)
        acc["raw_metrics"] = dict(metrics, speculation=sp)


def _mlx_dspark_stats(x, acc):
    """mlx-dspark 0.20.4: `x_mlx_dspark` at the root of a response (a stream's chunk is assumed to match). Verified keys:
    mode, accept_len, tokens_per_sec (END-TO-END, never read as decode), decode_tokens_per_sec, target_forwards, cap,
    prompt_tokens, cached_tokens, completion_tokens, context_tokens, prefill_seconds, decode_seconds, ttft_seconds,
    prefill_tokens_per_sec. accept_len = tokens kept per draft-verify cycle and target_forwards = cycles, stored as
    metrics.speculation (cycles, accepted_draft_tokens = (accept_len - 1) x cycles: Splash's tok/cycle = 1 + accepted /
    cycles then gives accept_len back)."""
    if not isinstance(x, dict) or not x:
        return
    num = lambda k: x[k] if isinstance(x.get(k), (int, float)) and not isinstance(x.get(k), bool) else None  # noqa: E731
    if num("decode_tokens_per_sec") is not None:
        acc["native_decode_tps"] = float(x["decode_tokens_per_sec"])
    if num("prefill_tokens_per_sec") is not None:
        acc["native_prefill_tps"] = float(x["prefill_tokens_per_sec"])
    if num("ttft_seconds") is not None:
        acc["native_ttft_ms"] = float(x["ttft_seconds"]) * 1000.0
    if acc.get("cached_tokens") is None and num("cached_tokens") is not None:
        acc["cached_tokens"] = int(x["cached_tokens"])
    if acc.get("prompt_tokens", 0) == 0 and num("prompt_tokens") is not None:
        acc["prompt_tokens"] = int(x["prompt_tokens"])
    if acc.get("exact_tokens", 0) == 0 and num("completion_tokens") is not None:
        acc["exact_tokens"] = int(x["completion_tokens"])
    cycles, accept_len = num("target_forwards"), num("accept_len")
    if cycles and accept_len is not None and cycles > 0:
        metrics = acc.get("raw_metrics") if isinstance(acc.get("raw_metrics"), dict) else {}
        sp = dict(metrics.get("speculation") or {})
        sp.update(cycles=int(cycles), accepted_draft_tokens=max(0, int(round((accept_len - 1) * cycles))))
        acc["raw_metrics"] = dict(metrics, speculation=sp)


def normalize(chunk, acc, current_time=None, start_time=None, ttft=None):
    """Update `acc` with telemetry extracted from an SSE response chunk."""
    if not isinstance(chunk, dict):
        return acc

    # 1. usage object
    usage = chunk.get("usage")
    if isinstance(usage, dict) and usage:
        acc["raw_usage"] = usage
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            rt = details.get("reasoning_tokens")
            if isinstance(rt, int):
                acc["reasoning_tokens"] = rt

        # A null count (some servers send "completion_tokens": null) counts as absent.
        for key, names in (("exact_tokens", ("completion_tokens", "output_tokens")),
                           ("prompt_tokens", ("prompt_tokens", "input_tokens"))):
            val = next((usage[n] for n in names if usage.get(n) is not None), acc.get(key, 0))
            acc[key] = val

        cached = usage.get("cached_tokens")
        if cached is None:
            pt_details = usage.get("prompt_tokens_details") or {}
            cached = pt_details.get("cached_tokens")
            if cached is None:
                in_details = usage.get("input_tokens_details") or {}
                cached = in_details.get("cached_tokens")
                if cached is None:
                    cached = pt_details.get("cache_read_input_tokens")
        if cached is not None:
            acc["cached_tokens"] = cached

        # Decode tok/s from usage
        usage_decode = _first_field(usage, DECODE_FIELDS)
        if usage_decode is not None and acc.get("native_decode_tps") is None:
            acc["native_decode_tps"] = float(usage_decode)
        elif acc.get("native_decode_tps") is None and usage.get("generation_duration") and acc.get("exact_tokens"):
            gen_dur = float(usage["generation_duration"])
            if gen_dur > 0:
                acc["native_decode_tps"] = acc["exact_tokens"] / gen_dur

        uncached = max(0, acc.get("prompt_tokens", 0) - (acc.get("cached_tokens") or 0))

        # Kapitsa metadata embedded in usage
        kapitsa = usage.get("kapitsa")
        if isinstance(kapitsa, dict):
            acc["is_kapitsa"] = True
            acc["kapitsa_meta"] = kapitsa
            if kapitsa.get("prefill_ms") is not None:
                pf_ms = float(kapitsa["prefill_ms"])
                if acc.get("native_ttft_ms") is None:
                    acc["native_ttft_ms"] = pf_ms
                if acc.get("native_prefill_tps") is None and pf_ms > 0 and uncached > 0:
                    acc["native_prefill_tps"] = (uncached * 1000.0) / pf_ms
            if acc.get("native_decode_tps") is None and kapitsa.get("decode_ms"):
                d_ms = float(kapitsa["decode_ms"])
                tokens = acc.get("exact_tokens", 0)
                if d_ms > 0 and tokens:
                    acc["native_decode_tps"] = ((tokens - 1) * 1000.0 / d_ms) if tokens > 1 else (tokens * 1000.0 / d_ms)

        # Prefill durations in usage
        if acc.get("native_prefill_tps") is None and usage.get("prompt_eval_duration") and uncached > 0:
            pe_dur = float(usage["prompt_eval_duration"])
            if pe_dur > 0:
                acc["native_prefill_tps"] = uncached / pe_dur

        if acc.get("native_prefill_tps") is None:
            usage_prefill = _first_field(usage, PREFILL_FIELDS)
            if usage_prefill is not None:
                raw_prefill = float(usage_prefill)
                prompt_tok = acc.get("prompt_tokens", 0)
                cached_tok = acc.get("cached_tokens") or 0
                if cached_tok > 0 and prompt_tok > 0:
                    acc["native_prefill_tps"] = raw_prefill * (uncached / prompt_tok)
                else:
                    acc["native_prefill_tps"] = raw_prefill

        # TTFT in usage
        if acc.get("native_ttft_ms") is None:
            for k, mult in TTFT_FIELDS:
                if k in usage:
                    acc["native_ttft_ms"] = float(usage[k]) * mult
                    break

    # 2. mtplx_stats (MTPLX)
    _mtplx_stats(chunk.get("mtplx_stats"), acc)

    # 2b. stats (LM Studio); its non-streaming usage carries the draft counters too
    _lmstudio_stats(chunk.get("stats"), acc)
    _draft_counts(usage, acc)

    # 2c. x_mlx_dspark (mlx-dspark)
    _mlx_dspark_stats(chunk.get("x_mlx_dspark"), acc)

    # 3. timings object (llama.cpp)
    timings = chunk.get("timings")
    if isinstance(timings, dict) and timings:
        acc["raw_timings"] = timings
        if acc.get("native_ttft_ms") is None and timings.get("prompt_ms") is not None:
            acc["native_ttft_ms"] = float(timings["prompt_ms"])
        if acc.get("exact_tokens", 0) == 0 and "predicted_n" in timings:
            acc["exact_tokens"] = int(timings["predicted_n"])
        if acc.get("prompt_tokens", 0) == 0 and "prompt_n" in timings:
            acc["prompt_tokens"] = int(timings["prompt_n"])
        if acc.get("cached_tokens") is None and "cache_n" in timings:
            acc["cached_tokens"] = int(timings["cache_n"])
        if acc.get("native_decode_tps") is None and timings.get("predicted_per_second") is not None:
            acc["native_decode_tps"] = float(timings["predicted_per_second"])
        if acc.get("native_prefill_tps") is None and timings.get("prompt_per_second") is not None:
            acc["native_prefill_tps"] = timings_prefill_tps(
                timings, acc.get("prompt_tokens", 0), acc.get("cached_tokens") or 0)

    # 4. MLX-style aliases (unverified; MTPLX's own keys are mtplx_stats and mtplx_progress)

    # 5. metrics (Splash)
    splash = chunk.get("metrics")
    if isinstance(splash, dict) and splash:
        acc["raw_metrics"] = splash
        req_lat = splash.get("request_latency") or {}
        if isinstance(req_lat, dict):
            if acc.get("native_ttft_ms") is None:
                ttft_val = req_lat.get("start_to_first_token_ms") or req_lat.get("ttft_ms")
                if ttft_val is not None:
                    acc["native_ttft_ms"] = float(ttft_val)

            if acc.get("native_decode_tps") is None:
                dec = _first_field(req_lat, ("stream_tokens_per_second",))
                if dec is not None:
                    acc["native_decode_tps"] = float(dec)
                elif req_lat.get("first_token_to_done_ms") and splash.get("decode", {}).get("tokens"):
                    f_ms = float(req_lat["first_token_to_done_ms"])
                    d_tok = float(splash["decode"]["tokens"])
                    if f_ms > 0 and d_tok > 0:
                        acc["native_decode_tps"] = d_tok * 1000.0 / f_ms

            if acc.get("native_prefill_tps") is None:
                if req_lat.get("start_to_first_token_ms"):
                    s_ms = float(req_lat["start_to_first_token_ms"])
                    p_tok = (splash.get("prefill") or {}).get("tokens")
                    if p_tok is None:
                        p_tok = max(0, acc.get("prompt_tokens", 0) - (acc.get("cached_tokens") or 0))
                    if s_ms > 0 and p_tok > 0:
                        acc["native_prefill_tps"] = float(p_tok) * 1000.0 / s_ms

        cache_obj = splash.get("cache")
        if isinstance(cache_obj, dict):
            if (acc.get("cached_tokens") is None or acc.get("cached_tokens") == 0) and "matched_tokens" in cache_obj:
                acc["cached_tokens"] = cache_obj["matched_tokens"]

    # 6. kapitsa root
    kapitsa_root = chunk.get("kapitsa")
    if isinstance(kapitsa_root, dict) and kapitsa_root:
        acc["is_kapitsa"] = True
        if acc.get("kapitsa_meta") is None:
            acc["kapitsa_meta"] = kapitsa_root
        if kapitsa_root.get("prefill_ms") is not None:
            pf_ms = float(kapitsa_root["prefill_ms"])
            if acc.get("native_ttft_ms") is None:
                acc["native_ttft_ms"] = pf_ms
            if acc.get("native_prefill_tps") is None and pf_ms > 0:
                uncached_pf = max(0, acc.get("prompt_tokens", 0) - (acc.get("cached_tokens") or 0))
                if uncached_pf > 0:
                    acc["native_prefill_tps"] = (uncached_pf * 1000.0) / pf_ms
        if acc.get("native_decode_tps") is None and kapitsa_root.get("decode_ms"):
            d_ms = float(kapitsa_root["decode_ms"])
            tokens = acc.get("exact_tokens", 0)
            if d_ms > 0 and tokens:
                acc["native_decode_tps"] = ((tokens - 1) * 1000.0 / d_ms) if tokens > 1 else (tokens * 1000.0 / d_ms)

    # 8. prompt_progress (Splash, llama.cpp)
    pp = chunk.get("prompt_progress")
    if isinstance(pp, dict) and ttft is None:
        try:
            nxt = {k: int(pp[k]) for k in ("total", "cache", "processed", "time_ms")}
        except (KeyError, TypeError, ValueError):
            nxt = None
        if nxt:
            last = acc.get("last_progress")
            if last:
                nxt["processed"] = max(nxt["processed"], last["processed"])
                nxt["time_ms"] = max(nxt["time_ms"], last["time_ms"])
            acc["last_progress"] = nxt

    # 9. kapitsa_progress (Kapitsa)
    kp = chunk.get("kapitsa_progress")
    if isinstance(kp, dict) and ttft is None and kp.get("phase") == 1:
        try:
            time_ms = int((current_time - start_time) * 1000) if current_time and start_time else 0
            nxt = {
                "total": int(kp["prompt_tokens"]),
                "cache": int(kp["cached_tokens"]),
                "processed": int(kp["processed_tokens"]),
                "time_ms": time_ms,
            }
        except (KeyError, TypeError, ValueError):
            nxt = None
        if nxt:
            last = acc.get("last_progress")
            if last:
                nxt["processed"] = max(nxt["processed"], last["processed"])
                nxt["time_ms"] = max(nxt["time_ms"], last["time_ms"])
            acc["last_progress"] = nxt

    # 10. mtplx_progress (MTPLX streaming decode tokens)
    _mtplx_progress(chunk.get("mtplx_progress"), acc)

    drop_nonpositive_native(acc)
    return acc


def drop_nonpositive_native(acc):
    """A zero or negative native duration or rate is "not measured" (a server that failed mid-request reports all-zero
    timings next to real token counts): None, so no 0.0 speed shows and no fake pause is derived from it."""
    for key in ("native_ttft_ms", "native_prefill_tps", "native_decode_tps"):
        v = acc.get(key)
        if v is not None and not v > 0:  # also NaN
            acc[key] = None


def finalize(acc, total_duration, ttft=None, token_count=0):
    """Post-stream adjustments: derive ttft fallback, drop cache-hit prefill, estimate kapitsa decode speed.

    Returns the updated ttft (seconds).
    """
    drop_nonpositive_native(acc)
    if ttft is None and acc.get("native_ttft_ms") is not None:
        ttft = acc["native_ttft_ms"] / 1000.0

    prompt_tokens = acc.get("prompt_tokens", 0)
    cached_tokens = acc.get("cached_tokens") or 0
    if acc.get("native_prefill_tps") is not None and prompt_tokens > 0 and (prompt_tokens - cached_tokens) < 10:
        acc["native_prefill_tps"] = None

    if acc.get("is_kapitsa") and acc.get("native_decode_tps") is None and ttft is not None:
        decode_duration = total_duration - ttft
        if decode_duration > 0 and token_count > 0:
            acc["native_decode_tps"] = (
                (token_count - 1) / decode_duration
                if token_count > 1
                else token_count / decode_duration
            )
    return ttft

