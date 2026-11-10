# MTPLX

## How it is detected
`GET /health` returns JSON with `ok: true` and `mlx_runtime`, `profile`, `chip`, `model`; `/v1/models` says
`owned_by: mtplx`. The version comes from the health data or the runtime path.

## Telemetry read
- `mtplx_stats` in the stream: prefill and decode tokens per second, token counts.
- `mtplx_progress`: running completion tokens and heartbeats.
- Cached prompt tokens are counted so the prefill speed is not inflated by them.

## Prefill progress
Yes, by polling: the benchmark sends `x-mtplx-request-id` and reads `GET /v1/mtplx/snapshot` while the request
prefills (the server sends only keep-alive comments then). No request field is added, so it cannot be rejected.

## Useful server flags
- `mtplx serve --model MODEL`: API server only, default `127.0.0.1:8000`; `--port N` changes it.
- `mtplx start`: interactive (pick model and mode), shares the server with the MTPLX app.
- `--model-search-dir DIR` (repeatable): where local models are looked up.
- `mtplx tune --model MODEL`: tune for the machine (see the MTPLX README).

## Sample report (quickstart, Qwen3.8-27B)

The quickstart report for Qwen3.8-27B on this server is in [`captures/mtplx.md`](captures/mtplx.md).
