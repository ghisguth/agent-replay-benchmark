# OMLX

## How it is detected
`/v1/models` says `owned_by: omlx`; `GET /admin/api/stats` gives `cli_prefix`, `engines` (mlx, mlx-lm with
versions and the mlx-lm commit) and uptime, and `/health` is the fallback. The label shows the OMLX version, or the
mlx-lm version when OMLX does not report its own.

## Telemetry read
The final chunk's `usage` carries OMLX's own measurements next to the OpenAI token counts:
- `time_to_first_token` (seconds, the native TTFT), `prompt_eval_duration` and `generation_duration`.
- `prompt_tokens_per_second` (prefill) and `generation_tokens_per_second` (decode).
- `prompt_tokens_details.cached_tokens`, the cached share of the prompt, and `completion_tokens_details.reasoning_tokens`.

Decode and prefill speeds are therefore native (the quickstart sample shows about 196 tok/s prefill and 43 tok/s decode);
only the live prefill progress is an estimate.

## Prefill progress
No: the live line shows an estimate (`~`).

## Useful server flags
Check your version: the OMLX command line and its context and cache options are documented with the server, and were
not verified here.

## Sample report (quickstart, Qwen3.8-27B)

The quickstart report for Qwen3.8-27B on this server is in [`captures/omlx.md`](captures/omlx.md).
