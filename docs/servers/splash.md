# Splash

## How it is detected
`GET /status` returns JSON with `schema_version`, `identity` and `instance`; `/v1/models` says `owned_by: splash`.
Splash exposes no version number, so the label carries the first 8 hex digits of `identity.cache.build_id`
(`src-<sha256>`); the full id is kept in the state header. `--server-version` overrides the label.

## Telemetry read
- `timings` (llama.cpp style, with `cache_n`) and `metrics`: `request_latency`, `decode`, `prefill`, `cache`.
- `metrics.speculation`: drafted, accepted and cycles when speculative decoding runs.
- `prompt_progress` chunks while the prompt is prefilled.

## Prefill progress
Yes, with `"return_progress": true` (same mechanism as llama.cpp, including the resend without it on HTTP 400).

## Useful server flags
Splash's own options (context size, prompt cache, speculative decoding) are documented with the server: check your
version. For a fair comparison note the Splash build id in the run, because results move between builds.

## Sample report (quickstart, Qwen3.8-27B)

Quickstart reports for Qwen3.8-27B on this server: [GGUF](captures/splash-gguf.md), [MLX](captures/splash-mlx.md).
