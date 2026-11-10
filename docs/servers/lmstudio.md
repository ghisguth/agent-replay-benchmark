# LM Studio

## How it is detected
`GET /api/v0/models` returns `data` entries with `arch`, `state`, `max_context_length`; no version is exposed.
The quantization, format (`gguf` or `mlx`), architecture and the loaded context come from the same call.

## Telemetry read
Chat requests go to `/api/v0/chat/completions` because `/v1` leaves out the `stats` object. In a stream the last
chunk (`choices: []`) carries `usage` and, with speculative decoding on, `stats.total_draft_tokens_count` and
`accepted_draft_tokens_count`: they give the Metrics row "Draft acceptance" (accepted ÷ drafted). A stream has no
speeds, no TTFT and no cached tokens, so decode, prefill and TTFT are measured on the client clock (`~`) and C% is `–`.
(A non-streaming response has `tokens_per_second` and `time_to_first_token` too, which the benchmark does not use.)
Because the cache is unknown, new prompt tokens are estimated as the prompt minus the previous prompt of the same
stream (the whole prompt for a first request); this feeds the `~` prefill figures and "Prefill by append size", and
the report's legend says so. If the `/api/v0` route is rejected the benchmark falls back to `/v1/chat/completions` for
the rest of the run and says so on stderr.

## Prefill progress
No: the live line shows an estimate (`~`).

## Useful server settings
- Start the server in the app (Developer tab) or with `lms server start`.
- Load the model with a context large enough for the preset: `lms load MODEL --context-length N` or the load dialog
  (check your version for the flag name).
- Keep one model loaded; a second one competes for memory.

## Sample report (quickstart, Qwen3.8-27B)

The quickstart report for Qwen3.8-27B on this server is in [`captures/lmstudio.md`](captures/lmstudio.md).
