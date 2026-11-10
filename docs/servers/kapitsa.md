# Kapitsa

## How it is detected
`GET /cluster` returns JSON with `proto` and `version` (e.g. `0.2.15`), plus `chip` and `model`;
`/v1/models` says `owned_by: nace`. The label shows the version, the state header the chip.

## Telemetry read
- `usage.kapitsa`: `prefill_ms`, `decode_ms`; the decode speed is derived from them.
- `kapitsa_progress` events while the prompt is prefilled.
- Empty keep-alive deltas are ignored for the time to first token.

## Prefill progress
Yes, but opt-in: the benchmark sends the request header `X-Kapitsa-Bench-Telemetry` instead of a body field.
`--no-prompt-progress` leaves the header out.

## Useful server flags
Kapitsa's own options are documented with the server: check your version.

## Sample report (quickstart, Qwen3.8-27B)

The quickstart report for Qwen3.8-27B on this server is in [`captures/kapitsa.md`](captures/kapitsa.md).
