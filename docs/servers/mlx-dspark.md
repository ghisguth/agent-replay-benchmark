# mlx-dspark

[mlx-dspark](https://github.com/ARahim3/mlx-dspark) (checked with 0.20.4) is an MLX server with speculative decoding:
a drafter proposes tokens and the target model verifies them, in mode `dflash` or `dspark`.

## How it is detected
Any one of: the response header `Server: mlx-dspark Python/3.x`, `owned_by: mlx-dspark` on a `/v1/models` entry, or an
`x_mlx_dspark` object on one. Version and chip come from `GET /doctor` (`environment.version`, `environment.chip.name`,
else `environment.device`); the catalog `/doctor` also lists is not read. The model id is a snapshot hash; when
`x_mlx_dspark.target` is a Hugging Face cache path (`.../models--ORG--NAME/snapshots/...`) the benchmark shows and names
files with `ORG/NAME` (file name form `ORG-NAME`), the drafter the same way in the header line, and the mode
(`dflash` or `dspark`) ends the state file name (`..._coding-large_dflash_...jsonl`) unless `--run-note` is given. The
local path (it contains a user name) never reaches a label, a file name or a state.

## Telemetry read
`x_mlx_dspark` in the last chunk of the stream, next to `usage` (verified on a streamed response, mode dspark):
- `decode_tokens_per_sec` is the native decode speed, `prefill_tokens_per_sec` the native prefill speed and
  `ttft_seconds` the native TTFT. `tokens_per_sec` is end to end (prefill included) and is never used as decode speed.
  When the whole prompt comes from the cache, `prefill_tokens_per_sec` is left out and the report uses the client clock.
- `accept_len` (tokens kept per draft-verify cycle) and `target_forwards` (cycles) fill the speculation figures
  (Tok/cycle, Cycles/s, ms/cycle in the context breakdown).
- `prompt_tokens`, `cached_tokens` and `completion_tokens` back up the OpenAI `usage` block.

## Prefill progress
No: the live line shows an estimate (`~`). During prefill the server sends a chunk with an empty `delta` every 15 s
(a keep-alive, no progress figure); these never count as the first token.

## Useful server settings
- Start command: `mlx-dspark serve --model MODEL` (placeholder: check the mlx-dspark README for your version).
- `GET /doctor` tells whether the target and drafter are installed and ready.
- Pick the mode when starting the server; the benchmark records whatever the server reports.

## Sample report (quickstart, Qwen3.8-27B)

Quickstart reports for Qwen3.8-27B on this server: [dflash](captures/mlx-dspark-dflash.md),
[dspark](captures/mlx-dspark-dspark.md).
