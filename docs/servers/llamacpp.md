# llama.cpp (llama-server)

## How it is detected
`GET /props` returns JSON with `build_info` (e.g. `b6123-1a2b3c4`), `total_slots` or `default_generation_settings`.
The version shown is the build number, the model file name comes from `model_path`.

## Telemetry read
- `timings` in the last SSE chunk: `prompt_ms`, `prompt_n`, `predicted_n`, `prompt_per_second`,
  `predicted_per_second`, and `cache_n` (cached prompt tokens) where the build sends it.
- `prompt_progress` chunks while the prompt is prefilled.
- OpenAI `usage` for the token counts.

## Prefill progress
Yes. The benchmark adds `"return_progress": true` to streaming requests. A server that answers HTTP 400 gets the request
again without it, and progress is off for the rest of the run. `--no-prompt-progress` turns it off.

## Useful server flags
- `-m FILE.gguf`: the model. `-ngl 99`: offload all layers to the GPU.
- `-c N` (`--ctx-size`): the context for all slots together. With `-np N` slots each gets `N / np`, so use `-np 1`
  for a long replay (the quickstart needs 24k, `large` 128k, `full` 236k).
- `--jinja`: use the model's chat template; needed for tool calls.
- `-fa on`: flash attention (older builds take plain `-fa`: check your version).
- `--cache-reuse N` and the default prompt cache: a replay appends to the previous prompt, so reuse decides the
  prefill cost. Check your version for the exact defaults.
- `--thinking on|off` sends `chat_template_kwargs.enable_thinking` (replay requests record it as true).
- `--reasoning-format`, `--reasoning-budget`: how thinking is returned and capped (check your version).

```bash
llama-server -m Qwen3.8-27B-Q8_0.gguf -c 262144 -np 1 -ngl 99 --jinja --port 8080
python3 replay_bench.py --preset quickstart --endpoint http://localhost:8080/v1
```

## Sample report (quickstart, Qwen3.8-27B)

The quickstart report for Qwen3.8-27B on this server is in [`captures/llamacpp.md`](captures/llamacpp.md).
