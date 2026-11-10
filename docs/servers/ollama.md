# Ollama

## How it is detected
`GET /api/version` returns `{"version": "0.x.y"}`. The model details come from `GET /api/ps` (loaded models:
quantization, family, parameter size, context length, runner, digest), with `POST /api/show` as the fallback.

## Telemetry read
Ollama 0.40 answers chat completions with a llama.cpp-style `timings` object next to the OpenAI `usage`:
- `usage.prompt_tokens_details.cached_tokens` for the cached share of the prompt (the cache column).
- `timings` (`prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`, `*_per_second`) for the native TTFT, prefill and
  decode speeds. Its `prompt_per_second` can count the whole prompt rather than the new tokens; the runner corrects for
  that (`timings_prefill_tps` in `bench/telemetry.py`).

## Prefill progress
No: Ollama sends nothing while it prefills a chat completion. The live line shows an estimate (`~`) from the recorded
prompt size and the median prefill speed so far.

## Useful server flags (environment variables of `ollama serve`)
- `OLLAMA_CONTEXT_LENGTH=N`: the default context. Ollama truncates prompts that do not fit, silently in some versions,
  so set it above the largest prompt of the preset (the quickstart needs 24k, `large` 128k).
- `OLLAMA_NUM_PARALLEL=1`: one request at a time, no context split between slots.
- `OLLAMA_KEEP_ALIVE=-1`: keep the model loaded between requests.
- `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE`: attention and KV cache type (check your version).
- `num_ctx` in a Modelfile does the same per model.

## Sample report (quickstart, Qwen3.8-27B)

The quickstart report for Qwen3.8-27B on this server is in [`captures/ollama.md`](captures/ollama.md).
