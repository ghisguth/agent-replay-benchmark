# Inference servers

One page per server: how the benchmark recognises it, what it reads from it, whether it reports prefill progress, and
the server flags that matter for a long-context replay. Each page ends with a sample `--preset quickstart` report.

| Server | Page | Detected by | Prefill progress | Native timings |
|---|---|---|---|---|
| llama.cpp | [llamacpp.md](llamacpp.md) | `/props` | real (`return_progress`) | `timings` |
| Splash | [splash.md](splash.md) | `/status` | real (`return_progress`) | `timings`, `metrics` |
| Kapitsa | [kapitsa.md](kapitsa.md) | `/cluster` | real (request header) | `usage.kapitsa` |
| Ollama | [ollama.md](ollama.md) | `/api/version` | estimate (`~`) | `timings`, cached tokens |
| LM Studio | [lmstudio.md](lmstudio.md) | `/api/v0/models` | estimate (`~`) | `stats` (draft counts only when streaming) |
| MTPLX | [mtplx.md](mtplx.md) | `/health` | real (polled snapshot) | `mtplx_stats` |
| OMLX | [omlx.md](omlx.md) | `owned_by` omlx, `/admin/api/stats` | estimate (`~`) | `usage` timings, cached tokens |
| mlx-dspark | [mlx-dspark.md](mlx-dspark.md) | `Server` header, `owned_by`, `/doctor` | estimate (`~`) | `x_mlx_dspark` |
| mlx_lm.server | [mlx_lm.md](mlx_lm.md) | weak: `Server: BaseHTTP/`, `/health`, no `owned_by` | estimate (`~`) | client clock, real cache counts |

Any other OpenAI-compatible server (vLLM, for one: `/version`) runs as "generic" or by its own probe, with the OpenAI
`usage` block and client-side timing. `--server-version TEXT` overrides the detected version in labels and file names.

Check what the benchmark sees before a long run: `python3 replay_bench.py --discover --endpoint URL` (no chat request; exit code 1 when nothing matched).
Everything the benchmark reads is in `bench/backend.py` (detection) and `bench/telemetry.py` (the field table).
Flags below are for the server, not the benchmark; ones marked "check your version" were not verified here.
