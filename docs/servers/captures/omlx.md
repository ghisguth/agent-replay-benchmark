
# Agent replay — suite coding

## Run

| Field     | Value                                                                         |
|-----------|-------------------------------------------------------------------------------|
| Server    | OMLX 0.32.3.dev8+g0d4d3d04f                                                   |
| Model     | `mlx-community--Qwen3.8-27B-4bit`                                             |
| Note      | splash-m1 engine                                                              |
| Device    | MacBook Pro M2 Max 64GB                                                       |
| Endpoint  | localhost:8134                                                                |
| Context   | 262,144 tokens (256k)                                                         |
| Suite     | coding, 5 requests                                                            |
| Requests  | 5 entries (main 5, subagent 0, compaction 0), max prompt 23729 tokens (23.2k) |
| Selection | no subagents, requests 1-5                                                    |
| Sampling  | temperature 0.6, reasoning effort none sent                                   |


## mlx-community--Qwen3.8-27B-4bit

### Results

3 PASS · 2 WARN · 0 FAIL · 0 SKIP · same tools as recorded 40% · in line with reference models 4/5 (80%)

| Case                  | Stream    |  TTFT |    Time |      In |     Out |  C% | Decode | Prefill | Verdict   | Reason / Preview                           |
|-----------------------|-----------|------:|--------:|--------:|--------:|----:|-------:|--------:|-----------|--------------------------------------------|
| **R001_main_e0_t000** | `main/e0` | 1m22s | *1m24s* |  18,470 |   *130* |  0% |   55.1 |   226.1 | ✅ PASS   | call `bash`                                |
| **R002_main_e0_t001** | `main/e0` | 3.29s | *4.38s* |  19,049 |    *71* | 97% |   65.2 |   180.0 | ✅ PASS   | call `read`                                |
| **R003_main_e0_t002** | `main/e0` | 16.5s | *52.9s* |  22,264 | *1,564* | 86% |   42.9 |   196.5 | ⚠️ WARN   | schema: todo list: JSON string, expecte…   |
| **R004_main_e0_t003** | `main/e0` | 3.81s | *12.9s* |  22,894 |   *424* | 97% |   46.6 |   174.4 | ✅ PASS   | *I've confirmed the source hasn't change…* |
| **R005_main_e0_t004** | `main/e0` | 6.48s | *18.7s* |  24,096 |   *532* | 95% |   43.4 |   188.2 | ⚠️ WARN   | schema: bash unknown key 'instruction'…    |
| **Total**             |           | 1m52s | *2m53s* | 106,773 | *2,721* | 77% |   46.6 |   188.2 | **3/2/0** |                                            |

### Metrics

| Metric                              |              Value |
|-------------------------------------|-------------------:|
| Requests                            |                  5 |
| Wall time *(sum of requests)*       |              2m53s |
| TTFT p50 / p90                      |      6.48s / 55.6s |
| Request time p50 / p90              |      18.7s / 1m12s |
| Prompt tokens                       |            106,773 |
| Cache hit *(cached ÷ prompt)*       |              77.4% |
| Output tokens                       |              2,721 |
| Output per request                  |                544 |
| Decode tok/s *(client)*             |               44.4 |
| End-to-end tok/s                    |               15.7 |
| Native decode tok/s p25 / p50 / p75 | 43.4 / 46.6 / 55.1 |
| Native prefill tok/s p50            |                188 |

### Anomalies

- **R003_main_e0_t002** WARN: schema: todo list: JSON string, expected array; off pattern: plan (reference models: look)
- **R005_main_e0_t004** WARN: schema: bash unknown key 'instruction' (omp drops it)


## Performance

### Prefill by append size

| New tokens | Requests | Sum new |  TTFT | tok/s |
|------------|---------:|--------:|------:|------:|
| < 1k       |        2 |   1,239 | 7.11s | 174.4 |
| 1–5k       |        2 |   4,440 | 22.9s | 193.6 |
| ≥ 5k       |        1 |  18,470 | 1m22s | 225.9 |


*TTFT = time to first token · C% = share of the prompt served from cache · P/W/F = PASS/WARN/FAIL counts · k = 1024*
*(context sizes, prompt and append sizes)*
*case table: Decode / Prefill are the server's own tok/s; Total row: TTFT, Time, In and Out are sums (TTFT = time spent*
*waiting for first tokens), Decode and Prefill are medians, C% is cached ÷ prompt.*
*~ = measured on the client clock (the server reports no speed of its own).*
*Metrics, context and comparison tables use the client clock: decode tok/s = output tokens ÷ (request time − TTFT),*
*end-to-end = output tokens ÷ request time, prefill = new prompt tokens ÷ TTFT (native = the server's own figures).*
*prefill by append size: requests with more than 16 new tokens (prompt − cached).*

State: `states/omlx-0.32.3.dev8-g0d4d3d04f_MBP-M2Max-64GB_mlx-community-Qwen3.8-27B-4bit_coding-quickstart_t0.6_splash-m1-engine_20261109-180044.jsonl`

