
# Agent replay — suite coding

## Run

| Field      | Value                                                                         |
|------------|-------------------------------------------------------------------------------|
| Server     | mlx-dspark 0.20.4                                                             |
| Model      | `mlx-community/Qwen3.8-27B-4bit`                                              |
| Model info | dflash, drafter incoai/Qwen3.8-27B-DFlash2                                    |
| Note       | dflash                                                                        |
| Device     | MacBook Pro M2 Max 64GB                                                       |
| Endpoint   | localhost:8134                                                                |
| Suite      | coding, 5 requests                                                            |
| Requests   | 5 entries (main 5, subagent 0, compaction 0), max prompt 23729 tokens (23.2k) |
| Selection  | no subagents, requests 1-5                                                    |
| Sampling   | temperature 0.6, reasoning effort none sent                                   |


## mlx-community/Qwen3.8-27B-4bit

### Results

5 PASS · 0 WARN · 0 FAIL · 0 SKIP · same tools as recorded 40% · in line with reference models 5/5 (100%)

| Case                  | Stream    |  TTFT |    Time |      In |     Out |  C% | Decode | Prefill | Verdict   | Reason / Preview                           |
|-----------------------|-----------|------:|--------:|--------:|--------:|----:|-------:|--------:|-----------|--------------------------------------------|
| **R001_main_e0_t000** | `main/e0` | 1m38s | *1m44s* |  18,147 |   *216* |  0% |   40.1 |   184.8 | ✅ PASS   | call `bash`, `bash`                       |
| **R002_main_e0_t001** | `main/e0` | 4.50s | *6.49s* |  18,726 |    *83* | 97% |   38.0 |   136.6 | ✅ PASS   | call `read`                                |
| **R003_main_e0_t002** | `main/e0` | 20.1s | *26.9s* |  21,941 |   *174* | 85% |   24.8 |   162.5 | ✅ PASS   | call `bash`                                |
| **R004_main_e0_t003** | `main/e0` | 4.64s | *18.1s* |  22,571 |   *426* | 97% |   31.1 |   142.8 | ✅ PASS   | *I've confirmed the src hasn't changed s…* |
| **R005_main_e0_t004** | `main/e0` | 8.23s | *23.6s* |  23,773 |   *399* | 95% |   25.6 |   151.6 | ✅ PASS   | call `bash`, `bash`                       |
| **Total**             |           | 2m16s | *2m59s* | 105,158 | *1,298* | 77% |   31.1 |   151.6 | **5/0/0** |                                            |

### Metrics

| Metric                              |              Value |
|-------------------------------------|-------------------:|
| Requests                            |                  5 |
| Wall time *(sum of requests)*       |              2m59s |
| TTFT p50 / p90                      |      8.23s / 1m07s |
| Request time p50 / p90              |      23.6s / 1m13s |
| Prompt tokens                       |            105,158 |
| Cache hit *(cached ÷ prompt)*       |              77.4% |
| Output tokens                       |              1,298 |
| Output per request                  |                260 |
| Decode tok/s *(client)*             |               30.4 |
| End-to-end tok/s                    |                7.3 |
| Native decode tok/s p25 / p50 / p75 | 25.6 / 31.1 / 38.0 |
| Native prefill tok/s p50            |                152 |


## Performance

### Prefill by append size

| New tokens | Requests | Sum new |  TTFT | tok/s |
|------------|---------:|--------:|------:|------:|
| < 1k       |        2 |   1,211 | 9.14s | 132.5 |
| 1–5k       |        2 |   4,419 | 28.4s | 155.8 |
| ≥ 5k       |        1 |  18,147 | 1m38s | 184.3 |


*TTFT = time to first token · C% = share of the prompt served from cache · P/W/F = PASS/WARN/FAIL counts · k = 1024*
*(context sizes, prompt and append sizes)*
*case table: Decode / Prefill are the server's own tok/s; Total row: TTFT, Time, In and Out are sums (TTFT = time spent*
*waiting for first tokens), Decode and Prefill are medians, C% is cached ÷ prompt.*
*~ = measured on the client clock (the server reports no speed of its own).*
*Metrics, context and comparison tables use the client clock: decode tok/s = output tokens ÷ (request time − TTFT),*
*end-to-end = output tokens ÷ request time, prefill = new prompt tokens ÷ TTFT (native = the server's own figures).*
*prefill by append size: requests with more than 16 new tokens (prompt − cached).*
*a cycle is one draft-and-verify step: tok/cycle = tokens kept per step, cycles/s (requests with ≥ 50 output tokens) is*
*the raw engine speed.*

State: `states/mlx-dspark-0.20.4_MBP-M2Max-64GB_mlx-community-Qwen3.8-27B-4bit_coding-quickstart_t0.6_dflash_20261109-194610.jsonl`

