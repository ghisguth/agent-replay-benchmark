
# Agent replay — suite coding

## Run

| Field     | Value                                                                         |
|-----------|-------------------------------------------------------------------------------|
| Server    | MTPLX 2.12.3                                                                  |
| Model     | `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16`                            |
| Device    | MacBook Pro M2 Max 64GB                                                       |
| Endpoint  | localhost:8134                                                                |
| Context   | 262,144 tokens (256k)                                                         |
| Suite     | coding, 5 requests                                                            |
| Requests  | 5 entries (main 5, subagent 0, compaction 0), max prompt 23729 tokens (23.2k) |
| Selection | no subagents, requests 1-5                                                    |
| Sampling  | temperature 0.6, reasoning effort none sent                                   |


## Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16

### Results

4 PASS · 1 WARN · 0 FAIL · 0 SKIP · same tools as recorded 40% · in line with reference models 4/5 (80%)

| Case                  | Stream    |  TTFT |    Time |      In |   Out |  C% | Decode | Prefill | Verdict   | Reason / Preview                           |
|-----------------------|-----------|------:|--------:|--------:|------:|----:|-------:|--------:|-----------|--------------------------------------------|
| **R001_main_e0_t000** | `main/e0` | 1m37s | *1m41s* |  18,117 | *103* |  0% |   28.7 |   187.3 | ✅ PASS   | call `bash`                                |
| **R002_main_e0_t001** | `main/e0` | 3.56s | *5.97s* |  18,679 |  *68* | 97% |   28.7 |   160.6 | ✅ PASS   | call `read`                                |
| **R003_main_e0_t002** | `main/e0` | 18.8s | *28.9s* |  21,876 | *270* | 85% |   27.0 |   170.6 | ⚠️ WARN   | off pattern: plan (reference models: lo…   |
| **R004_main_e0_t003** | `main/e0` | 2.25s | *13.2s* |  22,215 | *298* | 98% |   27.3 |   154.5 | ✅ PASS   | *src is unchanged since e8a3b6714 (empty…* |
| **R005_main_e0_t004** | `main/e0` | 6.17s | *11.5s* |  23,178 | *163* | 96% |   31.1 |   158.3 | ✅ PASS   | call `bash`                                |
| **Total**             |           | 2m08s | *2m40s* | 104,065 | *902* | 78% |   28.7 |   160.6 | **4/1/0** |                                            |

### Metrics

| Metric                              |              Value |
|-------------------------------------|-------------------:|
| Requests                            |                  5 |
| Wall time *(sum of requests)*       |              2m40s |
| TTFT p50 / p90                      |      6.17s / 1m06s |
| Request time p50 / p90              |      13.2s / 1m12s |
| Prompt tokens                       |            104,065 |
| Cache hit *(cached ÷ prompt)*       |              77.7% |
| Output tokens                       |                902 |
| Output per request                  |                180 |
| Decode tok/s *(client)*             |               27.9 |
| End-to-end tok/s                    |                5.6 |
| Native decode tok/s p25 / p50 / p75 | 27.3 / 28.7 / 28.7 |
| Native prefill tok/s p50            |                161 |

### Anomalies

- **R003_main_e0_t002** WARN: off pattern: plan (reference models: look)


## Performance

### Prefill by append size

| New tokens | Requests | Sum new |  TTFT | tok/s |
|------------|---------:|--------:|------:|------:|
| < 1k       |        3 |   1,864 | 12.0s | 155.6 |
| 1–5k       |        1 |   3,197 | 18.8s | 169.9 |
| ≥ 5k       |        1 |  18,117 | 1m37s | 187.0 |


*TTFT = time to first token · C% = share of the prompt served from cache · P/W/F = PASS/WARN/FAIL counts · k = 1024*
*(context sizes, prompt and append sizes)*
*case table: Decode / Prefill are the server's own tok/s; Total row: TTFT, Time, In and Out are sums (TTFT = time spent*
*waiting for first tokens), Decode and Prefill are medians, C% is cached ÷ prompt.*
*~ = measured on the client clock (the server reports no speed of its own).*
*Metrics, context and comparison tables use the client clock: decode tok/s = output tokens ÷ (request time − TTFT),*
*end-to-end = output tokens ÷ request time, prefill = new prompt tokens ÷ TTFT (native = the server's own figures).*
*prefill by append size: requests with more than 16 new tokens (prompt − cached).*

State: `states/mtplx-2.12.3_MBP-M2Max-64GB_Youssofal-Qwen3.8-27B-MTPLX-Optimized-Speed-FP16_coding-quickstart_t0.6_20261109-181350.jsonl`

