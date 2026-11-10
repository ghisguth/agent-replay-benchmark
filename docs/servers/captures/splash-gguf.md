
# Agent replay — suite coding

## Run

| Field     | Value                                                                         |
|-----------|-------------------------------------------------------------------------------|
| Server    | Splash 37c550ac                                                               |
| Model     | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`                                          |
| Note      | GGUF                                                                          |
| Device    | MacBook Pro M2 Max 64GB                                                       |
| Endpoint  | localhost:8133                                                                |
| Context   | 262,144 tokens (256k)                                                         |
| Suite     | coding, 5 requests                                                            |
| Requests  | 5 entries (main 5, subagent 0, compaction 0), max prompt 23729 tokens (23.2k) |
| Selection | no subagents, requests 1-5                                                    |
| Sampling  | temperature 0.6, reasoning effort none sent                                   |


## unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M

### Results

4 PASS · 1 WARN · 0 FAIL · 0 SKIP · same tools as recorded 40% · in line with reference models 4/5 (80%)

| Case                  | Stream    |  TTFT |    Time |      In |     Out |  C% | Decode | Prefill | Verdict   | Reason / Preview                         |
|-----------------------|-----------|------:|--------:|--------:|--------:|----:|-------:|--------:|-----------|------------------------------------------|
| **R001_main_e0_t000** | `main/e0` | 1m22s | *1m24s* |  18,107 |   *105* |  0% |   53.2 |   221.5 | ✅ PASS   | call `bash`                              |
| **R002_main_e0_t001** | `main/e0` | 3.34s | *4.42s* |  18,686 |    *68* | 97% |   62.1 |   182.0 | ✅ PASS   | call `read`                              |
| **R003_main_e0_t002** | `main/e0` | 16.7s | *33.6s* |  21,901 |   *680* | 85% |   40.1 |   195.0 | ⚠️ WARN   | off pattern: plan (reference models: lo… |
| **R004_main_e0_t003** | `main/e0` | 3.80s | *12.9s* |  22,531 |   *356* | 97% |   38.9 |   170.2 | ✅ PASS   | call `bash`, `read`                     |
| **R005_main_e0_t004** | `main/e0` | 7.15s | *17.5s* |  23,733 |   *379* | 95% |   36.4 |   173.5 | ✅ PASS   | call `bash`                              |
| **Total**             |           | 1m53s | *2m32s* | 104,958 | *1,588* | 77% |   40.1 |   182.0 | **4/1/0** |                                          |

### Metrics

| Metric                              |              Value |
|-------------------------------------|-------------------:|
| Requests                            |                  5 |
| Wall time *(sum of requests)*       |              2m32s |
| TTFT p50 / p90                      |      7.15s / 55.7s |
| Request time p50 / p90              |      17.5s / 1m04s |
| Prompt tokens                       |            104,958 |
| Cache hit *(cached ÷ prompt)*       |              77.3% |
| Output tokens                       |              1,588 |
| Output per request                  |                318 |
| Decode tok/s *(client)*             |               40.2 |
| End-to-end tok/s                    |               10.4 |
| Native decode tok/s p25 / p50 / p75 | 38.9 / 40.1 / 53.2 |
| Native prefill tok/s p50            |                182 |

### Anomalies

- **R003_main_e0_t002** WARN: off pattern: plan (reference models: look)


## Performance

### Prefill by append size

| New tokens | Requests | Sum new |  TTFT | tok/s |
|------------|---------:|--------:|------:|------:|
| < 1k       |        2 |   1,249 | 7.14s | 174.9 |
| 1–5k       |        2 |   4,482 | 23.8s | 188.2 |
| ≥ 5k       |        1 |  18,107 | 1m22s | 221.4 |


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

State: `states/splash-37c550ac_MBP-M2Max-64GB_unsloth-Qwen3.8-27B-GGUF-UD-Q4_K_M_coding-quickstart_t0.6_gguf_20261109-165041.jsonl`

