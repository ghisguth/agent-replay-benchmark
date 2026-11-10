
# Agent replay — suite coding

## Run

| Field     | Value                                                                         |
|-----------|-------------------------------------------------------------------------------|
| Server    | Kapitsa 0.2.31                                                                |
| Model     | `qwen3.8-27b`                                                                 |
| Device    | MacBook Pro M2 Max 64GB                                                       |
| Endpoint  | localhost:8133                                                                |
| Context   | 262,144 tokens (256k)                                                         |
| Suite     | coding, 5 requests                                                            |
| Requests  | 5 entries (main 5, subagent 0, compaction 0), max prompt 23729 tokens (23.2k) |
| Selection | no subagents, requests 1-5                                                    |
| Sampling  | temperature 0.6, reasoning effort none sent                                   |


## qwen3.8-27b

### Results

5 PASS · 0 WARN · 0 FAIL · 0 SKIP · same tools as recorded 40% · in line with reference models 5/5 (100%)

| Case                  | Stream    |  TTFT |    Time |      In |     Out |  C% | Decode | Prefill | Verdict   | Reason / Preview                           |
|-----------------------|-----------|------:|--------:|--------:|--------:|----:|-------:|--------:|-----------|--------------------------------------------|
| **R001_main_e0_t000** | `main/e0` | 1m22s | *1m25s* |  18,145 |   *199* |  0% |   62.4 |   221.9 | ✅ PASS   | call `bash`                                |
| **R002_main_e0_t001** | `main/e0` | 5.00s | *6.94s* |  18,724 |   *113* | 97% |   57.9 |   118.8 | ✅ PASS   | *Brief exists. Reading it in full.*        |
| **R003_main_e0_t002** | `main/e0` | 17.9s | *34.1s* |  21,939 |   *785* | 85% |   48.5 |   179.5 | ✅ PASS   | *Brief read. Checking prior state before…* |
| **R004_main_e0_t003** | `main/e0` | 5.27s | *16.4s* |  22,569 |   *616* | 97% |   55.3 |   120.7 | ✅ PASS   | *I've confirmed that 'src' hasn't change…* |
| **R005_main_e0_t004** | `main/e0` | 7.46s | *10.4s* |  23,771 |   *146* | 95% |   49.8 |   164.7 | ✅ PASS   | call `bash`                                |
| **Total**             |           | 1m57s | *2m33s* | 105,148 | *1,859* | 77% |   55.3 |   164.7 | **5/0/0** |                                            |

### Metrics

| Metric                              |              Value |
|-------------------------------------|-------------------:|
| Requests                            |                  5 |
| Wall time *(sum of requests)*       |              2m33s |
| TTFT p50 / p90                      |      7.46s / 56.3s |
| Request time p50 / p90              |      16.4s / 1m05s |
| Prompt tokens                       |            105,148 |
| Cache hit *(cached ÷ prompt)*       |              77.4% |
| Output tokens                       |              1,859 |
| Output per request                  |                372 |
| Decode tok/s *(client)*             |               52.7 |
| End-to-end tok/s                    |               12.2 |
| Native decode tok/s p25 / p50 / p75 | 49.8 / 55.3 / 57.9 |
| Native prefill tok/s p50            |                165 |


## Performance

### Prefill by append size

| New tokens | Requests | Sum new |  TTFT | tok/s |
|------------|---------:|--------:|------:|------:|
| < 1k       |        2 |   1,219 | 10.3s | 118.7 |
| 1–5k       |        2 |   4,427 | 25.4s | 174.2 |
| ≥ 5k       |        1 |  18,145 | 1m22s | 221.8 |


*TTFT = time to first token · C% = share of the prompt served from cache · P/W/F = PASS/WARN/FAIL counts · k = 1024*
*(context sizes, prompt and append sizes)*
*case table: Decode / Prefill are the server's own tok/s; Total row: TTFT, Time, In and Out are sums (TTFT = time spent*
*waiting for first tokens), Decode and Prefill are medians, C% is cached ÷ prompt.*
*~ = measured on the client clock (the server reports no speed of its own).*
*Metrics, context and comparison tables use the client clock: decode tok/s = output tokens ÷ (request time − TTFT),*
*end-to-end = output tokens ÷ request time, prefill = new prompt tokens ÷ TTFT (native = the server's own figures).*
*prefill by append size: requests with more than 16 new tokens (prompt − cached).*

State: `states/kapitsa-0.2.31_MBP-M2Max-64GB_qwen3.8-27b_coding-quickstart_t0.6_20261109-173205.jsonl`

