
# Agent replay — suite coding

## Run

| Field     | Value                                                                         |
|-----------|-------------------------------------------------------------------------------|
| Server    | llama.cpp b12117                                                              |
| Model     | `unsloth/Qwen3.8-27B-GGUF:Q4_K_M`                                             |
| Device    | MacBook Pro M2 Max 64GB                                                       |
| Endpoint  | localhost:8133                                                                |
| Context   | 262,144 tokens (256k)                                                         |
| Suite     | coding, 5 requests                                                            |
| Requests  | 5 entries (main 5, subagent 0, compaction 0), max prompt 23729 tokens (23.2k) |
| Selection | no subagents, requests 1-5                                                    |
| Sampling  | temperature 0.6, reasoning effort none sent                                   |


## unsloth/Qwen3.8-27B-GGUF:Q4_K_M

### Results

5 PASS · 0 WARN · 0 FAIL · 0 SKIP · same tools as recorded 40% · in line with reference models 5/5 (100%)

| Case                  | Stream    |  TTFT |    Time |      In |     Out |  C% | Decode | Prefill | Verdict   | Reason / Preview                          |
|-----------------------|-----------|------:|--------:|--------:|--------:|----:|-------:|--------:|-----------|-------------------------------------------|
| **R001_main_e0_t000** | `main/e0` | 1m57s | *2m01s* |  18,107 |   *140* |  0% |   34.4 |   155.0 | ✅ PASS   | call `bash`, `bash`                      |
| **R002_main_e0_t001** | `main/e0` | 4.51s | *6.09s* |  18,686 |    *70* | 97% |   43.9 |   130.1 | ✅ PASS   | call `read`                               |
| **R003_main_e0_t002** | `main/e0` | 23.8s | *35.3s* |  21,901 |   *265* | 85% |   22.8 |   135.7 | ✅ PASS   | call `bash`                               |
| **R004_main_e0_t003** | `main/e0` | 4.96s | *15.0s* |  22,531 |   *222* | 97% |   22.1 |   128.7 | ✅ PASS   | *'src' is unchanged since e8a3b6714, so…* |
| **R005_main_e0_t004** | `main/e0` | 9.19s | *22.3s* |  23,733 |   *334* | 95% |   25.4 |   132.1 | ✅ PASS   | call `bash`, `bash`                      |
| **Total**             |           | 2m39s | *3m20s* | 104,958 | *1,031* | 77% |   25.4 |   132.1 | **5/0/0** |                                           |

### Metrics

| Metric                              |              Value |
|-------------------------------------|-------------------:|
| Requests                            |                  5 |
| Wall time *(sum of requests)*       |              3m20s |
| TTFT p50 / p90                      |      9.19s / 1m20s |
| Request time p50 / p90              |      22.3s / 1m27s |
| Prompt tokens                       |            104,958 |
| Cache hit *(cached ÷ prompt)*       |              77.4% |
| Output tokens                       |              1,031 |
| Output per request                  |                206 |
| Decode tok/s *(client)*             |               25.6 |
| End-to-end tok/s                    |                5.2 |
| Native decode tok/s p25 / p50 / p75 | 22.8 / 25.4 / 34.4 |
| Native prefill tok/s p50            |                132 |


## Performance

### Prefill by append size

| New tokens | Requests | Sum new |  TTFT | tok/s |
|------------|---------:|--------:|------:|------:|
| < 1k       |        2 |   1,217 | 9.47s | 128.5 |
| 1–5k       |        2 |   4,425 | 32.9s | 134.3 |
| ≥ 5k       |        1 |  18,107 | 1m57s | 155.0 |


*TTFT = time to first token · C% = share of the prompt served from cache · P/W/F = PASS/WARN/FAIL counts · k = 1024*
*(context sizes, prompt and append sizes)*
*case table: Decode / Prefill are the server's own tok/s; Total row: TTFT, Time, In and Out are sums (TTFT = time spent*
*waiting for first tokens), Decode and Prefill are medians, C% is cached ÷ prompt.*
*~ = measured on the client clock (the server reports no speed of its own).*
*Metrics, context and comparison tables use the client clock: decode tok/s = output tokens ÷ (request time − TTFT),*
*end-to-end = output tokens ÷ request time, prefill = new prompt tokens ÷ TTFT (native = the server's own figures).*
*prefill by append size: requests with more than 16 new tokens (prompt − cached).*

State: `states/llamacpp-b12117_MBP-M2Max-64GB_unsloth-Qwen3.8-27B-GGUF-Q4_K_M_coding-quickstart_t0.6_20261109-171700.jsonl`

