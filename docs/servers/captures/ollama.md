
# Agent replay — suite coding

## Run

| Field      | Value                                                                         |
|------------|-------------------------------------------------------------------------------|
| Server     | ollama 0.40.15                                                                |
| Model      | `qwen3.8:27b-mlx-256k`                                                        |
| Model info | nvfp4 (safetensors, 27.8B, mlx)                                               |
| Device     | MacBook Pro M2 Max 64GB                                                       |
| Endpoint   | localhost:11434                                                               |
| Context    | 262,144 tokens (256k)                                                         |
| Suite      | coding, 5 requests                                                            |
| Requests   | 5 entries (main 5, subagent 0, compaction 0), max prompt 23729 tokens (23.2k) |
| Selection  | no subagents, requests 1-5                                                    |
| Sampling   | temperature 0.6, reasoning effort none sent                                   |


## qwen3.8:27b-mlx-256k

### Results

5 PASS · 0 WARN · 0 FAIL · 0 SKIP · same tools as recorded 40% · in line with reference models 5/5 (100%)

| Case                  | Stream    |  TTFT |    Time |      In |     Out |  C% | Decode | Prefill | Verdict   | Reason / Preview                           |
|-----------------------|-----------|------:|--------:|--------:|--------:|----:|-------:|--------:|-----------|--------------------------------------------|
| **R001_main_e0_t000** | `main/e0` | 2m27s | *2m32s* |  18,356 |   *137* |  0% |   24.2 |   125.5 | ✅ PASS   | call `bash`                                |
| **R002_main_e0_t001** | `main/e0` | 5.30s | *8.09s* |  18,917 |    *79* | 97% |   24.3 |   117.9 | ✅ PASS   | *The brief exists. Let me read it.*        |
| **R003_main_e0_t002** | `main/e0` | 27.5s | *45.3s* |  22,113 |   *450* | 86% |   24.9 |   118.0 | ✅ PASS   | *Brief exists. This is an instrument tas…* |
| **R004_main_e0_t003** | `main/e0` | 3.33s | *18.2s* |  22,451 |   *389* | 98% |   25.9 |   111.4 | ✅ PASS   | *src diff since e8a3b6714 is empty — so…*  |
| **R005_main_e0_t004** | `main/e0` | 8.88s | *15.6s* |  23,413 |   *162* | 96% |   23.0 |   113.1 | ✅ PASS   | call `bash`                                |
| **Total**             |           | 3m12s | *3m59s* | 105,250 | *1,217* | 78% |   24.3 |   117.9 | **5/0/0** |                                            |

### Metrics

| Metric                              |              Value |
|-------------------------------------|-------------------:|
| Requests                            |                  5 |
| Wall time *(sum of requests)*       |              3m59s |
| TTFT p50 / p90                      |      8.88s / 1m39s |
| Request time p50 / p90              |      18.2s / 1m49s |
| Prompt tokens                       |            105,250 |
| Cache hit *(cached ÷ prompt)*       |              77.7% |
| Output tokens                       |              1,217 |
| Output per request                  |                243 |
| Decode tok/s *(client)*             |               25.6 |
| End-to-end tok/s                    |                5.1 |
| Native decode tok/s p25 / p50 / p75 | 24.2 / 24.3 / 24.9 |
| Native prefill tok/s p50            |                118 |


## Performance

### Prefill by append size

| New tokens | Requests | Sum new |  TTFT | tok/s |
|------------|---------:|--------:|------:|------:|
| < 1k       |        3 |   1,876 | 17.5s | 107.1 |
| 1–5k       |        1 |   3,201 | 27.5s | 116.5 |
| ≥ 5k       |        1 |  18,356 | 2m27s | 125.2 |


*TTFT = time to first token · C% = share of the prompt served from cache · P/W/F = PASS/WARN/FAIL counts · k = 1024*
*(context sizes, prompt and append sizes)*
*case table: Decode / Prefill are the server's own tok/s; Total row: TTFT, Time, In and Out are sums (TTFT = time spent*
*waiting for first tokens), Decode and Prefill are medians, C% is cached ÷ prompt.*
*~ = measured on the client clock (the server reports no speed of its own).*
*Metrics, context and comparison tables use the client clock: decode tok/s = output tokens ÷ (request time − TTFT),*
*end-to-end = output tokens ÷ request time, prefill = new prompt tokens ÷ TTFT (native = the server's own figures).*
*prefill by append size: requests with more than 16 new tokens (prompt − cached).*

State: `states/ollama-0.40.15_MBP-M2Max-64GB_qwen3.8-27b-mlx-256k-nvfp4_coding-quickstart_t0.6_20261109-173734.jsonl`

