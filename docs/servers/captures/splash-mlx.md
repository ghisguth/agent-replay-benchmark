
# Agent replay — suite coding

## Run

| Field     | Value                                                                         |
|-----------|-------------------------------------------------------------------------------|
| Server    | Splash 37c550ac                                                               |
| Model     | `mlx-community/Qwen3.8-27B-4bit`                                              |
| Note      | MLX                                                                           |
| Device    | MacBook Pro M2 Max 64GB                                                       |
| Endpoint  | localhost:8133                                                                |
| Context   | 262,144 tokens (256k)                                                         |
| Suite     | coding, 5 requests                                                            |
| Requests  | 5 entries (main 5, subagent 0, compaction 0), max prompt 23729 tokens (23.2k) |
| Selection | no subagents, requests 1-5                                                    |
| Sampling  | temperature 0.6, reasoning effort none sent                                   |


## mlx-community/Qwen3.8-27B-4bit

### Results

4 PASS · 1 WARN · 0 FAIL · 0 SKIP · same tools as recorded 60% · in line with reference models 4/5 (80%)

| Case                  | Stream    |  TTFT |    Time |      In |     Out |  C% | Decode | Prefill | Verdict   | Reason / Preview                           |
|-----------------------|-----------|------:|--------:|--------:|--------:|----:|-------:|--------:|-----------|--------------------------------------------|
| **R001_main_e0_t000** | `main/e0` | 1m19s | *1m22s* |  18,107 |   *183* |  0% |   56.1 |   229.8 | ✅ PASS   | call `bash`                                |
| **R002_main_e0_t001** | `main/e0` | 3.19s | *5.98s* |  18,686 |   *134* | 97% |   47.8 |   190.4 | ✅ PASS   | *Brief exists. Reading it and fix-scope.*  |
| **R003_main_e0_t002** | `main/e0` | 16.1s | *47.3s* |  21,901 | *1,332* | 85% |   42.7 |   201.4 | ⚠️ WARN   | off pattern: plan (reference models: lo…   |
| **R004_main_e0_t003** | `main/e0` | 3.68s | *12.5s* |  22,531 |   *408* | 97% |   46.1 |   176.4 | ✅ PASS   | *Src hasn't moved since 'e8a3b6714'. Whi…* |
| **R005_main_e0_t004** | `main/e0` | 6.96s | *22.3s* |  23,733 |   *684* | 95% |   44.6 |   178.9 | ✅ PASS   | call `bash`                                |
| **Total**             |           | 1m49s | *2m50s* | 104,958 | *2,741* | 77% |   46.1 |   190.4 | **4/1/0** |                                            |

### Metrics

| Metric                              |              Value |
|-------------------------------------|-------------------:|
| Requests                            |                  5 |
| Wall time *(sum of requests)*       |              2m50s |
| TTFT p50 / p90                      |      6.96s / 53.8s |
| Request time p50 / p90              |      22.3s / 1m08s |
| Prompt tokens                       |            104,958 |
| Cache hit *(cached ÷ prompt)*       |              77.3% |
| Output tokens                       |              2,741 |
| Output per request                  |                548 |
| Decode tok/s *(client)*             |               44.7 |
| End-to-end tok/s                    |               16.1 |
| Native decode tok/s p25 / p50 / p75 | 44.6 / 46.1 / 47.8 |
| Native prefill tok/s p50            |                190 |

### Anomalies

- **R003_main_e0_t002** WARN: off pattern: plan (reference models: look)


## Performance

### Prefill by append size

| New tokens | Requests | Sum new |  TTFT | tok/s |
|------------|---------:|--------:|------:|------:|
| < 1k       |        2 |   1,249 | 6.88s | 181.6 |
| 1–5k       |        2 |   4,482 | 23.1s | 194.0 |
| ≥ 5k       |        1 |  18,107 | 1m19s | 229.7 |


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

State: `states/splash-37c550ac_MBP-M2Max-64GB_mlx-community-Qwen3.8-27B-4bit_coding-quickstart_t0.6_mlx_20261109-165345.jsonl`

