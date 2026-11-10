
# Agent replay — suite coding

## Run

| Field      | Value                                                                         |
|------------|-------------------------------------------------------------------------------|
| Server     | mlx-dspark 0.20.4                                                             |
| Model      | `mlx-community/Qwen3.8-27B-4bit`                                              |
| Model info | dspark, drafter RedHatAI/Qwen3.8-27B-speculator.dspark                        |
| Note       | dspark                                                                        |
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
| **R001_main_e0_t000** | `main/e0` | 1m40s | *1m48s* |  18,147 |   *234* |  0% |   29.6 |   182.2 | ✅ PASS   | call `bash`, `bash`                       |
| **R002_main_e0_t001** | `main/e0` | 4.45s | *9.62s* |  18,726 |   *118* | 97% |   22.2 |   135.6 | ✅ PASS   | *Brief exists. Reading it now.*            |
| **R003_main_e0_t002** | `main/e0` | 20.3s | *52.3s* |  21,941 |   *772* | 85% |   24.1 |   160.0 | ✅ PASS   | *I've reviewed the brief and identified…*  |
| **R004_main_e0_t003** | `main/e0` | 4.70s | *18.9s* |  22,571 |   *328* | 97% |   22.9 |   140.3 | ✅ PASS   | *I'm checking the P12.7 baseline rate an…* |
| **R005_main_e0_t004** | `main/e0` | 8.65s | *20.7s* |  23,773 |   *288* | 95% |   22.9 |   149.2 | ✅ PASS   | call `bash`                                |
| **Total**             |           | 2m18s | *3m29s* | 105,158 | *1,740* | 77% |   22.9 |   149.2 | **5/0/0** |                                            |

### Metrics

| Metric                              |              Value |
|-------------------------------------|-------------------:|
| Requests                            |                  5 |
| Wall time *(sum of requests)*       |              3m29s |
| TTFT p50 / p90                      |      8.65s / 1m08s |
| Request time p50 / p90              |      20.7s / 1m25s |
| Prompt tokens                       |            105,158 |
| Cache hit *(cached ÷ prompt)*       |              77.4% |
| Output tokens                       |              1,740 |
| Output per request                  |                348 |
| Decode tok/s *(client)*             |               24.5 |
| End-to-end tok/s                    |                8.3 |
| Native decode tok/s p25 / p50 / p75 | 22.9 / 22.9 / 24.1 |
| Native prefill tok/s p50            |                149 |


## Performance

### Prefill by append size

| New tokens | Requests | Sum new |  TTFT | tok/s |
|------------|---------:|--------:|------:|------:|
| < 1k       |        2 |   1,211 | 9.15s | 132.4 |
| 1–5k       |        2 |   4,419 | 29.0s | 152.6 |
| ≥ 5k       |        1 |  18,147 | 1m40s | 181.8 |


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

State: `states/mlx-dspark-0.20.4_MBP-M2Max-64GB_mlx-community-Qwen3.8-27B-4bit_coding-quickstart_t0.6_dspark_20261109-194953.jsonl`

