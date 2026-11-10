
# Agent replay — suite coding

## Run

| Field     | Value                                       |
|-----------|---------------------------------------------|
| Server    | mlx_lm.server                               |
| Model     | `mlx-community/Qwen3.8-27B-4bit`            |
| Device    | MacBook Pro M2 Max 64GB                     |
| Endpoint  | localhost:8080                              |
| Suite     | coding, 5 requests                          |
| Selection | no subagents, requests 1-5                  |
| Sampling  | temperature 0.6, reasoning effort none sent |


## mlx-community/Qwen3.8-27B-4bit

### Results

5 PASS · 0 WARN · 0 FAIL · 0 SKIP · same tools as recorded 0% · in line with reference models 5/5 (100%)

| Case                  | Stream    |  TTFT |    Time |      In |     Out |  C% | Decode | Prefill | Verdict   | Reason / Preview                           |
|-----------------------|-----------|------:|--------:|--------:|--------:|----:|-------:|--------:|-----------|--------------------------------------------|
| **R001_main_e0_t000** | `main/e0` | 2m12s | *2m26s* |  18,145 |   *229* |  0% |  ~15.6 |  ~138.0 | ✅ PASS   | call `bash`, `bash`                        |
| **R002_main_e0_t001** | `main/e0` | 4.90s | *11.8s* |  18,724 |   *108* | 97% |  ~15.7 |  ~118.5 | ✅ PASS   | *Brief exists. Reading it and the fix-sc…* |
| **R003_main_e0_t002** | `main/e0` | 25.2s | *4m25s* |  21,939 | *2,499* | 85% |  ~10.4 |  ~127.7 | ✅ PASS   | *Brief is present. Now I'll read the fil…* |
| **R004_main_e0_t003** | `main/e0` | 5.43s | *34.7s* |  22,569 |   *428* | 97% |  ~14.6 |  ~116.2 | ✅ PASS   | *Src hasn't changed since e8a3b6714, so…*  |
| **R005_main_e0_t004** | `main/e0` | 9.89s | *32.3s* |  23,771 |   *382* | 95% |  ~17.0 |  ~121.6 | ✅ PASS   | call `bash`                                |
| **Total**             |           | 2m57s | *8m10s* | 105,148 | *3,646* | 77% |  ~15.6 |  ~121.6 | **5/0/0** |                                            |

### Metrics

| Metric                        |         Value |
|-------------------------------|--------------:|
| Requests                      |             5 |
| Wall time *(sum of requests)* |         8m10s |
| TTFT p50 / p90                | 9.89s / 1m29s |
| Request time p50 / p90        | 34.7s / 3m38s |
| Prompt tokens                 |       105,148 |
| Cache hit *(cached ÷ prompt)* |         77.4% |
| Output tokens                 |         3,646 |
| Output per request            |           729 |
| Decode tok/s *(client)*       |          11.6 |
| End-to-end tok/s              |           7.4 |


## Performance

### Prefill by append size

| New tokens | Requests | Sum new |  TTFT | tok/s |
|------------|---------:|--------:|------:|------:|
| < 1k       |        2 |   1,211 | 10.3s | 117.3 |
| 1–5k       |        2 |   4,419 | 35.1s | 126.0 |
| ≥ 5k       |        1 |  18,145 | 2m12s | 138.0 |


*TTFT = time to first token · C% = share of the prompt served from cache · P/W/F = PASS/WARN/FAIL counts · k = 1024*
*(context sizes, prompt and append sizes)*
*case table: Decode / Prefill are the server's own tok/s; Total row: TTFT, Time, In and Out are sums (TTFT = time spent*
*waiting for first tokens), Decode and Prefill are medians, C% is cached ÷ prompt.*
*~ = measured on the client clock (the server reports no speed of its own).*
*Metrics, context and comparison tables use the client clock: decode tok/s = output tokens ÷ (request time − TTFT),*
*end-to-end = output tokens ÷ request time, prefill = new prompt tokens ÷ TTFT (native = the server's own figures).*
*prefill by append size: requests with more than 16 new tokens (prompt − cached).*

