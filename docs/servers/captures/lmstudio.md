
# Agent replay — suite coding

## Run

| Field      | Value                                       |
|------------|---------------------------------------------|
| Server     | LM Studio                                   |
| Model      | `qwen3.8-27b`                               |
| Model info | Q4_K_M (gguf)                               |
| Device     | MacBook Pro M2 Max 64GB                     |
| Endpoint   | localhost:8135                              |
| Context    | 262,144 tokens (256k)                       |
| Suite      | coding, 5 requests                          |
| Selection  | no subagents, requests 1-5                  |
| Sampling   | temperature 0.6, reasoning effort none sent |


## qwen3.8-27b

### Results

5 PASS · 0 WARN · 0 FAIL · 0 SKIP · same tools as recorded 40% · in line with reference models 5/5 (100%)

| Case                  | Stream    |  TTFT |    Time |      In |     Out | C% | Decode | Prefill | Verdict   | Reason / Preview                           |
|-----------------------|-----------|------:|--------:|--------:|--------:|---:|-------:|--------:|-----------|--------------------------------------------|
| **R001_main_e0_t000** | `main/e0` | 2m00s | *2m08s* |  18,144 |   *195* |  – |  ~25.1 |  ~151.2 | ✅ PASS   | call `bash`                                |
| **R002_main_e0_t001** | `main/e0` | 4.56s | *13.1s* |  18,726 |   *206* |  – |  ~24.2 |  ~127.6 | ✅ PASS   | *Brief exists — reading it now, plus fix…* |
| **R003_main_e0_t002** | `main/e0` | 24.3s | *4m59s* |  21,943 | *5,173* |  – |  ~18.9 |  ~132.3 | ✅ PASS   | *I'm starting with Step 0, which is canc…* |
| **R004_main_e0_t003** | `main/e0` | 5.01s | *22.1s* |  22,575 |   *336* |  – |  ~19.6 |  ~126.1 | ✅ PASS   | *'src/' hasn't moved since 'e8a3b6714',…*  |
| **R005_main_e0_t004** | `main/e0` | 9.36s | *30.5s* |  23,780 |   *425* |  – |  ~20.1 |  ~128.7 | ✅ PASS   | call `bash`                                |
| **Total**             |           | 2m43s | *8m12s* | 105,168 | *6,335* |  – |  ~20.1 |  ~128.7 | **5/0/0** |                                            |

### Metrics

| Metric                        |         Value |
|-------------------------------|--------------:|
| Requests                      |             5 |
| Wall time *(sum of requests)* |         8m12s |
| TTFT p50 / p90                | 9.36s / 1m22s |
| Request time p50 / p90        | 30.5s / 3m50s |
| Prompt tokens                 |       105,168 |
| Cache hit *(cached ÷ prompt)* |             – |
| Output tokens                 |         6,335 |
| Output per request            |         1,267 |
| Decode tok/s *(client)*       |          19.3 |
| End-to-end tok/s              |          12.9 |


## Performance

### Prefill by append size

| New tokens | Requests | Sum new |  TTFT | tok/s |
|------------|---------:|--------:|------:|------:|
| < 1k       |        2 |   1,214 | 9.57s | 126.8 |
| 1–5k       |        2 |   4,422 | 33.7s | 131.3 |
| ≥ 5k       |        1 |  18,144 | 2m00s | 151.2 |


*TTFT = time to first token · C% = share of the prompt served from cache · P/W/F = PASS/WARN/FAIL counts · k = 1024*
*(context sizes, prompt and append sizes)*
*case table: Decode / Prefill are the server's own tok/s; Total row: TTFT, Time, In and Out are sums (TTFT = time spent*
*waiting for first tokens), Decode and Prefill are medians, C% is cached ÷ prompt.*
*~ = measured on the client clock (the server reports no speed of its own).*
*Metrics, context and comparison tables use the client clock: decode tok/s = output tokens ÷ (request time − TTFT),*
*end-to-end = output tokens ÷ request time, prefill = new prompt tokens ÷ TTFT (native = the server's own figures).*
*prefill by append size: requests with more than 16 new tokens (prompt − cached).*
*no cache reported: new tokens ≈ prompt − previous prompt in the stream.*

