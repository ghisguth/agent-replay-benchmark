# mlx_lm.server

`mlx_lm.server` is the plain server of the [mlx-lm](https://github.com/ml-explore/mlx-lm) package. It reports no
timings of its own.

## How it is detected
A weak signature, tried last, and only when everything matches: the response header starts with `Server: BaseHTTP/`
(Python's built-in HTTP server, HTTP/1.0), `GET /health` returns exactly `{"status":"ok"}`, and no `/v1/models` entry has
`owned_by`. Any other server's detection wins over it, and a server that differs in any part stays "generic". There is
no version.

Model ids are Hugging Face repo names or local paths. A Hugging Face cache path is shown as `ORG/NAME`, any other path
as its last component, so a local path (it contains a user name) never reaches a label, a file name or a state.

## Telemetry read
No native telemetry: speeds are measured on the client clock (`~`). The server does report cached prompt tokens in
`usage`, so C% and the new-token counts are real, not estimated.

## Prefill progress
No: the live line shows an estimate (`~`).

## Useful server settings
- `mlx_lm.server --model MODEL --port N` (check your version for the prompt cache and context options).

## Sample report (quickstart, Qwen3.8-27B)

The quickstart report for Qwen3.8-27B on this server is in [`captures/mlx_lm.server.md`](captures/mlx_lm.server.md).
