# Agent notes for agent-replay-benchmark

Replays real coding-agent sessions (main agent, subagents, compactions) against OpenAI-compatible local servers, turn
by turn, to compare models, quants and engines at long context. This file is the one place for contributor and agent
notes: layout, rules and the design decisions below. User documentation is `README.md` and `docs/`.

## Layout
- `replay_bench.py`: the CLI. `--suite NAME`: `smoke` (the first four sanity cases), `sanity` (ten generic cases graded by validators, no replay), `coding` (replay `fixtures/conv-3.json.gz`, 470 requests), `judge` (one recorded match-judge case, `fixtures/judge.json.gz`); `--fixture FILE` overrides the suite's fixture. `--preset NAME` (quickstart = coding turns 1-5, small, medium, large = the default, xlarge, compaction, main, full, judge, smoke, sanity) is a named set of suite and selection options (`bench/presets.py`). One send per case, results keyed by case id. Servers: `--endpoint URL` (repeatable, `http(s)://host:port/v1`, default `http://localhost:8080/v1`), `--model SUBSTRING` filter, `--first-model`, `--api-key KEY`. `--discover` (alias `--list`) probes servers and prints models, quant, context, version, device and prefill-progress support without a chat request (exit 1 when nothing matched). Results go to stdout; state files are written by default to `states/` (or `--state-file FILE`, `--no-save-state`). Read back with `--state FILE` and compare with `--compare-state FILE` or `tools/replay_divergence.py`; both take a path or a name found in `states/`, `fixtures/refs/`, `fixtures/samples/`.
- `replay-bench`: POSIX sh launcher (finds the checkout through symlinks and a Python 3.9+, `.venv` first); same as `python3 replay_bench.py`. `pip install -e .` installs the same entry point as `replay-bench`.
- `bench/`: runner, transport, outputs, live line, display, telemetry, state (JSONL; a `route` line records an LM Studio /api/v0 to /v1 switch after the header), endpoint parsing, `backend.py` (server and model probes), `discovery.py`, `compare.py` (comparison against a baseline), `reference.py` (`--reference`: one answer against another run's answer, the agreement numbers), `report.py` (markdown & TUI tables), `perf.py` (context and append breakdown tables); `bench/cases/` suites and grading (`smoke.py`, `sanity.py`, `replay.py`, `grading.py`, `judge.py`).
- `tools/`: maintainer tools.
  - Results: `triage_states.py` (incremental triage of states/), `add_ref.py` (scrubs and adds a reference run, or a sample with `--sample`), `build_checks.py` (rebuilds the expectation sidecar from the reference states; counts distinct models), `replay_divergence.py` (divergence matrix and split turns).
  - Testing: `mock_server.py` (llamacpp, mtplx, generic, ollama, lmstudio, mlx-dspark, mlx_lm flavours on 127.0.0.1, used by `make integration`).
  - Capture and fixture build (`docs/capture-traces.md`): `proxy.py` (records request/response pairs), `omp_session.py`, `omp_requests.py`, `omp_tokens.py`, `omp_fixture.py` (session to fixture), `judge_fixture.py` (the `judge` fixture), `omp_session_pii_scan.py` with `pii_patterns.py` (PII scan), `omp-prompt-digest.ts` (independent check of omp's prompt digest).
- `fixtures/`: built replay fixtures (`conv-3.json.gz` for `coding`, `judge.json.gz` for `judge`), redacted, committed. `fixtures/refs/`: reference runs of high-fidelity models, the expected answers (host names scrubbed before adding; only these feed `conv-3.checks.json`). `fixtures/samples/`: shipped runs of other models and engines, among them the quickstart run of every engine on an M2 Max 64GB, to compare with `--compare-state`, never used for grading. `sources/` is gitignored scratch for raw inputs.
- `tests/`: offline unit tests (`make test`), `tests/integration/` runs the CLI against the mock server (`make integration`), `tests/data/` holds recorded server responses.
- `docs/`: `capture-traces.md` (capturing sessions and building fixtures), `servers/` (one page per supported server: detection, telemetry, flags; `servers/captures/` holds the quickstart reports).

## Rules
- Small commits with plain messages (`replay: …`, `fixture: …`, `docs: …`, `tools: …`, `tests: …`). Never force-push.
- **No raw transcripts, no secrets, no personal data in commits.** Raw agent sessions stay outside the repo or in
  `sources/` (gitignored). Only redacted fixtures are committed; the build rescans them and fails on any hit (see
  Redaction below). Host names in state files are scrubbed by `tools/add_ref.py` before a run is added.
- **Ask before running against a server someone else is using** (one run at a time keeps timings honest); never restart servers you do not own. Offline work
  (parsing, tokenizing, tests, the mock server) needs no permission.
- Python 3.9+, standard library only; match the existing code style in `bench/`. Keep tests runnable without a server
  (`make test lint integration`).
- Never guess a server's field names: base telemetry and detection on a recorded response (`tests/data/`, the server
  pages in `docs/servers/`).

## Design decisions

### Goal
Replay real coding-agent (omp) sessions turn by turn against OpenAI-compatible servers and compare models, quants and
engines on them: the main agent at up to about 240k context, its subagents and the compaction call. Each turn sends the
recorded history (exactly what the model saw then) and grades the answer against what the recorded model and the
reference models did. One run gives hundreds of paired samples instead of one end-to-end task score, and the context
is realistic, so speed and cache behaviour at depth are measured too.

### CLI decisions
- `--suite smoke|sanity|coding|judge`; `--preset NAME` is a named set of suite and selection options
  (`bench/presets.py`; the default is `large`); `--fixture FILE` replaces the suite's fixture.
- Selection: `--max-context LIMIT` keeps turns whose recorded prompt plus the 4096-token output reserve fits (k = 1024;
  below the first compaction a stream simply ends at the limit), `--no-subagents`, `--turns A-B` (1-based).
  Subagents run interleaved in the recorded order, so prefix-cache behaviour matches the real run.
- One send per case; to see run-to-run drift, compare two saved states.
- State: JSONL, one flushed line per finished case (`bench/state.py`); `--resume FILE` re-sends only missing or failed
  cases; `--state FILE` shows a run; `--compare-state FILE` compares with another run of the same suite and fixture
  digest (no `--baseline`). Names resolve in `states/`, `fixtures/refs/` and `fixtures/samples/`.
- A live run may cover several models; each is saved to its own file.
- Errors are one `error:` line with exit 2; `REPLAY_BENCH_DEBUG=1` gives tracebacks.

### Fixture format
Gzipped JSON: a shared message pool plus per-entry index lists, entries in chronological order (turns and compactions).
Each entry holds the stream, the message count, the recorded request parameters, the recorded answer and the recorded
prompt and output token counts. Payload = recorded parameters, messages and tools with `model` swapped; no temperature,
reasoning effort or thinking switch unless given (omp sent `enable_thinking: true`; `--thinking off` overrides it).
Cutoffs: 32768 output tokens, 1800 s per request. A case is SKIP when prompt plus reserve exceeds the server's context.

### Proof that a rebuilt request is exact
A fixture is built only from streams that rebuild exactly (`docs/capture-traces.md`):
1. Golden test: rebuilding the snapshot session's turns equals the captured requests, JSON-equal.
2. The rebuilt system prompt reproduces the digest omp recorded.
3. Each rebuilt request's prompt token count equals `contextSnapshot.promptTokens`, counted through the server's own
   request path (the server's default reasoning effort matters: without it every count is off by a constant).
The shipped coding fixture (`conv-3`) rebuilds 470 of 470 requests exactly; the redacted re-count differs from the
recorded count by a few tokens.

### Redaction
The fixtures are published, so every string passes through redaction rules kept outside git (`sources/redact.json`, a
list of `[regex, replacement]`): home paths to `/Users/standard.analyst` (`tools/omp_fixture.py`), e-mail addresses to `dev@example.com` or
`noreply@example.com`, machine and provider names to neutral placeholders, omp's base64 parse trees dropped. The build
rescans the redacted fixture and fails on any hit (`tools/pii_patterns.py`, with a tight allowlist). Redaction changes
token counts, so prompt tokens are re-counted afterwards. Real names are only ever given to the scanner through a
gitignored `--names` file. The user decides what session content may be published, since transcripts quote code.

### Grading (`bench/cases/grading.py`, `fixtures/conv-3.checks.json`)
- FAIL only for clearly broken answers: cut at the token cap, empty, invalid tool-call JSON, unknown tool, tool call
  written as text, repeated calls or text, a broken compaction.
- WARN for schema problems, early stops, off-pattern first steps and invalid syntax in written files.
- PASS otherwise. "Same tools as recorded" is information, not a verdict.
- The first action category (`look`, `wait`, `change`, `delegate`, `plan`, `<text>`) is accepted at a turn when the
  recording took it or at least two distinct reference models did. Several quants or runs of one model count once.
  Argument-level agreement is not built.
- `--reference STATE` (`bench/reference.py`) replaces the recording and the accepted steps with one run as the
  yardstick, `--match step|targets|calls|exact` says what is the same answer (first step category; same tools in order on the same
  targets: file without line range, set of bash programs, path; same calls in order, `i`
  ignored, whitespace-normalised text; exact text and `i`). It is a post-pass view over the normal grading, like
  `--compare-state`: stored verdicts stay the normal ones (the header records `reference`), FAIL stays FAIL, a
  compaction or a turn the reference has no usable answer to keeps its normal verdict. It applies to a live run, to
  `--state` and to the `--compare-state` baseline. Per-turn agreement (first step, tools, calls, text, args
  similarity, tokens, broken, loops; same targets for every level, and calls/targets by the reference's first tool) feeds a `vs reference` line and Metrics rows. `targets`, `calls` and `exact` warn when a
  side was not run at temperature 0.
- References (`fixtures/refs/`) are the expected answers and the only input of `tools/build_checks.py`. Samples
  (`fixtures/samples/`) are for comparison only.

### Servers
Detection, telemetry and prefill progress per server are in `bench/backend.py`, `bench/telemetry.py` and
`bench/transport.py`, documented in `docs/servers/`. Unknown servers run as generic with the OpenAI `usage` block and
client-side timing. Estimates are marked `~` in reports.

### Not built
- Teacher-forced token agreement (prompt logprobs) needs a Splash scoring path.
- `--only-subagents`: not needed so far.
