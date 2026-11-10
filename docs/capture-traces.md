This page applies to sessions recorded with omp (oh-my-pi); other agents are not supported.

# Capturing sessions and building fixtures

A replay fixture is built from a recorded omp session plus a snapshot of what the session does not store (system
prompt, tool schemas, request parameters). This page covers the whole chain: record, snapshot, prove, build.
Raw sessions, proxy dumps and server logs never go into git (`sources/` and `captures/` are gitignored); only the
redacted fixture is committed.

## 1. Record the omp session

Run the task in omp as usual, on the model under test. Sessions are written to
`~/.omp/agent/sessions/<cwd-slug>/<timestamp>_<id>.jsonl`, where the slug is the working directory with `/` turned
into `-`. Subagent transcripts go in the sibling directory with the same name minus `.jsonl`
(`<timestamp>_<id>/`). Copy both into `sources/` before working on them; never edit the originals.

What a session holds:
- Main `.jsonl`: user, assistant (`thinking`, `text`, `toolCall`) and `toolResult` messages, `usage`,
  `contextSnapshot` (`promptTokens`, `nonMessageTokens`, `toolSchemaTokens`, `compactionEpoch`), `compaction` entries
  (summary, `firstKeptEntryId`, `tokensBefore`/`tokensAfter`), `custom_message` and `custom` entries (skill
  prompts, date/cwd reminders, `system-prompt-digest`, todo state). Entries form a tree (`id` / `parentId`).
- Subagent `.jsonl`: a `session_init` entry with the full system prompt blocks, the task text, the agent name and
  its tool names, then the turns.

What it does not hold: the main agent's system prompt text (only its digest), the tool schemas, the request
parameters, how omp turns entries into chat messages, requests omp dropped, and the compaction request itself.
The tools rebuild those from the snapshot (section 2) and omp's source.

Check what a session contains:
```zsh
python3 tools/omp_session.py sources/<session>.jsonl --turns
```

## 2. Snapshot: system prompt, tools, request parameters

One short omp session through the capture proxy gives the missing pieces. Use the same omp version and the same
checkout as the session you want to replay (the prompt carries the cwd, AGENTS.md and the skill list).

Start the proxy (it forwards to the real server and saves `<ms>_req.json` / `<ms>_resp.json` pairs under
`captures/dumps/<topic>/`):
```zsh
python3 tools/proxy.py --target http://localhost:8133 --port 18133 --host 127.0.0.1 --topic my-snapshot -q
```
Flags: `--target` backend base URL (default `http://localhost:8080`), `--port` (default 18080), `--host`,
`--topic` or `--dump-dir`, `--ca-cert` for an HTTPS target, `--ipv4`, `-q`. The proxy buffers streamed answers, so
omp shows each turn's text only when it ends.

Add a provider in `~/.omp/agent/models.yml` that is a copy of the real one (keep its `compat` block and model list,
they change the request format) with `baseUrl: http://127.0.0.1:18133/v1`, and enable it in
`~/.omp/agent/config.yml`. In the checkout, pick that provider with `/model` and send two messages:
1. `say ok` (main agent request)
2. `Use the task tool once with a single task: "Reply with the single word ok and stop." Then reply ok.`
   (a default subagent's request)

Stop omp and the proxy. Build the snapshot:
```zsh
python3 tools/omp_requests.py snapshot --capture captures/dumps/my-snapshot \
  --session sources/<snapshot-session>.jsonl --retarget PROVIDER/MODEL -o sources/snapshot.json
```
- `--capture` the proxy dump directory.
- `--session` the omp session that produced the capture. Without it the snapshot is built from the capture alone;
  then give `--subagent-agent NAME` (the capture's subagent agent name) and `--omp-version`.
- `--retarget PROVIDER/MODEL` rewrites the `Model:` line and the body model to the provider and model the session
  to be replayed used. The prompt names model and provider, so keep the recorded ones.
- `--digest HEX` splits the main prompt into blocks that give this `system-prompt-digest`; `--split-main-at N`
  splits it into two blocks at a `\n\n` offset. `-o` writes the snapshot JSON.

Request parameters seen in a capture: `stream: true`, `stream_options.include_usage`, `max_completion_tokens`,
`chat_template_kwargs`, no temperature unless omp was configured with one.

## 3. Prove the rebuild is exact

A fixture is built only from streams that rebuild exactly.

1. Golden test: rebuild the snapshot session's own turns and compare with the captured requests, JSON-equal.
   ```zsh
   python3 tools/omp_requests.py golden --capture captures/dumps/my-snapshot --session sources/<snapshot-session>.jsonl
   ```
   `--snapshot FILE` uses a snapshot instead of one built from the capture.
2. System prompt digest: the snapshot's main prompt must reproduce the digest omp recorded in the session
   (`system-prompt-digest` entries).
   ```zsh
   python3 tools/omp_requests.py digest --snapshot sources/snapshot.json
   bun tools/omp-prompt-digest.ts blocks.json     # independent check: JSON list of block strings, joined by "\n\n"
   ```
   `tools/omp-prompt-digest.ts` is omp's wyhash chain; the Python port in `omp_requests.py` agrees with it.
   Case-specific prompts (AGENTS.md, skills as they were then) may need rebuilding from the checkout's history
   until the digest matches. A session whose checkout differed from the capture's gets its own snapshot from an
   existing one: `snapshot --from SNAP --context-file PATH=FILE` (the AGENTS.md embedded as `<file path="PATH">`,
   e.g. `git show REV:AGENTS.md` at the session's commit), `--enable-tool find` / `--disable-tool NAME` (the
   project's `.omp/config.yml` tool switches; the prompt's and the other tools' find-dependent lines follow),
   `--agent-blocking NAME=on|off` (`.omp/agents/NAME.md` `blocking:`), then `--digest HEX` to prove and split it.
3. Rebuild every turn: `python3 tools/omp_requests.py build --session S --snapshot SNAP --all` reports errors;
   `--stream`, `--turn`, `--compaction N` print one request; `--tool-source POOL=TOOL` takes a subagent tool's
   schema from a snapshot pool when the subagent's own is unknown (for example `main=grep`).
4. Token counts: each rebuilt request's prompt token count must equal `contextSnapshot.promptTokens`. The check
   counts through the server's own request path, so run it under Splash's venv python with a Splash checkout, the
   model's GGUF, and optionally the server log of the session's day (compares tool signatures too):
   ```zsh
   ~/Code/splash/.venv/bin/python tools/omp_tokens.py check --session sources/<session>.jsonl \
     --snapshot sources/snapshot.json --splash ~/Code/splash --gguf PATH/TO/model.gguf \
     --splash-log sources/<splash-day>.log --default-reasoning-effort medium --json sources/check.json
   ```
   Other flags: `--family`, `--limit N` (mismatching turns to list), `--notes`, `--no-compactions`,
   `--tool-source`. Use the server's real default reasoning effort, or every request is off by a constant.
   It prints a per-stream table and the mismatching turns; compaction calls are checked against the log by
   value. `omp_tokens.py count REQUEST.json ...` counts built request bodies directly.

## 4. Build and check the fixture

Redaction rules are a JSON list of `[regex, replacement]` pairs, applied in order to every string. They contain
real names and paths, so the file lives in gitignored `sources/redact.json` and is never committed:
```json
[
  ["/Users/realname", "/Users/dev"],
  ["realname@example\\.org", "dev@example.com"],
  ["\\bmachine-[a-z0-9-]+\\b", "host-a"],
  ["PRIVATE-PROVIDER", "LOCAL-8133"]
]
```
Scan the raw session first for secrets and personal data. `--names FILE` is a JSON list of regexes (personal
names, user ids, project paths) reported as `<id>:names`; keep it out of the repo too:
```zsh
python3 tools/omp_session_pii_scan.py sources/<session>.jsonl --names sources/names.json
```
A session argument may also be a session directory (its `.jsonl` plus the sibling subagent files).

Build, then check:
```zsh
~/Code/splash/.venv/bin/python tools/omp_fixture.py build --session sources/<session>.jsonl \
  --snapshot sources/snapshot.json --redact sources/redact.json --names sources/names.json --recount
python3 tools/omp_fixture.py check --fixture fixtures/<name>.json.gz --redact sources/redact.json --names sources/names.json
```
Give `--output fixtures/<name>.json.gz` (the shipped coding fixture is `conv-3.json.gz`); `--suite` and `--case` label the fixture, and `--redact` defaults to `sources/redact.json`. `--recount` re-counts prompt tokens on the redacted requests (redaction changes them) through
the Splash counter, so it needs the Splash venv python and `--splash`, `--gguf`, `--family`, `--effort` if the
defaults do not apply; without it the recorded counts are kept. The build rescans the redacted fixture and fails on
any PII or redact-pattern hit. `check` validates the structure, reconstructs every entry and scans again (the
redact and names files are optional; a missing redact file is skipped with a note).

Fixtures are gzipped JSON with a shared message pool; keep them small (well under 50 MB). The user decides what
session content may be published, since transcripts quote the checked-out code.
