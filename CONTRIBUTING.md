# Contributing

Layout, rules and design decisions are in [AGENTS.md](AGENTS.md); read it first.

- Run `make test lint integration` before sending a change. All three run offline (the integration tests use
  `tools/mock_server.py` on 127.0.0.1). `make bootstrap` sets up a `.venv` with ruff.
- Python 3.9+, standard library only, no runtime dependencies.
- No raw transcripts, secrets or personal data in commits. Ask before running anything against a server someone else
  is using.
- Bug reports: include the output of `replay-bench --discover --endpoint URL` for the server in question and the header
  line (first line) of the state file. Check that header for host names or paths you would rather not publish.
