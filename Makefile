# Development tasks. Run `make` for the list.
VENV   := .venv
PYTHON ?= $(shell if [ -x $(VENV)/bin/python ]; then echo $(VENV)/bin/python; else echo python3; fi)
RUFF   ?= $(shell if [ -x $(VENV)/bin/ruff ]; then echo $(VENV)/bin/ruff; elif command -v ruff >/dev/null 2>&1; then echo ruff; fi)

# `make smoke` settings (environment or command line): ENDPOINT, MODEL (optional), CA_FILE (optional), ARGS (extra flags)
ENDPOINT ?= http://localhost:8080/v1
MODEL    ?=
CA_FILE  ?=
ARGS     ?=

.DEFAULT_GOAL := help
.PHONY: smoke triage add-ref add-sample help bootstrap test integration lint clean distclean

help: ## Show this help
	@awk '\
	/^##@/ { \
		n++; type[n] = "group"; sub(/^##@[ ]*/, ""); text[n] = $$0; next \
	} \
	/^##  / { \
		n++; type[n] = "var"; sub(/^##  /, ""); text[n] = $$0; next \
	} \
	/^[a-zA-Z0-9_-]+:.*##/ { \
		n++; type[n] = "target"; \
		target = substr($$0, 1, index($$0, ":") - 1); \
		name[n] = target; \
		tlen = length(target); \
		if (tlen > max_len) max_len = tlen; \
		desc = $$0; \
		sub(/^.*##[ ]*/, "", desc); \
		text[n] = desc; \
		next \
	} \
	END { \
		width = max_len + 2; \
		if (width < 20) width = 20; \
		indent = "   "; \
		desc_col = 3 + width + 1; \
		var_indent = sprintf("%*s", desc_col, ""); \
		print "🎯 agent-replay-benchmark: make targets\n"; \
		for (i = 1; i <= n; i++) { \
			if (type[i] == "group") { \
				print "\n" text[i] \
			} else if (type[i] == "target") { \
				printf "%s%-*s %s\n", indent, width, name[i], text[i] \
			} else if (type[i] == "var") { \
				printf "%s%s\n", var_indent, text[i] \
			} \
		} \
		print "\nUsage: make <target> [VAR=value ...]" \
	}' $(MAKEFILE_LIST)

##@ 🛠️ Setup
bootstrap: ## Create .venv and install the package with dev extras (needs network)
	@[ -d $(VENV) ] || python3 -m venv $(VENV)
	$(VENV)/bin/pip install -e '.[dev]'

clean: ## Remove __pycache__ and integration temp output
	find . -name __pycache__ -type d -not -path './$(VENV)/*' -prune -exec rm -rf {} +
	rm -rf /tmp/replay-bench-it-*

distclean: clean ## clean, and remove .venv
	rm -rf $(VENV)

##@ 🧪 Tests
test: ## Unit tests (offline, no server)
	$(PYTHON) -m unittest discover -s tests

integration: ## End-to-end tests: real CLI against tools/mock_server.py on 127.0.0.1
	$(PYTHON) -m unittest discover -s tests/integration

lint: ## ruff check (skipped when ruff is not installed)
	@if [ -n "$(RUFF)" ]; then $(RUFF) check .; else echo "ruff not found: run 'make bootstrap' or install ruff"; fi

##@ 🚀 Runs (ask before running against a server someone else is using)
smoke: ## Smoke suite against a server
##  ENDPOINT=http://localhost:8080/v1  MODEL=substring  CA_FILE=path  ARGS='extra flags'
	$(PYTHON) replay_bench.py --suite smoke --endpoint '$(ENDPOINT)' $(if $(MODEL),--model '$(MODEL)') $(if $(CA_FILE),--ca-file '$(CA_FILE)') $(ARGS)

##@ 📊 Results (maintainer reference-set workflow)
triage: ## What is new in states/ since the last triage (ARGS='--all', '--quiet-warn', ...)
	$(PYTHON) tools/triage_states.py $(ARGS)

add-ref: ## Add a good run to the reference set (scrub, copy to fixtures/refs, rebuild checks)
##  FILE='states/a.jsonl states/b.jsonl'  ARGS='--dry-run'
	@[ -n "$(FILE)" ] || { echo "usage: make add-ref FILE=states/x.jsonl [ARGS=--dry-run]"; exit 2; }
	$(PYTHON) tools/add_ref.py $(ARGS) $(FILE)

add-sample: ## Add a run to the shipped samples (scrub, copy to fixtures/samples, checks untouched)
##  FILE='states/a.jsonl states/b.jsonl'  ARGS='--dry-run'
	@[ -n "$(FILE)" ] || { echo "usage: make add-sample FILE=states/x.jsonl [ARGS=--dry-run]"; exit 2; }
	$(PYTHON) tools/add_ref.py --sample $(ARGS) $(FILE)
