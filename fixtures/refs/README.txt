Reference runs (state files) of high-fidelity models = the expected answers; tools/build_checks.py reads every *.jsonl here (and nothing in fixtures/samples/) to build "accepted". A step is accepted when 2 distinct models (model id without org/quant/format tags) took it, or the recorded agent did.
Other shipped runs (to compare against, never graded against or used for grading) live in fixtures/samples/: make add-sample FILE=states/x.jsonl.
Add a run with: make add-ref FILE=states/x.jsonl   (tools/add_ref.py: scrubs host names, scans for personal data,
copies the file here, rebuilds fixtures/conv-3.checks.json; --dry-run first shows what would be replaced).
The host alias table is sources/hosts.json (gitignored, never committed); without it hosts become host-<sha1 prefix>.
Rebuild the sidecar alone with: python3 tools/build_checks.py

Answers that are FAIL, or WARN for anything but "off pattern", do not add to "accepted" (tools/build_checks.py broken()).
A run note (--run-note, e.g. MTP, DFlash2) is part of the header and stays when a file is added; the two llama.cpp Q8_0 runs carry MTP and DFlash2.
