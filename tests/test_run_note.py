"""--run-note TEXT: stored as `note` in the state header, shown in the run header line and Run table, in the
compare and performance "Run" labels (model · note, baseline · note), in the state file name (slugified) and in
triage; add_ref keeps it and --resume treats it as a presentation option. Mock servers and temp files only.

    python3 -m unittest tests.test_run_note
"""

import argparse
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import add_ref  # noqa: E402
import isolation  # noqa: E402,F401
import test_resume  # noqa: E402
import triage_states  # noqa: E402
from test_report import write_state  # noqa: E402
from test_resume import N_CASES, run_code  # noqa: E402
from test_state import SplashLike, behaviour  # noqa: E402

import replay_bench  # noqa: E402
from bench import cli_args, compare, perf, results, resume, state  # noqa: E402


def states_entries(paths):
    """The comparison entries of several state files, labelled together (what a run of several models gets)."""
    out = []
    for p in paths:
        raw, keys = state.load_state(p)
        out += compare.entries_from_run(keys, raw)
    return compare.run_labels(out)

TARGET = {"key": "h:1::m", "backend": "h:1", "name": "m", "url": "http://h:1/v1/chat/completions", "n_ctx": 65536,
          "backend_info": {"kind": "llamacpp", "version": "b1", "build": "b1-x", "details": {}}}


def first_line(path):
    with open(path) as f:
        return json.loads(f.readline())


def with_note(path, note, resume_note=None):
    """The state file with `note` added to its header (and a resume record carrying resume_note, when given)."""
    with open(path) as f:
        lines = f.read().splitlines()
    head = json.loads(lines[0])
    head["note"] = note
    lines[0] = json.dumps(head)
    if resume_note:
        lines.append(json.dumps({"type": "resume", "at": "t", "argv": [], "note": resume_note}))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


class Parsing(unittest.TestCase):
    def test_whitespace_collapses_and_empty_or_long_is_refused(self):
        self.assertEqual(cli_args.parse_run_note("  DFlash   2 "), "DFlash 2")
        for bad in ("", "   ", "x" * 61):
            with self.assertRaises(argparse.ArgumentTypeError):
                cli_args.parse_run_note(bad)

    def test_the_option_exists_and_is_a_presentation_option(self):
        args = replay_bench.build_parser().parse_args(["--run-note", "MTP"])
        self.assertEqual(args.run_note, "MTP")
        self.assertIn("run_note", resume.RESUME_PRESENTATION)
        self.assertNotIn("run_note", resume.RESUME_MATCH)


class HeaderAndFileName(unittest.TestCase):
    def test_header_carries_the_note_only_when_given(self):
        self.assertEqual(state.make_header("coding", TARGET, note="MTP")["note"], "MTP")
        self.assertNotIn("note", state.make_header("coding", TARGET))

    def test_file_name_has_the_slugified_lowercase_note_after_the_sampling_parts(self):
        when = datetime(2026, 11, 8, 12, 0, 0)
        path = state.default_path("states", TARGET["backend_info"], "m", "coding", when=when,
                                  request={"temperature": 0.6}, preset="large", note="DFlash 2")
        self.assertEqual(os.path.basename(path), "llamacpp-b1_m_coding-large_t0.6_dflash-2_20261108-120000.jsonl")
        plain = state.default_path("states", TARGET["backend_info"], "m", "coding", when=when, request={"temperature": 0.6})
        self.assertNotIn("dflash", plain)
        self.assertEqual(state.default_path("states", None, "m", "smoke", when=when, note="MTP/x!"),
                         os.path.join("states", "generic_m_smoke_mtp-x_20261108-120000.jsonl"))


class Labels(unittest.TestCase):
    def test_entries_take_the_note_from_the_header(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.jsonl")
            write_state(p, model="vendor/A")
            with_note(p, "MTP")
            (e,) = states_entries([p])
        self.assertEqual(e["note"], "MTP")
        self.assertEqual(e["label"], "vendor/A · MTP")

    def test_same_model_two_notes_are_two_runs_not_a_numbered_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.jsonl"), os.path.join(d, "b.jsonl")
            write_state(a, model="m")
            write_state(b, model="m")
            with_note(a, "MTP")
            with_note(b, "DFlash2")
            entries = states_entries([a, b])
        self.assertEqual([e["label"] for e in entries], ["m · MTP", "m · DFlash2"])
        self.assertEqual(compare.labels(entries), ["m · MTP", "m · DFlash2"])
        self.assertEqual(perf.run_names(entries), ["m · MTP", "m · DFlash2"])

    def test_distinct_models_keep_the_short_name_and_add_the_note(self):
        es = [compare.entry("h:1", "vendor/A", {}, note="MTP"), compare.entry("h:1", "vendor/B", {})]
        self.assertEqual(compare.labels(es), ["A · MTP", "B"])
        self.assertEqual(perf.run_names(es), ["A · MTP", "B"])

    def test_a_baseline_run_is_baseline_note(self):
        # comparison runs in perf table use base / this run
        es = [compare.entry("h:1", "m (baseline)", {}, label="MTP"), compare.entry("h:1", "m", {}, note="DFlash2")]
        self.assertEqual(perf.run_names(es), ["base", "this run"])

    def test_compare_state_report_shows_both_notes(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.jsonl"), os.path.join(d, "b.jsonl")
            write_state(a, model="m")
            write_state(b, model="m", speed=30.0)
            with_note(a, "DFlash2")
            with_note(b, "MTP")
            out, err, code = run_code("--state", a, "--compare-state", b, "--report-format", "markdown", "--no-mdcat")
        self.assertEqual(code, 0, err)
        self.assertIn("compared with: MTP", out)  # same server and model: the baseline's note
        self.assertIn("| Metric                              |                MTP |", out)
        self.assertIn("| Note ", out)
        self.assertIn("DFlash2", out)

    def test_no_note_no_change(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.jsonl"), os.path.join(d, "b.jsonl")
            write_state(a, model="m")
            write_state(b, model="m", speed=30.0)
            out, _, _ = run_code("--state", a, "--compare-state", b, "--report-format", "markdown", "--no-mdcat")
        self.assertNotIn("· MTP", out)
        self.assertNotIn("| Note ", out)
        self.assertIn("baseline", out)


class RunHeader(unittest.TestCase):
    def infos(self, note):
        return [results.run_info("coding", dict(TARGET), note=note)]

    def test_headline_and_run_table(self):
        info = self.infos("MTP")
        line = results.headline(info, 58)
        self.assertEqual(line, "Agent replay · suite coding (58 requests) · llama.cpp b1 · MTP · ctx 64k")
        text = results.headline(self.infos(None), 58)
        self.assertNotIn("MTP", text)
        rows = [b for b in results.run_blocks(info, ["m"], 58) if hasattr(b, "rows")]
        self.assertIn(("Note", "MTP"), [tuple(r) for r in rows[0].rows])

    def test_headline_and_run_table_model_info(self):
        t = dict(TARGET, model_info={"quant": "Q4_K_M", "format": "gguf", "parameter_size": "27.3B"})
        info = [results.run_info("coding", t)]
        line = results.headline(info, 58)
        self.assertIn("Q4_K_M (gguf, 27.3B)", line)
        rows = [b for b in results.run_blocks(info, ["m"], 58) if hasattr(b, "rows")]
        self.assertIn(("Model info", "Q4_K_M (gguf, 27.3B)"), [tuple(r) for r in rows[0].rows])

    def test_the_note_of_a_state_header_is_used_when_none_is_passed(self):
        t = dict(TARGET, header={"suite": "coding", "note": "DFlash2"})
        self.assertEqual(results.run_info("coding", t)["note"], "DFlash2")

    def test_several_models_show_the_note_in_the_model_cell(self):
        infos = self.infos("MTP") + [results.run_info("coding", dict(TARGET))]
        rows = results.run_blocks(infos, ["A · MTP", "B"], 5)[2].rows
        self.assertEqual([str(r[0]) for r in rows], ["A · MTP", "B"])


class Triage(unittest.TestCase):
    def test_head_line_shows_the_note_after_the_model(self):
        head = {"backend": "h:1", "model": "m", "suite": "coding", "note": "MTP", "request": {}}
        line = triage_states.head_line(head, None, 1)
        self.assertIn("m · MTP", line)
        self.assertNotIn("·", triage_states.head_line(dict(head, note=None), None, 1))


class AddRef(unittest.TestCase):
    def test_the_note_stays_in_the_header_of_the_added_file(self):
        with tempfile.TemporaryDirectory() as d:
            src, refs = os.path.join(d, "x.jsonl"), os.path.join(d, "refs")
            head = {"type": "run", "endpoint": "http://boxy.local:8133/v1", "backend": "boxy.local:8133", "argv": [],
                    "note": "DFlash2"}
            with open(src, "w") as f:
                f.write(json.dumps(head) + "\n" + json.dumps({"type": "result", "content": "ok"}) + "\n")
            with open(src) as f:
                text, _ = add_ref.scrub(f.read(), {"boxy.local": "host-1"})
            os.makedirs(refs)
            dest = os.path.join(refs, "x.jsonl")
            with open(dest, "w") as f:
                f.write(text)
            copied = first_line(dest)
        self.assertEqual(copied["note"], "DFlash2")
        self.assertEqual(copied["backend"], "host-1:8133")

    def test_the_refs_carry_their_notes(self):
        refs = os.path.join(ROOT, "fixtures", "refs")
        notes = {}
        for name in os.listdir(refs):
            if name.endswith(".jsonl") and "Q8_0_coding-large" in name and name.startswith("llamacpp"):
                notes[name.rsplit("_", 1)[1].split(".")[0]] = first_line(os.path.join(refs, name)).get("note")
        self.assertEqual(notes, {"20261108-161443": "MTP", "20261108-175742": "DFlash2"})
        with open(os.path.join(refs, "README.txt")) as f:
            self.assertNotIn("llama.cpp with MTP", f.read())  # the lines that told the two runs apart are gone


class ResumeNote(unittest.TestCase):
    setUp = test_resume.Resume.setUp  # the helpers of the resume tests, without re-running them
    first_run = test_resume.Resume.first_run
    records = test_resume.Resume.records

    def test_a_note_given_on_resume_replaces_the_recorded_one(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv, 3, "--run-note", "MTP")
            self.assertEqual(self.records()[0]["note"], "MTP")
            _, err, code = run_code("--resume", self.path)  # the recorded command carries --run-note MTP
            self.assertEqual(code, 0, err)
            self.assertEqual(state.read_blocks(self.path)[0]["header"]["note"], "MTP")

    def test_given_run_note_wins_and_is_not_a_match_option(self):
        with SplashLike(behaviour) as srv:
            self.first_run(srv, 3, "--run-note", "MTP")
            _, err, code = run_code("--resume", self.path, "--run-note", "DFlash2")
            self.assertEqual(code, 0, err)  # a presentation option: no "differs from the recorded run" error
        resumes = [r for r in self.records() if r["type"] == "resume"]
        self.assertEqual(resumes[-1]["note"], "DFlash2")
        self.assertEqual(state.read_blocks(self.path)[0]["header"]["note"], "DFlash2")
        self.assertEqual(len(state.read_blocks(self.path)[0]["results"]), N_CASES)


class DecodedModelLabel(unittest.TestCase):
    MI = {"label": "mlx-community/Qwen3.8-27B-4bit", "mode": "dflash", "drafter": "incoai/Qwen3.8-27B-DFlash2"}
    HASH = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"

    def test_label_and_file_name_use_org_name_and_the_mode(self):
        self.assertEqual(state.model_label(self.HASH, self.MI), "mlx-community/Qwen3.8-27B-4bit")
        when = datetime(2026, 11, 8, 12, 0, 0)
        info = {"kind": "mlx-dspark", "version": "0.20.4"}
        name = os.path.basename(state.default_path("states", info, self.HASH, "coding", when=when, model_info=self.MI))
        self.assertEqual(name, "mlx-dspark-0.20.4_mlx-community-Qwen3.8-27B-4bit_coding_dflash_20261108-120000.jsonl")
        noted = os.path.basename(state.default_path("states", info, self.HASH, "coding", when=when, model_info=self.MI,
                                                    note="DFlash 2"))
        self.assertIn("_dflash-2_", noted)  # an explicit note wins over the mode

    def test_headline_names_mode_and_drafter(self):
        text = results.model_info_text(self.MI)
        self.assertEqual(text, "dflash, drafter incoai/Qwen3.8-27B-DFlash2")
        self.assertNotIn("/Users", text)


if __name__ == "__main__":
    unittest.main()
