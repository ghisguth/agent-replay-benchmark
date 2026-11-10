"""--server-version, state-name resolution (states/ + fixtures/refs), add_ref symlinks, triage of reference symlinks."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import add_ref  # noqa: E402
import isolation  # noqa: E402,F401
import triage_states  # noqa: E402
from test_replay import synthetic_fixture, write_fixture  # noqa: E402
from test_state import SplashLike, behaviour, lines, run_main  # noqa: E402

import replay_bench  # noqa: E402
from bench import backend  # noqa: E402
from bench.resume import RESUME_PRESENTATION, resolve_state  # noqa: E402
from bench.utils import UserError  # noqa: E402


def touch(path, text="{}\n"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


class ServerVersion(unittest.TestCase):
    def test_override_keeps_detected_as_build_id(self):
        info = {"kind": "splash", "version": "1a2b3c4d", "build": "src-x"}
        backend.apply_version_override(info, "1.6.0-m1")
        self.assertEqual((info["version"], info["build_id"]), ("1.6.0-m1", "1a2b3c4d"))

    def test_existing_build_id_and_same_value_and_none(self):
        info = {"version": "a", "build_id": "keep"}
        backend.apply_version_override(info, "2")
        self.assertEqual((info["version"], info["build_id"]), ("2", "keep"))
        info = {"version": "2"}
        backend.apply_version_override(info, "2")
        self.assertNotIn("build_id", info)
        info = {"version": None}
        backend.apply_version_override(info, "2")
        self.assertNotIn("build_id", info)
        self.assertIsNone(backend.apply_version_override(None, "2"))
        self.assertEqual(backend.apply_version_override({"version": "a"}, None), {"version": "a"})

    def test_classified_presentation(self):
        self.assertIn("server_version", RESUME_PRESENTATION)

    def test_bad_value_rejected(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            replay_bench.main(["--server-version", "a b"])

    def test_end_to_end_header_and_file_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = write_fixture(synthetic_fixture(), tmp)
            states = os.path.join(tmp, "states")
            with mock.patch.object(replay_bench, "STATES_DIR", states), SplashLike(behaviour) as srv:
                run_main("--suite", "coding", "--fixture", fixture, "--endpoint", srv.url, "--no-mdcat",
                         "--server-version", "1.6.0-m1")
            (name,) = os.listdir(states)
            self.assertTrue(name.startswith("splash-1.6.0-m1_"), name)
            server = lines(os.path.join(states, name))[0]["server"]
            self.assertEqual((server["version"], server["build_id"]), ("1.6.0-m1", "1a2b3c4d"))


class ResolveState(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.states = os.path.join(self.tmp.name, "states")
        self.refs = os.path.join(self.tmp.name, "fixtures", "refs")
        self.samples = os.path.join(self.tmp.name, "fixtures", "samples")
        os.makedirs(self.states)
        os.makedirs(self.refs)
        os.makedirs(self.samples)

    def res(self, name):
        return resolve_state(name, states_dir=self.states, refs_dir=self.refs, samples_dir=self.samples)

    def test_samples_are_searched_last(self):
        s = touch(os.path.join(self.samples, "sample-one.jsonl"))
        r = touch(os.path.join(self.refs, "same-name.jsonl"))
        touch(os.path.join(self.samples, "same-name.jsonl"))
        self.assertEqual(self.res("sample-one.jsonl"), s)
        self.assertEqual(self.res("sample-o"), s)
        self.assertEqual(self.res("same-name.jsonl"), r)  # exact names: states/, refs/, then samples/
        with self.assertRaises(UserError) as cm:
            self.res("same-")  # substring: every directory counts, ambiguity lists them
        self.assertIn("refs", str(cm.exception))
        self.assertIn("samples", str(cm.exception))

    def test_exact_names_in_both_dirs(self):
        a = touch(os.path.join(self.states, "a-one.jsonl"))
        r = touch(os.path.join(self.refs, "r-one.jsonl"))
        self.assertEqual(self.res("a-one.jsonl"), a)
        self.assertEqual(self.res("r-one.jsonl"), r)
        self.assertEqual(self.res(a), a)

    def test_unique_prefix_and_substring(self):
        touch(os.path.join(self.states, "llamacpp-b1_qwen_20261108.jsonl"))
        r = touch(os.path.join(self.refs, "splash-1.6.0-m1_glimmer_20261107.jsonl"))
        self.assertEqual(self.res("splash"), r)
        self.assertTrue(self.res("qwen").endswith("llamacpp-b1_qwen_20261108.jsonl"))

    def test_ambiguous_lists_candidates(self):
        touch(os.path.join(self.states, "x-1.jsonl"))
        touch(os.path.join(self.refs, "x-2.jsonl"))
        with self.assertRaises(UserError) as cm:
            self.res("x-")
        self.assertIn("ambiguous", str(cm.exception))
        self.assertIn("x-1.jsonl", str(cm.exception))
        self.assertIn("x-2.jsonl", str(cm.exception))

    def test_symlink_and_target_count_once(self):
        r = touch(os.path.join(self.refs, "ref-a.jsonl"))
        os.symlink(os.path.relpath(r, self.states), os.path.join(self.states, "ref-a.jsonl"))
        self.assertTrue(self.res("ref-a").endswith("ref-a.jsonl"))
        self.assertEqual(os.path.realpath(self.res("ref-")), os.path.realpath(r))

    def test_not_found_with_hint(self):
        touch(os.path.join(self.refs, "something.jsonl"))
        with self.assertRaises(UserError) as cm:
            self.res("nothing-like-it")
        self.assertIn("state file not found", str(cm.exception))
        with self.assertRaises(UserError) as cm:
            self.res("somethin.jsnl")
        self.assertIn("did you mean", str(cm.exception))


class AddRefLink(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.states = os.path.join(self.tmp.name, "states")
        self.refs = os.path.join(self.tmp.name, "fixtures", "refs")
        os.makedirs(self.refs)
        self.src = touch(os.path.join(self.states, "x.jsonl"), "orig\n")
        self.dest = touch(os.path.join(self.refs, "x.jsonl"), "scrubbed\n")

    def test_regular_file_is_replaced_by_relative_symlink(self):
        kept = add_ref.link_source(self.src, self.dest, self.states)
        self.assertEqual(kept, os.path.join(self.states, "review", "refs-originals", "x.jsonl"))
        self.assertTrue(os.path.islink(self.src))
        self.assertFalse(os.path.isabs(os.readlink(self.src)))
        with open(self.src) as f:
            self.assertEqual(f.read(), "scrubbed\n")
        with open(kept) as f:
            self.assertEqual(f.read(), "orig\n")

    def test_skips_symlink_and_outside_states(self):
        add_ref.link_source(self.src, self.dest, self.states)
        self.assertIsNone(add_ref.link_source(self.src, self.dest, self.states))
        out = touch(os.path.join(self.tmp.name, "elsewhere", "y.jsonl"))
        self.assertIsNone(add_ref.link_source(out, self.dest, self.states))
        self.assertFalse(os.path.islink(out))

    def test_main_dry_run_leaves_source(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = add_ref.main(["--dry-run", "--hosts", os.path.join(self.tmp.name, "h.json"), "--refs-dir", self.refs,
                               "--states-dir", self.states, "--force", self.src])
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.islink(self.src))


class TriageRefs(unittest.TestCase):
    def test_ref_symlink_shown_as_ref_and_review_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            states, refs = os.path.join(tmp, "states"), os.path.join(tmp, "refs")
            hdr = json.dumps({"type": "run", "suite": "smoke", "backend": "h:1", "model": "m"}) + "\n"
            r = touch(os.path.join(refs, "ref-a.jsonl"), hdr)
            os.makedirs(states)
            os.symlink(os.path.relpath(r, states), os.path.join(states, "ref-a.jsonl"))
            touch(os.path.join(states, "review", "refs-originals", "ref-a.jsonl"), hdr)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                triage_states.main(["--states-dir", states, "--refs-dir", refs])
            text = out.getvalue()
            self.assertIn("ref  ref-a.jsonl", text)
            self.assertNotIn("smoke", text)
            self.assertNotIn("unchanged", text)

    def test_sample_symlink_shown_as_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            states, samples = os.path.join(tmp, "states"), os.path.join(tmp, "samples")
            hdr = json.dumps({"type": "run", "suite": "smoke", "backend": "h:1", "model": "m"}) + "\n"
            r = touch(os.path.join(samples, "s-a.jsonl"), hdr)
            os.makedirs(states)
            os.symlink(os.path.relpath(r, states), os.path.join(states, "s-a.jsonl"))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                triage_states.main(["--states-dir", states, "--refs-dir", os.path.join(tmp, "refs"), "--samples-dir", samples])
            self.assertIn("sample  s-a.jsonl", out.getvalue())
            self.assertNotIn("smoke", out.getvalue())


if __name__ == "__main__":
    unittest.main()
