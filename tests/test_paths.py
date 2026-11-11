import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from bench import paths, resume  # noqa: E402
from bench.utils import UserError  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(**kw):
    """A clean environment for the states folder lookup, with the given variables set."""
    base = {k: v for k, v in os.environ.items() if k not in ("REPLAY_BENCH_STATES_DIR", "XDG_DATA_HOME", "LOCALAPPDATA")}
    base.update(kw)
    return mock.patch.dict(os.environ, base, clear=True)


class UserStatesDirTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.home, ignore_errors=True))

    def installed(self, **env):
        stack = [_env(HOME=self.home, USERPROFILE=self.home, **env), mock.patch.object(paths, "is_checkout", return_value=False)]
        for m in stack:
            m.start()
            self.addCleanup(m.stop)

    def test_env_override_wins_and_expands_home(self):
        with _env(HOME=self.home, REPLAY_BENCH_STATES_DIR="~/mine"), mock.patch.object(paths, "is_checkout", return_value=True):
            self.assertEqual(paths.user_states_dir(), os.path.join(self.home, "mine"))
        with _env(REPLAY_BENCH_STATES_DIR="/x/y"), mock.patch.object(paths, "is_checkout", return_value=False):
            self.assertEqual(paths.user_states_dir(), "/x/y")

    def test_checkout_uses_repo_states(self):
        with _env(HOME=self.home, XDG_DATA_HOME="/xdg"), mock.patch.object(paths, "is_checkout", return_value=True):
            self.assertEqual(paths.user_states_dir(), os.path.join(paths.ROOT, "states"))
            self.assertEqual(paths.default_states_dir(), paths.user_states_dir())

    def test_installed_default_and_xdg(self):
        self.installed()
        self.assertEqual(paths.user_states_dir(), os.path.join(self.home, ".local", "share", "replay-bench", "states"))
        with _env(HOME=self.home, XDG_DATA_HOME="/data/xdg"):
            self.assertEqual(paths.user_states_dir(), os.path.join("/data/xdg", "replay-bench", "states"))

    def test_relative_xdg_is_ignored(self):
        self.installed(XDG_DATA_HOME="relative/dir")
        self.assertEqual(paths.user_states_dir(), os.path.join(self.home, ".local", "share", "replay-bench", "states"))

    def test_windows(self):
        self.installed()
        with mock.patch.object(paths, "_is_windows", return_value=True):
            with _env(HOME=self.home, LOCALAPPDATA="/win/local"):
                self.assertEqual(paths.user_states_dir(), os.path.join("/win/local", "replay-bench", "states"))
            with _env(HOME=self.home):
                self.assertEqual(paths.user_states_dir(), os.path.join(self.home, "AppData", "Local", "replay-bench", "states"))

    def test_nothing_created_by_lookup_or_import(self):
        home = tempfile.mkdtemp()
        env = {"HOME": home, "PATH": os.environ.get("PATH", ""), "PYTHONPATH": ROOT}
        code = "import replay_bench, bench.paths as p; p.user_states_dir(); p.legacy_states_dir()"
        subprocess.run([sys.executable, "-c", code], env=env, cwd=home, check=True)
        for _dirpath, dirs, _files in os.walk(home):
            self.assertNotIn("replay-bench", dirs)
            self.assertNotIn("states", dirs)

    def test_tilde(self):
        with _env(HOME=self.home):
            self.assertEqual(paths.tilde(os.path.join(self.home, "a", "b")), os.path.join("~", "a", "b"))
            self.assertEqual(paths.tilde("/elsewhere/x"), "/elsewhere/x")
            self.assertEqual(paths.tilde(self.home + "x/y"), self.home + "x/y")


class LookupOrderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.states = os.path.join(self.tmp, "data", "states")
        self.cwd = os.path.join(self.tmp, "work")
        self.refs = os.path.join(self.tmp, "refs")
        self.samples = os.path.join(self.tmp, "samples")
        for d in (self.states, os.path.join(self.cwd, "states"), self.refs, self.samples):
            os.makedirs(d)
        old = os.getcwd()
        os.chdir(self.cwd)
        self.addCleanup(os.chdir, old)
        m = mock.patch.dict(os.environ, {"REPLAY_BENCH_STATES_DIR": self.states})
        m.start()
        self.addCleanup(m.stop)

    def touch(self, *parts):
        p = os.path.join(*parts)
        open(p, "w").close()
        return p

    def resolve(self, name):
        return resume.resolve_state(name, states_dir=self.states, refs_dir=self.refs, samples_dir=self.samples)

    def test_order(self):
        s = self.touch(self.states, "run_a.jsonl")
        lg = self.touch(self.cwd, "states", "run_a.jsonl")
        r = self.touch(self.refs, "run_a.jsonl")
        sm = self.touch(self.samples, "run_a.jsonl")
        self.assertEqual(self.resolve("run_a.jsonl"), s)
        os.remove(s)
        self.assertEqual(os.path.realpath(self.resolve("run_a.jsonl")), os.path.realpath(lg))
        os.remove(lg)
        self.assertEqual(self.resolve("run_a.jsonl"), r)
        os.remove(r)
        self.assertEqual(self.resolve("run_a.jsonl"), sm)

    def test_path_as_given_first(self):
        self.touch(self.states, "x.jsonl")
        own = self.touch(self.cwd, "x.jsonl")
        self.assertEqual(self.resolve("x.jsonl"), "x.jsonl")
        self.assertTrue(os.path.samefile("x.jsonl", own))

    def test_legacy_substring_fallback(self):
        lg = self.touch(self.cwd, "states", "old-run_20260101.jsonl")
        self.assertEqual(os.path.realpath(self.resolve("old-run")), os.path.realpath(lg))

    def test_legacy_skipped_when_same_folder(self):
        with mock.patch.dict(os.environ, {"REPLAY_BENCH_STATES_DIR": "states"}):
            self.assertIsNone(paths.legacy_states_dir())

    def test_not_found_lists_folders(self):
        with self.assertRaises(UserError) as cm:
            self.resolve("nothing-here")
        msg = str(cm.exception)
        self.assertIn("searched:", msg)
        for d in (self.states, os.path.join(self.cwd, "states"), self.refs, self.samples):
            self.assertIn(os.path.basename(d), msg)


if __name__ == "__main__":
    unittest.main()
