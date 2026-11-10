"""The ./replay-bench launcher: POSIX sh, executable, finds the checkout through symlinks and a Python 3.9+."""

import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LAUNCHER = os.path.join(ROOT, "replay-bench")
SH = shutil.which("sh")


@unittest.skipUnless(SH, "no sh")
class Launcher(unittest.TestCase):
    def run_sh(self, path, *args, env=None):
        return subprocess.run([SH, path, *args], capture_output=True, text=True, timeout=60, env=env)

    def test_is_executable_posix_sh(self):
        self.assertTrue(os.access(LAUNCHER, os.X_OK))
        with open(LAUNCHER) as f:
            self.assertEqual(f.readline().strip(), "#!/bin/sh")

    def test_help(self):
        r = self.run_sh(LAUNCHER, "--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--suite", r.stdout)

    def test_through_a_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            link = os.path.join(tmp, "replay-bench")
            os.symlink(LAUNCHER, link)
            r = self.run_sh(link, "--help")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("--suite", r.stdout)

    def test_no_python_says_how_to_get_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            os.mkdir(bin_dir)
            for tool in ("dirname", "readlink"):  # what the launcher needs besides the shell
                found = shutil.which(tool)
                if found:
                    os.symlink(found, os.path.join(bin_dir, tool))
            copy = os.path.join(tmp, "replay-bench")  # no .venv beside it either
            shutil.copy(LAUNCHER, copy)
            r = self.run_sh(copy, "--help", env={"PATH": bin_dir})
            self.assertEqual(r.returncode, 1)
            self.assertIn("make bootstrap", r.stderr)
            self.assertIn("Python 3.9", r.stderr)


if __name__ == "__main__":
    unittest.main()
