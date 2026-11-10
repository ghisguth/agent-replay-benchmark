import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

SCAN = os.path.join(os.path.dirname(__file__), "..", "tools", "omp_session_pii_scan.py")


def run(*args):
    return subprocess.run([sys.executable, SCAN, *args], capture_output=True, text=True, check=True).stdout


class PiiScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.session = os.path.join(self.tmp.name, "s.jsonl")
        with open(self.session, "w") as f:
            f.write(json.dumps({"text": "key sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2 in /Users/someone/Code/x, ask Zorgon"}) + "\n")

    def test_reports_secret_and_home_path(self):
        out = run(self.session)
        self.assertIn("secret:openai/anthropic key", out)
        self.assertIn("id:home path", out)
        self.assertNotIn("id:names", out)

    def test_names_file(self):
        names = os.path.join(self.tmp.name, "names.json")
        with open(names, "w") as f:
            json.dump(["zorgon"], f)
        self.assertIn("id:names", run(self.session, "--names", names))


if __name__ == "__main__":
    unittest.main()
