import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import add_ref  # noqa: E402
import isolation  # noqa: E402,F401


def state_text(host="boxy.local", extra=""):
    header = {"type": "run", "endpoint": "http://%s:8133/v1" % host, "backend": "%s:8133" % host,
              "argv": ["--endpoint", "http://%s:8133/v1" % host]}
    return json.dumps(header) + "\n" + json.dumps({"type": "result", "content": "ssh %s. and %s ok %s" % (host, host, extra)}) + "\n"


class AddRefTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.refs = os.path.join(self.tmp.name, "refs")
        self.hosts = os.path.join(self.tmp.name, "hosts.json")
        self.src = os.path.join(self.tmp.name, "x.jsonl")

    def write(self, text):
        with open(self.src, "w") as f:
            f.write(text)

    def run_main(self, *args):
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = add_ref.main(["--hosts", self.hosts, "--refs-dir", self.refs, *args, self.src])
        return rc, err.getvalue()

    def test_replaces_with_and_without_trailing_dot(self):
        out, found = add_ref.scrub(state_text(), {"boxy.local": "host-7"})
        self.assertEqual(found, {"boxy.local": "host-7"})
        self.assertNotIn("boxy", out)
        self.assertIn("ssh host-7 and host-7 ok", out)
        self.assertIn("http://host-7:8133/v1", out)

    def test_hash_fallback_without_table(self):
        out, found = add_ref.scrub(state_text(), None)
        self.assertRegex(found["boxy.local"], r"^host-[0-9a-f]{8}$")
        self.assertNotIn("boxy", out)

    def test_next_free_number(self):
        aliases = {"a.local": "host-mbp1", "b.local": "host-1"}
        self.assertEqual(add_ref.assign("c.local", aliases), "host-2")
        self.assertEqual(aliases["c.local"], "host-2")

    def test_existing_placeholder_left_alone(self):
        text = state_text(host="host-mbp1")
        out, found = add_ref.scrub(text, {})
        self.assertEqual(out, text)
        self.assertEqual(found, {})

    def test_trailing_dot_host_is_the_same_host(self):
        aliases = {"boxy.local": "host-mbp1"}
        text = state_text(host="boxy.local.") + state_text(host="boxy.local")
        out, found = add_ref.scrub(text, aliases)
        self.assertEqual(found, {"boxy.local": "host-mbp1"})
        self.assertEqual(aliases, {"boxy.local": "host-mbp1"})
        self.assertIn("http://host-mbp1:8133/v1", out)
        self.assertNotIn("host-1", out)
        self.assertNotIn("host-mbp1.:", out)
        self.assertNotIn("boxy", out)

    def test_localhost_left_alone(self):
        text = state_text(host="localhost")
        self.assertEqual(add_ref.scrub(text, {})[0], text)

    def test_blocked_hit_exits_nonzero(self):
        self.write(state_text(extra="mail me@example.com"))
        with open(self.hosts, "w") as f:
            json.dump({}, f)
        rc, err = self.run_main("--dry-run")
        self.assertEqual(rc, 1)
        self.assertIn("pii:email", err)
        self.assertFalse(os.path.exists(self.refs))

    def test_allowed_hits_pass_dry_run(self):
        self.write(state_text(extra="/Users/standard.analyst " + "a" * 64 + " MacBook"))
        rc, _ = self.run_main("--dry-run")
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(self.refs))

    def test_sample_copies_without_rebuilding_checks(self):
        self.write(state_text())
        checks = os.path.join(self.tmp.name, "fx.checks.json")
        with mock.patch("add_ref.build_checks.main", side_effect=AssertionError("checks rebuilt")):
            rc, err = self.run_main("--sample", "--checks", checks)
        self.assertEqual(rc, 0, err)
        self.assertTrue(os.path.isfile(os.path.join(self.refs, "x.jsonl")))
        self.assertFalse(os.path.exists(checks))

    def test_refuses_overwrite(self):
        self.write(state_text())
        os.makedirs(self.refs)
        with open(os.path.join(self.refs, "x.jsonl"), "w") as f:
            f.write("old")
        rc, err = self.run_main()
        self.assertEqual(rc, 1)
        self.assertIn("already exists", err)
        with open(os.path.join(self.refs, "x.jsonl")) as f:
            self.assertEqual(f.read(), "old")


if __name__ == "__main__":
    unittest.main()
