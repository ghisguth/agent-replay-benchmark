import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from tools import replay_divergence  # noqa: E402


def make_result(cid, stream="main", kind="turn", epoch=0, tool_name=None, args=None, eval_result="PASS",
                in_line=True, prompt_tokens=1000, recorded_tools=None):
    tool_calls = []
    if tool_name:
        tool_calls.append({
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args) if isinstance(args, dict) else (args or "{}"),
            }
        })
    return {
        "id": cid,
        "stream": stream,
        "kind": kind,
        "epoch": epoch,
        "prompt_tokens": prompt_tokens,
        "eval_result": eval_result,
        "tool_names": [tool_name] if tool_name else [],
        "tool_calls": tool_calls,
        "recorded_tools": recorded_tools or ([tool_name] if tool_name else []),
        "in_line": in_line,
    }


def write_state(path, model="model-a", device="dev1", server_version="v1.0", suite="coding",
                fixture_file="conv-3.json.gz", fixture_sha="11112222333344445555666677778888",
                results=None):
    with open(path, "w", encoding="utf-8") as f:
        header = {
            "type": "run",
            "schema": 1,
            "suite": suite,
            "fixture": {"file": fixture_file, "sha256": fixture_sha} if fixture_file else None,
            "model": model,
            "device": {"name": device} if device else None,
            "server": {"version": server_version} if server_version else None,
            "backend": "test-backend",
        }
        f.write(json.dumps(header) + "\n")
        for r in (results or []):
            rec = {"type": "result", **r}
            f.write(json.dumps(rec) + "\n")
        f.write(json.dumps({"type": "end"}) + "\n")


class DivergenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        # 10 cases:
        # R001: read path "file1.py:10-20"
        # R002: edit path "file2.py:5-10"
        # R003: write path "file3.py"
        # R004: bash (stream main)
        # R005: read path "file4.py"
        # R006: edit path "file5.py"
        # R007: write path "file6.py"
        # R008: bash (stream subagent)
        # R009: read path "file7.py"
        # R010: bash
        # In line: R1 and R2 have 8 True, 2 False (80%); R3 has 6 True, 4 False (60%).

        def r1_results():
            return [
                make_result("R001", "main", tool_name="read", args={"path": "file1.py:10-20"}, in_line=True),
                make_result("R002", "main", tool_name="edit", args={"path": "file2.py:5-10"}, in_line=True),
                make_result("R003", "main", tool_name="write", args={"path": "file3.py"}, in_line=True),
                make_result("R004", "main", tool_name="bash", args={"command": "echo 1"}, in_line=True),
                make_result("R005", "S1", tool_name="read", args={"path": "file4.py"}, in_line=True),
                make_result("R006", "S1", tool_name="edit", args={"path": "file5.py"}, in_line=True),
                make_result("R007", "S1", tool_name="write", args={"path": "file6.py"}, in_line=True),
                make_result("R008", "S1", tool_name="bash", args={"command": "echo 2"}, in_line=True),
                make_result("R009", "main", tool_name="read", args={"path": "file7.py"}, in_line=False),
                make_result("R010", "main", tool_name="bash", args={"command": "echo 3"}, in_line=False),
            ]

        def r2_results():
            # Identical to R1
            return r1_results()

        def r3_results():
            # Differs on 2 turns:
            # R004: calls "read" instead of "bash"
            # R008: calls "write" instead of "bash"
            # Differs on path on R001: calls "other.py" instead of "file1.py:10-20"
            # Matches path on R002 despite line suffix: calls "file2.py:40-50" vs "file2.py:5-10"
            res = r1_results()
            res[0] = make_result("R001", "main", tool_name="read", args={"path": "other.py"}, in_line=True)
            res[1] = make_result("R002", "main", tool_name="edit", args={"path": "file2.py:40-50"}, in_line=True)
            res[3] = make_result("R004", "main", tool_name="read", args={"path": "foo.py"}, in_line=False)
            res[7] = make_result("R008", "S1", tool_name="write", args={"path": "bar.py"}, in_line=False)
            res[8]["in_line"] = False
            res[9]["in_line"] = False
            return res

        self.p1 = os.path.join(self.tmp.name, "r1.jsonl")
        self.p2 = os.path.join(self.tmp.name, "r2.jsonl")
        self.p3 = os.path.join(self.tmp.name, "r3.jsonl")

        write_state(self.p1, model="model-a", device="dev1", server_version="v1.0", results=r1_results())
        write_state(self.p2, model="model-a", device="dev1", server_version="v1.0", results=r2_results())
        write_state(self.p3, model="model-b", device="dev2", server_version="v2.0", results=r3_results())

    def run_tool(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = replay_divergence.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_matrix_numbers(self):
        code, out, _ = self.run_tool("--format", "markdown", self.p1, self.p2, self.p3)
        self.assertEqual(code, 0)

        # Label disambiguation: r1 and r2 have same model@device, so stems "r1" and "r2" are used.
        # r3 gets "model-b@dev2".
        self.assertIn("## Summary matrix", out)
        self.assertIn("r1", out)
        self.assertIn("r2", out)
        self.assertIn("model-b@dev2", out)

        # Matrix values:
        # r1 vs r2: 100% (noise floor)
        self.assertIn("100% (noise floor)", out)
        # r1 vs r3 and r2 vs r3: 8/10 -> 80%
        self.assertIn("80%", out)
        # Consensus: 8/10 (80%) for r1, r2; 6/10 (60%) for r3
        self.assertIn("8/10 (80%)", out)
        self.assertIn("6/10 (60%)", out)

    def test_split_rows_and_filters(self):
        # Default min-split 2: exactly R004 and R008 should appear in split turns
        code, out, _ = self.run_tool("--format", "markdown", self.p1, self.p2, self.p3)
        self.assertEqual(code, 0)
        self.assertIn("## Split turns", out)
        self.assertIn("R004", out)
        self.assertIn("R008", out)
        self.assertNotIn("R001", out)
        self.assertNotIn("R002", out)

        # Filter --stream main: only R004 should appear
        code, out, _ = self.run_tool("--format", "markdown", "--stream", "main", self.p1, self.p2, self.p3)
        self.assertEqual(code, 0)
        self.assertIn("R004", out)
        self.assertNotIn("R008", out)

        # Filter --stream subagent: only R008 should appear
        code, out, _ = self.run_tool("--format", "markdown", "--stream", "subagent", self.p1, self.p2, self.p3)
        self.assertEqual(code, 0)
        self.assertNotIn("R004", out)
        self.assertIn("R008", out)

        # Filter --min-split 3: R004 and R008 each have only 2 distinct tools (bash vs read, bash vs write)
        code, out, _ = self.run_tool("--format", "markdown", "--min-split", "3", self.p1, self.p2, self.p3)
        self.assertEqual(code, 0)
        self.assertIn("No split turns found (min-split 3).", out)

    def test_args_path_mismatch(self):
        code, out, _ = self.run_tool("--format", "markdown", "--args", self.p1, self.p2, self.p3)
        self.assertEqual(code, 0)
        self.assertIn("## Path arguments", out)
        # 7 path turns: R001, R002, R003, R005, R006, R007, R009.
        # 6 match (R002 matches because line suffixes :5-10 and :40-50 stripped to file2.py).
        # 1 differs: R001 (file1.py vs other.py).
        self.assertIn("Paths: 6/7 matching (86%), 1 differing.", out)
        self.assertIn("R001", out)
        self.assertIn("file1.py", out)
        self.assertIn("other.py", out)
        # R002 should not be in the differing table
        self.assertNotIn("file2.py", out)

    def test_json_shape(self):
        code, out, _ = self.run_tool("--format", "json", "--args", self.p1, self.p2, self.p3)
        self.assertEqual(code, 0)
        data = json.loads(out)

        self.assertEqual(data["suite"], "coding")
        self.assertEqual(len(data["runs"]), 3)
        self.assertEqual([r["label"] for r in data["runs"]], ["r1", "r2", "model-b@dev2"])

        # Check matrix
        matrix = data["matrix"]
        self.assertEqual(len(matrix), 3)
        # r1 vs r2 is noise floor
        self.assertTrue(matrix[0]["matches"]["r2"]["noise_floor"])
        self.assertEqual(matrix[0]["matches"]["r2"]["percent"], 100.0)
        # r1 vs r3 is 80%
        self.assertFalse(matrix[0]["matches"]["model-b@dev2"]["noise_floor"])
        self.assertEqual(matrix[0]["matches"]["model-b@dev2"]["percent"], 80.0)
        # In line
        self.assertEqual(matrix[0]["in_line"], {"matching": 8, "total": 10, "percent": 80.0})
        self.assertEqual(matrix[2]["in_line"], {"matching": 6, "total": 10, "percent": 60.0})

        # Check splits
        splits = data["splits"]
        self.assertEqual(len(splits), 2)
        self.assertEqual([s["id"] for s in splits], ["R004", "R008"])
        self.assertEqual(splits[0]["tools"], {"r1": "bash", "r2": "bash", "model-b@dev2": "read"})
        self.assertEqual(splits[1]["tools"], {"r1": "bash", "r2": "bash", "model-b@dev2": "write"})

        # Check args
        args_data = data["args"]
        self.assertEqual(args_data["total"], 7)
        self.assertEqual(args_data["matching"], 6)
        self.assertEqual(args_data["differing_count"], 1)
        self.assertEqual(len(args_data["differing"]), 1)
        self.assertEqual(args_data["differing"][0]["id"], "R001")
        self.assertEqual(args_data["differing"][0]["tool"], "read")
        self.assertEqual(args_data["differing"][0]["paths"], {
            "r1": "file1.py",
            "r2": "file1.py",
            "model-b@dev2": "other.py",
        })

    def test_run_from_another_fixture_skipped(self):
        p_other = os.path.join(self.tmp.name, "other_fixture.jsonl")
        write_state(p_other, model="model-c", fixture_sha="9999999999999999", results=[])

        code, out, err = self.run_tool(self.p1, self.p2, p_other)
        self.assertEqual(code, 0)
        self.assertIn("skipped from divergence", err)
        self.assertIn("fixture", err)
        self.assertIn("other_fixture.jsonl", err)
        # The skipped run does not appear in matrix
        self.assertNotIn("model-c", out)

    def test_run_from_another_suite_skipped(self):
        p_other = os.path.join(self.tmp.name, "other_suite.jsonl")
        write_state(p_other, model="model-c", suite="smoke", results=[])

        code, out, err = self.run_tool(self.p1, self.p2, p_other)
        self.assertEqual(code, 0)
        self.assertIn("skipped from divergence", err)
        self.assertIn("suite", err)
        self.assertNotIn("model-c", out)

    def test_label_overrides(self):
        code, out, _ = self.run_tool(f"REF={self.p1}", f"CAND={self.p3}")
        self.assertEqual(code, 0)
        self.assertIn("REF", out)
        self.assertIn("CAND", out)


if __name__ == "__main__":
    unittest.main()
