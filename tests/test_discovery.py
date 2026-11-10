"""Discovery probes: which extra GETs each kind of server gets. In-process mock servers on 127.0.0.1 count them.

    python3 -m unittest tests.test_discovery
"""

import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401
from test_backend import LLAMACPP, OLLAMA, VLLM, Mock, load_data, models  # noqa: E402

from bench import discovery  # noqa: E402


def discover(routes, filters=None):
    with Mock(routes) as srv, contextlib.redirect_stdout(io.StringIO()):
        targets = discovery.discover_and_report([{"backend": "m", "base_url": srv.url}], filters)
        return targets, list(srv.hits)


def two_models(owner, **extra):
    return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": owner, **extra} for i in ("A", "B")]}


class ProbeCounts(unittest.TestCase):
    def test_vllm_gets_no_extra_requests(self):
        _, hits = discover({**VLLM, "/v1/models": {"data": [{"id": "m", "owned_by": "vllm"}]}})
        self.assertNotIn("/api/ps", hits)
        self.assertEqual(hits.count("/v1/models"), 2)  # discover_models, then detect
        self.assertEqual(hits.count("/api/v0/models"), 1)  # detect()'s own LM Studio probe, not the n_ctx probe

    def test_generic_server_without_n_ctx_gets_no_n_ctx_probes(self):
        targets, hits = discover({"/v1/models": models("mock")})
        self.assertEqual(targets[0]["n_ctx"], None)
        self.assertNotIn("/api/ps", hits)
        self.assertEqual(hits.count("/status"), 1)  # detect()'s Splash probe only

    def test_discover_models_is_one_request(self):
        with Mock({"/v1/models": models("mock")}) as srv:
            self.assertEqual(discovery.discover_models(srv.url), [{"id": "m", "n_ctx": None}])
            self.assertEqual(srv.hits, ["/v1/models"])

    def test_ollama_ps_once_and_only_selected_models_probed(self):
        ps = load_data("ollama_ps.json")
        ps["models"].append({**ps["models"][0], "name": "B", "model": "B"})
        ps["models"][0]["name"] = ps["models"][0]["model"] = "A"
        routes = {**OLLAMA, "/v1/models": two_models("library"), "/api/ps": ps}
        targets, hits = discover(routes, filters=["A"])
        self.assertEqual(hits.count("/api/ps"), 1)
        self.assertNotIn("/api/show", hits)
        self.assertEqual([t["name"] for t in targets], ["A"])
        self.assertEqual(targets[0]["model_info"]["quant"], "Q4_K_M")
        self.assertEqual(targets[0]["n_ctx"], 262144)

    def test_ollama_unlisted_model_uses_show_not_the_loaded_one(self):
        ps = {"models": [{"name": "A", "details": {"quantization_level": "Q4_K_M"}, "context_length": 4096}]}
        show = {"details": {"quantization_level": "Q8_0"}, "parameters": "num_ctx 8192"}
        routes = {**OLLAMA, "/v1/models": two_models("library"), "/api/ps": ps, "/api/show": show}
        targets, hits = discover(routes, filters=["B"])
        self.assertEqual(targets[0]["model_info"], {"quant": "Q8_0", "context_length": 8192})
        self.assertEqual(hits.count("/api/ps"), 1)

    def test_lmstudio_models_fetched_for_probes_once(self):
        v0 = load_data("lmstudio_v0_models.json")
        routes = {"/v1/models": models("organization_owner"), "/api/v0/models": v0}
        routes["/v1/models"]["data"][0]["id"] = "qwen3.8-27b-mlx"
        targets, hits = discover(routes)
        self.assertEqual(targets[0]["model_info"]["quant"], "4bit")
        self.assertTrue(targets[0]["url"].endswith("/api/v0/chat/completions"))
        self.assertTrue(targets[0]["fallback_url"].endswith("/v1/chat/completions"))
        self.assertEqual(hits.count("/api/v0/models"), 2)  # detect(), and one fetch for all model probes


def run_discover(routes, *args):
    """(stdout, exit code) of `replay_bench.py --discover` against a stub server; also the paths it requested."""
    import replay_bench
    with Mock(routes) as srv:
        out, code = io.StringIO(), 0
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            try:
                replay_bench.main(["--discover", "--endpoint", srv.url, *args])
            except SystemExit as e:
                code = e.code or 0
        return out.getvalue(), code, list(srv.hits)


class DiscoverCommand(unittest.TestCase):
    def test_ollama(self):
        ps = load_data("ollama_ps.json")
        ps["models"][0]["name"] = ps["models"][0]["model"] = "A"
        out, code, hits = run_discover({**OLLAMA, "/v1/models": two_models("library"), "/api/ps": ps}, "--first-model")
        self.assertEqual(code, 0)
        self.assertIn("ollama 0.6.2", out)
        self.assertIn("Q4_K_M", out)
        self.assertIn("prefill progress  no", out)
        self.assertIn("context 256k", out)
        self.assertFalse([h for h in hits if "chat" in h])

    def test_lmstudio(self):
        v0 = load_data("lmstudio_v0_models.json")
        routes = {"/v1/models": models("organization_owner"), "/api/v0/models": v0}
        routes["/v1/models"]["data"][0]["id"] = "qwen3.8-27b-mlx"
        out, code, hits = run_discover(routes)
        self.assertEqual(code, 0)
        self.assertIn("LM Studio", out)
        self.assertIn("quant 4bit", out)
        self.assertIn("format mlx", out)
        self.assertFalse([h for h in hits if "chat" in h])

    def test_generic_and_llamacpp_progress(self):
        out, code, _ = run_discover({"/v1/models": models("mock")})
        self.assertEqual(code, 0)
        self.assertIn("generic", out)
        self.assertIn("prefill progress  no", out)
        out, code, _ = run_discover({**LLAMACPP})
        self.assertIn("llama.cpp b6123", out)
        self.assertIn("prefill progress  yes (request field)", out)
        self.assertIn("context 64k", out)

    def test_model_filter_exact_and_substring(self):
        from bench.discovery import select_models
        ms = [{"id": "Foo"}, {"id": "Foo:xxx"}, {"id": "Bar"}]
        ids = lambda f: [m["id"] for m in select_models(ms, f)]
        self.assertEqual(ids(["foo"]), ["Foo", "Foo:xxx"])
        self.assertEqual(ids(["exact:Foo"]), ["Foo"])
        self.assertEqual(ids(["EXACT:foo"]), ["Foo"])
        self.assertEqual(ids(["exact:Foo", "bar"]), ["Foo", "Bar"])
        self.assertEqual(ids(["exact:Fo"]), [])

    def test_list_with_exact_filter(self):
        routes = {"/v1/models": {"object": "list", "data": [{"id": "Foo"}, {"id": "Foo:xxx"}]}}
        out, code, _ = run_discover(routes, "--model", "exact:Foo")
        self.assertEqual(code, 0)
        self.assertIn("1/2 selected", out)

    def test_exit_codes(self):
        _, code, _ = run_discover({"/v1/models": models("mock")}, "--model", "nothing-like-this")
        self.assertEqual(code, 1)
        import replay_bench
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                replay_bench.main(["--list", "--endpoint", "http://127.0.0.1:9/v1"])  # nothing listens
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
