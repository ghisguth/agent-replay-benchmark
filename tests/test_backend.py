"""Backend classification against in-process mock servers (127.0.0.1, ephemeral port) that answer with the shapes
of the real servers (Splash, Kapitsa, llama.cpp, ollama, LM Studio, vLLM). No real server is contacted.

    python3 -m unittest tests.test_backend
"""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isolation  # noqa: E402,F401

from bench import backend  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_data(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


BUILD_ID = "src-" + "1a2b3c4d5e6f7a8b" * 4


def models(owned_by, **extra):
    return {"object": "list", "data": [{"id": "m", "object": "model", "created": 0, "owned_by": owned_by, **extra}]}


# path -> JSON body, per server kind (what each real server answers, trimmed)
SPLASH = {
    "/v1/models": models("splash", max_model_len=262144, context_length=262144, reasoning=True),
    "/health": {"status": "ok"},
    "/status": {"schema_version": 8, "ready": True,
                "identity": {"cache": {"build_id": BUILD_ID, "block_tokens": 32, "dtype": "int8"},
                             "kv": {"format": "int8", "quantization": "symmetric_int8"}},
                "instance": {"id": "abc", "pid": 1, "model": "qwen-27b", "host": "127.0.0.1", "port": 8133,
                             "started_at": "2026-11-07T00:00:00Z"}},
}
KAPITSA = {
    "/v1/models": models("nace", context_length=262144, context_fit=200000),
    "/health": {"status": "ok"},
    "/cluster": {"proto": 1, "id": "mac-1", "name": "studio", "chip": "Apple M2 Max", "ram_gb": 96, "budget_gb": 40,
                 "model": "qwen3.8-27b", "version": "0.2.15", "port": 8133},
}
LLAMACPP = {
    "/v1/models": models("llamacpp", meta={"n_ctx": 65536}),
    "/health": {"status": "ok"},
    "/props": {"default_generation_settings": {"n_ctx": 65536}, "total_slots": 4, "model_path": "/models/qwen-Q4.gguf",
               "build_info": "b6123-1a2b3c4"},
}
OLLAMA = {"/v1/models": models("library"), "/api/version": {"version": "0.6.2"}}
LMSTUDIO = {
    "/v1/models": models("organization_owner"),
    "/api/v0/models": {"object": "list", "data": [
        {"id": "qwen", "object": "model", "type": "llm", "arch": "qwen3", "quantization": "Q4_K_M", "state": "loaded",
         "max_context_length": 40960},
        {"id": "other", "object": "model", "type": "llm", "arch": "llama", "state": "not-loaded",
         "max_context_length": 8192}]},
}
VLLM = {"/v1/models": models("vllm", max_model_len=32768), "/health": {}, "/version": {"version": "0.6.3"}}
MTPLX = {
    "/v1/models": models("mtplx"),
    "/health": {
        "ok": True,
        "model": "qwen3.8-27b",
        "chip": "Apple M2 Max",
        "generation_mode": "mtp",
        "runtime_mode": "turbo MTP",
        "profile": {"name": "turbo"},
        "depth": 3,
        "mlx_runtime": {
            "ok": True,
            "path": "/opt/homebrew/var/mtplx/venv-2.12.3/lib/python3.13/site-packages/mlx/core.cpython-313-darwin.so",
            "version": "0.32.2",
        },
    },
}
OMLX = {
    "/v1/models": models("omlx"),
    "/health": {
        "status": "healthy",
        "default_model": "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0",
        "engine_pool": {"model_count": 15, "loaded_count": 1, "final_ceiling": 42111181398, "current_model_memory": 16857268416},
        "mcp": None,
    },
    "/admin/api/stats": {
        "cli_prefix": "omlx",
        "uptime_seconds": 1474.7,
        "engines": {
            "mlx": {"name": "mlx", "version": "0.32.4", "commit": None, "url": None},
            "mlx-lm": {
                "name": "mlx-lm",
                "version": "0.32.3.dev8+g0d4d3d04f",
                "commit": "0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d",
                "url": "https://github.com/ml-explore/mlx-lm/commit/0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d",
            },
        },
    },
}
GENERIC = {"/v1/models": models("someone")}


class Mock:
    def __init__(self, routes, headers=None):
        self.hits = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.hits.append(self.path)
                if self.path not in routes:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(routes[self.path]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    return

            def do_POST(self):
                outer.hits.append(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                if length > 0:
                    self.rfile.read(length)
                if self.path not in routes:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(routes[self.path]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.url = "http://127.0.0.1:%d/v1" % self.httpd.server_address[1]

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def detect(routes, **kw):
    with Mock(routes, **kw) as srv:
        return backend.detect(srv.url, timeout=2), srv


class Classification(unittest.TestCase):
    def test_splash(self):
        info, _ = detect(SPLASH, headers={"Server": "Splash"})
        self.assertEqual((info["kind"], info["version"], info["build"]), ("splash", "1a2b3c4d", BUILD_ID))
        self.assertEqual(info["details"]["schema_version"], 8)
        self.assertEqual(info["details"]["model"], "qwen-27b")
        self.assertEqual(info["details"]["kv_format"], "int8")
        self.assertEqual(info["details"]["server"], "Splash")

    def test_splash_models_only(self):
        info, _ = detect({"/v1/models": SPLASH["/v1/models"]})  # /status unreachable: kind from owned_by
        self.assertEqual((info["kind"], info["version"]), ("splash", None))

    def test_kapitsa(self):
        info, _ = detect(KAPITSA)
        self.assertEqual((info["kind"], info["version"]), ("kapitsa", "0.2.15"))
        self.assertEqual(info["details"]["chip"], "Apple M2 Max")

    def test_kapitsa_models_only(self):
        info, _ = detect({"/v1/models": KAPITSA["/v1/models"]})
        self.assertEqual((info["kind"], info["version"]), ("kapitsa", None))

    def test_llamacpp(self):
        info, _ = detect(LLAMACPP)
        self.assertEqual((info["kind"], info["version"], info["build"]), ("llamacpp", "b6123", "b6123-1a2b3c4"))
        self.assertEqual(info["details"]["total_slots"], 4)
        self.assertEqual(info["details"]["model_file"], "qwen-Q4.gguf")

    def test_ollama(self):
        info, _ = detect(OLLAMA)
        self.assertEqual((info["kind"], info["version"]), ("ollama", "0.6.2"))

    def test_parse_ollama_ps_fixture(self):
        fixture = load_data("ollama_ps.json")
        info = backend.parse_ollama_ps(fixture, "qwen3.8:27b-256k")
        self.assertEqual(info, {
            "quant": "Q4_K_M",
            "format": "gguf",
            "family": "qwen35",
            "parameter_size": "27.3B",
            "context_length": 262144,
            "runner": "llamacpp",
            "digest": "4010bd202712ea6bee89a163733011c461085c2bd1154caa2a49bcabb7c5eaa6",
        })
        # Tag match with or without :latest
        self.assertEqual(backend.parse_ollama_ps(fixture, "qwen3.8:27b-256k:latest"), info)
        # No name given: the first loaded model
        self.assertEqual(backend.parse_ollama_ps(fixture, None), info)
        # A name that is not loaded matches nothing, not the single loaded model
        self.assertIsNone(backend.parse_ollama_ps(fixture, "other:q8"))
        self.assertIsNone(backend.parse_lmstudio_models(load_data("lmstudio_v0_models.json"), "other-model"))

    def test_ollama_model_info_probe(self):
        fixture = {
            "models": [{
                "name": "qwen3.8:27b-256k", "model": "qwen3.8:27b-256k",
                "details": {"quantization_level": "Q4_K_M", "format": "gguf"},
                "context_length": 262144,
            }],
        }
        with Mock({"/api/ps": fixture}) as srv:
            info = backend.ollama_model_info(srv.url, "qwen3.8:27b-256k")
            self.assertEqual(info, {"quant": "Q4_K_M", "format": "gguf", "context_length": 262144})

    def test_ollama_model_info_absent_does_not_raise(self):
        with Mock({"/v1/models": models("library")}) as srv:
            # Server returns 404 for /api/ps and /api/show; must return None and not raise
            self.assertIsNone(backend.ollama_model_info(srv.url, "qwen3.8:27b-256k"))

    def test_ollama_model_info_unlisted_model_uses_show(self):
        ps = {"models": [{"name": "A:q4", "details": {"quantization_level": "Q4_K_M"}, "context_length": 4096}]}
        show = {"details": {"quantization_level": "Q8_0"}, "parameters": "num_ctx 8192"}
        with Mock({"/api/ps": ps, "/api/show": show}) as srv:
            self.assertEqual(backend.ollama_model_info(srv.url, "A:q4"), {"quant": "Q4_K_M", "context_length": 4096})
            self.assertEqual(backend.ollama_model_info(srv.url, "B:q8"), {"quant": "Q8_0", "context_length": 8192})

    def test_parse_ollama_show(self):
        fixture = {
            "parameters": "stop \"<|im_start|>\"\nstop \"<|im_end|>\"\nnum_ctx 262144",
            "details": {
                "parent_model": "",
                "format": "safetensors",
                "family": "qwen3_5",
                "families": ["qwen3_5"],
                "parameter_size": "27.8B",
                "quantization_level": "nvfp4",
                "runner": "mlx",
            },
            "model_info": {
                "general.architecture": "qwen3_5",
                "general.file_type": 39,
                "qwen3_5.context_length": 262144,
            },
        }
        info = backend.parse_ollama_show(fixture)
        self.assertEqual(info, {
            "quant": "nvfp4",
            "format": "safetensors",
            "family": "qwen3_5",
            "parameter_size": "27.8B",
            "runner": "mlx",
            "context_length": 262144,
        })

    def test_ollama_model_info_fallback_to_show(self):
        show_fixture = {
            "details": {"quantization_level": "nvfp4", "format": "safetensors"},
            "parameters": "num_ctx 131072",
        }
        with Mock({"/api/ps": {"models": []}, "/api/show": show_fixture}) as srv:
            info = backend.ollama_model_info(srv.url, "qwen3.8:27b-nvfp4")
            self.assertEqual(info, {"quant": "nvfp4", "format": "safetensors", "context_length": 131072})

    def test_parse_lmstudio_models(self):
        fixture = load_data("lmstudio_v0_models.json")
        info = backend.parse_lmstudio_models(fixture, "qwen3.8-27b-mlx")
        self.assertEqual(info, {
            "quant": "4bit",
            "format": "mlx",
            "family": "qwen3_5",
            "context_length": 262144,
        })
        # Matches when name is None and single loaded model
        self.assertEqual(backend.parse_lmstudio_models(fixture, None), info)

    def test_lmstudio_model_info_probe(self):
        fixture = {
            "data": [{
                "id": "qwen3.8-27b-mlx",
                "arch": "qwen3_5",
                "compatibility_type": "mlx",
                "quantization": "4bit",
                "state": "loaded",
                "max_context_length": 262144,
            }],
        }
        with Mock({"/api/v0/models": fixture}) as srv:
            info = backend.lmstudio_model_info(srv.url, "qwen3.8-27b-mlx")
            self.assertEqual(info, {
                "quant": "4bit",
                "format": "mlx",
                "family": "qwen3_5",
                "context_length": 262144,
            })

    def test_lmstudio(self):
        info, _ = detect(LMSTUDIO)
        self.assertEqual((info["kind"], info["version"]), ("lmstudio", None))
        self.assertEqual((info["details"]["models"], info["details"]["loaded"]), (2, 1))

    def test_vllm(self):
        info, _ = detect(VLLM)
        self.assertEqual((info["kind"], info["version"]), ("vllm", "0.6.3"))

    def test_mtplx(self):
        info, _ = detect(MTPLX)
        self.assertEqual((info["kind"], info["version"]), ("mtplx", "2.12.3"))
        self.assertEqual(info["details"]["chip"], "Apple M2 Max")
        self.assertEqual(info["details"]["generation_mode"], "mtp")
        self.assertEqual(info["details"]["profile"], "turbo")
        self.assertEqual(info["details"]["mlx_version"], "0.32.2")
        self.assertEqual(info["details"]["depth"], 3)
        self.assertEqual(info["details"]["owned_by"], "mtplx")

    def test_mtplx_models_only(self):
        info, _ = detect({"/v1/models": MTPLX["/v1/models"]})
        self.assertEqual((info["kind"], info["version"]), ("mtplx", None))
        self.assertEqual(info["details"]["owned_by"], "mtplx")

    def test_omlx(self):
        info, _ = detect(OMLX)
        self.assertEqual((info["kind"], info["version"], info["build"]),
                         ("omlx", "0.32.3.dev8+g0d4d3d04f", "0d4d3d04"))
        self.assertEqual(info["details"]["mlx_lm_version"], "0.32.3.dev8+g0d4d3d04f")
        self.assertEqual(info["details"]["mlx_version"], "0.32.4")
        self.assertEqual(info["details"]["cli_prefix"], "omlx")
        self.assertEqual(info["details"]["owned_by"], "omlx")

    def test_omlx_models_only(self):
        info, _ = detect({"/v1/models": OMLX["/v1/models"]})
        self.assertEqual((info["kind"], info["version"]), ("omlx", None))
        self.assertEqual(info["details"]["owned_by"], "omlx")

    def test_generic(self):
        info, srv = detect(GENERIC)
        self.assertEqual((info["kind"], info["version"], info["build"]), ("generic", None, None))
        self.assertIn("/status", srv.hits)  # every probe was tried

    def test_unreachable_server_is_generic_and_fast(self):
        with Mock(GENERIC) as srv:
            url = srv.url
        info = backend.detect(url, timeout=1)  # the server is closed now
        self.assertEqual(info["kind"], "generic")
        self.assertFalse(info["details"]["reachable"])

    def test_junk_answers_never_raise(self):
        info, _ = detect({"/v1/models": [1, 2], "/status": "text", "/props": [], "/cluster": {"proto": 1},
                          "/api/version": {"version": 3}, "/api/v0/models": {"data": "x"}, "/version": None})
        self.assertEqual(info["kind"], "generic")

    def test_label(self):
        self.assertEqual(backend.label({"kind": "splash", "version": "1a2b3c4d"}), "Splash 1a2b3c4d")
        self.assertEqual(backend.label({"kind": "llamacpp", "version": "b6123"}), "llama.cpp b6123")
        self.assertEqual(backend.label({"kind": "lmstudio", "version": None}), "LM Studio")
        self.assertEqual(backend.label({"kind": "mtplx", "version": "2.12.3"}), "MTPLX 2.12.3")
        self.assertEqual(backend.label({"kind": "omlx", "version": "0.32.3"}), "OMLX 0.32.3")
        self.assertEqual(backend.label(None), "generic")


HASH = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
HF = "/Users/USER/.cache/huggingface/hub/models--%s--%s/snapshots/%s"
DSPARK_TARGET = HF % ("mlx-community", "Qwen3.8-27B-4bit", HASH)
DSPARK_DRAFTER = HF % ("incoai", "Qwen3.8-27B-DFlash2", "015e795645c74b1a0eeef3b570031fb62e769bc5")
DSPARK = {
    "/v1/models": {"object": "list", "data": [{
        "id": HASH, "object": "model", "created": 1794277626, "owned_by": "mlx-dspark",
        "display_name": HASH + " (mlx-dspark dflash)",
        "x_mlx_dspark": {"mode": "dflash", "target": DSPARK_TARGET, "drafter": DSPARK_DRAFTER}}]},
    "/doctor": {"ok": True, "problems": [], "environment": {
        "version": "0.20.4", "platform": "Darwin", "device": "Apple M2 Max",
        "chip": {"name": "Apple M2 Max", "generation": "M2", "gpu_cores": 38}},
        "models": [{"id": "qwen3.8-27b", "target": "mlx-community/Qwen3.8-27B-4bit", "mode": "dflash", "ready": True}]},
}
MLX_LM = {
    "/v1/models": {"object": "list", "data": [{"id": "mlx-community/Qwen3.8-27B-4bit", "object": "model", "created": 1}]},
    "/health": {"status": "ok"},
}


class MlxServers(unittest.TestCase):
    def test_mlx_dspark_by_server_header_with_doctor_details(self):
        info, srv = detect({k: v for k, v in DSPARK.items()}, headers={"Server": "mlx-dspark Python/3.11.18"})
        self.assertEqual((info["kind"], info["version"]), ("mlx-dspark", "0.20.4"))
        self.assertEqual((info["details"]["chip"], info["details"]["mode"]), ("Apple M2 Max", "dflash"))
        self.assertEqual(backend.label(info), "mlx-dspark 0.20.4")
        self.assertIn("/doctor", srv.hits)

    def test_mlx_dspark_by_owned_by_or_model_object_alone(self):
        for models_body in (models("mlx-dspark"), models("x", x_mlx_dspark={"mode": "dspark"})):
            info, _ = detect({"/v1/models": models_body})
            self.assertEqual(info["kind"], "mlx-dspark")
            self.assertIsNone(info["version"])  # no /doctor

    def test_mlx_dspark_label_and_drafter_never_carry_the_path(self):
        entry = DSPARK["/v1/models"]["data"][0]
        self.assertEqual(backend.model_extras("mlx-dspark", entry), {
            "label": "mlx-community/Qwen3.8-27B-4bit", "mode": "dflash", "drafter": "incoai/Qwen3.8-27B-DFlash2"})
        # not a hash id: the id stays the label; a hash without an HF cache target stays as it is
        named = dict(entry, id="my-model")
        self.assertNotIn("label", backend.model_extras("mlx-dspark", named))
        local = {"id": HASH, "x_mlx_dspark": {"mode": "dspark", "target": "/Users/USER/models/qwen"}}
        self.assertEqual(backend.model_extras("mlx-dspark", local), {"mode": "dspark"})  # the hash id stays as it is
        self.assertNotIn("/Users", json.dumps(backend.model_extras("mlx-dspark", local)))

    def test_hf_label(self):
        self.assertEqual(backend.hf_label(DSPARK_TARGET), "mlx-community/Qwen3.8-27B-4bit")
        self.assertEqual(backend.hf_label("/Users/USER/models/Qwen3.8-4bit"), "Qwen3.8-4bit")
        self.assertEqual(backend.hf_label("/Users/USER/models/Qwen3.8-4bit/"), "Qwen3.8-4bit")
        self.assertEqual(backend.hf_label("mlx-community/Qwen3.8-4bit"), "mlx-community/Qwen3.8-4bit")
        self.assertEqual(backend.hf_label("C:\\models\\q"), "q")
        self.assertIsNone(backend.hf_label(""))

    def test_mlx_lm_needs_the_whole_weak_signature(self):
        info, _ = detect(MLX_LM)  # the Mock's default Server header is BaseHTTP/0.6 Python/x
        self.assertEqual(info["kind"], "mlx_lm")
        self.assertEqual(backend.label(info), "mlx_lm.server")
        info, _ = detect(MLX_LM, headers={"Server": "uvicorn"})
        self.assertEqual(info["kind"], "generic")
        info, _ = detect({"/v1/models": MLX_LM["/v1/models"], "/health": {"status": "ok", "x": 1}})
        self.assertEqual(info["kind"], "generic")
        info, _ = detect({"/v1/models": MLX_LM["/v1/models"]})  # no /health
        self.assertEqual(info["kind"], "generic")
        owned = {"/v1/models": models("someone"), "/health": {"status": "ok"}}
        info, _ = detect(owned)
        self.assertEqual(info["kind"], "generic")

    def test_mlx_lm_never_overrides_another_detection(self):
        for routes in (dict(SPLASH), dict(LLAMACPP), dict(MTPLX)):
            info, _ = detect(routes)
            self.assertNotEqual(info["kind"], "mlx_lm")
        no_owner = dict(LLAMACPP, **{"/v1/models": {"object": "list", "data": [{"id": "m", "object": "model"}]}})
        info, _ = detect(no_owner)
        self.assertEqual(info["kind"], "llamacpp")

    def test_mlx_lm_model_ids(self):
        self.assertEqual(backend.model_extras("mlx_lm", {"id": "/Users/USER/m/Qwen-4bit"}), {"label": "Qwen-4bit"})
        self.assertEqual(backend.model_extras("mlx_lm", {"id": DSPARK_TARGET}), {"label": "mlx-community/Qwen3.8-27B-4bit"})
        self.assertEqual(backend.model_extras("mlx_lm", {"id": "mlx-community/Qwen-4bit"}), {})


if __name__ == "__main__":
    unittest.main()
