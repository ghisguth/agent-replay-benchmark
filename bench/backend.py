"""Backend detection: which inference server answers at an endpoint, and which version of it.

A few cheap GETs with short timeouts, outside /v1 where the servers keep their own endpoints. The shapes are those
of the real servers (read from their sources):

  Splash     GET /status: JSON with schema_version, identity (identity.cache.build_id "src-<sha256>") and instance;
             /v1/models owned_by "splash". The only version Splash exposes is that build id (first 8 hex digits).
  Kapitsa    GET /cluster: JSON with proto, version ("0.2.15"), chip, model; /v1/models owned_by "nace".
  llama.cpp  GET /props: JSON with build_info ("b6123-1a2b3c4"), total_slots, default_generation_settings.
  ollama     GET /api/version: {"version": "0.6.2"}.
  LM Studio  GET /api/v0/models: {"data": [{"arch", "state", "type", ...}]}; no version is exposed.
  vLLM       GET /version: {"version": "0.6.3"}.
  MTPLX      GET /health: JSON with ok, mlx_runtime, model, chip, profile; /v1/models owned_by "mtplx".
  OMLX       /v1/models owned_by "omlx"; GET /admin/api/stats: JSON with cli_prefix, engines (mlx, mlx-lm).
  mlx-dspark Response header `Server: mlx-dspark Python/3.11.18`; /v1/models owned_by "mlx-dspark" with an
             x_mlx_dspark {mode, target, drafter} object; GET /doctor: environment {version, device, chip{name}}.
  mlx_lm     Weak signature, tried last: `Server: BaseHTTP/0.6 Python/3.11.18`, GET /health exactly {"status":"ok"},
             /v1/models entries without owned_by. Never overrides another detection.

Anything else is "generic". detect() never raises: a probe that fails is a probe that found nothing.
"""

import json
import re
import urllib.error
import urllib.request

from .auth import auth_header
from .tls import ssl_context

PROBE_TIMEOUT = 3.0  # seconds per GET

NAMES = {"splash": "Splash", "kapitsa": "Kapitsa", "llamacpp": "llama.cpp", "ollama": "ollama",
         "lmstudio": "LM Studio", "vllm": "vLLM", "mtplx": "MTPLX", "omlx": "OMLX", "mlx-dspark": "mlx-dspark",
         "mlx_lm": "mlx_lm.server", "generic": "generic"}


_HF_CACHE = re.compile(r"models--([^/\\]+?)--([^/\\]+)")
_HASH_ID = re.compile(r"^[0-9a-f]{40}$")


def hf_label(text):
    """A model reference as a label that never carries a local path: `ORG/NAME` for a Hugging Face cache path
    (`.../models--ORG--NAME/snapshots/<hash>`), the last path component for any other path, a plain id (an HF repo
    name) unchanged. None for an empty or non-string value."""
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    m = _HF_CACHE.search(text)
    if m:
        return "%s/%s" % (m.group(1), m.group(2))
    if re.match(r"^(?:[/~\\]|[A-Za-z]:[/\\])", text) or "\\" in text:
        parts = [p for p in re.split(r"[/\\]+", text) if p]
        return parts[-1] if parts else None
    return text


def model_extras(kind, entry):
    """The label details of one /v1/models entry of an MLX server: {"label": ORG/NAME (only when it differs from the
    id), "mode": dflash/dspark, "drafter": ORG/NAME} for mlx-dspark (the id is a snapshot hash there), {"label"} for
    mlx_lm (a local path id becomes its HF name or basename). Local paths never reach labels, file names or states."""
    if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
        return {}
    mid, out = entry["id"], {}
    x = entry.get("x_mlx_dspark") if kind == "mlx-dspark" else None
    if isinstance(x, dict):
        target = hf_label(x.get("target"))
        if _HASH_ID.match(mid) and target and "/" in target:
            out["label"] = target
        if isinstance(x.get("mode"), str) and x["mode"]:
            out["mode"] = x["mode"]
        drafter = hf_label(x.get("drafter"))
        if drafter:
            out["drafter"] = drafter
    if "label" not in out and kind in ("mlx-dspark", "mlx_lm"):
        label = hf_label(mid)
        if label and label != mid:
            out["label"] = label
    return out


def _get(url, timeout):
    """GET url: {"reachable", "status", "headers", "json"}; json is None unless the body is JSON."""
    out = {"reachable": False, "status": None, "headers": {}, "json": None}
    try:
        req = urllib.request.Request(url, headers={**auth_header(), "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as r:
            body = r.read(1 << 20)
            out.update(reachable=True, status=r.status, headers={k.lower(): v for k, v in r.headers.items()})
    except urllib.error.HTTPError as e:  # answered, but not 2xx: reachable, nothing found
        out.update(reachable=True, status=e.code)
        e.close()
        return out
    except Exception:
        return out
    try:
        out["json"] = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        pass
    return out


def _post(url, payload, timeout):
    """POST url with JSON payload: {"reachable", "status", "headers", "json"}."""
    out = {"reachable": False, "status": None, "headers": {}, "json": None}
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={**auth_header(), "Content-Type": "application/json", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as r:
            body = r.read(1 << 20)
            out.update(reachable=True, status=r.status, headers={k.lower(): v for k, v in r.headers.items()})
    except urllib.error.HTTPError as e:
        out.update(reachable=True, status=e.code)
        e.close()
        return out
    except Exception:
        return out
    try:
        out["json"] = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        pass
    return out


def _info(kind, version=None, build=None, **details):
    return {"kind": kind, "version": version, "build": build,
            "details": {k: v for k, v in details.items() if v is not None}}


def _splash(root, timeout):
    r = _get(root + "/status", timeout)
    j = r["json"]
    if not (isinstance(j, dict) and "schema_version" in j and isinstance(j.get("identity"), dict)
            and isinstance(j.get("instance"), dict)):
        return None
    cache = j["identity"].get("cache") or {}
    build_id = cache.get("build_id")
    short = build_id[4:12] if isinstance(build_id, str) and build_id.startswith("src-") else (
        build_id[:8] if isinstance(build_id, str) and build_id else None)
    return _info("splash", short, build_id, schema_version=j["schema_version"],
                 model=j["instance"].get("model"), kv_format=(j["identity"].get("kv") or {}).get("format"),
                 server=r["headers"].get("server"))


def _kapitsa(root, timeout):
    j = _get(root + "/cluster", timeout)["json"]
    if not (isinstance(j, dict) and "proto" in j and isinstance(j.get("version"), str)):
        return None
    return _info("kapitsa", j["version"], None, proto=j["proto"], chip=j.get("chip"), model=j.get("model"))


def _llamacpp(root, timeout):
    j = _get(root + "/props", timeout)["json"]
    if not isinstance(j, dict) or not any(k in j for k in ("build_info", "default_generation_settings",
                                                           "total_slots", "role")):
        return None
    build = j.get("build_info") if isinstance(j.get("build_info"), str) else None
    version = build.split("-", 1)[0] if build else None
    path = j.get("model_path")
    return _info("llamacpp", version, build, total_slots=j.get("total_slots"), role=j.get("role"),
                 model_file=path.replace("\\", "/").rsplit("/", 1)[-1] if isinstance(path, str) else None)


def _ollama(root, timeout):
    j = _get(root + "/api/version", timeout)["json"]
    if not (isinstance(j, dict) and isinstance(j.get("version"), str)):
        return None
    return _info("ollama", j["version"])


def parse_ollama_ps(data, model_name=None):
    """Extract model_info from an Ollama GET /api/ps response: {quant, format, family, parameter_size,
    context_length, runner, digest} (only fields present). Matches model_name if given."""
    if not isinstance(data, dict):
        return None
    models = data.get("models")
    if not (isinstance(models, list) and models):
        return None

    matched = None
    if model_name:
        mn_lower = model_name.lower()
        mn_base = mn_lower[:-7] if mn_lower.endswith(":latest") else mn_lower
        for m in models:
            if not isinstance(m, dict):
                continue
            name = (m.get("name") or m.get("model") or "")
            n_lower = name.lower()
            n_base = n_lower[:-7] if n_lower.endswith(":latest") else n_lower
            if name == model_name or n_lower == mn_lower or n_base == mn_base:
                matched = m
                break
    elif isinstance(models[0], dict):
        matched = models[0]  # no name given: the first loaded model
    if not matched:
        return None

    details = matched.get("details") if isinstance(matched.get("details"), dict) else {}
    info = {}
    quant = details.get("quantization_level") or matched.get("quantization_level")
    if quant and isinstance(quant, str):
        info["quant"] = quant
    for k in ("format", "family", "parameter_size"):
        v = details.get(k)
        if v and isinstance(v, str):
            info[k] = v
    for k in ("context_length", "runner", "digest"):
        v = matched.get(k)
        if v is not None and v != "":
            info[k] = v
    return info or None


def parse_ollama_show(data):
    """Extract model_info from an Ollama POST /api/show response."""
    if not isinstance(data, dict):
        return None
    details = data.get("details") if isinstance(data.get("details"), dict) else {}
    info = {}
    quant = details.get("quantization_level")
    if quant and isinstance(quant, str):
        info["quant"] = quant
    for k in ("format", "family", "parameter_size", "runner"):
        v = details.get(k)
        if v and isinstance(v, str):
            info[k] = v
    ctx = None
    mi = data.get("model_info")
    if isinstance(mi, dict):
        for k, v in mi.items():
            if k.endswith(".context_length") and isinstance(v, int):
                ctx = v
                break
    if ctx is None and isinstance(data.get("parameters"), str):
        m = re.search(r"num_ctx\s+(\d+)", data["parameters"])
        if m:
            ctx = int(m.group(1))
    if ctx is not None:
        info["context_length"] = ctx
    return info or None


_UNSET = object()


def fetch_ollama_ps(root, timeout=PROBE_TIMEOUT):
    """GET /api/ps once: the parsed JSON, or None."""
    base = root[:-3] if root.endswith("/v1") else root.rstrip("/")
    return _get(base + "/api/ps", timeout).get("json")


def fetch_lmstudio_models(root, timeout=PROBE_TIMEOUT):
    """GET /api/v0/models once: the parsed JSON, or None."""
    base = root[:-3] if root.endswith("/v1") else root.rstrip("/")
    return _get(base + "/api/v0/models", timeout).get("json")


def ollama_model_info(root, model_name=None, timeout=PROBE_TIMEOUT, ps=_UNSET):
    """Query Ollama's GET /api/ps (loaded models; `ps`: an already fetched response), falling back to POST /api/show
    when the model is not listed there. Never raises."""
    try:
        base = root[:-3] if root.endswith("/v1") else root.rstrip("/")
        if ps is _UNSET:
            ps = fetch_ollama_ps(root, timeout)
        res = parse_ollama_ps(ps, model_name)
        if res:
            return res
        if model_name:
            r_show = _post(base + "/api/show", {"model": model_name}, timeout)
            return parse_ollama_show(r_show.get("json"))
        return None
    except Exception:
        return None


def parse_lmstudio_models(data, model_name=None):
    """Extract model_info from an LM Studio GET /api/v0/models response: {quant, format, family, context_length}."""
    if not isinstance(data, dict):
        return None
    models = data.get("data")
    if not (isinstance(models, list) and models):
        return None
    matched = None
    if model_name:
        mn_lower = model_name.lower()
        for m in models:
            if isinstance(m, dict) and (m.get("id") == model_name or (m.get("id") or "").lower() == mn_lower):
                matched = m
                break
    else:  # no name given: the only loaded model, or the only model
        loaded = [m for m in models if isinstance(m, dict) and m.get("state") == "loaded"]
        if len(loaded) == 1:
            matched = loaded[0]
        elif len(models) == 1 and isinstance(models[0], dict):
            matched = models[0]
    if not matched:
        return None

    info = {}
    quant = matched.get("quantization")
    if quant and isinstance(quant, str):
        info["quant"] = quant
    fmt = matched.get("compatibility_type")
    if fmt and isinstance(fmt, str):
        info["format"] = fmt
    arch = matched.get("arch")
    if arch and isinstance(arch, str):
        info["family"] = arch
    ctx = matched.get("loaded_context_length") or matched.get("max_context_length")
    if isinstance(ctx, int):
        info["context_length"] = ctx
    return info or None


def lmstudio_model_info(root, model_name=None, timeout=PROBE_TIMEOUT, models=_UNSET):
    """Query LM Studio's GET /api/v0/models (`models`: an already fetched response) for model details. Never raises."""
    try:
        if models is _UNSET:
            models = fetch_lmstudio_models(root, timeout)
        return parse_lmstudio_models(models, model_name)
    except Exception:
        return None


def _lmstudio(root, timeout):
    j = _get(root + "/api/v0/models", timeout)["json"]
    data = j.get("data") if isinstance(j, dict) else None
    if not (isinstance(data, list) and data and all(isinstance(m, dict) for m in data)
            and any("arch" in m or "state" in m or "max_context_length" in m for m in data)):
        return None
    return _info("lmstudio", models=len(data), loaded=sum(1 for m in data if m.get("state") == "loaded"))


def _vllm(root, timeout):
    j = _get(root + "/version", timeout)["json"]
    if not (isinstance(j, dict) and isinstance(j.get("version"), str)):
        return None
    return _info("vllm", j["version"])


def _mtplx(root, timeout):
    j = _get(root + "/health", timeout)["json"]
    if not (isinstance(j, dict) and j.get("ok") is True
            and any(k in j for k in ("mlx_runtime", "generation_mode", "runtime_mode", "profile"))):
        return None
    mlx = j.get("mlx_runtime") or {}
    path = mlx.get("path") or ""
    m = re.search(r"venv-(\d+\.\d+(?:\.\d+)?)", path)
    version = j.get("version") or (m.group(1) if m else mlx.get("version"))
    profile = (j.get("profile") or {}).get("name")
    return _info("mtplx", version=version, build=None,
                 model=j.get("model"), chip=j.get("chip"),
                 generation_mode=j.get("generation_mode"), runtime_mode=j.get("runtime_mode"),
                 profile=profile, mlx_version=mlx.get("version"), depth=j.get("depth"))


def _omlx(root, timeout):
    r = _get(root + "/admin/api/stats", timeout)
    j = r["json"]
    if not (isinstance(j, dict) and (j.get("cli_prefix") == "omlx" or "engines" in j or "active_models" in j)):
        rh = _get(root + "/health", timeout)
        jh = rh["json"]
        if not (isinstance(jh, dict) and "engine_pool" in jh):
            return None
        j = jh

    version = j.get("version")
    engines = j.get("engines") if isinstance(j.get("engines"), dict) else {}
    mlx_lm = engines.get("mlx-lm") if isinstance(engines.get("mlx-lm"), dict) else {}
    mlx = engines.get("mlx") if isinstance(engines.get("mlx"), dict) else {}
    mlx_lm_ver = mlx_lm.get("version")
    mlx_ver = mlx.get("version")

    if not version:
        rv = _get(root + "/admin/api/version", timeout)
        jv = rv["json"]
        if isinstance(jv, dict) and isinstance(jv.get("version"), str):
            version = jv["version"]
        elif isinstance(jv, str):
            version = jv.strip()

    commit = mlx_lm.get("commit")
    build = commit[:8] if isinstance(commit, str) and len(commit) >= 8 else None

    if not version:
        version = mlx_lm_ver or mlx_ver

    return _info("omlx", version=version, build=build,
                 mlx_lm_version=mlx_lm_ver, mlx_version=mlx_ver,
                 cli_prefix=j.get("cli_prefix"),
                 uptime_seconds=j.get("uptime_seconds"))


def _models_data(models):
    j = models.get("json") if isinstance(models, dict) else None
    data = j.get("data") if isinstance(j, dict) else None
    return [m for m in data if isinstance(m, dict)] if isinstance(data, list) else []


def _is_mlx_dspark(models):
    data = _models_data(models)
    if str(models["headers"].get("server") or "").startswith("mlx-dspark"):
        return True
    return any(m.get("owned_by") == "mlx-dspark" or isinstance(m.get("x_mlx_dspark"), dict) for m in data)


def _mlx_dspark(root, timeout, models):
    """mlx-dspark: version and device from GET /doctor (environment.version, environment.chip.name or .device), the
    mode (dflash/dspark) from the first model's x_mlx_dspark. /doctor lists a whole catalog; only environment is read."""
    env = (_get(root + "/doctor", timeout)["json"] or {}).get("environment")
    env = env if isinstance(env, dict) else {}
    chip = env.get("chip") if isinstance(env.get("chip"), dict) else {}
    data = _models_data(models)
    extras = model_extras("mlx-dspark", data[0]) if data else {}
    return _info("mlx-dspark", version=env.get("version") if isinstance(env.get("version"), str) else None,
                 chip=chip.get("name") or env.get("device"), mode=extras.get("mode"),
                 server=models["headers"].get("server"))


def _mlx_lm(root, timeout, models):
    """mlx_lm.server: only the weak signature (see the module doc); None unless every part matches."""
    if not str(models["headers"].get("server") or "").startswith("BaseHTTP/"):
        return None
    data = _models_data(models)
    if not data or any(m.get("owned_by") for m in data):
        return None
    if _get(root + "/health", timeout)["json"] != {"status": "ok"}:
        return None
    return _info("mlx_lm", server=models["headers"].get("server"))


PROBES = (_splash, _kapitsa, _llamacpp, _ollama, _lmstudio, _vllm, _mtplx, _omlx)
OWNED_BY = {"splash": "splash", "nace": "kapitsa", "llamacpp": "llamacpp", "vllm": "vllm", "mtplx": "mtplx", "omlx": "omlx"}


def detect(base_url, timeout=PROBE_TIMEOUT):
    """Classify the server at base_url (ends in /v1): {"kind", "version", "build", "details"}. Never raises."""
    try:
        root = base_url[:-3] if base_url.endswith("/v1") else base_url.rstrip("/")
        models = _get(base_url + "/models", timeout)
        if not models["reachable"]:
            return _info("generic", reachable=False)
        owner = None
        data = (models["json"] or {}).get("data") if isinstance(models["json"], dict) else None
        if isinstance(data, list) and data and isinstance(data[0], dict):
            owner = data[0].get("owned_by")
        if owner == "omlx":
            found = _omlx(root, timeout)
            if found:
                found["details"]["owned_by"] = owner
                return found
            return _info("omlx", owned_by=owner)
        if _is_mlx_dspark(models):
            found = _mlx_dspark(root, timeout, models)
            if owner and "owned_by" not in found["details"]:
                found["details"]["owned_by"] = owner
            return found
        for probe in PROBES:
            found = probe(root, timeout)
            if found:
                if owner and "owned_by" not in found["details"]:
                    found["details"]["owned_by"] = owner
                return found
        if owner == "mtplx":
            found = _mtplx(root, timeout)
            if found:
                found["details"]["owned_by"] = owner
                return found
        if owner is None:
            found = _mlx_lm(root, timeout, models)
            if found:
                return found
        return _info(OWNED_BY.get(owner, "generic"), owned_by=owner)
    except Exception:
        return _info("generic")


def apply_version_override(info, version):
    """--server-version: set info["version"] to `version`; the detected one is kept as info["build_id"] when it
    differs and no build_id is set. Returns info (a dict is changed in place; None stays None)."""
    if not version or not isinstance(info, dict):
        return info
    detected = info.get("version")
    if detected and detected != version and not info.get("build_id"):
        info["build_id"] = detected
    info["version"] = version
    return info


def label(info):
    """Short text for discovery output: 'Splash 1a2b3c4d', 'llama.cpp b6123', 'generic'."""
    info = info or {}
    name = NAMES.get(info.get("kind"), "generic")
    return name + (" " + info["version"] if info.get("version") else "")
