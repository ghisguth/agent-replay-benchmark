"""Endpoint/model discovery.

The list of models returned by each endpoint's GET <base>/models (base ends in /v1) is the source of
truth: endpoints that are down are excluded, and the model filters only ever narrow what a live
endpoint actually reports.
"""

import json
import urllib.request

from . import backend, device
from .auth import auth_header
from .colors import Clr
from .tls import ssl_context


def _model_n_ctx(m):
    """Extract a model's runtime context window from a /v1/models entry.

    Prefers the loaded model's ``meta.n_ctx``; falls back to the configured
    ``--ctx-size`` in the (un)loaded slot's launch args or preset (e.g. llama-server
    router mode), then runtime fields like ``max_model_len`` (e.g. oMLX/vLLM),
    and finally fallback context fields like ``context_length``. Returns int or None.
    """
    meta = m.get('meta') or {}
    if isinstance(meta.get('n_ctx'), int):
        return meta['n_ctx']

    args = (m.get('status') or {}).get('args') or []
    for i, a in enumerate(args):
        if a in ('--ctx-size', '-c') and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except (TypeError, ValueError):
                pass

    preset = (m.get('status') or {}).get('preset') or ''
    if preset:
        import re
        m_ctx = re.search(r'^(?:ctx-size|c)\s*=\s*(\d+)', preset, re.MULTILINE)
        if m_ctx:
            try:
                return int(m_ctx.group(1))
            except (TypeError, ValueError):
                pass

    for key in ('max_model_len', 'max_context_length', 'max_tokens', 'context_length', 'n_ctx'):
        if isinstance(m.get(key), int):
            return m[key]

    return None


def discover_models(base_url):
    """Return the models served by an endpoint (base_url ends in /v1) as ``[{"id", "n_ctx"}]`` (None if down).

    One GET <base>/models and nothing else (wait_for_server polls this): the context window a server does not put
    there is filled in by fill_n_ctx once the kind of server is known.

    The llama router advertises a synthetic "default" placeholder alias for its idle
    slot alongside the actual loaded models — skip it so it isn't benchmarked twice.
    """
    url = f"{base_url}/models"
    try:
        req = urllib.request.Request(url, headers=dict(auth_header()))
        with urllib.request.urlopen(req, timeout=10, context=ssl_context()) as response:
            data = json.loads(response.read().decode('utf-8'))
        if data.get('data'):
            entries = [{"id": m['id'], "n_ctx": _model_n_ctx(m), **({"entry": m} if m.get("x_mlx_dspark") else {})}
                       for m in data['data']]
        elif data.get('models'):
            entries = [{"id": m['name'], "n_ctx": _model_n_ctx(m)} for m in data['models']]
        else:
            entries = []
        return [e for e in entries if e["id"] != "default"]
    except Exception as e:
        _close_error(e)
        return None


def _get_json(url):
    try:
        req = urllib.request.Request(url, headers=dict(auth_header()))
        with urllib.request.urlopen(req, timeout=2, context=ssl_context()) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        _close_error(e)
        return None


def fill_n_ctx(root, entries, kind, ps=None, lm_models=None):
    """Fill the missing ``n_ctx`` of `entries` from the server's own endpoints outside /v1, one GET per endpoint and
    only for the kinds that have it: Splash/Kapitsa /status, Ollama /api/ps (`ps`, when already fetched), LM Studio
    /api/v0/models (`lm_models`). Other servers (vLLM, llama.cpp, generic) get no extra request."""
    if not any(e["n_ctx"] is None for e in entries):
        return
    if kind in ("splash", "kapitsa"):
        s_data = _get_json(f"{root}/status")
        if isinstance(s_data, dict):
            s_ctx = (
                s_data.get('maximum_context_tokens')
                or (s_data.get('memory_plan') or {}).get('budget', {}).get('kv_virtual_tokens')
                or (s_data.get('budget') or {}).get('kv_virtual_tokens')
            )
            if isinstance(s_ctx, int):
                for e in entries:
                    if e["n_ctx"] is None:
                        e["n_ctx"] = s_ctx
    elif kind in ("ollama", "lmstudio"):
        if kind == "ollama":
            data, parse = ps, backend.parse_ollama_ps
        else:
            data, parse = lm_models, backend.parse_lmstudio_models
        for e in entries:
            if e["n_ctx"] is None:
                mi = parse(data, e["id"])
                if mi and isinstance(mi.get("context_length"), int):
                    e["n_ctx"] = mi["context_length"]


def _close_error(e):
    """An HTTPError holds the response socket open until closed."""
    if isinstance(e, urllib.error.HTTPError):
        e.close()


EXACT_PREFIX = 'exact:'


def _is_selected(model, model_filters):
    if not model_filters:
        return True
    low = model.lower()
    for f in model_filters:
        if f.lower().startswith(EXACT_PREFIX):
            if f[len(EXACT_PREFIX):].lower() == low:
                return True
        elif f.lower() in low:
            return True
    return False


def select_models(models, model_filters=None, first_model=False):
    """The models to benchmark on one endpoint, in the server's /v1/models order.

    --model filters first (case-insensitive substring, or whole-id equality for `exact:NAME`), then
    --first-model keeps only the first one left.
    """
    selected = [m for m in models if _is_selected(m["id"], model_filters)]
    return selected[:1] if first_model else selected


def discover_and_report(routers, model_filters, spinner=None, first_model=False):
    """Probe every endpoint, print a discovery report, and return execution targets.

    Prints each endpoint's status (up/down) and every model it serves, marking which
    models are selected by --model / --first-model. Only reachable endpoints and
    selected models become execution targets.
    """
    print(f"{Clr.BLD}{Clr.MAG}Discovery{Clr.RST}")

    execution_targets = []
    for router in routers:
        if spinner:
            spinner.start(f"probing {router['backend']} ...")
        models = discover_models(router["base_url"])
        if spinner:
            spinner.stop()

        if models is None:
            print(f"  {Clr.BRED}●{Clr.RST} {Clr.WHT}{router['backend']:<24}{Clr.RST} {Clr.BRED}down{Clr.RST} {Clr.DIM}({router['base_url']}){Clr.RST}")
            continue
        if not models:
            print(f"  {Clr.BYEL}●{Clr.RST} {Clr.WHT}{router['backend']:<24}{Clr.RST} {Clr.BYEL}up, but serves no models{Clr.RST}")
            continue

        selected = select_models(models, model_filters, first_model)
        if not selected:
            print(f"  {Clr.BYEL}●{Clr.RST} {Clr.WHT}{router['backend']:<24}{Clr.RST} {Clr.BYEL}no model matches --model "
                  f"{', '.join(model_filters)}{Clr.RST} {Clr.DIM}(serves: {', '.join(m['id'] for m in models)}){Clr.RST}")
            continue
        selected_ids = {m["id"] for m in selected}
        if spinner:
            spinner.start(f"detecting the server at {router['backend']} ...")
        info = backend.detect(router["base_url"])
        dev = device.server(router["base_url"], info)
        if spinner:
            spinner.stop()
        print(f"  {Clr.BGRN}●{Clr.RST} {Clr.WHT}{router['backend']:<24}{Clr.RST} {Clr.BLD}{backend.label(info)}{Clr.RST}"
              f"{(' ' + Clr.DIM + '· ' + dev['name'] + Clr.RST) if dev else ''} "
              f"{Clr.DIM}({len(selected)}/{len(models)} selected){Clr.RST}")
        root = router["base_url"][:-3] if router["base_url"].endswith("/v1") else router["base_url"].rstrip("/")
        kind = info.get("kind")
        ps = lm_models = None  # fetched once per endpoint, the models are matched locally
        if kind == "ollama":
            ps = backend.fetch_ollama_ps(root)
        elif kind == "lmstudio":
            lm_models = backend.fetch_lmstudio_models(root)
        fill_n_ctx(root, models, kind, ps, lm_models)
        for model in models:
            name = model["id"]
            model_info = None
            if name in selected_ids:  # the probes are for the models that run
                if kind == "ollama":
                    model_info = backend.ollama_model_info(root, name, ps=ps)
                elif kind == "lmstudio":
                    model_info = backend.lmstudio_model_info(root, name, models=lm_models)
                if kind in ("mlx-dspark", "mlx_lm"):
                    model_info = backend.model_extras(kind, model.get("entry") or {"id": name}) or None
                if model_info and model.get("n_ctx") is None and model_info.get("context_length"):
                    model["n_ctx"] = model_info["context_length"]
            n_ctx = model.get("n_ctx")
            ctx_str = f" {Clr.DIM}[ctx {n_ctx // 1024}k]{Clr.RST}" if n_ctx else ""
            if name in selected_ids:
                print(f"      {Clr.BGRN}✓{Clr.RST} {Clr.BLD}{name}{Clr.RST}{ctx_str}")
                lmstudio = kind == "lmstudio"
                chat_url = f"{root}/api/v0/chat/completions" if lmstudio else f"{router['base_url']}/chat/completions"
                target = {
                    "key": f"{router['backend']}::{name}",
                    "backend": router["backend"],
                    "name": name,
                    "url": chat_url,
                    "n_ctx": n_ctx,
                    "backend_info": info,
                    "device": dev,
                    "model_info": model_info,
                }
                if lmstudio:  # transport.post_with_fallback: used if /api/v0 does not take the first request
                    target["fallback_url"] = f"{router['base_url']}/chat/completions"
                execution_targets.append(target)
            else:
                print(f"      {Clr.DIM}· {name}{ctx_str}{Clr.RST}")

    print()
    return execution_targets


def _prefill_progress(target):
    from .transport import PromptProgress  # transport imports this module
    how = PromptProgress(True).sends(target)
    return {True: "yes (request field)", "header": "yes (request header)", "snapshot": "yes (snapshot polling)"}.get(how, "no")


def describe_targets(targets):
    """Print what discovery learned about the selected models, per server: kind, version, build, device, prefill
    progress; per model: quant, context, family, runner (whatever the server's probes returned). No request is sent."""
    seen = set()
    for t in targets:
        if t["backend"] not in seen:
            seen.add(t["backend"])
            info = t.get("backend_info") or {}
            dev = t.get("device")
            print(f"{Clr.BLD}{t['backend']}{Clr.RST}")
            for key, value in (
                    ("server", backend.label(info)), ("version", info.get("version") or "-"),
                    ("build", info.get("build") or "-"), ("device", (dev or {}).get("name") or "-"),
                    ("prefill progress", _prefill_progress(t))):
                print(f"  {key:<18}{value}")
        mi = t.get("model_info") or {}
        parts = [f"{k} {mi[k]}" for k in ("quant", "family", "runner", "format", "parameter_size") if mi.get(k)]
        ctx = t.get("n_ctx") or mi.get("context_length")
        parts.insert(0, f"context {ctx // 1024}k" if ctx else "context -")
        print(f"  {Clr.BGRN}✓{Clr.RST} {t['name']}: " + ", ".join(parts))
    print()
