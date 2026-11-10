"""Endpoint URLs: parsing, validation and short labels (offline, no requests)."""

import argparse
import urllib.parse

DEFAULT_ENDPOINT = "http://localhost:8080/v1"  # llama-server's default port


def parse_endpoint(value):
    """Validate an --endpoint URL and return {"backend": label, "base_url": "<scheme>://host[:port]/v1"}.

    Accepted: http:// or https:// with a host, and a path of either /v1 (optionally with a trailing slash)
    or nothing at all, in which case /v1 is appended. Any other path is rejected, so a pasted
    /v1/chat/completions URL fails loudly instead of being silently rewritten. The label is host:port
    (the scheme's default port when none is given, a trailing dot on the host dropped).
    """
    raw = str(value).strip()
    u = urllib.parse.urlsplit(raw)
    if u.scheme not in ("http", "https"):
        raise argparse.ArgumentTypeError(
            "invalid endpoint %r: the scheme must be http:// or https:// (e.g. http://localhost:8080/v1)" % value)
    if not u.hostname:
        raise argparse.ArgumentTypeError("invalid endpoint %r: no host" % value)
    if u.query or u.fragment or u.username or u.password:
        raise argparse.ArgumentTypeError("invalid endpoint %r: no credentials, query or fragment allowed" % value)
    try:
        port = u.port
    except ValueError:
        raise argparse.ArgumentTypeError("invalid endpoint %r: bad port" % value)
    path = u.path.rstrip("/")
    if path not in ("", "/v1"):
        raise argparse.ArgumentTypeError(
            "invalid endpoint %r: the path must be /v1 (or empty, /v1 is then appended), not %r" % (value, u.path))
    host = u.hostname.rstrip(".")
    shown = "[%s]" % host if ":" in host else host
    netloc = u.netloc.rsplit("@", 1)[-1]
    port_str = ":%d" % (port or (443 if u.scheme == "https" else 80))
    return {
        "backend": shown + port_str,
        "base_url": "%s://%s/v1" % (u.scheme, netloc),
    }


def resolve_endpoints(values):
    """Parse the repeatable --endpoint values (default endpoint when none), dropping exact repeats.

    Raises argparse.ArgumentTypeError when two different URLs would get the same label.
    """
    routers = []
    by_label = {}
    for v in (values or [DEFAULT_ENDPOINT]):
        r = v if isinstance(v, dict) else parse_endpoint(v)
        prev = by_label.get(r["backend"])
        if prev is not None:
            if prev["base_url"] == r["base_url"]:
                continue
            raise argparse.ArgumentTypeError(
                "endpoints %s and %s have the same label %s" % (prev["base_url"], r["base_url"], r["backend"]))
        by_label[r["backend"]] = r
        routers.append(r)
    return routers
