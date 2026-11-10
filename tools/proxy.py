#!/usr/bin/env python3
"""Capture proxy: forwards every request to an OpenAI-compatible server and dumps each request/response
pair as <ts>_req.json / <ts>_resp.json under captures/dumps[/<topic>].

    tools/proxy.py                                              # :18080 -> http://localhost:8080
    tools/proxy.py --target https://example.org:8132 --port 18132 --topic my-run

Request bodies may be chunked (the .NET HttpClient sends JsonContent that way). HTTPS targets trust the
system trust store plus --ca-cert.
"""

import argparse
import http.client
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench.tls import ssl_context  # noqa: E402

# Defaults
TARGET_URL = "http://localhost:8080"
LISTEN_PORT = 18080
DUMP_DIR = os.path.join(ROOT, "captures", "dumps")

_ts_lock = threading.Lock()
_last_ts = 0


def _unique_timestamp():
    """Millisecond timestamp, bumped so concurrent requests never share a dump name."""
    global _last_ts
    with _ts_lock:
        _last_ts = max(int(time.time() * 1000), _last_ts + 1)
        return _last_ts


def _pretty(body):
    """The body as indented JSON when it is JSON, else as text."""
    try:
        return json.dumps(json.loads(body.decode('utf-8')), indent=2, ensure_ascii=False)
    except ValueError:
        return body.decode('utf-8', errors='ignore')


def ipv4_target(target_url):
    """Rewrite the target to its IPv4 address; returns (url, original host:port for the Host header).
    Some .local names resolve to an IPv6 link-local address first and connects there can stall for minutes."""
    u = urllib.parse.urlsplit(target_url)
    ip = socket.getaddrinfo(u.hostname, u.port, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    netloc = f"{ip}:{u.port}" if u.port else ip
    return urllib.parse.urlunsplit((u.scheme, netloc, u.path, u.query, u.fragment)), u.netloc


class ThoughtSignatures:
    """Gemini 3 returns a thought_signature (extra_content.google) on each tool call and rejects a later
    request whose assistant tool call lacks it. Clients that drop unknown fields
    cannot send it back, so the proxy remembers it by tool-call id and puts it back."""

    def __init__(self):
        self.by_id = {}
        self.lock = threading.Lock()

    def learn(self, resp_body):
        try:
            resp = json.loads(resp_body)
        except ValueError:
            return
        for choice in resp.get("choices") or [] if isinstance(resp, dict) else []:
            for tc in (choice.get("message") or {}).get("tool_calls") or []:
                if tc.get("id") and tc.get("extra_content"):
                    with self.lock:
                        self.by_id[tc["id"]] = tc["extra_content"]

    def restore(self, req_body):
        try:
            req = json.loads(req_body)
        except ValueError:
            return req_body
        changed = False
        for m in req.get("messages") or [] if isinstance(req, dict) else []:
            for tc in m.get("tool_calls") or []:
                extra = self.by_id.get(tc.get("id"))
                if extra and "extra_content" not in tc:
                    tc["extra_content"] = extra
                    changed = True
        return json.dumps(req).encode("utf-8") if changed else req_body


def make_handler(target_url, dump_dir, ctx, quiet, host_header=None, signatures=None):
    class DumpProxyHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # the default http.server access log would bury the one line per saved file
            pass

        def _read_body(self):
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                parts = []
                while True:
                    size = int(self.rfile.readline().split(b";")[0].strip(), 16)
                    if size == 0:
                        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                            pass
                        return b"".join(parts)
                    parts.append(self.rfile.read(size))
                    self.rfile.readline()
            content_length = int(self.headers.get('Content-Length', 0))
            return self.rfile.read(content_length) if content_length > 0 else b""

        def handle_proxy(self, method):
            url = f"{target_url.rstrip('/')}/{self.path.lstrip('/')}"
            req_body = self._read_body()
            if signatures is not None and req_body:
                req_body = signatures.restore(req_body)

            # Filter out hop-by-hop headers and let urllib manage the body length
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ('host', 'content-length', 'transfer-encoding', 'connection',
                                            'accept-encoding')}
            if host_header:
                headers['Host'] = host_header

            # Generate unique timestamp for matching pairs
            timestamp = _unique_timestamp()
            os.makedirs(dump_dir, exist_ok=True)

            if req_body:
                req_file = os.path.join(dump_dir, f"{timestamp}_req.json")
                with open(req_file, "w", encoding="utf-8") as f:
                    f.write(_pretty(req_body))
                if not quiet:
                    print(f"[+] [{method}] -> Saved request payload: {req_file}")

            req = urllib.request.Request(url, data=req_body if req_body else None, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=3600, context=ctx) as response:
                    resp_body = response.read()
                    status_code = response.status
                    resp_headers = response.info()
            except urllib.error.HTTPError as e:
                resp_body = e.read()
                status_code = e.code
                resp_headers = e.info()
            except (OSError, http.client.HTTPException) as e:
                print(f"[-] Connection to backend failed: {e}")
                self.send_error(502, f"Bad Gateway: {e}")
                return

            if signatures is not None and resp_body:
                signatures.learn(resp_body)

            if resp_body and req_body:
                resp_file = os.path.join(dump_dir, f"{timestamp}_resp.json")
                with open(resp_file, "w", encoding="utf-8") as f:
                    f.write(_pretty(resp_body))
                if not quiet:
                    print(f"[+] [{status_code}] <- Saved response payload: {resp_file}")

            # Send response back to original client
            self.send_response(status_code)
            for key, val in resp_headers.items():
                if key.lower() not in ('transfer-encoding', 'content-length', 'connection', 'content-encoding'):
                    self.send_header(key, val)
            self.send_header('Content-Length', str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        def do_POST(self):
            self.handle_proxy("POST")

        def do_GET(self):
            self.handle_proxy("GET")

    return DumpProxyHandler


def run():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=TARGET_URL, help="Backend base URL (default: %(default)s)")
    ap.add_argument("--port", type=int, default=LISTEN_PORT, help="Listen port (default: %(default)s)")
    ap.add_argument("--host", default="", help="Listen address (default: all interfaces)")
    ap.add_argument("--topic", help="Dump into captures/dumps/<topic>/")
    ap.add_argument("--dump-dir", help="Dump directory (overrides --topic)")
    ap.add_argument("--ca-cert", help="Extra CA bundle for an HTTPS target")
    ap.add_argument("--ipv4", action="store_true",
                    help="Connect to the target's IPv4 address (keeps its name in the Host header)")
    ap.add_argument("--gemini-thought-signatures", action="store_true",
                    help="Put Gemini's tool-call thought_signature back on later requests (for clients that drop it)")
    ap.add_argument("-q", "--quiet", action="store_true", help="Don't print a line per saved file")
    a = ap.parse_args()
    dump_dir = a.dump_dir or (os.path.join(DUMP_DIR, a.topic) if a.topic else DUMP_DIR)

    target, host_header = ipv4_target(a.target) if a.ipv4 else (a.target, None)
    httpd = ThreadingHTTPServer((a.host, a.port),
                                make_handler(target, dump_dir, ssl_context(a.ca_cert), a.quiet, host_header,
                                             ThoughtSignatures() if a.gemini_thought_signatures else None))
    print(f"[*] Proxy running on port {a.port} -> Forwarding to {target}"
          + (f" (Host: {host_header})" if host_header else ""))
    print(f"[*] Dumps directory: {os.path.abspath(dump_dir)}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Proxy stopped.")


if __name__ == "__main__":
    run()
