"""TLS settings shared by every request replay_bench makes.

HTTPS endpoints are verified against the system trust store. Optional: --ca-file FILE adds a CA bundle on
top of it (a server with a private CA), --insecure turns certificate verification off.
"""

import ssl

_ca_file = None
_insecure = False
_ctx = None


def configure(ca_file=None, insecure=False):
    """Select the TLS options (from --ca-file / --insecure); resets the cached context."""
    global _ca_file, _insecure, _ctx
    _ca_file, _insecure, _ctx = ca_file, bool(insecure), None


def ssl_context(path=None):
    """System trust store plus the configured extra CA bundle (or *path*); unverified with --insecure."""
    global _ctx
    if path is None and _ctx is not None:
        return _ctx
    ctx = ssl.create_default_context()
    if _insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    extra = path or _ca_file
    if extra:
        ctx.load_verify_locations(cafile=extra)
    if path is None:
        _ctx = ctx
    return ctx
