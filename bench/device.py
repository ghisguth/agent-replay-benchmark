"""The hardware a run ran on, as a short human name: `MacBook Pro M2 Max 64GB`, `Linux Mint 22.3 32GB NVIDIA GeForce
GTX 1080 8192MiB`.

The server's hardware is what matters, and only the server knows it. It is known when the server is this machine
(the endpoint host is localhost or this host's name), when the server reports it (Kapitsa's /cluster `chip`), or
when the user names it with --device. A remote server that reports nothing gets no device (None), not this machine's.

Detection only reads the model name, chip, memory, OS and GPU name/memory; never serial numbers or UUIDs. Every probe
is best effort with a short timeout: detection never raises.
"""

import ipaddress
import platform
import re
import shutil
import socket
import subprocess
import urllib.parse

TIMEOUT = 5.0  # seconds per probe

_override = None
_local = None  # cached local() result


def configure(override):
    """--device TEXT: the server's hardware, named by the user (wins over detection)."""
    global _override
    _override = (override or "").strip() or None


def _run(argv):
    if not shutil.which(argv[0]):
        return None
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT)
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _gb(n_bytes):
    return "%dGB" % round(n_bytes / (1 << 30))


def _mac():
    model = chip = mem = None
    text = _run(["system_profiler", "SPHardwareDataType"]) or ""
    for line in text.splitlines():
        key, _, value = line.strip().partition(":")
        value = value.strip()
        if key == "Model Name":
            model = value
        elif key in ("Chip", "Processor Name"):
            chip = value
        elif key == "Memory":
            mem = value.replace(" ", "")
    if not model:
        model = (_run(["sysctl", "-n", "hw.model"]) or "").strip() or "Mac"
    if not chip:
        chip = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip() or None
    if not mem:
        raw = (_run(["sysctl", "-n", "hw.memsize"]) or "").strip()
        mem = _gb(int(raw)) if raw.isdigit() else None
    chip = re.sub(r"^Apple\s+", "", chip) if chip else None
    return " ".join(p for p in (model, chip, mem) if p)


def _os_release():
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"') or None
    except OSError:
        pass
    return None


def _linux_mem():
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return _gb(int(line.split()[1]) * 1024)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _nvidia():
    """`NVIDIA GeForce GTX 1080 8192MiB` per GPU, joined with ` + ` (None without nvidia-smi)."""
    text = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    gpus = []
    for line in (text or "").splitlines():
        name, _, mem = line.partition(",")
        if name.strip():
            gpus.append(" ".join(p for p in (name.strip(), mem.strip().replace(" ", "")) if p))
    return " + ".join(gpus) or None


def _linux():
    return " ".join(p for p in (_os_release() or "Linux", _linux_mem(), _nvidia()) if p)


def local():
    """This machine's hardware name (cached)."""
    global _local
    if _local is None:
        try:
            system = platform.system()
            if system == "Darwin":
                _local = _mac()
            elif system == "Linux":
                _local = _linux()
            else:
                _local = " ".join(p for p in (platform.system(), platform.release(), platform.machine()) if p)
        except Exception:
            _local = platform.system() or "unknown"
    return _local


def _host_names():
    names = set()
    found = []
    for fn in (socket.gethostname, socket.getfqdn):
        try:
            found.append(fn())
        except OSError:
            pass
    if platform.system() == "Darwin":  # the Bonjour name (`name.local`) differs from gethostname() on many Macs
        found.append(_run(["scutil", "--get", "LocalHostName"]) or "")
    for n in found:
        n = n.strip().lower().rstrip(".")
        if n:
            names.add(n)
            names.add(n.split(".")[0])
    return names


def is_local(url):
    """True when the endpoint URL points at this machine."""
    host = (urllib.parse.urlparse(url).hostname or "").lower().rstrip(".")
    if not host or host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_unspecified
    except ValueError:
        pass
    return host in _host_names() or host.split(".")[0] in _host_names()


def server(url, info=None):
    """{"name", "source"} of the server's hardware (source: "device-flag", "server" or "local"), or None."""
    if _override:
        return {"name": _override, "source": "device-flag"}
    chip = ((info or {}).get("details") or {}).get("chip")
    if chip:
        return {"name": str(chip), "source": "server"}
    if is_local(url):
        return {"name": local(), "source": "local"}
    return None


_ABBREV = (("MacBook Pro", "MBP-"), ("MacBook Air", "MBA-"), ("Mac mini", "Macmini-"), ("Mac Studio", "MacStudio-"),
           ("NVIDIA GeForce ", ""), ("NVIDIA ", ""), ("GeForce ", ""), (" + ", "+"))


def short(name):
    """The device part of a state file name: `MBP-M2Max-64GB`, `LinuxMint22.3-32GB-GTX1080-8192MiB`."""
    s = str(name or "")
    for long_, abbr in _ABBREV:
        s = s.replace(long_, abbr)
    s = re.sub(r"\s*\b(\d+(?:GB|MiB))\b\s*", r"-\1-", s)  # memory sizes separate the parts
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Za-z0-9.+]+", "-", s).replace("-+", "+").strip("-.+")
    return s[:48].rstrip("-.+") or None
