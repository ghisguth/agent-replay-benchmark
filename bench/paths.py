"""Where the shipped data lives, for a source checkout and for an installed package alike.

``fixtures/`` (with ``refs/`` and ``samples/``) is at the repository root in a checkout, and inside the ``bench``
package (``bench/fixtures``) when installed: pyproject.toml maps the root directory to ``bench.fixtures``. Run output (``states/``) goes to the repository's ``states/`` in a checkout and to
``./states`` under the current directory when installed.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(ROOT if os.path.isfile(os.path.join(ROOT, "pyproject.toml")) else _PACKAGE_DIR, "fixtures")
REFS_DIR = os.path.join(FIXTURES_DIR, "refs")
SAMPLES_DIR = os.path.join(FIXTURES_DIR, "samples")


def is_checkout():
    """True when running from a source checkout (pyproject.toml beside the package), not from site-packages."""
    return os.path.isfile(os.path.join(ROOT, "pyproject.toml"))


def default_states_dir():
    return os.path.join(ROOT, "states") if is_checkout() else os.path.join(os.getcwd(), "states")
