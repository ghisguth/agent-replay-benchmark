"""ANSI color codes and small text-formatting helpers."""

import contextlib


class Clr:
    RST = "\033[0m"
    BLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GRN = "\033[32m"
    YEL = "\033[33m"
    BLU = "\033[34m"
    MAG = "\033[35m"
    CYN = "\033[36m"
    WHT = "\033[37m"
    BCYN = "\033[96m"
    BGRN = "\033[92m"
    BYEL = "\033[93m"
    BRED = "\033[91m"
    BMAG = "\033[95m"
    BBLU = "\033[94m"
    REV = "\033[7m"


_DEFAULTS = {k: v for k, v in vars(Clr).items() if k.isupper()}


def configure(enabled):
    """Turn the ANSI codes on or off for everything that formats with Clr (off: empty strings)."""
    for k, v in _DEFAULTS.items():
        setattr(Clr, k, v if enabled else "")


live_enabled = False  # stderr's own setting: the live line may carry colors even where the report on stdout has none


def configure_live(on):
    """Whether what is drawn on stderr (the live line) is colored: stderr is a terminal and NO_COLOR is not set,
    whatever the report format or where stdout goes."""
    global live_enabled
    live_enabled = bool(on)


@contextlib.contextmanager
def live():
    """Colors on for what is built inside when stderr takes them, even if the report on stdout has none (markdown, a
    pipe). Used around the live line only; the report itself is never drawn inside it."""
    on = Clr.RST != "" or not live_enabled
    if on:
        yield
        return
    configure(True)
    try:
        yield
    finally:
        configure(False)
