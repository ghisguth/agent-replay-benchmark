"""Terminal spinner used while requests are in flight."""

import sys
import threading
import time

from . import colors
from .colors import Clr


class ProgressAnimation:
    def __init__(self):
        self.frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.running = False
        self.thread = None
        self.label = ""
        self.state = None
        self.formatter = None
        self.lock = threading.Lock()

    @staticmethod
    def enabled():
        """Progress goes to stderr and only when stderr is a terminal (stdout stays data)."""
        try:
            return sys.stderr.isatty()
        except (AttributeError, ValueError):
            return False

    def start(self, label, state=None, up_tokens=0, formatter=None):
        """Start the spinner. `formatter(label, state)`, when given, builds the whole text after the spinner frame.
        If omitted with `state`, `live.live_line()` renders the active prefill/decode line."""
        if not self.enabled():
            return
        self.label = label
        self.state = state
        self.formatter = formatter
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        i = 0
        while self.running:
            with self.lock:
                lbl = self.label
            with colors.live():  # stderr's own colors, not the report format's
                if self.formatter:
                    lbl = self.formatter(lbl, self.state)
                elif self.state:
                    from . import live
                    lbl = live.live_line(lbl, self.state, time.time())
                # Autowrap off while drawing: a line wider than the terminal is clipped, never wrapped (a wrapped line
                # leaves a copy behind on every redraw, since \r only returns to the start of the last row).
                sys.stderr.write(f"\r\033[?7l\033[K{Clr.BCYN}{self.frames[i % len(self.frames)]}{Clr.RST} {lbl}\033[?7h")
            sys.stderr.flush()
            i += 1
            time.sleep(0.08)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            self.thread = None
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
