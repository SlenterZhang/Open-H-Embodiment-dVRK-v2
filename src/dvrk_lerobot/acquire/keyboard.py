"""Non-blocking single-key input from the terminal."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty


class Keyboard:
    """Read single key presses without waiting for Enter.

    On a terminal, stdin is switched to cbreak mode (Ctrl+C still works) and
    restored on exit. A pipe is read as-is; end of input reads as 'q'.
    """

    def __init__(self, fd: int | None = None) -> None:
        self._fd = sys.stdin.fileno() if fd is None else fd
        self._saved: list | None = None

    def __enter__(self):
        if os.isatty(self._fd):
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def read(self) -> str | None:
        """Return one pressed key in lower case, or None if nothing is waiting."""
        ready, _, _ = select.select([self._fd], [], [], 0)
        if not ready:
            return None
        data = os.read(self._fd, 1)
        return data.decode(errors="ignore").lower() if data else "q"
