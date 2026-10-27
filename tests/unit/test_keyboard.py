"""Non-blocking key reader."""

import os

from dvrk_lerobot.acquire.keyboard import Keyboard


def test_keys_are_read_one_at_a_time_and_eof_quits() -> None:
    read_fd, write_fd = os.pipe()
    try:
        with Keyboard(read_fd) as keyboard:
            assert keyboard.read() is None
            os.write(write_fd, b"sE")
            assert keyboard.read() == "s"
            assert keyboard.read() == "e"
            assert keyboard.read() is None
            os.close(write_fd)
            assert keyboard.read() == "q"
    finally:
        os.close(read_fd)
