"""Legacy host timestamps plus one diagnostic CSV row per saved stereo frame."""
from __future__ import annotations

import csv
from dataclasses import fields
from pathlib import Path

from dvrk_lerobot.acquire.v4l2 import FrameTiming


def driver_log_path(timestamps: Path) -> Path:
    return timestamps.with_name(timestamps.stem.removesuffix("_timestamps") + "_driver_timestamps.csv")


class StereoTimestampLog:
    def __init__(self, timestamps: Path) -> None:
        self._host = timestamps.open("w", encoding="utf-8")
        try:
            self._driver = driver_log_path(timestamps).open("w", encoding="utf-8", newline="")
            names = ["frame_index", *[field.name for field in fields(FrameTiming)],
                     "driver_to_dequeue_ns", "encoder_write_start_monotonic_ns",
                     "encoder_write_end_monotonic_ns"]
            self._csv = csv.DictWriter(self._driver, fieldnames=names)
            self._csv.writeheader()
        except BaseException:
            self._host.close()
            if hasattr(self, "_driver"):
                self._driver.close()
            raise

    def write(self, index: int, timing: FrameTiming, encode_start: int, encode_end: int) -> None:
        # Preserve the old after-decode wall-clock timestamps for convert.
        self._host.write(f"{timing.host_decode_complete_ns}\n")
        self._csv.writerow({"frame_index": index, **timing.row(),
                            "encoder_write_start_monotonic_ns": encode_start,
                            "encoder_write_end_monotonic_ns": encode_end})

    def close(self) -> None:
        try:
            self._host.close()
        finally:
            self._driver.close()
