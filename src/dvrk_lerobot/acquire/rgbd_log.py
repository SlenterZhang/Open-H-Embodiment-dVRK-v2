"""Per-frameset timing logs for the D435i."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SENSOR_LOG = "external_sensor_timestamps.csv"
HEADER = "host_ns,color_ms,depth_ms,color_domain,depth_domain,color_frame,depth_frame\n"


class TimestampLog:
    """Write host receive times, plus a CSV of sensor times beside them."""

    def __init__(self, timestamps: Path) -> None:
        self._host = timestamps.open("w", encoding="utf-8")
        self._sensor = timestamps.with_name(SENSOR_LOG).open("w", encoding="utf-8")
        self._sensor.write(HEADER)

    def write(self, host_ns: int, color: Any, depth: Any) -> None:
        self._host.write(f"{host_ns}\n")
        self._sensor.write(
            f"{host_ns},{color.get_timestamp()},{depth.get_timestamp()},"
            f"{color.get_frame_timestamp_domain()},{depth.get_frame_timestamp_domain()},"
            f"{color.get_frame_number()},{depth.get_frame_number()}\n"
        )

    def close(self) -> None:
        self._host.close()
        self._sensor.close()
