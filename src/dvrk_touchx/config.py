"""YAML configuration for TouchX teleoperation of one PSM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class ConfigError(ValueError):
    """Configuration file is missing a value or contains an invalid one."""


@dataclass(frozen=True)
class TeleopConfig:
    """ROS namespaces of both devices and the TouchX-to-PSM mapping."""

    touchx: str
    psm: str
    rotation: tuple[tuple[float, ...], ...]
    scale: float
    jaw_open: float
    jaw_closed: float
    max_error: float
    max_age: float


def load_config(path: str | Path) -> TeleopConfig:
    """Read and validate a teleoperation YAML file."""
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"configuration not found: {source}")
    return parse_config(yaml.safe_load(source.read_text(encoding="utf-8")))


def parse_config(raw: Any) -> TeleopConfig:
    """Validate an already parsed YAML mapping."""
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    expected = set(TeleopConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ConfigError(
            f"missing keys {sorted(expected - set(raw))}, unknown keys {sorted(set(raw) - expected)}"
        )
    rotation = np.asarray(raw["rotation"], dtype=float)
    if rotation.shape != (3, 3):
        raise ConfigError("rotation must be a 3x3 matrix")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
        raise ConfigError("rotation must be a proper rotation (orthonormal, det +1)")
    for key in ("scale", "max_error", "max_age"):
        if float(raw[key]) <= 0:
            raise ConfigError(f"{key} must be positive")
    if float(raw["jaw_closed"]) >= float(raw["jaw_open"]):
        raise ConfigError("jaw_closed must be smaller than jaw_open")
    return TeleopConfig(
        touchx=str(raw["touchx"]).rstrip("/"),
        psm=str(raw["psm"]).rstrip("/"),
        rotation=tuple(tuple(row) for row in rotation.tolist()),
        scale=float(raw["scale"]),
        jaw_open=float(raw["jaw_open"]),
        jaw_closed=float(raw["jaw_closed"]),
        max_error=float(raw["max_error"]),
        max_age=float(raw["max_age"]),
    )
