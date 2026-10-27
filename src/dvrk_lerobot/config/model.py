"""Immutable configuration objects shared by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

SIDES = ("left", "right")


@dataclass(frozen=True)
class ArmConfig:
    """One dVRK arm and the three CRTK topics it publishes."""

    name: str
    js_topic: str
    cp_topic: str
    jaw_topic: str


@dataclass(frozen=True)
class CameraConfig:
    """One V4L2 device and the format requested from it."""

    side: str
    device: str
    width: int
    height: int
    fps: int
    fourcc: str
    jpeg_quality: int


@dataclass(frozen=True)
class AlignmentConfig:
    """Tolerances used when matching camera frames and robot samples."""

    stereo_tolerance_ms: float
    robot_tolerance_ms: float


@dataclass(frozen=True)
class DatasetConfig:
    """Output rate and H.264 settings for the LeRobot export."""

    fps: int
    robot_type: str
    video_codec: str
    video_crf: int
    video_preset: str


@dataclass(frozen=True)
class AppConfig:
    """Everything needed by the recorder, aligner, and exporter."""

    arms: tuple[ArmConfig, ...]
    cameras: dict[str, CameraConfig]
    alignment: AlignmentConfig
    dataset: DatasetConfig

    @property
    def camera_sides(self) -> tuple[str, str]:
        return SIDES
