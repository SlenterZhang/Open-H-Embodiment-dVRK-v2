"""YAML parsing and validation for acquisition configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dvrk_lerobot.config.model import (
    SIDES,
    AlignmentConfig,
    AppConfig,
    ArmConfig,
    CameraConfig,
    DatasetConfig,
)

X264_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
}


class ConfigError(ValueError):
    """Configuration file is missing a value or contains an invalid one."""


def load_config(path: str | Path) -> AppConfig:
    """Read *path* and return a validated configuration."""
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"configuration not found: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source}: {exc}") from exc
    return parse_config(raw)


def parse_config(raw: Any) -> AppConfig:
    """Validate an already parsed YAML mapping."""
    root = _mapping(raw, "root")
    _only(root, {"dataset", "arms", "cameras", "alignment"}, "root")

    dataset = _parse_dataset(_mapping(root.get("dataset"), "dataset"))
    arms = _parse_arms(root.get("arms"))
    cameras = _parse_cameras(root.get("cameras"), dataset.fps)
    alignment = _parse_alignment(_mapping(root.get("alignment"), "alignment"))
    return AppConfig(arms=arms, cameras=cameras, alignment=alignment, dataset=dataset)


def _parse_dataset(raw: dict[str, Any]) -> DatasetConfig:
    _only(raw, {"fps", "robot_type", "video_codec", "video_crf", "video_preset"}, "dataset")
    fps = _integer(raw.get("fps"), "dataset.fps", 1, 240)
    codec = _string(raw.get("video_codec"), "dataset.video_codec")
    if codec != "h264":
        raise ConfigError("dataset.video_codec must be 'h264'")
    preset = _string(raw.get("video_preset"), "dataset.video_preset")
    if preset not in X264_PRESETS:
        raise ConfigError(f"dataset.video_preset must be one of {sorted(X264_PRESETS)}")
    return DatasetConfig(
        fps=fps,
        robot_type=_string(raw.get("robot_type"), "dataset.robot_type"),
        video_codec=codec,
        video_crf=_integer(raw.get("video_crf"), "dataset.video_crf", 0, 51),
        video_preset=preset,
    )


def _parse_arms(raw: Any) -> tuple[ArmConfig, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("arms must be a non-empty list")
    arms: list[ArmConfig] = []
    for index, item in enumerate(raw):
        where = f"arms[{index}]"
        arm = _mapping(item, where)
        _only(arm, {"name", "js_topic", "cp_topic", "jaw_topic"}, where)
        name = _string(arm.get("name"), f"{where}.name")
        if not name.replace("_", "").isalnum():
            raise ConfigError(f"{where}.name must contain only letters, digits, and underscores")
        arms.append(
            ArmConfig(
                name=name,
                js_topic=_string(arm.get("js_topic"), f"{where}.js_topic"),
                cp_topic=_string(arm.get("cp_topic"), f"{where}.cp_topic"),
                jaw_topic=_string(arm.get("jaw_topic"), f"{where}.jaw_topic"),
            )
        )
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise ConfigError("arm names must be unique")
    return tuple(arms)


def _parse_cameras(raw: Any, fps: int) -> dict[str, CameraConfig]:
    mapping = _mapping(raw, "cameras")
    _only(mapping, set(SIDES), "cameras")
    cameras: dict[str, CameraConfig] = {}
    for side in SIDES:
        where = f"cameras.{side}"
        item = _mapping(mapping.get(side), where)
        _only(item, {"device", "width", "height", "fps", "fourcc", "crf"}, where)
        width = _integer(item.get("width"), f"{where}.width", 2, 8192)
        height = _integer(item.get("height"), f"{where}.height", 2, 8192)
        if width % 2 or height % 2:
            raise ConfigError(f"{where} width and height must be even for H.264 yuv420p")
        camera_fps = _integer(item.get("fps"), f"{where}.fps", 1, 240)
        if camera_fps != fps:
            raise ConfigError(
                f"{where}.fps ({camera_fps}) must equal dataset.fps ({fps}); "
                "the exporter writes one output frame per stereo pair"
            )
        fourcc = _string(item.get("fourcc"), f"{where}.fourcc").upper()
        if len(fourcc) != 4:
            raise ConfigError(f"{where}.fourcc must have exactly four characters")
        cameras[side] = CameraConfig(
            side=side,
            device=_string(item.get("device"), f"{where}.device"),
            width=width,
            height=height,
            fps=camera_fps,
            fourcc=fourcc,
            crf=_integer(item.get("crf"), f"{where}.crf", 0, 51),
        )
    if cameras["left"].device == cameras["right"].device:
        raise ConfigError("left and right cameras must use different devices")
    return cameras


def _parse_alignment(raw: dict[str, Any]) -> AlignmentConfig:
    _only(raw, {"stereo_tolerance_ms", "robot_tolerance_ms"}, "alignment")
    stereo = _number(raw.get("stereo_tolerance_ms"), "alignment.stereo_tolerance_ms", 0.0, 1000.0)
    robot = _number(raw.get("robot_tolerance_ms"), "alignment.robot_tolerance_ms", 0.0, 1000.0)
    return AlignmentConfig(stereo_tolerance_ms=stereo, robot_tolerance_ms=robot)


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    return value


def _only(mapping: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"{where} has unknown keys: {', '.join(unknown)}")


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")
    return value.strip()


def _integer(value: Any, where: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer")
    if not low <= value <= high:
        raise ConfigError(f"{where} must be in [{low}, {high}]")
    return value


def _number(value: Any, where: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number")
    number = float(value)
    if not low <= number <= high:
        raise ConfigError(f"{where} must be in [{low}, {high}]")
    return number
