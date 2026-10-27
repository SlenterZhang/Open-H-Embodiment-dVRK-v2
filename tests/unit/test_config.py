"""Configuration parsing and validation."""

from pathlib import Path

import pytest

from dvrk_lerobot.config import ConfigError, load_config, parse_config

ROOT = Path(__file__).resolve().parents[2]


def _raw() -> dict:
    return {
        "dataset": {
            "fps": 30,
            "robot_type": "dvrk",
            "video_codec": "h264",
            "video_crf": 18,
            "video_preset": "medium",
        },
        "arms": [
            {"name": "PSM1", "js_topic": "/PSM1/measured_js", "cp_topic": "/PSM1/measured_cp"},
        ],
        "cameras": {
            "left": {
                "device": "/dev/video0",
                "width": 640,
                "height": 480,
                "fps": 30,
                "fourcc": "MJPG",
                "jpeg_quality": 95,
            },
            "right": {
                "device": "/dev/video2",
                "width": 640,
                "height": 480,
                "fps": 30,
                "fourcc": "MJPG",
                "jpeg_quality": 95,
            },
        },
        "alignment": {"stereo_tolerance_ms": 15.0, "robot_tolerance_ms": 25.0},
    }


def test_repository_config_is_valid() -> None:
    config = load_config(ROOT / "configs" / "record.yaml")
    assert config.dataset.fps == 30
    assert config.arms[0].name == "PSM1"
    assert config.cameras["left"].device == "/dev/video0"


def test_unknown_key_is_rejected() -> None:
    raw = _raw()
    raw["dataset"]["extra"] = True
    with pytest.raises(ConfigError, match="unknown keys"):
        parse_config(raw)


def test_camera_rate_must_match_dataset() -> None:
    raw = _raw()
    raw["cameras"]["right"]["fps"] = 15
    with pytest.raises(ConfigError, match="must equal dataset.fps"):
        parse_config(raw)


def test_odd_frame_size_is_rejected() -> None:
    raw = _raw()
    raw["cameras"]["left"]["width"] = 641
    with pytest.raises(ConfigError, match="must be even"):
        parse_config(raw)


def test_duplicate_arm_names_are_rejected() -> None:
    raw = _raw()
    raw["arms"].append(dict(raw["arms"][0]))
    with pytest.raises(ConfigError, match="unique"):
        parse_config(raw)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
