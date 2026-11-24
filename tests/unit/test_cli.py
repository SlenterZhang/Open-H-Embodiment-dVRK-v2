"""CLI argument handling and run directory selection."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dvrk_lerobot.acquire.recorder import _resume_number
from dvrk_lerobot.cli import _acquisition_config, _parser, _session_paths
from dvrk_lerobot.config import load_config


def test_run_directory_expands_to_episodes_in_numeric_order(tmp_path: Path) -> None:
    run = tmp_path / "20261027-140000"
    for number in (1, 2, 10):
        (run / f"episode_{number}").mkdir(parents=True)
        (run / f"episode_{number}" / "meta.json").write_text("{}")
    single = run / "episode_2"
    paths = _session_paths([str(run), str(single)])
    assert [(path.name, from_run) for path, from_run in paths] == [
        ("episode_1", True),
        ("episode_2", True),
        ("episode_10", True),
        ("episode_2", False),
    ]


def test_empty_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="neither an episode nor a run"):
        _session_paths([str(tmp_path)])


@pytest.mark.parametrize("command,extra", [
    ("record", ["--task", "needle pick"]),
    ("convert", ["--sessions", "raw", "--output", "dataset"]),
    ("doctor", []),
])
def test_realsense_can_be_skipped_per_command(command: str, extra: list[str]) -> None:
    config_path = Path(__file__).parents[2] / "configs" / "record.yaml"
    enabled = _parser().parse_args([command, "-c", str(config_path), *extra])
    skipped = _parser().parse_args([command, "-c", str(config_path), *extra, "--no-realsense"])
    assert _acquisition_config(enabled).realsense is not None
    assert _acquisition_config(skipped).realsense is None


def test_resume_continues_after_highest_episode(tmp_path: Path) -> None:
    configured = load_config(Path(__file__).parents[2] / "configs" / "record.yaml")
    config = replace(configured, realsense=None)
    meta = {"task": "needle pick", "fps": 30, "arms": [{"name": arm.name} for arm in config.arms]}
    for number in (1, 9, 109):
        episode = tmp_path / f"episode_{number}"
        episode.mkdir()
        (episode / "meta.json").write_text(json.dumps(meta))
    assert _resume_number(tmp_path, config, "needle pick") == 109
    with pytest.raises(ValueError, match="task differs"):
        _resume_number(tmp_path, config, "another task")
    with pytest.raises(ValueError, match="RealSense setting differs"):
        _resume_number(tmp_path, configured, "needle pick")


def test_resume_rejects_missing_run(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[2] / "configs" / "record.yaml")
    with pytest.raises(ValueError, match="does not exist"):
        _resume_number(tmp_path / "missing", config, "needle pick")
