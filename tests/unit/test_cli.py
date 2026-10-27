"""Export input expansion."""

from pathlib import Path

import pytest

from dvrk_lerobot.cli import _session_paths


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
