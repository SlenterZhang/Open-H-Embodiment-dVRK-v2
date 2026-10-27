"""Raw session writing, loading, and validation."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dvrk_lerobot.acquire.session import RawSession, load_session

META = {
    "task": "needle transfer",
    "operator": "tester",
    "notes": "",
    "fps": 10,
    "clock": "system",
    "arms": [{"name": "PSM1", "js_topic": "/PSM1/measured_js", "cp_topic": "/PSM1/measured_cp"}],
}


def _jpeg(path: Path) -> None:
    assert cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))


def test_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "session"
    session = RawSession.create(root, META)
    session.add_robot("PSM1", "jp", 1, 2, [0.1, 0.2])
    session.add_robot("PSM1", "cp", 3, 4, [0, 0, 0, 0, 0, 0, 1])
    for side in ("left", "right"):
        _jpeg(root / "frames" / side / "100.jpg")
    session.finalize(
        status="ok",
        ended_ns=200,
        frame_counts={"left": 1, "right": 1},
        camera_info={},
        joint_names={"PSM1": ["j1", "j2"]},
        frame_ids={"PSM1": "PSM1_base"},
    )

    loaded = load_session(root)
    assert loaded.task == "needle transfer"
    assert len(loaded.robot) == 2
    assert loaded.frames["left"][0].ts_ns == 100
    assert loaded.meta["joint_names"]["PSM1"] == ["j1", "j2"]
    assert loaded.meta["robot_counts"] == {"jp": 1, "cp": 1}


def test_existing_directory_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "session"
    root.mkdir()
    with pytest.raises(FileExistsError):
        RawSession.create(root, META)


def test_malformed_robot_line_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "session"
    session = RawSession.create(root, META)
    session.add_robot("PSM1", "jp", 1, 2, [0.1])
    for side in ("left", "right"):
        _jpeg(root / "frames" / side / "100.jpg")
    session.finalize(
        status="ok",
        ended_ns=200,
        frame_counts={"left": 1, "right": 1},
        camera_info={},
        joint_names={},
        frame_ids={},
    )
    with (root / "robot.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"t_ns": 1}) + "\n")
    with pytest.raises(ValueError, match="malformed sample"):
        load_session(root)
