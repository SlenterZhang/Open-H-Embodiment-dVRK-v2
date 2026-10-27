"""Raw session writing, loading, and validation."""

import json
from pathlib import Path

import numpy as np
import pytest

from dvrk_lerobot.acquire.session import RawSession, load_session, video_files
from dvrk_lerobot.dataset.video import finish_encoder, start_encoder

META = {
    "task": "needle transfer",
    "notes": "",
    "fps": 10,
    "clock": "system",
    "arms": [{"name": "PSM1", "js_topic": "/PSM1/measured_js", "cp_topic": "/PSM1/measured_cp"}],
}


def _videos(root: Path, frames: int, stamps: list[int]) -> None:
    for side in ("left", "right"):
        video, timestamps = video_files(root, side)
        process = start_encoder(video, width=16, height=16, fps=10, crf=18, preset="ultrafast")
        for _ in range(frames):
            process.stdin.write(np.zeros((16, 16, 3), dtype=np.uint8).tobytes())
        finish_encoder(process, "test encode")
        timestamps.write_text("".join(f"{stamp}\n" for stamp in stamps), encoding="utf-8")


def test_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "session"
    session = RawSession.create(root, META)
    session.add_robot("PSM1", "jp", 1, 2, [0.1, 0.2])
    session.add_robot("PSM1", "cp", 3, 4, [0, 0, 0, 0, 0, 0, 1])
    session.add_robot("PSM1", "jaw", 5, 6, [0.3])
    _videos(root, 2, [100, 200])
    session.finalize(
        status="ok",
        ended_ns=200,
        frame_counts={"left": 2, "right": 2},
        camera_info={},
        joint_names={"PSM1": ["j1", "j2"]},
        frame_ids={"PSM1": "PSM1_base"},
    )

    loaded = load_session(root)
    assert loaded.task == "needle transfer"
    assert len(loaded.robot) == 3
    assert [(frame.ts_ns, frame.index) for frame in loaded.frames["left"]] == [(100, 0), (200, 1)]
    assert loaded.videos["right"] == root / "video" / "right.mkv"
    assert loaded.meta["joint_names"]["PSM1"] == ["j1", "j2"]
    assert loaded.meta["robot_counts"] == {"jp": 1, "cp": 1, "jaw": 1}


def test_existing_directory_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "session"
    root.mkdir()
    with pytest.raises(FileExistsError):
        RawSession.create(root, META)


def test_malformed_robot_line_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "session"
    session = RawSession.create(root, META)
    session.add_robot("PSM1", "jp", 1, 2, [0.1])
    _videos(root, 1, [100])
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


def test_timestamps_are_truncated_to_encoded_frames(tmp_path: Path) -> None:
    root = tmp_path / "session"
    session = RawSession.create(root, META)
    session.add_robot("PSM1", "jp", 1, 2, [0.1])
    session.close()
    _videos(root, 2, [100, 200, 300])
    loaded = load_session(root)
    assert [frame.ts_ns for frame in loaded.frames["left"]] == [100, 200]
