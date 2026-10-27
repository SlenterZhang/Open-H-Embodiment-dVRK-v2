"""End-to-end raw session to LeRobot v2.1 export."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from dvrk_lerobot.acquire.session import RawSession, load_session, video_files
from dvrk_lerobot.align.sync import AlignOptions, align_session
from dvrk_lerobot.config import parse_config
from dvrk_lerobot.dataset.video import finish_encoder, start_encoder
from dvrk_lerobot.dataset.writer import ExportError, export_dataset
from dvrk_lerobot.tools.verify import verify_dataset

FPS = 10
FRAMES = 6


def _config() -> dict:
    return {
        "dataset": {
            "fps": FPS,
            "robot_type": "dvrk",
            "video_codec": "h264",
            "video_crf": 18,
            "video_preset": "medium",
        },
        "arms": [
            {
                "name": "PSM1",
                "js_topic": "/PSM1/measured_js",
                "cp_topic": "/PSM1/measured_cp",
                "jaw_topic": "/PSM1/jaw/measured_js",
            }
        ],
        "cameras": {
            "left": {
                "device": "left",
                "width": 64,
                "height": 48,
                "fps": FPS,
                "fourcc": "MJPG",
                "crf": 18,
            },
            "right": {
                "device": "right",
                "width": 64,
                "height": 48,
                "fps": FPS,
                "fourcc": "MJPG",
                "crf": 18,
            },
        },
        "alignment": {"stereo_tolerance_ms": 15.0, "robot_tolerance_ms": 25.0},
    }


def _make_session(root: Path, task: str, phase: float) -> Path:
    meta = {
        "task": task,
        "fps": FPS,
        "arms": [
            {"name": "PSM1", "js_topic": "/PSM1/measured_js", "cp_topic": "/PSM1/measured_cp"}
        ],
    }
    session = RawSession.create(root, meta)
    base = 1_000_000_000
    stamps = [base + index * (1_000_000_000 // FPS) for index in range(FRAMES)]
    for side, offset in (("left", 0), ("right", 5_000_000)):
        video, timestamps = video_files(root, side)
        process = start_encoder(video, width=64, height=48, fps=FPS, crf=18, preset="ultrafast")
        for index in range(FRAMES):
            process.stdin.write(np.full((48, 64, 3), (index * 30, 80, 120), np.uint8).tobytes())
        finish_encoder(process, "test encode")
        timestamps.write_text("".join(f"{stamp + offset}\n" for stamp in stamps))
    for index, timestamp in enumerate(stamps):
        session.add_robot(
            "PSM1",
            "jp",
            timestamp,
            timestamp,
            [phase + index * 0.1, phase + index * 0.2],
        )
        session.add_robot(
            "PSM1",
            "cp",
            timestamp + 1_000_000,
            timestamp + 1_000_000,
            [phase, index * 0.01, 0, 0, 0, 0, 1],
        )
        session.add_robot("PSM1", "jaw", timestamp, timestamp, [index * 0.05])
    session.finalize(
        status="ok",
        ended_ns=base + FRAMES * 100_000_000,
        frame_counts={"left": FRAMES, "right": FRAMES},
        camera_info={},
        joint_names={"PSM1": ["PSM1_1_joint", "PSM1_2_joint"]},
        frame_ids={"PSM1": "PSM1_base"},
    )
    return root


def test_export_and_verify(tmp_path: Path) -> None:
    config = parse_config(_config())
    options = AlignOptions(fps=FPS, stereo_tolerance_ms=15.0, robot_tolerance_ms=25.0)
    sessions = [
        _make_session(tmp_path / "raw" / "first", "needle transfer", 0.0),
        _make_session(tmp_path / "raw" / "second", "needle transfer", 1.0),
    ]
    episodes = [align_session(load_session(path), config.arms, options) for path in sessions]
    result = export_dataset(episodes, config, tmp_path / "dataset")

    assert result.episodes == 2
    assert result.frames == 2 * FRAMES
    summary = verify_dataset(tmp_path / "dataset")
    assert summary["frames"] == 2 * FRAMES

    info = json.loads((tmp_path / "dataset" / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == 2
    assert info["total_videos"] == 4
    assert info["features"]["observation.images.left"]["shape"] == [48, 64, 3]
    assert info["features"]["observation.state"]["shape"] == [10]

    table = pq.read_table(tmp_path / "dataset" / "data" / "chunk-000" / "episode_000000.parquet")
    assert table.num_rows == FRAMES
    assert table.schema.field("observation.state").type.list_size == 10
    state = np.array(table["observation.state"].to_pylist())
    action = np.array(table["action"].to_pylist())
    assert np.allclose(action[:-1], state[1:])
    assert (
        tmp_path
        / "dataset"
        / "videos"
        / "chunk-000"
        / "observation.images.left"
        / "episode_000000.mp4"
    ).is_file()

    with pytest.raises(ExportError, match="already exists"):
        export_dataset(episodes, config, tmp_path / "dataset")
