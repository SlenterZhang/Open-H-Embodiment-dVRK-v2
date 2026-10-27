"""Stereo pairing and robot-state alignment."""

from pathlib import Path

import numpy as np
import pytest

from dvrk_lerobot.acquire.session import Frame, RobotSample, Session
from dvrk_lerobot.align.sync import AlignOptions, align_session, pair_frames
from dvrk_lerobot.config import ArmConfig

ARM = ArmConfig("PSM1", "/PSM1/measured_js", "/PSM1/measured_cp")
OPTIONS = AlignOptions(fps=10, stereo_tolerance_ms=15.0, robot_tolerance_ms=25.0)


def _frame(ts_ns: int, name: str) -> Frame:
    return Frame(ts_ns=ts_ns, path=Path(f"/tmp/{name}.jpg"))


def _sample(ts_ns: int, kind: str, values: list[float]) -> RobotSample:
    return RobotSample(t_ns=ts_ns, stamp_ns=ts_ns, kind=kind, arm="PSM1", values=tuple(values))


def _session() -> Session:
    left = [_frame(0, "l0"), _frame(100_000_000, "l1"), _frame(200_000_000, "l2")]
    right = [_frame(5_000_000, "r0"), _frame(105_000_000, "r1"), _frame(205_000_000, "r2")]
    robot = []
    for index, timestamp in enumerate((0, 100_000_000, 200_000_000)):
        robot.append(_sample(timestamp + 1_000_000, "jp", [index, index + 0.5]))
        robot.append(_sample(timestamp + 2_000_000, "cp", [index, 0, 0, 0, 0, 0, 1]))
    meta = {"task": "test", "joint_names": {"PSM1": ["j1", "j2"]}}
    return Session(
        root=Path("/tmp/session"), meta=meta, robot=robot, frames={"left": left, "right": right}
    )


def test_pair_frames_are_greedy_and_one_to_one() -> None:
    left = [_frame(0, "a"), _frame(40_000_000, "b")]
    right = [_frame(5_000_000, "c"), _frame(45_000_000, "d")]
    pairs = pair_frames(left, right, 15_000_000)
    assert [(a.ts_ns, b.ts_ns) for a, b in pairs] == [
        (0, 5_000_000),
        (40_000_000, 45_000_000),
    ]


def test_alignment_builds_state_action_and_timestamps() -> None:
    episode = align_session(_session(), [ARM], OPTIONS)
    assert episode.state.shape == (3, 9)
    assert episode.state_names[0] == "PSM1.jp.j1"
    assert episode.state_names[-1] == "PSM1.cp.qw"
    assert np.allclose(episode.action[:-1], episode.state[1:])
    assert np.allclose(episode.action[-1], episode.state[-1])
    assert np.allclose(episode.timestamps, [0.0, 0.1, 0.2], atol=1e-6)
    assert len(episode.left) == 3
    assert any("aligned frames: 3" in note for note in episode.notes)


def test_alignment_skips_frames_without_robot_data() -> None:
    session = _session()
    session.robot = [
        sample
        for sample in session.robot
        if not (sample.kind == "jp" and sample.stamp_ns == 101_000_000)
    ]
    episode = align_session(session, [ARM], OPTIONS)
    assert len(episode.state) == 2
    assert any("dropped for robot: 1" in note for note in episode.notes)


def test_alignment_requires_two_frames() -> None:
    session = _session()
    session.frames["right"] = session.frames["right"][:1]
    with pytest.raises(ValueError, match="fewer than two"):
        align_session(session, [ARM], OPTIONS)


def test_missing_cp_samples_are_rejected() -> None:
    session = _session()
    session.robot = [sample for sample in session.robot if sample.kind != "cp"]
    with pytest.raises(ValueError, match="no cp samples"):
        align_session(session, [ARM], OPTIONS)
