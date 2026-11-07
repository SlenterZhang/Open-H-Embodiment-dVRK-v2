"""Slot-grid placement of stereo frames and robot state."""

from pathlib import Path

import numpy as np
import pytest

from dvrk_lerobot.acquire.session import Frame, RobotSample, Session
from dvrk_lerobot.align.sync import AlignOptions, align_session, place_frames, slot_grid
from dvrk_lerobot.config import ArmConfig

ARM = ArmConfig("PSM1", "/PSM1/measured_js", "/PSM1/measured_cp", "/PSM1/jaw/measured_js")
OPTIONS = AlignOptions(fps=10, stereo_tolerance_ms=15.0, robot_tolerance_ms=25.0)
MS = 1_000_000
SLOT = 100 * MS


def _frames(*times_ms: float) -> list[Frame]:
    return [Frame(ts_ns=round(t * MS), index=i) for i, t in enumerate(times_ms)]


def _samples(t_ms: float, index: float, stamp_ms: float | None = None) -> list[RobotSample]:
    t_ns = round(t_ms * MS)
    stamp = t_ns if stamp_ms is None else round(stamp_ms * MS)
    return [
        RobotSample(t_ns, stamp, "jp", "PSM1", (index, index + 0.5)),
        RobotSample(t_ns, stamp, "cp", "PSM1", (index, 0, 0, 0, 0, 0, 1)),
        RobotSample(t_ns, stamp, "jaw", "PSM1", (index * 0.1,)),
    ]


def _session(left=(0, 100, 200), right=(5, 105, 205), robot=None) -> Session:
    # The grid of the default eyes is -50, 50, 150 ms, just after these samples.
    robot = [(-60, 0), (40, 1), (140, 2)] if robot is None else robot
    samples = [sample for row in robot for sample in _samples(*row)]
    return Session(
        root=Path("/tmp/session"),
        meta={"task": "test", "joint_names": {"PSM1": ["j1", "j2"]}},
        robot=samples,
        frames={"left": _frames(*left), "right": _frames(*right)},
        videos={"left": Path("/tmp/left.mkv"), "right": Path("/tmp/right.mkv")},
    )


def _grid_ms(left, right) -> list[float]:
    grid = slot_grid(np.array(left) * MS, np.array(right) * MS, SLOT)
    return (grid / MS).tolist()


def test_alignment_builds_state_action_and_timestamps() -> None:
    episode = align_session(_session(), [ARM], OPTIONS)
    assert episode.state.shape == (3, 10)
    assert episode.state_names[0] == "PSM1.jp.j1"
    assert episode.state_names[2] == "PSM1.jaw"
    assert np.allclose(episode.state[:, 2], [0.0, 0.1, 0.2])
    assert episode.state_names[-1] == "PSM1.cp.qw"
    assert np.allclose(episode.action[:-1], episode.state[1:])
    assert np.allclose(episode.action[-1], episode.state[-1])
    assert np.allclose(episode.timestamps, [0.0, 0.1, 0.2], atol=1e-6)
    assert episode.left == [0, 1, 2]
    assert episode.right == [0, 1, 2]


def test_grid_spans_only_where_both_eyes_have_frames() -> None:
    assert _grid_ms([0, 100, 200, 300], [5, 105]) == [-50, 50]
    assert _grid_ms([0, 100, 200, 300], [130, 230, 330]) == [50, 150, 250]


def test_empty_slot_repeats_the_previous_frame() -> None:
    times = np.array([0, 100, 400, 500]) * MS
    grid = np.array([-50, 50, 150, 250, 350, 450]) * MS
    positions, repeats = place_frames(times, grid, SLOT)
    assert positions.tolist() == [0, 1, 1, 1, 2, 3]
    assert repeats.tolist() == [False, False, True, True, False, False]


def test_second_frame_in_a_slot_is_unused() -> None:
    times = np.array([0, 10, 100]) * MS
    positions, repeats = place_frames(times, np.array([-50, 50]) * MS, SLOT)
    assert positions.tolist() == [0, 2]
    assert not repeats.any()


def test_a_stall_keeps_the_episode_duration() -> None:
    left, right = (0, 100, 500, 600), (5, 105, 505, 605)
    robot = [(-60, 0), (40, 1), (140, 2), (240, 3), (340, 4), (440, 5), (540, 6)]
    episode = align_session(_session(left, right, robot), [ARM], OPTIONS)
    assert episode.left == [0, 1, 1, 1, 1, 2, 3]
    assert episode.right == [0, 1, 1, 1, 1, 2, 3]
    assert np.isclose(episode.timestamps[-1], 0.6)
    assert np.allclose(episode.state[:, 2], np.arange(7) * 0.1)
    assert any("3 slots repeated" in note for note in episode.notes)
    assert any("longest gap 400 ms" in note for note in episode.notes)


def test_state_never_comes_from_after_the_slot_instant() -> None:
    robot = [(-60, 0), (51, 1), (149, 2), (151, 3)]
    episode = align_session(_session(robot=robot), [ARM], OPTIONS)
    assert np.allclose(episode.state[:, 2], [0.0, 0.0, 0.2])


def test_alignment_uses_receive_time_not_the_header_stamp() -> None:
    robot = [(-60, 0, 1000), (40, 1, 1100), (140, 2, 1200)]
    episode = align_session(_session(robot=robot), [ARM], OPTIONS)
    assert np.allclose(episode.state[:, 2], [0.0, 0.1, 0.2])


def test_robot_gap_holds_and_reports_without_removing_rows() -> None:
    episode = align_session(_session(robot=[(-60, 0), (140, 2)]), [ARM], OPTIONS)
    assert len(episode.state) == 3
    assert np.allclose(episode.state[:, 2], [0.0, 0.0, 0.2])
    assert any(
        "robot sample age" in note and "1 frames over 25 ms" in note for note in episode.notes
    )


def test_stereo_skew_is_reported_without_removing_rows() -> None:
    episode = align_session(_session(right=(0, 130, 230)), [ARM], OPTIONS)
    assert episode.left == [0, 1, 2] and episode.right == [0, 1, 2]
    assert any("stereo skew" in note and "2 frames over 15 ms" in note for note in episode.notes)


def test_alignment_requires_two_slots() -> None:
    with pytest.raises(ValueError, match="fewer than two"):
        align_session(_session(right=(5,)), [ARM], OPTIONS)


def test_missing_cp_samples_are_rejected() -> None:
    session = _session()
    session.robot = [sample for sample in session.robot if sample.kind != "cp"]
    with pytest.raises(ValueError, match="no cp samples"):
        align_session(session, [ARM], OPTIONS)


def test_missing_jaw_samples_are_rejected() -> None:
    session = _session()
    session.robot = [sample for sample in session.robot if sample.kind != "jaw"]
    with pytest.raises(ValueError, match="no jaw samples"):
        align_session(session, [ARM], OPTIONS)
