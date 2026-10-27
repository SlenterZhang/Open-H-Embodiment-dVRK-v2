"""Align raw robot samples and stereo frames into fixed-rate episodes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dvrk_lerobot.acquire.session import Frame, RobotSample, Session
from dvrk_lerobot.config import ArmConfig
from dvrk_lerobot.dataset.schema import POSE_NAMES, state_names

# Per-arm state blocks, in observation.state order.
KINDS = ("jp", "jaw", "cp")


@dataclass(frozen=True)
class AlignOptions:
    """Target rate and matching tolerances."""

    fps: int
    stereo_tolerance_ms: float
    robot_tolerance_ms: float


@dataclass
class Episode:
    """One aligned episode, ready to be written to LeRobot format."""

    session: Path
    task: str
    state: np.ndarray
    action: np.ndarray
    timestamps: np.ndarray
    left: list[Path]
    right: list[Path]
    state_names: list[str]
    notes: list[str]


def align_session(session: Session, arms: Sequence[ArmConfig], options: AlignOptions) -> Episode:
    """Pair stereo frames, attach nearest robot state, and build actions."""
    recorded_fps = int(session.meta.get("fps", options.fps))
    if recorded_fps != options.fps:
        raise ValueError(
            f"{session.root}: recorded at {recorded_fps} fps, export config uses {options.fps}"
        )
    tolerance_ns = round(options.stereo_tolerance_ms * 1_000_000)
    pairs = pair_frames(session.frames["left"], session.frames["right"], tolerance_ns)
    if len(pairs) < 2:
        raise ValueError(f"{session.root}: fewer than two stereo pairs")

    joint_names = session.meta.get("joint_names", {})
    names = state_names(arms, joint_names)
    series = {arm.name: _arm_series(session.robot, arm.name) for arm in arms}
    robot_tolerance = round(options.robot_tolerance_ms * 1_000_000)

    rows: list[np.ndarray] = []
    capture_ns: list[int] = []
    left_paths: list[Path] = []
    right_paths: list[Path] = []
    skew_ns: list[int] = []
    missing_robot = 0
    for left, right in pairs:
        timestamp = (left.ts_ns + right.ts_ns) // 2
        vector: list[np.ndarray] = []
        pair_skew = 0
        for arm in arms:
            picked = []
            for kind in KINDS:
                times, values = series[arm.name][kind]
                index = _nearest(times, timestamp, robot_tolerance)
                if index is None:
                    break
                picked.append((int(times[index]), values[index]))
            if len(picked) != len(KINDS):
                vector = []
                break
            vector.extend(values for _, values in picked)
            stamps = [stamp for stamp, _ in picked]
            pair_skew = max(pair_skew, max(stamps) - min(stamps))
        if not vector:
            missing_robot += 1
            continue
        rows.append(np.concatenate(vector).astype(np.float32))
        capture_ns.append(timestamp)
        left_paths.append(left.path)
        right_paths.append(right.path)
        skew_ns.append(pair_skew)

    if len(rows) < 2:
        raise ValueError(f"{session.root}: fewer than two aligned frames")
    state = np.stack(rows, axis=0)
    if state.shape[1] != len(names):
        raise ValueError(
            f"{session.root}: state has {state.shape[1]} values but {len(names)} names"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{session.root}: non-finite robot state")
    action = np.concatenate((state[1:], state[-1:]), axis=0)
    timestamps = (np.arange(len(state), dtype=np.float64) / options.fps).astype(np.float32)
    notes = _notes(
        len(session.frames["left"]), pairs, missing_robot, capture_ns, skew_ns, options.fps
    )
    return Episode(
        session=session.root,
        task=session.task,
        state=state,
        action=action,
        timestamps=timestamps,
        left=left_paths,
        right=right_paths,
        state_names=names,
        notes=notes,
    )


def pair_frames(
    left: Sequence[Frame], right: Sequence[Frame], tolerance_ns: int
) -> list[tuple[Frame, Frame]]:
    """Greedily pair each left frame with an unused nearby right frame."""
    pairs: list[tuple[Frame, Frame]] = []
    index = 0
    for left_frame in left:
        while index < len(right) and right[index].ts_ns < left_frame.ts_ns - tolerance_ns:
            index += 1
        if index >= len(right):
            break
        best = index
        if index + 1 < len(right) and abs(right[index + 1].ts_ns - left_frame.ts_ns) <= abs(
            right[index].ts_ns - left_frame.ts_ns
        ):
            best = index + 1
        if abs(right[best].ts_ns - left_frame.ts_ns) <= tolerance_ns:
            pairs.append((left_frame, right[best]))
            index = best + 1
    return pairs


def _arm_series(
    samples: Sequence[RobotSample], arm: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in KINDS:
        rows = [sample for sample in samples if sample.arm == arm and sample.kind == kind]
        if not rows:
            raise ValueError(f"no {kind} samples for {arm}")
        rows.sort(key=_sample_time)
        times = np.array([_sample_time(sample) for sample in rows], dtype=np.int64)
        values = np.array([sample.values for sample in rows], dtype=np.float64)
        unique, first = np.unique(times, return_index=True)
        result[kind] = (unique, values[first])
    if result["cp"][1].shape[1] != len(POSE_NAMES):
        raise ValueError(f"{arm}: measured_cp must have 7 values, not {result['cp'][1].shape[1]}")
    if result["jaw"][1].shape[1] != 1:
        raise ValueError(f"{arm}: jaw/measured_js must have 1 value, not {result['jaw'][1].shape[1]}")
    return result


def _sample_time(sample: RobotSample) -> int:
    return sample.stamp_ns if sample.stamp_ns > 0 else sample.t_ns


def _nearest(times: np.ndarray, target: int, tolerance: int) -> int | None:
    index = int(np.searchsorted(times, target))
    best: int | None = None
    best_delta = tolerance + 1
    for candidate in (index - 1, index):
        if 0 <= candidate < len(times):
            delta = abs(int(times[candidate]) - target)
            if delta < best_delta:
                best = candidate
                best_delta = delta
    return best if best_delta <= tolerance else None


def _notes(
    left_count: int,
    pairs: Sequence[tuple[Frame, Frame]],
    missing_robot: int,
    capture_ns: Sequence[int],
    skew_ns: Sequence[int],
    fps: int,
) -> list[str]:
    notes = [
        (
            f"left frames: {left_count}, stereo pairs: {len(pairs)}, "
            f"aligned frames: {len(capture_ns)}, dropped for robot: {missing_robot}"
        )
    ]
    if len(capture_ns) >= 2:
        span = (capture_ns[-1] - capture_ns[0]) / 1e9
        expected = (len(capture_ns) - 1) / fps
        if expected > 0 and abs(span - expected) / expected > 0.10:
            notes.append(
                f"capture span {span:.3f}s differs from nominal {expected:.3f}s; "
                "hardware frame drops are possible"
            )
    if skew_ns:
        notes.append(f"median jp/jaw/cp header skew: {np.median(skew_ns) / 1e6:.1f} ms")
    return notes
