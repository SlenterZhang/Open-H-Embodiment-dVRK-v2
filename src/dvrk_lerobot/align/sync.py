"""Place raw robot samples and stereo frames on one fixed-rate slot grid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dvrk_lerobot.acquire.session import SIDES, RobotSample, Session
from dvrk_lerobot.config import ArmConfig
from dvrk_lerobot.dataset.schema import POSE_NAMES, state_names

KINDS = ("jp", "jaw", "cp")  # per-arm state blocks, in observation.state order


@dataclass(frozen=True)
class AlignOptions:
    """Target rate and the tolerances the alignment report counts against."""

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
    videos: dict[str, Path]
    left: list[int]
    right: list[int]
    state_names: list[str]
    notes: list[str]


def align_session(session: Session, arms: Sequence[ArmConfig], options: AlignOptions) -> Episode:
    """Fill every grid slot with both eyes and the held robot state; rows are never removed."""
    recorded_fps = int(session.meta.get("fps", options.fps))
    if recorded_fps != options.fps:
        raise ValueError(
            f"{session.root}: recorded at {recorded_fps} fps, convert config uses {options.fps}"
        )
    slot_ns = round(1e9 / options.fps)
    times = {
        side: np.array([frame.ts_ns for frame in session.frames[side]], dtype=np.int64)
        for side in SIDES
    }
    grid = slot_grid(times["left"], times["right"], slot_ns)
    if len(grid) < 2:
        raise ValueError(f"{session.root}: fewer than two grid slots")
    placed = {side: place_frames(times[side], grid, slot_ns) for side in SIDES}

    names = state_names(arms, session.meta.get("joint_names", {}))
    blocks: list[np.ndarray] = []
    ages: list[np.ndarray] = []
    for arm in arms:
        series = _arm_series(session.robot, arm.name)
        for kind in KINDS:
            stamps, values = series[kind]
            index = hold_last_index(stamps, grid)
            blocks.append(values[index])
            ages.append(np.abs(grid - stamps[index]))
    state = np.concatenate(blocks, axis=1).astype(np.float32)
    if state.shape[1] != len(names):
        raise ValueError(
            f"{session.root}: state has {state.shape[1]} values but {len(names)} names"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{session.root}: non-finite robot state")
    action = np.concatenate((state[1:], state[-1:]), axis=0)
    timestamps = (np.arange(len(state), dtype=np.float64) / options.fps).astype(np.float32)
    return Episode(
        session=session.root,
        task=session.task,
        state=state,
        action=action,
        timestamps=timestamps,
        videos=session.videos,
        left=[session.frames["left"][position].index for position in placed["left"][0]],
        right=[session.frames["right"][position].index for position in placed["right"][0]],
        state_names=names,
        notes=_notes(times, placed, np.max(ages, axis=0), options),
    )


def slot_grid(left: np.ndarray, right: np.ndarray, slot_ns: int) -> np.ndarray:
    """Slot instants from where both eyes have started to where the first one ends."""
    # Half a slot early centres frames in their slots, so jitter does not cross a boundary.
    base = min(left[0], right[0]) - slot_ns // 2
    first = (max(left[0], right[0]) - base) // slot_ns
    last = (min(left[-1], right[-1]) - base) // slot_ns
    return base + np.arange(first, last + 1, dtype=np.int64) * slot_ns


def place_frames(
    times: np.ndarray, grid: np.ndarray, slot_ns: int
) -> tuple[np.ndarray, np.ndarray]:
    """Give each slot the first frame inside it, else repeat the frame before it."""
    first = np.searchsorted(times, grid)
    inside = first < len(times)
    inside[inside] = times[first[inside]] < grid[inside] + slot_ns
    return np.where(inside, first, first - 1), ~inside


def hold_last_index(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index of the last sample at or before each instant, or the first sample."""
    return np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 1)


def _arm_series(
    samples: Sequence[RobotSample], arm: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each kind's receive times and values; receive time shares the cameras' clock."""
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in KINDS:
        rows = [sample for sample in samples if sample.arm == arm and sample.kind == kind]
        if not rows:
            raise ValueError(f"no {kind} samples for {arm}")
        rows.sort(key=lambda sample: sample.t_ns)
        times = np.array([sample.t_ns for sample in rows], dtype=np.int64)
        result[kind] = (times, np.array([sample.values for sample in rows], dtype=np.float64))
    if result["cp"][1].shape[1] != len(POSE_NAMES):
        raise ValueError(f"{arm}: measured_cp must have 7 values, not {result['cp'][1].shape[1]}")
    if result["jaw"][1].shape[1] != 1:
        raise ValueError(f"{arm}: jaw/measured_js must have 1 value, not {result['jaw'][1].shape[1]}")
    return result


def _notes(
    times: dict[str, np.ndarray],
    placed: dict[str, tuple[np.ndarray, np.ndarray]],
    robot_age_ns: np.ndarray,
    options: AlignOptions,
) -> list[str]:
    """Report what the grid had to repeat, skip, or hold; nothing here changes the rows."""
    rows = len(robot_age_ns)
    notes = [f"grid: {rows} frames at {options.fps} fps ({rows / options.fps:.2f} s)"]
    for side in SIDES:
        positions, repeats = placed[side]
        unused = len(times[side]) - len(np.unique(positions))
        gap = np.max(np.diff(times[side]), initial=0) / 1e6
        notes.append(
            f"{side}: {len(times[side])} frames captured, {int(repeats.sum())} slots repeated "
            f"({100 * repeats.mean():.1f}%), {unused} frames unused, longest gap {gap:.0f} ms"
        )
    real = ~(placed["left"][1] | placed["right"][1])
    if real.any():
        left, right = (times[side][placed[side][0][real]] for side in SIDES)
        skew = np.abs(left - right) / 1e6
        notes.append(_spread("stereo skew", skew, options.stereo_tolerance_ms))
    notes.append(_spread("robot sample age", robot_age_ns / 1e6, options.robot_tolerance_ms))
    return notes


def _spread(label: str, values_ms: np.ndarray, tolerance_ms: float) -> str:
    over = int(np.count_nonzero(values_ms > tolerance_ms))
    return (
        f"{label}: median {np.median(values_ms):.1f} ms, worst {values_ms.max():.1f} ms, "
        f"{over} frames over {tolerance_ms:g} ms"
    )
