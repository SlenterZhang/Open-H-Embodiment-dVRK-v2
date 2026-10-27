"""Raw on-disk session: robot JSONL plus stereo H.264 videos with frame timestamps."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from dvrk_lerobot.dataset.video import count_frames

RAW_FORMAT = "dvrk-raw-v2"
SIDES = ("left", "right")


@dataclass(frozen=True)
class Frame:
    """One captured camera frame: wall-clock timestamp and index in the side's video."""

    ts_ns: int
    index: int


@dataclass(frozen=True)
class RobotSample:
    """One measured_js, jaw/measured_js, or measured_cp sample from ROS 2."""

    t_ns: int
    stamp_ns: int
    kind: str
    arm: str
    values: tuple[float, ...]


@dataclass
class Session:
    """A loaded raw session, ready for alignment."""

    root: Path
    meta: dict[str, Any]
    robot: list[RobotSample]
    frames: dict[str, list[Frame]]
    videos: dict[str, Path]

    @property
    def task(self) -> str:
        return str(self.meta.get("task", ""))


class RawSession:
    """Append-only writer for one raw session directory."""

    def __init__(self, root: Path, meta: dict[str, Any]) -> None:
        self.root = root
        self.meta = meta
        self._handle = (root / "robot.jsonl").open("w", encoding="utf-8", buffering=1)
        self._counts = {"jp": 0, "cp": 0, "jaw": 0}
        self._closed = False

    @classmethod
    def create(cls, root: Path, meta: dict[str, Any]) -> RawSession:
        """Create *root*, its frame directories, and the initial meta.json."""
        if root.exists():
            raise FileExistsError(f"session directory already exists: {root}")
        (root / "video").mkdir(parents=True)
        meta = dict(meta)
        meta["format"] = RAW_FORMAT
        meta["status"] = "recording"
        meta.setdefault("created_ns", 0)
        _write_meta(root, meta)
        return cls(root, meta)

    def add_robot(
        self, arm: str, kind: str, t_ns: int, stamp_ns: int, values: list[float] | tuple[float, ...]
    ) -> None:
        """Append one sample to robot.jsonl."""
        if self._closed:
            raise RuntimeError("session is already closed")
        if kind not in self._counts:
            raise ValueError(f"unknown sample kind: {kind}")
        row = {
            "t_ns": int(t_ns),
            "stamp_ns": int(stamp_ns),
            "kind": kind,
            "arm": arm,
            "values": [float(value) for value in values],
        }
        self._handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._counts[kind] += 1

    def finalize(
        self,
        *,
        status: str,
        ended_ns: int,
        frame_counts: dict[str, int],
        camera_info: dict[str, Any],
        joint_names: dict[str, list[str]],
        frame_ids: dict[str, str],
    ) -> None:
        """Write final counts and metadata, then close the JSONL file."""
        if self._closed:
            return
        self.meta.update(
            {
                "status": status,
                "ended_ns": int(ended_ns),
                "frame_counts": {side: int(frame_counts.get(side, 0)) for side in SIDES},
                "robot_counts": dict(self._counts),
                "camera_info": camera_info,
                "joint_names": joint_names,
                "frame_ids": frame_ids,
            }
        )
        _write_meta(self.root, self.meta)
        self._handle.close()
        self._closed = True

    def close(self) -> None:
        """Close the writer without touching metadata."""
        if not self._closed:
            self._handle.close()
            self._closed = True


def load_session(path: str | Path) -> Session:
    """Read and validate one raw session from disk."""
    root = Path(path)
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        raise ValueError(f"not a raw session (missing meta.json): {root}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != RAW_FORMAT:
        raise ValueError(f"{root}: unsupported raw format {meta.get('format')!r}")
    arms = {str(item["name"]) for item in meta.get("arms", [])}
    robot = _load_robot(root / "robot.jsonl", arms)
    frames = {side: _load_frames(*video_files(root, side)) for side in SIDES}
    videos = {side: video_files(root, side)[0] for side in SIDES}
    return Session(root=root, meta=meta, robot=robot, frames=frames, videos=videos)


def video_files(root: Path, side: str) -> tuple[Path, Path]:
    """Paths of one side's raw video and its per-frame timestamp file."""
    return root / "video" / f"{side}.mkv", root / "video" / f"{side}_timestamps.txt"


def _load_robot(path: Path, arms: set[str]) -> list[RobotSample]:
    if not path.is_file():
        raise ValueError(f"missing robot log: {path}")
    samples: list[RobotSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                sample = RobotSample(
                    t_ns=int(row["t_ns"]),
                    stamp_ns=int(row["stamp_ns"]),
                    kind=str(row["kind"]),
                    arm=str(row["arm"]),
                    values=tuple(float(value) for value in row["values"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: malformed sample: {exc}") from exc
            if sample.kind not in ("jp", "cp", "jaw"):
                raise ValueError(f"{path}:{line_no}: unknown kind {sample.kind!r}")
            if arms and sample.arm not in arms:
                raise ValueError(f"{path}:{line_no}: unknown arm {sample.arm!r}")
            samples.append(sample)
    if not samples:
        raise ValueError(f"{path}: no robot samples")
    return samples


def _load_frames(video: Path, timestamps: Path) -> list[Frame]:
    if not video.is_file() or not timestamps.is_file():
        raise ValueError(f"missing raw video or timestamps: {video}")
    try:
        stamps = [int(line) for line in timestamps.read_text(encoding="utf-8").split()]
    except ValueError as exc:
        raise ValueError(f"{timestamps}: malformed timestamp") from exc
    # After a crash the two can differ by the frames still buffered; keep the common prefix.
    stamps = stamps[: count_frames(video)]
    if not stamps:
        raise ValueError(f"no frames in {video}")
    for previous, current in pairwise(stamps):
        if current <= previous:
            raise ValueError(f"frame timestamps are not increasing in {timestamps}")
    return [Frame(ts_ns=stamp, index=index) for index, stamp in enumerate(stamps)]


def _write_meta(root: Path, meta: dict[str, Any]) -> None:
    path = root / "meta.json"
    temporary = root / "meta.json.tmp"
    temporary.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
