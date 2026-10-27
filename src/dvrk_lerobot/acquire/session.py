"""Raw on-disk session: robot JSONL plus timestamped stereo JPEG frames."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

RAW_FORMAT = "dvrk-raw-v1"
SIDES = ("left", "right")


@dataclass(frozen=True)
class Frame:
    """One captured camera frame and its wall-clock timestamp."""

    ts_ns: int
    path: Path


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
        root.mkdir(parents=True)
        for side in SIDES:
            (root / "frames" / side).mkdir(parents=True)
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
    frames = {side: _load_frames(root / "frames" / side) for side in SIDES}
    return Session(root=root, meta=meta, robot=robot, frames=frames)


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


def _load_frames(directory: Path) -> list[Frame]:
    if not directory.is_dir():
        raise ValueError(f"missing frame directory: {directory}")
    frames: list[Frame] = []
    for path in directory.glob("*.jpg"):
        try:
            ts_ns = int(path.stem)
        except ValueError as exc:
            raise ValueError(f"unexpected frame name: {path.name}") from exc
        frames.append(Frame(ts_ns=ts_ns, path=path))
    if not frames:
        raise ValueError(f"no frames in {directory}")
    frames.sort(key=lambda frame: frame.ts_ns)
    for previous, current in pairwise(frames):
        if current.ts_ns <= previous.ts_ns:
            raise ValueError(f"frame timestamps are not increasing in {directory}")
    return frames


def _write_meta(root: Path, meta: dict[str, Any]) -> None:
    path = root / "meta.json"
    temporary = root / "meta.json.tmp"
    temporary.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
