"""Write one LeRobot v2.1 episode table to parquet."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_episode_table(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray,
    episode_index: int,
    task_index: int,
    index_offset: int,
) -> None:
    """Build the fixed LeRobot frame columns and write compressed parquet."""
    frames = len(state)
    if state.shape != action.shape or len(timestamp) != frames:
        raise ValueError("state, action, and timestamp lengths disagree")
    dimension = int(state.shape[1])
    arrays = [
        _vector(state, dimension),
        _vector(action, dimension),
        pa.array(np.asarray(timestamp, dtype=np.float32), type=pa.float32()),
        pa.array(np.arange(frames, dtype=np.int64), type=pa.int64()),
        pa.array(np.full(frames, episode_index, dtype=np.int64), type=pa.int64()),
        pa.array(np.arange(index_offset, index_offset + frames, dtype=np.int64), type=pa.int64()),
        pa.array(np.full(frames, task_index, dtype=np.int64), type=pa.int64()),
    ]
    names = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    table = pa.Table.from_arrays(arrays, names=names)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _vector(values: np.ndarray, dimension: int) -> pa.FixedSizeListArray:
    data = np.ascontiguousarray(values, dtype=np.float32)
    flat = pa.array(data.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dimension)
