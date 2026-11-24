"""Write one LeRobot v2.1 episode table to parquet."""

from __future__ import annotations

import os
from contextlib import closing
from itertools import islice
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dvrk_lerobot.dataset.depth import read_frames as read_depth_frames


def write_episode_table(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray,
    episode_index: int,
    task_index: int,
    index_offset: int,
    depth_source: Path | None = None,
    depth_indices: list[int] | None = None,
    depth_shape: tuple[int, int] | None = None,
) -> None:
    """Build the fixed LeRobot frame columns and write compressed parquet."""
    frames = len(state)
    if state.shape != action.shape or len(timestamp) != frames:
        raise ValueError("state, action, and timestamp lengths disagree")
    if depth_source is not None:
        if depth_shape is None or depth_indices is None or len(depth_indices) != frames:
            raise ValueError("depth source, shape, and one frame index per row are required")
        height, width = depth_shape
        source = read_depth_frames(depth_source, depth_indices, width, height)
        with closing(source):
            _write_depth_batches(
                path, source, frames, state, action, timestamp,
                episode_index, task_index, index_offset,
            )
        return
    table = _table(state, action, timestamp, episode_index, task_index, index_offset)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _table(
    state: np.ndarray, action: np.ndarray, timestamp: np.ndarray,
    episode_index: int, task_index: int, index_offset: int,
    depth_values: np.ndarray | None = None, frame_offset: int = 0,
) -> pa.Table:
    frames = len(state)
    dimension = int(state.shape[1])
    arrays = [
        _vector(state, dimension),
        _vector(action, dimension),
    ]
    names = ["observation.state", "action"]
    if depth_values is not None:
        height, width = depth_values.shape[1:]
        rows = _vector(depth_values.reshape(-1, width), width)
        arrays.append(pa.FixedSizeListArray.from_arrays(rows, height))
        names.append("observation.depth.external")
    arrays.extend([
        pa.array(np.asarray(timestamp, dtype=np.float32), type=pa.float32()),
        pa.array(np.arange(frame_offset, frame_offset + frames, dtype=np.int64), type=pa.int64()),
        pa.array(np.full(frames, episode_index, dtype=np.int64), type=pa.int64()),
        pa.array(np.arange(index_offset, index_offset + frames, dtype=np.int64), type=pa.int64()),
        pa.array(np.full(frames, task_index, dtype=np.int64), type=pa.int64()),
    ])
    names.extend([
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ])
    return pa.Table.from_arrays(arrays, names=names)


def _write_depth_batches(
    path: Path, source, frames: int, state: np.ndarray, action: np.ndarray,
    timestamp: np.ndarray, episode_index: int, task_index: int, index_offset: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    offset = 0
    try:
        while offset < frames:
            batch = list(islice(source, min(8, frames - offset)))
            if not batch:
                raise ValueError("depth video ended before all aligned rows")
            stop = offset + len(batch)
            values = np.stack(batch).astype(np.float32)
            table = _table(
                state[offset:stop], action[offset:stop], timestamp[offset:stop],
                episode_index, task_index, index_offset + offset, values, offset,
            )
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            offset = stop
    finally:
        if writer is not None:
            writer.close()
    os.replace(temporary, path)


def _vector(values: np.ndarray, dimension: int) -> pa.FixedSizeListArray:
    data = np.ascontiguousarray(values, dtype=np.float32)
    flat = pa.array(data.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dimension)
