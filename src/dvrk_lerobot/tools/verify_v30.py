"""Read-only structural checks for an exported LeRobot v3.0 dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from dvrk_lerobot.dataset import v30, video


class VerifyV30Error(RuntimeError):
    """The dataset does not satisfy the expected v3.0 structure."""


def verify_dataset_v30(root: Path) -> dict[str, Any]:
    """Validate info.json, tasks/episodes parquet, data slices, and video spans."""
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != v30.CODEBASE_VERSION:
        raise VerifyV30Error(f"unexpected codebase_version: {info.get('codebase_version')}")
    fps = int(info["fps"])
    features = info["features"]
    dimension = int(features["observation.state"]["shape"][0])
    if list(features["action"]["names"]) != list(features["observation.state"]["names"]):
        raise VerifyV30Error("action names differ from observation.state")
    tasks = pq.read_table(root / v30.TASKS_PATH)
    task_indices = set(tasks["task_index"].to_pylist())
    if b"pandas" not in (tasks.schema.metadata or {}):
        raise VerifyV30Error("tasks.parquet lacks the pandas 'task' index")
    episodes = pq.read_table(root / "meta" / "episodes").to_pylist()
    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]

    offset = 0
    for position, episode in enumerate(episodes):
        index, length = int(episode["episode_index"]), int(episode["length"])
        if index != position or episode["dataset_from_index"] != offset:
            raise VerifyV30Error(f"episode {index}: index or dataset_from_index out of order")
        if episode["dataset_to_index"] - offset != length:
            raise VerifyV30Error(f"episode {index}: dataset_to_index disagrees with length")
        data = root / info["data_path"].format(
            chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
        )
        _check_slice(pq.read_table(data), episode, dimension, fps, task_indices)
        for key in video_keys:
            _check_video(root, info, features[key], episode, key, length / fps)
        offset += length

    if offset != int(info["total_frames"]) or len(episodes) != int(info["total_episodes"]):
        raise VerifyV30Error("info.json totals do not match episodes metadata")
    if len(task_indices) != int(info["total_tasks"]):
        raise VerifyV30Error("info.json total_tasks does not match tasks.parquet")
    return {
        "root": str(root),
        "format": v30.CODEBASE_VERSION,
        "episodes": len(episodes),
        "frames": offset,
        "tasks": len(task_indices),
        "fps": fps,
        "state_dimension": dimension,
    }


def _check_slice(table, episode: dict, dimension: int, fps: int, task_indices: set) -> None:
    index, length = int(episode["episode_index"]), int(episode["length"])
    rows = table.filter(table["episode_index"].to_numpy() == index)
    if rows.num_rows != length:
        raise VerifyV30Error(f"episode {index}: {rows.num_rows} rows, expected {length}")
    if rows.schema.field("observation.state").type.list_size != dimension:
        raise VerifyV30Error(f"episode {index}: state is not a fixed-size list")
    start = int(episode["dataset_from_index"])
    if not np.array_equal(rows["index"].to_numpy(), np.arange(start, start + length)):
        raise VerifyV30Error(f"episode {index}: global index does not match dataset_from_index")
    if not np.array_equal(rows["frame_index"].to_numpy(), np.arange(length)):
        raise VerifyV30Error(f"episode {index}: frame_index column is not 0..N-1")
    if not np.allclose(rows["timestamp"].to_numpy(), np.arange(length) / fps, atol=1e-4):
        raise VerifyV30Error(f"episode {index}: timestamps are not k/fps")
    if not set(rows["task_index"].to_pylist()) <= task_indices:
        raise VerifyV30Error(f"episode {index}: unknown task_index")


def _check_video(root: Path, info: dict, feature: dict, episode: dict, key: str, span: float):
    index = episode["episode_index"]
    relative = info["video_path"].format(
        video_key=key,
        chunk_index=episode[f"videos/{key}/chunk_index"],
        file_index=episode[f"videos/{key}/file_index"],
    )
    if not (root / relative).is_file():
        raise VerifyV30Error(f"missing video: {relative}")
    probed = video.probe(root / relative)
    if (probed["width"], probed["height"]) != (feature["shape"][1], feature["shape"][0]):
        raise VerifyV30Error(f"{relative}: unexpected video dimensions")
    start, end = episode[f"videos/{key}/from_timestamp"], episode[f"videos/{key}/to_timestamp"]
    # Every frame timestamp k/fps (k < length) must fall inside [from, to).
    if end - start < span - 1e-4:
        raise VerifyV30Error(f"episode {index}: {key} span {end - start:.4f}s < {span:.4f}s")
