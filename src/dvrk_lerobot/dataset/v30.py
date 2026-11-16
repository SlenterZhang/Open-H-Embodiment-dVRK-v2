"""Repack an exported LeRobot v2.1 dataset into the v3.0 layout read by lerobot 0.6.0.

Mirrors lerobot's ``convert_dataset_v21_to_v30`` using only pyarrow and ffmpeg.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dvrk_lerobot.dataset.v30_video import concat_videos, duration_s

CODEBASE_VERSION = "v3.0"
CHUNKS_SIZE = 1000
DATA_FILE_SIZE_IN_MB = 100
VIDEO_FILE_SIZE_IN_MB = 200
_FILE = "chunk-{chunk_index:03d}/file-{file_index:03d}"
DATA_PATH = "data/" + _FILE + ".parquet"
VIDEO_PATH = "videos/{video_key}/" + _FILE + ".mp4"
EPISODES_PATH = "meta/episodes/" + _FILE + ".parquet"
TASKS_PATH = "meta/tasks.parquet"


def convert_v21_to_v30(source: Path, destination: Path) -> None:
    """Write a v3.0 copy of the v2.1 dataset *source* into the new directory *destination*."""
    info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = _jsonl(source / "meta" / "episodes.jsonl")
    episode_stats = _jsonl(source / "meta" / "episodes_stats.jsonl")
    tasks = _jsonl(source / "meta" / "tasks.jsonl")
    destination.mkdir(parents=True)

    rows = _pack_data(source, destination, len(episodes))
    video_keys = sorted(k for k, ft in info["features"].items() if ft["dtype"] == "video")
    for key in video_keys:
        for row, extra in zip(rows, _pack_videos(source, destination, key), strict=True):
            row.update(extra)
    for row, legacy, stats in zip(rows, episodes, episode_stats, strict=True):
        if not row["episode_index"] == legacy["episode_index"] == stats["episode_index"]:
            raise ValueError("episode indices disagree between data and metadata")
        row.update({"tasks": legacy["tasks"], "length": legacy["length"]})
        for feature, values in stats["stats"].items():
            row.update({f"stats/{feature}/{name}": value for name, value in values.items()})
        row.update({"meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
    _write_parquet(destination / EPISODES_PATH.format(chunk_index=0, file_index=0), rows)
    _write_tasks(destination / TASKS_PATH, tasks)
    shutil.copy2(source / "meta" / "stats.json", destination / "meta" / "stats.json")
    _write_json(destination / "meta" / "info.json", _info(info))


def _pack_data(source: Path, destination: Path, count: int) -> list[dict[str, Any]]:
    paths = sorted((source / "data").glob("*/*.parquet"))
    if len(paths) != count:
        raise ValueError(f"{len(paths)} episode parquet files for {count} episodes")
    rows: list[dict[str, Any]] = []
    tables: list[pa.Table] = []
    chunk, file, size, frames = 0, 0, 0.0, 0
    for episode_index, path in enumerate(paths):
        episode_size = _size_mb(path)
        if size + episode_size >= DATA_FILE_SIZE_IN_MB and tables:
            _write_table(destination / DATA_PATH.format(chunk_index=chunk, file_index=file), tables)
            chunk, file = _next_file(chunk, file)
            size, tables = 0.0, []
        table = pq.read_table(path)
        rows.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": chunk,
                "data/file_index": file,
                "dataset_from_index": frames,
                "dataset_to_index": frames + table.num_rows,
            }
        )
        tables.append(table)
        size += episode_size
        frames += table.num_rows
    if tables:
        _write_table(destination / DATA_PATH.format(chunk_index=chunk, file_index=file), tables)
    return rows


def _pack_videos(source: Path, destination: Path, key: str) -> list[dict[str, Any]]:
    paths = sorted((source / "videos").glob(f"*/{key}/*.mp4"))
    rows: list[dict[str, Any]] = []
    pending: list[Path] = []
    chunk, file, size, duration = 0, 0, 0.0, 0.0

    def flush() -> None:
        target = destination / VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
        concat_videos(pending, target)

    for path in paths:
        episode_size = _size_mb(path)
        if size + episode_size >= VIDEO_FILE_SIZE_IN_MB and pending:
            flush()
            chunk, file = _next_file(chunk, file)
            size, duration, pending = 0.0, 0.0, []
        episode_duration = duration_s(path)
        rows.append(
            {
                f"videos/{key}/chunk_index": chunk,
                f"videos/{key}/file_index": file,
                f"videos/{key}/from_timestamp": duration,
                f"videos/{key}/to_timestamp": duration + episode_duration,
            }
        )
        pending.append(path)
        size += episode_size
        duration += episode_duration
    if pending:
        flush()
    return rows


def _info(legacy: dict[str, Any]) -> dict[str, Any]:
    """Build the v3.0 info.json in the key order lerobot 0.6.0 writes."""
    fps = int(legacy["fps"])
    features = {key: dict(value) for key, value in legacy["features"].items()}
    for feature in features.values():
        if feature["dtype"] == "video":
            info = dict(feature["info"])
            info["is_depth_map"] = info.pop("video.is_depth_map", False)
            feature["info"] = info
        else:
            feature["fps"] = fps
    return {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "features": features,
        "total_episodes": int(legacy["total_episodes"]),
        "total_frames": int(legacy["total_frames"]),
        "total_tasks": int(legacy["total_tasks"]),
        "chunks_size": CHUNKS_SIZE,
        "data_files_size_in_mb": DATA_FILE_SIZE_IN_MB,
        "video_files_size_in_mb": VIDEO_FILE_SIZE_IN_MB,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "robot_type": legacy["robot_type"],
        "splits": legacy["splits"],
    }


def _write_tasks(path: Path, tasks: list[dict[str, Any]]) -> None:
    """Write tasks.parquet with the pandas index metadata lerobot's ``tasks.loc[task]`` needs."""
    table = pa.table(
        {
            "task_index": pa.array([int(t["task_index"]) for t in tasks], pa.int64()),
            "task": pa.array([str(t["task"]) for t in tasks], pa.string()),
        }
    )
    column = {"metadata": None}
    pandas = {
        "index_columns": ["task"],
        "column_indexes": [],
        "columns": [
            {"name": "task_index", "field_name": "task_index", "pandas_type": "int64",
             "numpy_type": "int64", **column},
            {"name": "task", "field_name": "task", "pandas_type": "unicode",
             "numpy_type": "object", **column},
        ],
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.2.3",
    }  # fmt: skip
    table = table.replace_schema_metadata({b"pandas": json.dumps(pandas).encode()})
    pq.write_table(table, path)


def _next_file(chunk: int, file: int) -> tuple[int, int]:
    return (chunk + 1, 0) if file == CHUNKS_SIZE - 1 else (chunk, file + 1)


def _write_table(path: Path, tables: list[pa.Table]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(tables), path)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=4, allow_nan=False) + "\n", encoding="utf-8")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024**2)
