"""Write meta/*.json and meta/*.jsonl for a LeRobot v2.1 dataset."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dvrk_lerobot.dataset.stats import aggregate, to_jsonable


def write_metadata(
    root: Path,
    *,
    info: Mapping[str, Any],
    tasks: Sequence[tuple[int, str]],
    episodes: Sequence[Mapping[str, Any]],
    episode_stats: Sequence[Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    """Write info.json, tasks.jsonl, episodes.jsonl, and both stats files."""
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    _json(meta / "info.json", dict(info))
    _jsonl(meta / "tasks.jsonl", [{"task_index": index, "task": task} for index, task in tasks])
    _jsonl(meta / "episodes.jsonl", [dict(item) for item in episodes])
    _jsonl(
        meta / "episodes_stats.jsonl",
        [
            {"episode_index": index, "stats": to_jsonable(stats)}
            for index, stats in enumerate(episode_stats)
        ],
    )
    combined = aggregate(episode_stats)
    _json(meta / "stats.json", to_jsonable(combined))


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    _text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _text(path, "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def _text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
