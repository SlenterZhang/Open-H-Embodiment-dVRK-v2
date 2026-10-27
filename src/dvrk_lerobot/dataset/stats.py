"""Numeric and video statistics compatible with LeRobot v2.1 metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

VIDEO_STAT_MAX_FRAMES = 1000


def numeric_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """Compute min/max/mean/std/count over frame-major *values*."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim == 0:
        raise ValueError("stats need at least one dimension")
    keepdims = data.ndim == 1
    return {
        "min": np.min(data, axis=0, keepdims=keepdims),
        "max": np.max(data, axis=0, keepdims=keepdims),
        "mean": np.mean(data, axis=0, keepdims=keepdims),
        "std": np.std(data, axis=0, keepdims=keepdims),
        "count": np.array([len(data)], dtype=np.int64),
    }


def video_stats(paths: Sequence[Path]) -> dict[str, np.ndarray]:
    """Channel statistics in [0, 1], shape (3, 1, 1), from sampled frames."""
    import cv2

    if not paths:
        raise ValueError("no video frames to summarize")
    indices = _sample_indices(len(paths))
    frames: list[np.ndarray] = []
    for index in indices:
        image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode frame: {paths[index]}")
        image = image[:, :, ::-1]
        height, width = image.shape[:2]
        if max(height, width) >= 300:
            factor = max(1, int(max(height, width) / 150))
            image = image[::factor, ::factor]
        frames.append(image.astype(np.float32) / 255.0)
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f"sampled frames have inconsistent shapes: {sorted(shapes)}")
    data = np.stack(frames, axis=0)
    reduced = {
        "min": data.min(axis=(0, 1, 2), keepdims=True),
        "max": data.max(axis=(0, 1, 2), keepdims=True),
        "mean": data.mean(axis=(0, 1, 2), keepdims=True),
        "std": data.std(axis=(0, 1, 2), keepdims=True),
    }
    stats = {key: value.reshape(3, 1, 1) for key, value in reduced.items()}
    stats["count"] = np.array([len(frames)], dtype=np.int64)
    return stats


def aggregate(
    episodes: Sequence[Mapping[str, Mapping[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    """Pool per-episode statistics into one dataset-level table."""
    if not episodes:
        return {}
    keys = set(episodes[0])
    for index, stats in enumerate(episodes[1:], start=1):
        if set(stats) != keys:
            raise ValueError(f"episode {index} has a different set of stats")
    return {key: _aggregate_feature([stats[key] for stats in episodes]) for key in sorted(keys)}


def _aggregate_feature(stats: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    means = np.stack([item["mean"] for item in stats])
    variances = np.stack([np.square(item["std"]) for item in stats])
    counts = np.stack([item["count"] for item in stats]).astype(np.float64)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)
    total_count = float(counts.sum())
    total_mean = (means * counts).sum(axis=0) / total_count
    delta = means - total_mean
    total_variance = ((variances + np.square(delta)) * counts).sum(axis=0) / total_count
    return {
        "min": np.min(np.stack([item["min"] for item in stats]), axis=0),
        "max": np.max(np.stack([item["max"] for item in stats]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": np.array([int(total_count)], dtype=np.int64),
    }


def to_jsonable(stats: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
    """Convert numpy statistics to JSON-safe nested lists."""
    payload: dict[str, dict[str, Any]] = {}
    for key, values in stats.items():
        converted: dict[str, Any] = {}
        for name, array in values.items():
            data = np.asarray(array, dtype=np.float64)
            if not np.all(np.isfinite(data)):
                raise ValueError(f"non-finite {name} for feature {key}")
            converted[name] = data.tolist() if name != "count" else [int(v) for v in data.ravel()]
        payload[key] = converted
    return payload


def _sample_indices(length: int) -> list[int]:
    minimum = min(100, length)
    count = max(minimum, min(int(length**0.75), VIDEO_STAT_MAX_FRAMES))
    count = min(count, length)
    return sorted({round(value) for value in np.linspace(0, length - 1, count)})
