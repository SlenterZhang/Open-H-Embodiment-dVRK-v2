"""Export aligned episodes as LeRobot Dataset v3.0 (lerobot 0.6.0) via a v2.1 staging copy."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from dvrk_lerobot.align.sync import Episode
from dvrk_lerobot.config import AppConfig
from dvrk_lerobot.dataset.v30 import convert_v21_to_v30
from dvrk_lerobot.dataset.writer import ExportError, ExportResult, export_dataset


def export_dataset_v30(
    episodes: Sequence[Episode], config: AppConfig, output: Path
) -> ExportResult:
    """Write v2.1 to ``<output>.v2.1``, repack it into *output* as v3.0, then drop the staging copy."""
    destination = output.expanduser().absolute()
    staging = destination.with_name(destination.name + ".v2.1")
    partial = destination.with_name(destination.name + ".partial")
    for path in (destination, partial):
        if path.exists() or path.is_symlink():
            raise ExportError(f"output already exists: {path}")
    legacy = export_dataset(episodes, config, staging)
    try:
        convert_v21_to_v30(legacy.root, partial)
    except Exception as exc:
        raise ExportError(
            f"v3.0 repack failed: {exc}; v2.1 export kept at {staging}, partial at {partial}"
        ) from exc
    os.replace(partial, destination)
    shutil.rmtree(staging)
    return ExportResult(root=destination, episodes=legacy.episodes, frames=legacy.frames)
