"""End-to-end raw session to LeRobot v3.0 export."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from test_export import FPS, FRAMES, _config, _make_session

from dvrk_lerobot.acquire.session import load_session
from dvrk_lerobot.align.sync import AlignOptions, align_session
from dvrk_lerobot.config import parse_config
from dvrk_lerobot.dataset.writer import ExportError
from dvrk_lerobot.dataset.writer_v30 import export_dataset_v30
from dvrk_lerobot.tools.verify import verify_dataset


def test_export_v30_and_verify(tmp_path: Path) -> None:
    config = parse_config(_config())
    options = AlignOptions(fps=FPS, stereo_tolerance_ms=15.0, robot_tolerance_ms=25.0)
    sessions = [
        _make_session(tmp_path / "raw" / "first", "needle transfer", 0.0),
        _make_session(tmp_path / "raw" / "second", "needle pick", 1.0, stall_slots=2),
    ]
    episodes = [align_session(load_session(path), config.arms, options) for path in sessions]
    root = tmp_path / "dataset"
    result = export_dataset_v30(episodes, config, root)

    assert result.frames == 2 * FRAMES + 2
    assert verify_dataset(root)["frames"] == 2 * FRAMES + 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dataset", "raw"]

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert "total_videos" not in info and "total_chunks" not in info
    assert info["features"]["observation.state"]["fps"] == FPS
    assert info["features"]["observation.images.left"]["info"]["is_depth_map"] is False

    data = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
    assert data["index"].to_pylist() == list(range(2 * FRAMES + 2))
    episodes_meta = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    rows = episodes_meta.to_pylist()
    assert [row["dataset_from_index"] for row in rows] == [0, FRAMES]
    assert [row["tasks"] for row in rows] == [["needle transfer"], ["needle pick"]]
    left = "videos/observation.images.left"
    assert rows[1][f"{left}/from_timestamp"] == pytest.approx(FRAMES / FPS)
    assert np.array(rows[0]["stats/observation.images.left/mean"]).shape == (3, 1, 1)
    assert (root / "videos" / "observation.images.left" / "chunk-000" / "file-000.mp4").is_file()

    with pytest.raises(ExportError, match="already exists"):
        export_dataset_v30(episodes, config, root)
