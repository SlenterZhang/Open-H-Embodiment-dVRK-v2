"""Raw D435i color: lossless mode decodes exactly; the YAML switch selects it."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from dvrk_lerobot.config import ConfigError, load_config
from dvrk_lerobot.dataset.video import finish_encoder, read_frames, start_lossless_encoder


def test_lossless_color_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (48, 64, 3), dtype=np.uint8) for _ in range(5)]
    path = tmp_path / "external.mkv"
    process = start_lossless_encoder(path, width=64, height=48, fps=30)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    finish_encoder(process, "encode test")
    decoded = list(read_frames(path, range(len(frames))))
    assert all(np.array_equal(a, b) for a, b in zip(decoded, frames, strict=True))


def test_color_lossless_setting(tmp_path):
    assert not load_config(Path("configs/record.yaml")).realsense.color_lossless
    raw = yaml.safe_load(Path("configs/record.yaml").read_text())
    raw["realsense"]["color_lossless"] = True
    path = tmp_path / "record.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).realsense.color_lossless
    raw["realsense"]["color_lossless"] = "yes"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="must be a boolean"):
        load_config(path)
