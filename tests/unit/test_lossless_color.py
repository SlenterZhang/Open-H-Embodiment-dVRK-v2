"""The raw D435i color video must decode back to the exact BGR frames written."""

import numpy as np

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
