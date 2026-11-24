"""Lossless uint16 depth frames stored as FFV1 gray16le video."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from dvrk_lerobot.dataset.video import VideoError, _binary


def start_encoder(path: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{width}x{height}",
        "-framerate", str(fps), "-i", "-", "-vsync", "passthrough",
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray16le", str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def read_frames(source: Path, indices: Sequence[int], width: int, height: int) -> Iterator[np.ndarray]:
    """Decode selected raw depth frames without reducing them to eight bits."""
    if not indices:
        return
    frame_size = width * height * 2
    command = [
        _binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(source), "-f", "rawvideo", "-pix_fmt", "gray16le", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    position = 0
    try:
        for index in indices:
            if index < position:
                raise VideoError(f"{source}: depth frame indices must increase")
            frame = b""
            while position <= index:
                frame = process.stdout.read(frame_size)
                if len(frame) != frame_size:
                    raise VideoError(f"{source}: ends before depth frame {index}")
                position += 1
            yield np.frombuffer(frame, dtype="<u2").reshape(height, width)
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if process.stderr is not None:
            process.stderr.close()


def probe(source: Path) -> dict:
    """Verify that the raw depth stream retained its lossless pixel format."""
    from dvrk_lerobot.dataset.video import probe as probe_video

    info = probe_video(source)
    if info["codec"] != "ffv1" or info["pix_fmt"] not in ("gray16le", "gray16be"):
        raise VideoError(f"{source}: expected FFV1 16-bit grayscale depth")
    return info
