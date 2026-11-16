"""ffmpeg helpers for packing per-episode MP4s into v3.0 multi-episode files."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from dvrk_lerobot.dataset import video


def concat_videos(paths: list[Path], target: Path) -> None:
    """Stream-copy MP4s end to end with ffmpeg's concat demuxer, as lerobot does."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False) as listing:
        listing.write("ffconcat version 1.0\n")
        listing.writelines(f"file '{path.resolve()}'\n" for path in paths)
    command = [
        video._binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "concat", "-safe", "0", "-i", listing.name,
        "-map", "0:v", "-c", "copy", "-movflags", "+faststart", str(target),
    ]  # fmt: skip
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=3600)
    finally:
        Path(listing.name).unlink()
    if result.returncode != 0:
        raise video.VideoError(f"concat {target} failed: {video._tail(result.stderr)}")


def duration_s(path: Path) -> float:
    """Video stream duration in seconds, the value lerobot uses for from/to timestamps."""
    command = [
        video._binary("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration", "-of", "csv=p=0", str(path),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, check=False, timeout=60)
    if result.returncode != 0:
        raise video.VideoError(f"cannot read duration of {path}: {video._tail(result.stderr)}")
    return float(result.stdout.decode().strip())
