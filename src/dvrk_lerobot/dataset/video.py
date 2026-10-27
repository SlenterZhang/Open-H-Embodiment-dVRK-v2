"""H.264 encoding and probing through the system ffmpeg binaries."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class VideoError(RuntimeError):
    """ffmpeg or ffprobe failed, or produced unexpected metadata."""


def encode(
    frames: Sequence[Path], destination: Path, *, fps: int, crf: int, preset: str
) -> dict[str, Any]:
    """Encode JPEG *frames* to an H.264 yuv420p MP4 and return its metadata."""
    if not frames:
        raise VideoError(f"no frames to encode for {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".frames-", dir=destination.parent) as temporary:
        directory = Path(temporary)
        for index, frame in enumerate(frames):
            os.symlink(frame.resolve(), directory / f"{index:06d}.jpg")
        command = [
            _binary("ffmpeg"),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-framerate",
            str(fps),
            "-i",
            str(directory / "%06d.jpg"),
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        _run(command, f"encode {destination}")
    info = probe(destination)
    if info["codec"] != "h264" or info["pix_fmt"] != "yuv420p":
        raise VideoError(f"{destination}: unexpected codec {info['codec']}/{info['pix_fmt']}")
    if info["fps"] != fps:
        raise VideoError(f"{destination}: encoded at {info['fps']} fps, expected {fps}")
    return info


def probe(path: Path) -> dict[str, Any]:
    """Read width, height, codec, pixel format, fps, and channels."""
    command = [
        _binary("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,pix_fmt,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise VideoError(f"cannot probe {path}: {exc}") from exc
    if result.returncode != 0:
        raise VideoError(f"cannot probe {path}: {_tail(result.stderr)}")
    try:
        stream = json.loads(result.stdout.decode("utf-8"))["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        codec = str(stream["codec_name"])
        pix_fmt = str(stream["pix_fmt"])
        fps = _fps(stream.get("avg_frame_rate", "0/0"))
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VideoError(f"{path}: unusable ffprobe output") from exc
    if width < 1 or height < 1:
        raise VideoError(f"{path}: invalid dimensions {width}x{height}")
    return {
        "width": width,
        "height": height,
        "codec": codec,
        "pix_fmt": pix_fmt,
        "fps": fps,
        "channels": _channels(pix_fmt),
    }


def _run(command: list[str], action: str) -> None:
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=3600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise VideoError(f"{action}: {exc}") from exc
    if result.returncode != 0:
        raise VideoError(f"{action} failed: {_tail(result.stderr)}")


def _binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise VideoError(f"{name} not found on PATH; install ffmpeg")
    return path


def _tail(data: bytes, limit: int = 1200) -> str:
    text = data.decode("utf-8", errors="replace").strip()
    return text[-limit:] if text else "no error output"


def _fps(value: str) -> int:
    numerator, _, denominator = str(value).partition("/")
    try:
        top = float(numerator)
        bottom = float(denominator or 1)
    except ValueError:
        return 0
    return round(top / bottom) if bottom else 0


def _channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "mono" in pix_fmt:
        return 1
    if "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    return 3
