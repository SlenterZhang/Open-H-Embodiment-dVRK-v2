"""H.264 encoding and probing through the system ffmpeg binaries."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class VideoError(RuntimeError):
    """ffmpeg or ffprobe failed, or produced unexpected metadata."""


def start_encoder(
    destination: Path, *, width: int, height: int, fps: int, crf: int, preset: str
) -> subprocess.Popen:
    """Start ffmpeg reading raw BGR frames on stdin and writing H.264 yuv420p."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-framerate", str(fps), "-i", "-",
        # One output frame per input frame: frame i must match timestamp line i.
        "-vsync", "passthrough",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
    ]  # fmt: skip
    if destination.suffix == ".mp4":
        command += ["-movflags", "+faststart"]
    command.append(str(destination))
    try:
        return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise VideoError(f"cannot start ffmpeg for {destination}: {exc}") from exc


def finish_encoder(process: subprocess.Popen, action: str) -> None:
    """Close stdin, wait for ffmpeg to flush, and raise if it failed."""
    _, stderr = process.communicate(timeout=600)
    if process.returncode != 0:
        raise VideoError(f"{action} failed: {_tail(stderr)}")


def read_frames(source: Path, indices: Sequence[int]) -> Iterator[np.ndarray]:
    """Decode *source* sequentially and yield the BGR frames at increasing *indices*."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise VideoError(f"cannot open {source}")
    position = 0
    try:
        for index in indices:
            if index < position:
                raise VideoError(f"{source}: frame indices must increase")
            while position <= index:
                ok, frame = capture.read()
                if not ok:
                    raise VideoError(f"{source}: ends before frame {index}")
                position += 1
            yield frame
    finally:
        capture.release()


def encode(
    source: Path, indices: Sequence[int], destination: Path, *, fps: int, crf: int, preset: str
) -> dict[str, Any]:
    """Re-encode the selected frames of *source* to an H.264 yuv420p MP4."""
    if not indices:
        raise VideoError(f"no frames to encode for {destination}")
    process: subprocess.Popen | None = None
    try:
        for frame in read_frames(source, indices):
            if process is None:
                height, width = frame.shape[:2]
                process = start_encoder(
                    destination, width=width, height=height, fps=fps, crf=crf, preset=preset
                )
            process.stdin.write(frame.data)
    except BrokenPipeError:
        pass  # ffmpeg exited early; finish_encoder reports its error
    except BaseException:
        if process is not None:
            process.kill()
        raise
    finish_encoder(process, f"encode {destination}")
    info = probe(destination)
    if info["codec"] != "h264" or info["pix_fmt"] != "yuv420p":
        raise VideoError(f"{destination}: unexpected codec {info['codec']}/{info['pix_fmt']}")
    if info["fps"] != fps:
        raise VideoError(f"{destination}: encoded at {info['fps']} fps, expected {fps}")
    return info


def count_frames(path: Path) -> int:
    """Count video packets; libx264 writes exactly one packet per frame."""
    command = [
        _binary("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, check=False, timeout=600)
    if result.returncode != 0:
        raise VideoError(f"cannot count frames in {path}: {_tail(result.stderr)}")
    return int(result.stdout.decode().strip() or 0)


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
