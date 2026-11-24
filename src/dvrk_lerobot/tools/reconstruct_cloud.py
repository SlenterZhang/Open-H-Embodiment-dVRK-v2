"""Export aligned RGB-D recordings to colored PLY files in camera coordinates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

from dvrk_lerobot.dataset import depth, video
from dvrk_lerobot.dataset.cloud import rays, write_ply


def camera_calibration(info: dict) -> dict:
    """Read the matching device's color profile without starting capture."""
    context = rs.context()
    for device in context.query_devices():
        if device.get_info(rs.camera_info.serial_number) != info["serial"]:
            continue
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() != rs.stream.color or profile.fps() != info["fps"]:
                    continue
                p = profile.as_video_stream_profile()
                if (p.width(), p.height()) != (info["width"], info["height"]):
                    continue
                intr = p.get_intrinsics()
                return {
                    "serial": info["serial"], "width": intr.width, "height": intr.height,
                    "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
                    "model": str(intr.model).split(".")[-1], "coeffs": list(intr.coeffs),
                    "source": "current connected device color profile; not recording-time calibration",
                    "read_at_utc": datetime.now(timezone.utc).isoformat(),
                }
    raise RuntimeError("Matching RealSense camera/color profile unavailable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--read-camera-calibration", action="store_true",
                        help="Save current matching camera calibration and exit")
    args = parser.parse_args()
    meta = json.loads((args.episode / "meta.json").read_text())
    info = meta["realsense"]
    if not info["depth_aligned_to_color"]:
        raise ValueError("This exporter requires depth aligned to color")
    if args.read_camera_calibration:
        calibration = camera_calibration(info)
        args.calibration.parent.mkdir(parents=True, exist_ok=True)
        with args.calibration.open("x") as handle:
            json.dump(calibration, handle, indent=2)
        print(json.dumps(calibration, indent=2))
        return
    calibration = json.loads(args.calibration.read_text())
    for key in ("serial", "width", "height"):
        if calibration[key] != info[key]:
            raise ValueError(f"Calibration {key} does not match recording")
    scale = float(info["depth_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid depth scale")
    folder = args.episode / "video"
    depth_path, color_path = folder / "external_depth.mkv", folder / "external.mkv"
    for stream in (depth.probe(depth_path), video.probe(color_path)):
        if (stream["width"], stream["height"]) != (info["width"], info["height"]):
            raise ValueError("Video dimensions do not match calibration")
    timestamps = [int(x) for x in (folder / "external_timestamps.txt").read_text().split()]
    count = len(timestamps)
    if not count or any(video.count_frames(p) != count for p in (depth_path, color_path)):
        raise ValueError("RGB/depth/timestamp frame counts do not match")
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("frame_*.ply")) or (args.output / "manifest.json").exists():
        raise FileExistsError("Output already contains an export; use a fresh directory")
    unit_rays = rays(calibration)
    indices = list(range(count))
    depths = depth.read_frames(depth_path, indices, info["width"], info["height"])
    colors = video.read_frames(color_path, indices)
    frames = []
    try:
        for index, (raw, bgr) in enumerate(zip(depths, colors, strict=True)):
            valid = raw != info.get("depth_invalid_value", 0)
            xyz = unit_rays[valid] * (raw[valid].astype(np.float32) * scale)[:, None]
            name = f"frame_{index:06d}.ply"
            write_ply(args.output / name, xyz, bgr[valid][:, ::-1])
            frames.append({"file": name, "frame_index": index,
                           "timestamp_ns": timestamps[index], "points": len(xyz)})
            if index % 50 == 0 or index == count - 1:
                print(f"Exported {index + 1}/{count}: {len(xyz)} points", flush=True)
    finally:
        depths.close()
        colors.close()
    manifest = {
        "episode": str(args.episode.resolve()), "units": "meters",
        "coordinate_frame": "color camera: x right, y down, z forward",
        "calibration": calibration, "depth_scale_m_per_unit": scale,
        "processing": "SDK deprojection; invalid depth omitted; no filtering or frame fusion",
        "color": "same-index recorded RGB frame; source H.264 is lossy",
        "frames": frames,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
