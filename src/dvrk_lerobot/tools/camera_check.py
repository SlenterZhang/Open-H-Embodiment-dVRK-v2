"""Camera-only check: rates, sensor timing, depth overlay, and offsets between cameras.

Needs no robot. Flash a light or move something visible to every camera during
the run: offsets come from matching brightness changes between the streams.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from dvrk_lerobot.acquire.camera import Camera
from dvrk_lerobot.config import load_config

MAX_LAG_MS = 200
MIN_CORRELATION = 0.5


def main() -> int:
    """Capture all cameras together for a few seconds and print a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="configs/record.yaml")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--out", default="camera_check", help="directory for preview images")
    args = parser.parse_args()
    config = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    series: dict[str, tuple[list[int], list[float]]] = {}
    deadline = time.monotonic() + args.seconds
    workers = [threading.Thread(target=_endoscope, args=(config.cameras[s], deadline, out, series)) for s in config.camera_sides]
    workers.append(threading.Thread(target=_realsense, args=(config.realsense, deadline, out, series)))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    print("\n== offsets from brightness changes (host time, what convert uses) ==")
    names = sorted(series)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            lag, score = _offset(series[first], series[second])
            verdict = "ok" if score >= MIN_CORRELATION else "UNRELIABLE: no shared change seen"
            print(f"{second} later than {first} by {lag:+.1f} ms (correlation {score:.2f}, {verdict})")
    print(f"\npreview images: {out.resolve()}")
    return 0


def _endoscope(config, deadline: float, out: Path, series: dict) -> None:
    camera = Camera(config)
    try:
        info = camera.open()
    except Exception as exc:  # noqa: BLE001 - report and let the other cameras run
        print(f"FAIL {config.side}: {exc}")
        return
    capture, times, levels, frame = camera._capture, [], [], None
    while time.monotonic() < deadline:
        ok, image = capture.read()
        stamp = time.time_ns()
        if not ok:
            print(f"FAIL {config.side}: frame read failed")
            break
        times.append(stamp)
        levels.append(_level(image))
        frame = image
    camera.stop()
    if frame is not None:
        cv2.imwrite(str(out / f"{config.side}.png"), frame)
    series[config.side] = (times, levels)
    gaps = np.diff(times) / 1e6
    print(
        f"\n== {config.side} ({config.device}) ==\n{info['width']}x{info['height']} {info['fourcc']}, "
        f"mean {1000 / gaps.mean():.2f} fps, period med {np.median(gaps):.1f} max {gaps.max():.1f} ms"
    )


def _realsense(config, deadline: float, out: Path, series: dict) -> None:
    import pyrealsense2 as rs

    pipeline, settings = rs.pipeline(), rs.config()
    if config.serial:
        settings.enable_device(config.serial)
    settings.enable_stream(rs.stream.color, config.width, config.height, rs.format.bgr8, config.fps)
    settings.enable_stream(rs.stream.depth, config.width, config.height, rs.format.z16, config.fps)
    try:
        profile = pipeline.start(settings)
    except RuntimeError as exc:
        print(f"FAIL external: {config.width}x{config.height}@{config.fps}: {exc}")
        return
    align = rs.align(rs.stream.color)
    rows, levels, color, depth = [], [], None, None
    try:
        while time.monotonic() < deadline:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            stamp = time.time_ns()
            raw_depth_ms = frames.get_depth_frame().get_timestamp()
            aligned = align.process(frames)
            color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            rows.append((
                stamp, color_frame.get_timestamp(), depth_frame.get_timestamp(), raw_depth_ms,
                color_frame.get_frame_number(), depth_frame.get_frame_number(),
            ))
            color = np.asanyarray(color_frame.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data()).copy()
            levels.append(_level(color))
    except RuntimeError as exc:
        print(f"FAIL external: started, then {exc}")
    finally:
        device = profile.get_device()
        pipeline_domain = color_frame.get_frame_timestamp_domain() if rows else None
        report_device = (
            device.get_info(rs.camera_info.serial_number),
            device.get_info(rs.camera_info.usb_type_descriptor),
            device.first_color_sensor().get_option(rs.option.auto_exposure_priority),
        )
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        extrinsics = profile.get_stream(rs.stream.depth).get_extrinsics_to(profile.get_stream(rs.stream.color))
        scale = device.first_depth_sensor().get_depth_scale()
        pipeline.stop()
    if not rows:
        return
    data = np.array(rows, dtype=np.float64)
    series["external"] = ([row[0] for row in rows], levels)
    hosts, skew = np.diff(data[:, 0]) / 1e6, np.abs(data[:, 1] - data[:, 2])
    latency = data[:, 0] / 1e6 - data[:, 1]
    print(
        f"\n== external (D435i {report_device[0]}, USB {report_device[1]}) ==\n"
        f"{config.width}x{config.height}, auto-exposure priority {report_device[2]:.0f}, domain {pipeline_domain}\n"
        f"host: mean {1000 / hosts.mean():.2f} fps, period med {np.median(hosts):.1f} max {hosts.max():.1f} ms\n"
        f"sensor color: mean {1000 / np.diff(data[:, 1]).mean():.2f} fps, depth: mean {1000 / np.diff(data[:, 2]).mean():.2f} fps\n"
        f"color-depth skew: med {np.median(skew):.2f} p99 {np.percentile(skew, 99):.2f} max {skew.max():.2f} ms\n"
        f"aligned depth keeps its sensor timestamp: {bool(np.all(data[:, 2] == data[:, 3]))}\n"
        f"frame numbers skipped: color {int((np.diff(data[:, 4]) - 1).sum())}, depth {int((np.diff(data[:, 5]) - 1).sum())}\n"
        f"host minus color sensor: med {np.median(latency):.1f} std {latency.std():.2f} ms\n"
        f"depth valid {100 * (depth > 0).mean():.1f}%, scale {scale} m/unit, "
        f"median {np.median(depth[depth > 0]) * scale if (depth > 0).any() else 0:.3f} m\n"
        f"color intrinsics fx={stream.fx:.1f} fy={stream.fy:.1f} cx={stream.ppx:.1f} cy={stream.ppy:.1f} {stream.model}\n"
        f"depth-to-color translation (mm): {np.round(np.array(extrinsics.translation) * 1000, 2).tolist()}"
    )
    _save_overlay(color, depth, out)


def _save_overlay(color: np.ndarray, depth: np.ndarray, out: Path) -> None:
    """Save color, a depth blend, and depth edges drawn on color."""
    valid = depth > 0
    low, high = np.percentile(depth[valid], (2, 98)) if valid.any() else (0, 1)
    scaled = np.clip((depth.astype(np.float32) - low) / max(high - low, 1) * 255, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    heat[~valid] = 0
    edges = cv2.Canny(scaled, 40, 120) > 0
    outlined = color.copy()
    outlined[edges] = (0, 255, 0)
    cv2.imwrite(str(out / "external_color.png"), color)
    cv2.imwrite(str(out / "external_blend.png"), cv2.addWeighted(color, 0.5, heat, 0.5, 0))
    cv2.imwrite(str(out / "external_edges.png"), outlined)


def _level(image: np.ndarray) -> float:
    return float(image[::8, ::8].mean())


def _offset(first: tuple[list[int], list[float]], second: tuple[list[int], list[float]]) -> tuple[float, float]:
    """Return how many ms *second* lags *first*, and the correlation of their changes."""
    start, stop = max(first[0][0], second[0][0]), min(first[0][-1], second[0][-1])
    grid = np.arange(start, stop, 1_000_000, dtype=np.float64)
    a = np.gradient(np.interp(grid, first[0], first[1]))
    b = np.gradient(np.interp(grid, second[0], second[1]))
    a, b = (a - a.mean()) / (a.std() + 1e-9), (b - b.mean()) / (b.std() + 1e-9)
    scores = {
        lag: float(np.mean(a[max(0, -lag):len(a) - max(0, lag)] * b[max(0, lag):len(b) - max(0, -lag)]))
        for lag in range(-MAX_LAG_MS, MAX_LAG_MS + 1)
    }
    best = max(scores, key=scores.get)
    return float(best), scores[best]


if __name__ == "__main__":
    raise SystemExit(main())
