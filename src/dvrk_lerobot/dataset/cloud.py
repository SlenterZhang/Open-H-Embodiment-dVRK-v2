"""Calibrated depth deprojection and binary colored PLY output."""

from pathlib import Path

import numpy as np


def rays(calibration: dict) -> np.ndarray:
    """Use SDK deprojection to account for the calibrated distortion model."""
    import pyrealsense2 as rs

    intr = rs.intrinsics()
    for key in ("width", "height", "fx", "fy", "ppx", "ppy", "coeffs"):
        setattr(intr, key, calibration[key])
    intr.model = getattr(rs.distortion, calibration["model"])
    return np.asarray([
        rs.rs2_deproject_pixel_to_point(intr, [u, v], 1.0)
        for v in range(intr.height) for u in range(intr.width)
    ], dtype=np.float32).reshape(intr.height, intr.width, 3)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write unorganized binary little-endian XYZ/RGB vertices."""
    points = np.empty(len(xyz), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    for i, name in enumerate(("x", "y", "z")):
        points[name] = xyz[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        points[name] = rgb[:, i]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment meters; color camera frame: x right, y down, z forward\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("xb") as handle:
        handle.write(header.encode("ascii"))
        points.tofile(handle)


