"""TouchX-to-PSM pose mapping, following the dVRK MTM-to-PSM teleoperation."""

from __future__ import annotations

import numpy as np

# (position [m], 3x3 rotation), both expressed in a fixed reference frame.
Pose = tuple[np.ndarray, np.ndarray]


def quat_to_matrix(q) -> np.ndarray:
    """ROS quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to a unit ROS quaternion (x, y, z, w)."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4, (r[1, 0] - r[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


class Retarget:
    """Relative mapping captured at engage time, as in dVRK mtsTeleOperationPSM.

    `rotation` is the TouchX base frame: it maps TouchX coordinates into the PSM
    command frame (display/camera frame in dVRK terms). Translation is scaled;
    orientation changes are applied 1:1 in that common frame.
    """

    def __init__(self, rotation: np.ndarray, scale: float) -> None:
        self.rotation = np.asarray(rotation, dtype=float)
        self.scale = scale
        self._touch0: Pose | None = None
        self._psm0: Pose | None = None

    @property
    def engaged(self) -> bool:
        return self._touch0 is not None

    def engage(self, touch: Pose, psm: Pose) -> None:
        """Capture the TouchX and PSM start poses; the PSM goal starts at `psm`."""
        self._touch0 = self._to_common(touch)
        self._psm0 = psm

    def release(self) -> None:
        self._touch0 = None
        self._psm0 = None

    def goal(self, touch: Pose) -> Pose:
        """PSM goal: p = p0 + s * dp, R = (R_touch * R_touch0^T) * R_psm0."""
        if self._touch0 is None or self._psm0 is None:
            raise RuntimeError("retarget is not engaged")
        position, rotation = self._to_common(touch)
        position0, rotation0 = self._touch0
        psm_position, psm_rotation = self._psm0
        goal_position = psm_position + self.scale * (position - position0)
        goal_rotation = rotation @ rotation0.T @ psm_rotation
        return goal_position, goal_rotation

    def _to_common(self, touch: Pose) -> Pose:
        position, rotation = touch
        return self.rotation @ position, self.rotation @ rotation
