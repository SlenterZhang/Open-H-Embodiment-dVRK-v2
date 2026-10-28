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


def fit_rotation(touch: np.ndarray, psm: np.ndarray) -> np.ndarray:
    """Kabsch fit of R with R @ touch[i] ~ psm[i] (N x 3 displacements, N >= 2).

    Each pair is normalized so it counts by direction only; scale is ignored.
    """
    a = touch / np.linalg.norm(touch, axis=1, keepdims=True)
    b = psm / np.linalg.norm(psm, axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(a.T @ b)
    if s[1] < 1e-2 * s[0]:
        raise ValueError("need at least two non-parallel movements")
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def _axis_angle(r: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit axis and angle (rad) of a rotation matrix."""
    angle = float(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)))
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if np.linalg.norm(axis) < 1e-9:  # 0 or pi: (r + I) / 2 = axis axis^T at pi
        b = (r + np.eye(3)) / 2
        axis = b[:, np.argmax(np.diag(b))]
    return axis / np.linalg.norm(axis), angle


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


class Retarget:
    """Relative position, absolute orientation, as in dVRK mtsTeleOperationPSM.

    `rotation` is the TouchX base frame: it maps TouchX coordinates into the PSM
    command frame (display/camera frame in dVRK terms). Translation is scaled from
    the engage pose; the tool orientation equals the stylus orientation times `tip`
    times a roll about the tool axis, set at the first engage to the current tool roll.
    It is reached from the engage orientation at `align_rate` (rad/s) first.
    """

    def __init__(self, rotation: np.ndarray, tip: np.ndarray, scale: float, align_rate: float) -> None:
        self.rotation = np.asarray(rotation, dtype=float)
        self.tip = np.asarray(tip, dtype=float)
        self.scale = scale
        self.align_rate = align_rate
        self.aligned = False
        self._touch0: np.ndarray | None = None
        self._psm0: np.ndarray | None = None
        self._orientation: np.ndarray | None = None
        self._roll: np.ndarray | None = None  # kept across clutches

    @property
    def engaged(self) -> bool:
        return self._touch0 is not None

    def engage(self, touch: Pose, psm: Pose) -> None:
        """Capture the start poses; the goal starts at `psm` and turns toward the stylus."""
        self._touch0 = self.rotation @ touch[0]
        self._psm0 = psm[0]
        self._orientation = psm[1]
        self.aligned = False
        if self._roll is None:
            m = (self.rotation @ touch[1] @ self.tip).T @ psm[1]
            self._roll = _rotation(np.array([0.0, 0.0, 1.0]), np.arctan2(m[1, 0] - m[0, 1], m[0, 0] + m[1, 1]))

    def release(self) -> None:
        self._touch0 = None
        self._psm0 = None

    def goal(self, touch: Pose, dt: float) -> Pose:
        """PSM goal: p = p0 + s * dp, R -> rotation * R_touch * tip * roll (rate limited until aligned)."""
        if self._touch0 is None or self._psm0 is None:
            raise RuntimeError("retarget is not engaged")
        position = self._psm0 + self.scale * (self.rotation @ touch[0] - self._touch0)
        target = self.rotation @ touch[1] @ self.tip @ self._roll
        if not self.aligned:
            axis, angle = _axis_angle(self._orientation.T @ target)
            step = self.align_rate * dt
            if angle > step:
                target = self._orientation @ _rotation(axis, step)
            else:
                self.aligned = True
        self._orientation = target
        return position, target
