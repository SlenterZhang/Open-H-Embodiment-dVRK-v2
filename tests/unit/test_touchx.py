"""TouchX retargeting math and configuration validation."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from dvrk_touchx.calibrate import to_yaml
from dvrk_touchx.config import parse_config
from dvrk_touchx.retarget import Retarget, fit_rotation, matrix_to_quat, quat_to_matrix

ROOT = Path(__file__).resolve().parents[2]
ISI = np.diag([-1.0, 1.0, -1.0])


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_quaternion_round_trip():
    for rotation in (np.eye(3), _rz(np.pi), _rx(2.5) @ _rz(-1.0), ISI @ _rx(np.pi / 2)):
        assert np.allclose(quat_to_matrix(matrix_to_quat(rotation)), rotation)


def test_goal_equals_psm_pose_at_engage():
    retarget = Retarget(ISI, 0.2)
    psm = (np.array([0.01, -0.02, -0.1]), _rx(0.3))
    touch = (np.array([0.05, 0.02, -0.03]), _rz(0.7))
    retarget.engage(touch, psm)
    position, rotation = retarget.goal(touch)
    assert np.allclose(position, psm[0])
    assert np.allclose(rotation, psm[1])


def test_translation_is_scaled_and_rotated_into_psm_frame():
    retarget = Retarget(ISI, 0.2)
    psm = (np.zeros(3), np.eye(3))
    retarget.engage((np.zeros(3), np.eye(3)), psm)
    # TouchX +X (user's right) is -X in the ISI frame (X points to the user's left).
    position, _ = retarget.goal((np.array([0.1, 0.0, 0.0]), np.eye(3)))
    assert np.allclose(position, [-0.02, 0.0, 0.0])


def test_rotation_delta_is_applied_in_common_frame():
    retarget = Retarget(ISI, 0.2)
    psm_rotation = _rx(0.4)
    retarget.engage((np.zeros(3), _rz(0.2)), (np.zeros(3), psm_rotation))
    # Rotating the stylus about TouchX Y (up) rotates the PSM tool about ISI Y (up).
    turn = np.array([[np.cos(0.5), 0, np.sin(0.5)], [0, 1, 0], [-np.sin(0.5), 0, np.cos(0.5)]])
    _, rotation = retarget.goal((np.zeros(3), turn @ _rz(0.2)))
    assert np.allclose(rotation, turn @ psm_rotation)


def test_goal_requires_engage():
    retarget = Retarget(ISI, 0.2)
    with pytest.raises(RuntimeError):
        retarget.goal((np.zeros(3), np.eye(3)))


def test_fit_rotation_recovers_rotation_from_noisy_scaled_moves():
    truth = ISI @ _rz(0.3) @ _rx(-0.2)
    touch = np.array([[0.04, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.05], [0.02, 0.02, 0.0]])
    noise = np.random.default_rng(0).normal(scale=1e-4, size=touch.shape)
    rotation = fit_rotation(touch, 0.2 * touch @ truth.T + noise)
    assert np.allclose(rotation, truth, atol=1e-2)


def test_fit_rotation_from_two_directions_is_proper():
    touch = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotation = fit_rotation(touch, touch @ ISI.T)
    assert np.allclose(rotation, ISI)


def test_fit_rotation_rejects_parallel_moves():
    with pytest.raises(ValueError):
        fit_rotation(np.array([[1.0, 0, 0], [2.0, 0, 0]]), np.array([[0, 1.0, 0], [0, 3.0, 0]]))


def test_fitted_yaml_passes_config_validation():
    raw = _raw()
    raw.update(yaml.safe_load(to_yaml(ISI @ _rz(0.123) @ _rx(0.456))))
    parse_config(raw)


def _raw() -> dict:
    return yaml.safe_load((ROOT / "configs" / "touchx.yaml").read_text())


def test_shipped_config_parses():
    config = parse_config(_raw())
    assert config.psm == "/PSM1"


@pytest.mark.parametrize(
    "key, value",
    [
        ("rotation", [[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        ("rotation", [[1, 0], [0, 1]]),
        ("scale", 0),
        ("jaw_closed", 1.0),
    ],
)
def test_invalid_config_is_rejected(key, value):
    raw = _raw()
    raw[key] = value
    with pytest.raises(ValueError):
        parse_config(raw)


def test_unknown_key_is_rejected():
    raw = _raw()
    raw["extra"] = 1
    with pytest.raises(ValueError):
        parse_config(raw)
