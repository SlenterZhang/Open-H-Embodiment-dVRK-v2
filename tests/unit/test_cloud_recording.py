"""Cloud geometry, synchronization, backpressure, and CLI opt-in behavior."""

import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dvrk_lerobot.acquire import cloud
from dvrk_lerobot.acquire.camera import CameraError
from dvrk_lerobot.acquire.realsense import RealSense
from dvrk_lerobot.cli import _acquisition_config, _parser
from dvrk_lerobot.config import ConfigError, load_config
from dvrk_lerobot.dataset.cloud import rays


def calibration():
    return {"width": 2, "height": 2, "fx": 2., "fy": 2., "ppx": 0., "ppy": 0.,
                "model": "none", "coeffs": [0.] * 5}


def writer(root):
    info = {"serial": "test", "depth_scale": 0.001, "color_intrinsics": calibration()}
    return cloud.CloudWriter(root, info, rays(calibration()))


def test_geometry_color_timestamps_and_drain(tmp_path):
    output = writer(tmp_path / "cloud")
    raw = np.array([[1000, 0], [2000, 3000]], dtype=np.uint16)
    bgr = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
    output.submit(0, 123, raw, bgr)
    output.submit(1, 456, raw, bgr)
    raw[:] = 0  # Queued frames must own their buffers.
    bgr[:] = 0
    output.close()
    data = (output.root / "frame_000000.ply").read_bytes().split(b"end_header\n", 1)[1]
    points = np.frombuffer(data, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    np.testing.assert_allclose(points["xyz"], [[0, 0, 1], [0, 1, 2], [1.5, 1.5, 3]])
    np.testing.assert_array_equal(points["rgb"], [[30, 20, 10]] * 3)
    assert (output.root / "timestamps.csv").read_text().splitlines()[1:] == [
        "0,123,3", "1,456,3",
    ]
    meta = json.loads((output.root / "metadata.json").read_text())
    assert meta["frame_count"] == 2 and meta["status"] == "ok"


def test_write_failure_is_reported(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(cloud, "write_ply", fail)
    output = writer(tmp_path / "cloud")
    output.submit(0, 1, np.ones((2, 2), np.uint16), np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(CameraError, match="disk full"):
        output.close()
    assert json.loads((output.root / "metadata.json").read_text())["status"] == "failed"


def test_backpressure_fails_instead_of_dropping_frames(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = cloud.write_ply
    def slow(*args):
        entered.set()
        assert release.wait(5)
        original(*args)
    monkeypatch.setattr(cloud, "write_ply", slow)
    output = writer(tmp_path / "cloud")
    raw, bgr = np.ones((2, 2), np.uint16), np.zeros((2, 2, 3), np.uint8)
    try:
        output.submit(0, 0, raw, bgr)
        assert entered.wait(5)
        for index in range(1, 31):
            output.submit(index, index, raw, bgr)
        with pytest.raises(CameraError, match="queue full"):
            output.submit(31, 31, raw, bgr)
    finally:
        release.set()
        with pytest.raises(CameraError, match="queue full"):
            output.close()
    assert output.count == 31


@pytest.mark.parametrize("flags,expected", [([], False), (["--record-cloud"], True),
                                          (["--no-record-cloud"], False)])
def test_cli_switch(flags, expected):
    args = _parser().parse_args(["record", "--task", "test", *flags])
    assert _acquisition_config(args).realsense.record_cloud is expected


def test_cloud_requires_realsense():
    args = _parser().parse_args(["record", "--task", "test", "--record-cloud", "--no-realsense"])
    with pytest.raises(ConfigError, match="requires RealSense"):
        _acquisition_config(args)


@pytest.mark.parametrize("enabled", [False, True])
def test_episode_wiring_and_reset(tmp_path, enabled):
    config = load_config(Path("configs/record.yaml")).realsense
    camera = RealSense(replace(config, width=16, height=16, record_cloud=enabled))
    intr = dict(calibration(), width=16, height=16)
    camera._info = {"serial": "test", "depth_scale": 0.001, "color_intrinsics": intr}
    camera._unit_rays = rays(intr) if enabled else None
    for number in range(2):
        root = tmp_path / str(number)
        (root / "video").mkdir(parents=True)
        camera.begin(root / "video/external.mkv", root / "video/external_depth.mkv",
                     root / "video/external_timestamps.txt")
        raw, bgr = np.full((16, 16), 1000, np.uint16), np.zeros((16, 16, 3), np.uint8)
        stop = threading.Event()
        def frame(values):
            return SimpleNamespace(
                get_data=lambda: values, get_timestamp=lambda: 1000.,
                get_frame_timestamp_domain=lambda: "test", get_frame_number=lambda: 1,
            )
        frameset = SimpleNamespace(get_color_frame=lambda values=bgr: frame(values),
                                  get_depth_frame=lambda values=raw: frame(values))
        def capture(stop=stop, frameset=frameset, **kwargs):
            stop.set()  # Return one frame, then exit the capture loop.
            return frameset
        camera._pipeline = SimpleNamespace(wait_for_frames=capture)
        camera._align = SimpleNamespace(process=lambda frames: frames)
        camera._loop(stop)
        camera.check()
        camera.detach()
        assert camera.end() == 1
        assert camera.end() == 0
        assert (root / "cloud/frame_000000.ply").exists() == enabled
        if enabled:
            timestamp = (root / "video/external_timestamps.txt").read_text().strip()
            cloud_row = (root / "cloud/timestamps.csv").read_text().splitlines()[1]
            assert cloud_row == f"0,{timestamp},256"


def test_yaml_cloud_setting_and_cli_override(tmp_path):
    raw = yaml.safe_load(Path("configs/record.yaml").read_text())
    raw["realsense"]["record_cloud"] = True
    path = tmp_path / "record.yaml"
    path.write_text(yaml.safe_dump(raw))
    args = _parser().parse_args(["record", "--task", "test", "-c", str(path)])
    assert _acquisition_config(args).realsense.record_cloud
    args.record_cloud = False
    assert not _acquisition_config(args).realsense.record_cloud
    raw["realsense"]["record_cloud"] = "false"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="must be a boolean"):
        load_config(path)
