"""RGB-D alignment and exact depth-array export without physical cameras."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

from dvrk_lerobot.acquire.rgbd_log import TimestampLog
from dvrk_lerobot.acquire.session import Frame, RobotSample, Session
from dvrk_lerobot.align.sync import AlignOptions, Episode, align_session
from dvrk_lerobot.config import load_config
from dvrk_lerobot.config.model import ArmConfig, RealSenseConfig
from dvrk_lerobot.dataset.parquet import write_episode_table
from dvrk_lerobot.dataset.writer import export_dataset
from dvrk_lerobot.tools.verify import verify_dataset


class RGBDTests(unittest.TestCase):
    def test_missing_external_frame_repeats_on_the_grid(self) -> None:
        base = 1_000_000_000
        period = 33_333_333
        left = [Frame(base + n * period, n) for n in range(3)]
        right = [Frame(frame.ts_ns + 5_000_000, frame.index) for frame in left]
        external = [Frame(left[n].ts_ns + 10_000_000, i) for i, n in enumerate((0, 2))]
        robot = []
        for frame in left:
            stamp = frame.ts_ns + 2_500_000
            for kind, values in (
                ("jp", (float(frame.index),)),
                ("jaw", (0.1,)),
                ("cp", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
            ):
                robot.append(RobotSample(stamp, stamp, kind, "PSM1", values))
        session = Session(
            root=Path("synthetic"),
            meta={"fps": 30, "task": "pick", "joint_names": {"PSM1": ["joint"]},
                  "realsense": {"height": 2, "width": 2, "depth_scale": 0.001}},
            robot=robot,
            frames={"left": left, "right": right, "external": external},
            videos={key: Path(f"{key}.mkv") for key in ("left", "right", "external")},
            depth=Path("depth.mkv"),
        )
        arm = ArmConfig("PSM1", "/js", "/cp", "/jaw")
        options = AlignOptions(30, 15, 25, external_tolerance_ms=15, require_external=True)
        episode = align_session(session, (arm,), options)
        self.assertEqual(episode.left, [0, 1, 2])
        self.assertEqual(episode.right, [0, 1, 2])
        self.assertEqual(episode.external, [0, 0, 1])  # slot 1 has no frameset: repeat
        self.assertEqual(episode.state.shape, (3, 9))
        self.assertEqual(episode.depth_shape, (2, 2))

    def test_depth_parquet_preserves_every_uint16_value(self) -> None:
        images = [
            np.array([[0, 1], [65534, 65535]], dtype=np.uint16),
            np.array([[1000, 2048], [32768, 4095]], dtype=np.uint16),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.parquet"
            def frames(*_args):
                yield from images

            with patch("dvrk_lerobot.dataset.parquet.read_depth_frames", side_effect=frames):
                write_episode_table(
                    path, state=np.zeros((2, 1), np.float32),
                    action=np.zeros((2, 1), np.float32),
                    timestamp=np.array([0, 1 / 30], np.float32),
                    episode_index=0, task_index=0, index_offset=0,
                    depth_source=Path("synthetic.mkv"), depth_indices=[0, 1], depth_shape=(2, 2),
                )
            rows = pq.read_table(path)["observation.depth.external"].to_pylist()
            self.assertEqual(rows, [image.tolist() for image in images])

    def test_depth_parquet_frame_index_continues_across_batches(self) -> None:
        images = [np.full((2, 2), i, np.uint16) for i in range(20)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.parquet"
            def frames(*_args):
                yield from images

            with patch("dvrk_lerobot.dataset.parquet.read_depth_frames", side_effect=frames):
                write_episode_table(
                    path, state=np.zeros((20, 1), np.float32), action=np.zeros((20, 1), np.float32),
                    timestamp=np.arange(20, dtype=np.float32) / 30,
                    episode_index=0, task_index=0, index_offset=5,
                    depth_source=Path("synthetic.mkv"), depth_indices=list(range(20)), depth_shape=(2, 2),
                )
            table = pq.read_table(path)
            self.assertEqual(table["frame_index"].to_pylist(), list(range(20)))
            self.assertEqual(table["index"].to_pylist(), list(range(5, 25)))

    def test_timestamp_log_keeps_host_file_and_adds_sensor_times(self) -> None:
        class FakeFrame:
            def __init__(self, ms: float, number: int) -> None:
                self.ms, self.number = ms, number
            def get_timestamp(self) -> float:
                return self.ms
            def get_frame_timestamp_domain(self) -> str:
                return "global_time"
            def get_frame_number(self) -> int:
                return self.number

        with tempfile.TemporaryDirectory() as directory:
            host = Path(directory) / "external_timestamps.txt"
            log = TimestampLog(host)
            log.write(123, FakeFrame(1000.5, 7), FakeFrame(1000.25, 9))
            log.close()
            self.assertEqual(host.read_text().split(), ["123"])
            lines = host.with_name("external_sensor_timestamps.csv").read_text().splitlines()
            self.assertEqual(lines[1], "123,1000.5,1000.25,global_time,global_time,7,9")

    def test_rgbd_export_has_three_videos_and_depth(self) -> None:
        images = [np.full((16, 16), 1000 + i, np.uint16) for i in range(2)]
        config = load_config(Path(__file__).parents[2] / "configs" / "record.yaml")
        config = replace(config, realsense=RealSenseConfig(None, 16, 16, 30, 18))
        episode = Episode(
            session=Path("synthetic"), task="pick",
            state=np.zeros((2, 1), np.float32), action=np.zeros((2, 1), np.float32),
            timestamps=np.array([0, 1 / 30], np.float32),
            videos={side: Path(f"{side}.mkv") for side in ("left", "right", "external")},
            left=[0, 1], right=[0, 1], external=[0, 1],
            depth=Path("external_depth.mkv"), depth_scale=0.001, depth_shape=(16, 16),
            state_names=["PSM1.jp.joint"], notes=[],
        )

        def encode(_source, _indices, destination, **_kwargs):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"video")
            external = "external" in str(destination)
            return {
                "width": 16 if external else 1920, "height": 16 if external else 1080,
                "codec": "h264", "pix_fmt": "yuv420p", "fps": 30, "channels": 3,
            }

        def depth_frames(*_args):
            yield from images

        tiny_stats = {
            "min": np.array([0.0]), "max": np.array([1.0]),
            "mean": np.array([0.5]), "std": np.array([0.5]),
            "count": np.array([2]),
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            with (
                patch("dvrk_lerobot.dataset.video.encode", side_effect=encode),
                patch("dvrk_lerobot.dataset.video.probe", side_effect=lambda path: encode(None, None, path)),
                patch("dvrk_lerobot.dataset.parquet.read_depth_frames", side_effect=depth_frames),
                patch("dvrk_lerobot.dataset.stats.video_stats", return_value=tiny_stats),
                patch("dvrk_lerobot.dataset.stats.depth_stats", return_value=tiny_stats),
            ):
                export_dataset([episode], config, output)
                verify_dataset(output)
            table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
            self.assertEqual(table["observation.depth.external"].to_pylist(), [i.tolist() for i in images])


if __name__ == "__main__":
    unittest.main()
