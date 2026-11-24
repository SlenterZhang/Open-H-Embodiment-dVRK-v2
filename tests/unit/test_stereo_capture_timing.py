"""Buffer metadata ownership, clock semantics and saved-frame correspondence."""
from __future__ import annotations

import ctypes as ct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from dvrk_lerobot.acquire import v4l2
from dvrk_lerobot.acquire.stereo_log import StereoTimestampLog
from dvrk_lerobot.config.model import CameraConfig
from dvrk_lerobot.tools.stereo_timing import _steps, analyze_session


def sample_timing(index=0, flags=0x12000):
    stamp = 10_000_000_000 + index * 33_333_333
    clock, source = v4l2.timestamp_flags(flags)
    return v4l2.FrameTiming(
        stamp, index, flags, clock, source, 0, 100, False, 1,
        1_700_000_000_000_000_000 + stamp + 1_000_000, stamp + 1_000_000,
        100, stamp - 1_000_000,
        1_700_000_000_000_000_000 + stamp + 3_000_000, stamp + 3_000_000,
    )


class DriverCaptureTests(unittest.TestCase):
    def test_unknown_clock_is_not_assumed_to_be_realtime(self):
        self.assertEqual(v4l2.timestamp_flags(0), ("unknown", "end_of_frame"))
        self.assertIsNone(sample_timing(flags=0).row()["driver_to_dequeue_ns"])
        self.assertIsNone(sample_timing(flags=0x4000).row()["driver_to_dequeue_ns"])
        self.assertEqual(sample_timing().row()["driver_to_dequeue_ns"], 1_000_000)

    def test_metadata_survives_requeue_and_uses_same_image_buffer(self):
        capture = v4l2.V4L2Capture(CameraConfig("left", "/dev/fake", 4, 2, 30, "MJPG", 18))
        ok, encoded = cv2.imencode(".jpg", np.full((2, 4, 3), 100, dtype=np.uint8))
        self.assertTrue(ok)
        capture._fd, capture._buffers = 123, [encoded.tobytes()]
        capture._previous_sequence = 0xFFFFFFFF
        def ioctl(request, buf):
            if request == v4l2.DQBUF:
                buf.index, buf.bytesused, buf.flags, buf.sequence = 0, len(encoded), 0x12000, 0
                buf.timestamp.sec, buf.timestamp.usec = 12, 345678
            elif request == v4l2.QBUF:
                # QBUF can rewrite its argument; saved DQBUF metadata must survive.
                ct.memset(ct.addressof(buf), 0, ct.sizeof(buf))
            else:
                self.fail(f"unexpected ioctl {request}")
        with patch.object(capture, "_ioctl", side_effect=ioctl), patch.object(
            v4l2.select, "select", return_value=([123], [], [])
        ):
            image, timing = capture.read()
        self.assertEqual(image.shape, (2, 4, 3))
        self.assertEqual(timing.driver_timestamp_ns, 12_345_678_000)
        self.assertEqual(timing.driver_sequence, 0)
        self.assertEqual(timing.sequence_step, 1)
        self.assertEqual(timing.driver_clock, "monotonic")
        self.assertEqual(timing.driver_timestamp_source, "start_of_exposure")

    def test_decode_failure_still_requeues_buffer(self):
        capture = v4l2.V4L2Capture(CameraConfig("left", "/dev/fake", 4, 2, 30, "MJPG", 18))
        capture._fd, capture._buffers = 123, [b"invalid jpeg"]
        requests = []
        def ioctl(request, buf):
            requests.append(request)
            if request == v4l2.DQBUF:
                buf.index, buf.bytesused = 0, 12
        with patch.object(capture, "_ioctl", side_effect=ioctl), patch.object(
            v4l2.select, "select", return_value=([123], [], [])
        ), self.assertRaisesRegex(RuntimeError, "decoded frame"):
            capture.read()
        self.assertEqual(requests, [v4l2.DQBUF, v4l2.QBUF])

    def test_sequence_wraparound(self):
        self.assertEqual(_steps([0xFFFFFFFE, 0xFFFFFFFF, 0, 1]),
                         {"missing_between_saved_frames": 0, "duplicates": 0, "resets_or_backwards": 0})
        self.assertEqual(_steps([10, 13])["missing_between_saved_frames"], 2)


class TimingReportTests(unittest.TestCase):
    def make_session(self, root, right_flags=0x12000):
        video = root / "video"
        video.mkdir()
        for side, flags in (("left", 0x12000), ("right", right_flags)):
            log = StereoTimestampLog(video / f"{side}_timestamps.txt")
            for i in range(3):
                timing = sample_timing(i, flags)
                if side == "right":
                    timing = replace(timing,
                        driver_timestamp_ns=timing.driver_timestamp_ns + 100_000,
                        host_decode_complete_ns=timing.host_decode_complete_ns + 8_000_000,
                        host_decode_complete_monotonic_ns=timing.host_decode_complete_monotonic_ns + 8_000_000)
                log.write(i, timing, timing.host_decode_complete_monotonic_ns + 10_000,
                          timing.host_decode_complete_monotonic_ns + 100_000)
            log.close()

    def test_driver_and_host_differences_are_separated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_session(root)
            with patch("dvrk_lerobot.tools.stereo_timing.count_frames", return_value=3):
                result = analyze_session(root)
            self.assertEqual(result["stereo_driver"]["nearest_absolute_gap_ms"]["median"], 0.1)
            self.assertEqual(result["stereo_host_decode"]["nearest_absolute_gap_ms"]["median"], 8.0)
            self.assertTrue(result["streams"]["left"]["video_csv_host_indices_match"])

    def test_unknown_clock_or_different_source_blocks_comparison(self):
        for flags in (0, 0x2000):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_session(root, flags)
                with patch("dvrk_lerobot.tools.stereo_timing.count_frames", return_value=3):
                    result = analyze_session(root)
                self.assertFalse(result["stereo_driver"]["available"])

    def test_frame_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_session(root)
            with (
                patch("dvrk_lerobot.tools.stereo_timing.count_frames", return_value=2),
                self.assertRaisesRegex(ValueError, "frame counts disagree"),
            ):
                analyze_session(root)

    def test_legacy_timestamp_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_session(root)
            (root / "video/left_timestamps.txt").write_text("1\n2\n3\n")
            with (
                patch("dvrk_lerobot.tools.stereo_timing.count_frames", return_value=3),
                self.assertRaisesRegex(ValueError, "legacy timestamps differ"),
            ):
                analyze_session(root)

    def test_old_metadata_is_not_reconstructed(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "old recordings"),
        ):
            analyze_session(Path(directory))


if __name__ == "__main__":
    unittest.main()
