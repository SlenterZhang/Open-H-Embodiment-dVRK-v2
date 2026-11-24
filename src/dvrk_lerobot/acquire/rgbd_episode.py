"""Encoder and cloud-writer lifetime for one RealSense recording episode."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from dvrk_lerobot.acquire.camera import CameraError
from dvrk_lerobot.acquire.cloud import CloudWriter
from dvrk_lerobot.acquire.rgbd_log import TimestampLog
from dvrk_lerobot.dataset import depth
from dvrk_lerobot.dataset.video import finish_encoder, start_lossless_encoder


@dataclass
class Episode:
    color: Path
    depth: Path
    color_process: subprocess.Popen
    depth_process: subprocess.Popen
    timestamps: TimestampLog
    cloud: CloudWriter | None = None
    count: int = 0

    @classmethod
    def create(cls, color, depth_path, timestamps, config, info, unit_rays):
        processes = []
        log = None
        try:
            processes.append(start_lossless_encoder(
                color, width=config.width, height=config.height, fps=config.fps,
            ))
            processes.append(depth.start_encoder(
                depth_path, config.width, config.height, config.fps,
            ))
            log = TimestampLog(timestamps)
            cloud = CloudWriter(color.parent.parent / "cloud", info, unit_rays) if config.record_cloud else None
            return cls(color, depth_path, *processes, log, cloud)
        except Exception:
            if log is not None:
                log.close()
            for process in processes:
                process.kill()
                process.communicate()
            raise

    def close(self) -> int:
        error = None
        try:
            self.timestamps.close()
        except OSError as exc:
            error = exc
        for process, path in ((self.color_process, self.color), (self.depth_process, self.depth)):
            try:
                finish_encoder(process, f"encode {path}")
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                error = exc
        if self.cloud is not None:
            try:
                self.cloud.close()
                if self.cloud.count != self.count:
                    raise CameraError("point-cloud count differs from RGB-D frame count")
            except (OSError, RuntimeError) as exc:
                error = exc
        if error:
            raise CameraError(f"D435i: {error}") from error
        return self.count
