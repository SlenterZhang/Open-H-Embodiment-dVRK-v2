"""Single-plane Linux V4L2 capture with the metadata from the decoded buffer.

The ctypes layouts mirror linux/videodev2.h (generic Linux ioctl ABI). No second
camera handle is opened: image, timestamp, flags and sequence come from DQBUF.
"""
from __future__ import annotations

import ctypes as ct
import errno
import fcntl
import mmap
import os
import select
import time
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from dvrk_lerobot.config import CameraConfig

_U32 = ct.c_uint32
_CAPTURE, _MMAP = 1, 1


class _Capability(ct.Structure):
    _fields_ = [("driver", ct.c_char * 16), ("card", ct.c_char * 32),
                ("bus_info", ct.c_char * 32), ("version", _U32),
                ("capabilities", _U32), ("device_caps", _U32), ("reserved", _U32 * 3)]


class _PixFormat(ct.Structure):
    _fields_ = [(name, _U32) for name in (
        "width", "height", "pixelformat", "field", "bytesperline", "sizeimage",
        "colorspace", "priv", "flags", "ycbcr_enc", "quantization", "xfer_func")]


class _FormatUnion(ct.Union):
    # v4l2_window contains pointers and sets the union's native alignment.
    _fields_ = [("pix", _PixFormat), ("raw", ct.c_uint8 * 200), ("alignment", ct.c_void_p)]


class _Format(ct.Structure):
    _fields_ = [("type", _U32), ("fmt", _FormatUnion)]


class _Fraction(ct.Structure):
    _fields_ = [("numerator", _U32), ("denominator", _U32)]


class _CaptureParm(ct.Structure):
    _fields_ = [("capability", _U32), ("capturemode", _U32),
                ("timeperframe", _Fraction), ("extendedmode", _U32),
                ("readbuffers", _U32), ("reserved", _U32 * 4)]


class _ParmUnion(ct.Union):
    _fields_ = [("capture", _CaptureParm), ("raw", ct.c_uint8 * 200)]


class _StreamParm(ct.Structure):
    _fields_ = [("type", _U32), ("parm", _ParmUnion)]


class _RequestBuffers(ct.Structure):
    _fields_ = [(name, _U32) for name in ("count", "type", "memory", "capabilities", "reserved")]


class _Timeval(ct.Structure):
    _fields_ = [("sec", ct.c_long), ("usec", ct.c_long)]


class _Timecode(ct.Structure):
    _fields_ = [("type", _U32), ("flags", _U32), ("frames", ct.c_uint8),
                ("seconds", ct.c_uint8), ("minutes", ct.c_uint8),
                ("hours", ct.c_uint8), ("userbits", ct.c_uint8 * 4)]


class _BufferMemory(ct.Union):
    _fields_ = [("offset", _U32), ("userptr", ct.c_ulong),
                ("planes", ct.c_void_p), ("fd", ct.c_int32)]


class _Buffer(ct.Structure):
    _fields_ = [("index", _U32), ("type", _U32), ("bytesused", _U32),
                ("flags", _U32), ("field", _U32), ("timestamp", _Timeval),
                ("timecode", _Timecode), ("sequence", _U32), ("memory", _U32),
                ("m", _BufferMemory), ("length", _U32), ("reserved2", _U32),
                ("request_fd", ct.c_int32)]


def _request(number: int, structure: type, direction: int = 3) -> int:
    return (direction << 30) | (ct.sizeof(structure) << 16) | (ord("V") << 8) | number


QUERYCAP = _request(0, _Capability, 2)
S_FMT = _request(5, _Format)
REQBUFS = _request(8, _RequestBuffers)
QUERYBUF = _request(9, _Buffer)
QBUF = _request(15, _Buffer)
DQBUF = _request(17, _Buffer)
STREAMON = _request(18, ct.c_int, 1)
STREAMOFF = _request(19, ct.c_int, 1)
S_PARM = _request(22, _StreamParm)


def timestamp_flags(flags: int) -> tuple[str, str]:
    """Decode the UAPI flags; UNKNOWN must never be guessed to be realtime."""
    clock_bits, source_bits = flags & 0xE000, flags & 0x70000
    clock = {0: "unknown", 0x2000: "monotonic", 0x4000: "copy"}.get(
        clock_bits, f"unknown_0x{clock_bits:x}")
    source = {0: "end_of_frame", 0x10000: "start_of_exposure"}.get(
        source_bits, f"unknown_0x{source_bits:x}")
    return clock, source


@dataclass(frozen=True)
class FrameTiming:
    driver_timestamp_ns: int
    driver_sequence: int
    driver_flags: int
    driver_clock: str
    driver_timestamp_source: str
    driver_buffer_index: int
    driver_bytesused: int
    driver_error: bool
    sequence_step: int | None
    host_dequeue_ns: int
    host_dequeue_monotonic_ns: int
    host_clock_pair_span_ns: int
    host_read_start_monotonic_ns: int
    host_decode_complete_ns: int
    host_decode_complete_monotonic_ns: int

    def row(self) -> dict[str, Any]:
        values = asdict(self)
        values["driver_to_dequeue_ns"] = (
            self.host_dequeue_monotonic_ns - self.driver_timestamp_ns
            if self.driver_clock == "monotonic" and self.driver_timestamp_ns > 0 else None
        )
        return values


class V4L2Capture:
    """MMAP capture. Copy each dequeued buffer before requeueing or decoding it."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._fd: int | None = None
        self._buffers: list[mmap.mmap] = []
        self._streaming = False
        self._previous_sequence: int | None = None
        self._stride = 0
        self.info: dict[str, Any] = {}

    def _ioctl(self, request: int, value: Any) -> None:
        if self._fd is None:
            raise RuntimeError("V4L2 device is closed")
        while True:
            try:
                fcntl.ioctl(self._fd, request, value, True)
                return
            except InterruptedError:
                continue

    def open(self) -> dict[str, Any]:
        c = self.config
        if c.fourcc not in ("MJPG", "JPEG", "YUYV", "UYVY"):
            raise RuntimeError(f"driver metadata capture does not decode {c.fourcc}")
        self._fd = os.open(c.device, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            cap = _Capability()
            self._ioctl(QUERYCAP, cap)
            caps = cap.device_caps if cap.capabilities & 0x80000000 else cap.capabilities
            if not caps & 1 or not caps & 0x04000000:
                raise RuntimeError("single-plane VIDEO_CAPTURE and STREAMING are required")
            fmt = _Format(type=_CAPTURE)
            fmt.fmt.pix.width, fmt.fmt.pix.height = c.width, c.height
            fmt.fmt.pix.pixelformat = int.from_bytes(c.fourcc.encode("ascii"), "little")
            self._ioctl(S_FMT, fmt)
            pix = fmt.fmt.pix
            actual_fourcc = int(pix.pixelformat).to_bytes(4, "little").decode("ascii")
            if (pix.width, pix.height, actual_fourcc) != (c.width, c.height, c.fourcc):
                raise RuntimeError(f"negotiated {pix.width}x{pix.height} {actual_fourcc}, "
                                   f"expected {c.width}x{c.height} {c.fourcc}")
            self._stride = pix.bytesperline
            parm = _StreamParm(type=_CAPTURE)
            parm.parm.capture.timeperframe = _Fraction(1, c.fps)
            self._ioctl(S_PARM, parm)
            period = parm.parm.capture.timeperframe
            if not period.numerator or not period.denominator:
                raise RuntimeError("driver returned an invalid frame period")
            request = _RequestBuffers(count=2, type=_CAPTURE, memory=_MMAP)
            self._ioctl(REQBUFS, request)
            if not request.count:
                raise RuntimeError("driver allocated no MMAP buffers")
            for index in range(request.count):
                buf = _Buffer(index=index, type=_CAPTURE, memory=_MMAP)
                self._ioctl(QUERYBUF, buf)
                self._buffers.append(mmap.mmap(
                    self._fd, buf.length, flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=buf.m.offset))
                self._ioctl(QBUF, buf)
            self._ioctl(STREAMON, ct.c_int(_CAPTURE))
            self._streaming = True
            self.info = {
                "device": c.device, "width": int(pix.width), "height": int(pix.height),
                "fps": period.denominator / period.numerator, "fourcc": actual_fourcc,
                "capture_backend": "v4l2_mmap", "driver": cap.driver.decode(errors="replace"),
                "card": cap.card.decode(errors="replace"), "bus_info": cap.bus_info.decode(errors="replace"),
                "driver_version": int(cap.version), "buffer_count": int(request.count),
                "timestamp_source": "host_decode_complete_time_ns",
                "driver_timing_log": f"{c.side}_driver_timestamps.csv",
                "driver_timing_schema_version": 1,
                "driver_timestamp_resolution_ns": 1000,
            }
            return dict(self.info)
        except BaseException:
            self.close()
            raise

    def read(self, timeout: float = 5.0) -> tuple[np.ndarray, FrameTiming]:
        start = time.monotonic_ns()
        deadline = time.monotonic() + timeout
        buf = _Buffer(type=_CAPTURE, memory=_MMAP)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self._fd], [], [], max(0, remaining))[0]:
                raise TimeoutError(f"{self.config.device}: no frame in {timeout:g}s")
            try:
                self._ioctl(DQBUF, buf)
                break
            except OSError as exc:
                if exc.errno != errno.EAGAIN:
                    raise
        mono_before = time.monotonic_ns()
        host = time.time_ns()
        mono_after = time.monotonic_ns()
        try:
            if buf.index >= len(self._buffers) or not 0 < buf.bytesused <= len(self._buffers[buf.index]):
                raise RuntimeError("invalid V4L2 buffer index or payload length")
            payload = self._buffers[buf.index][:buf.bytesused]
        finally:
            queued = _Buffer(index=buf.index, type=_CAPTURE, memory=_MMAP)
            self._ioctl(QBUF, queued)
        image = self._decode(payload)
        decoded = time.time_ns()
        decoded_mono = time.monotonic_ns()
        clock, source = timestamp_flags(buf.flags)
        sequence = int(buf.sequence)
        step = None if self._previous_sequence is None else (sequence - self._previous_sequence) & 0xFFFFFFFF
        self._previous_sequence = sequence
        timing = FrameTiming(
            int(buf.timestamp.sec) * 1_000_000_000 + int(buf.timestamp.usec) * 1000,
            sequence, int(buf.flags), clock, source, int(buf.index), int(buf.bytesused),
            bool(buf.flags & 0x40), step, host, (mono_before + mono_after) // 2,
            mono_after - mono_before, start, decoded, decoded_mono,
        )
        return image, timing

    def _decode(self, payload: bytes) -> np.ndarray:
        c = self.config
        if c.fourcc in ("MJPG", "JPEG"):
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            stride = self._stride or c.width * 2
            if len(payload) < stride * c.height or stride < c.width * 2:
                raise RuntimeError("truncated packed YUV frame")
            packed = np.frombuffer(payload, dtype=np.uint8, count=stride * c.height)
            packed = packed.reshape(c.height, stride)[:, :c.width * 2].copy().reshape(c.height, c.width, 2)
            code = cv2.COLOR_YUV2BGR_YUYV if c.fourcc == "YUYV" else cv2.COLOR_YUV2BGR_UYVY
            image = cv2.cvtColor(packed, code)
        if image is None or image.shape != (c.height, c.width, 3):
            raise RuntimeError("decoded frame does not match the configured color dimensions")
        return np.ascontiguousarray(image)

    def close(self) -> None:
        if self._fd is None:
            return
        try:
            if self._streaming:
                try:
                    self._ioctl(STREAMOFF, ct.c_int(_CAPTURE))
                except OSError:
                    pass
        finally:
            self._streaming = False
            for buffer in self._buffers:
                buffer.close()
            self._buffers.clear()
            os.close(self._fd)
            self._fd = None
