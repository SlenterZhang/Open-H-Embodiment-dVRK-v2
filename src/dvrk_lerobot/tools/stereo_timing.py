"""Compare saved stereo frames at the driver, dequeue and decode boundaries."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from dvrk_lerobot.acquire.session import Frame
from dvrk_lerobot.acquire.stereo_log import driver_log_path
from dvrk_lerobot.dataset.video import count_frames


def pair_frames(left: list[Frame], right: list[Frame], tolerance_ns: int) -> list[tuple[Frame, Frame]]:
    """Greedily pair each left frame with an unused nearby right frame."""
    pairs: list[tuple[Frame, Frame]] = []
    index = 0
    for frame in left:
        while index < len(right) and right[index].ts_ns < frame.ts_ns - tolerance_ns:
            index += 1
        if index >= len(right):
            break
        best = index
        if index + 1 < len(right) and abs(right[index + 1].ts_ns - frame.ts_ns) <= abs(
            right[index].ts_ns - frame.ts_ns
        ):
            best = index + 1
        if abs(right[best].ts_ns - frame.ts_ns) <= tolerance_ns:
            pairs.append((frame, right[best]))
            index = best + 1
    return pairs


def _summary_ns(values) -> dict[str, float] | None:
    values = np.asarray(values, dtype=np.int64)
    if not len(values):
        return None
    return {name: float(np.percentile(values, percentile) / 1e6)
            for name, percentile in (("min", 0), ("median", 50), ("p99", 99), ("max", 100))}


def _steps(sequences: list[int]) -> dict[str, int]:
    steps = [(b - a) & 0xFFFFFFFF for a, b in pairwise(sequences)]
    return {
        "missing_between_saved_frames": sum(step - 1 for step in steps if 1 < step < 0x80000000),
        "duplicates": sum(step == 0 for step in steps),
        "resets_or_backwards": sum(step >= 0x80000000 for step in steps),
    }


def _series_pairing(left: list[int], right: list[int], tolerance_ms: float) -> dict[str, Any]:
    if not left or not right:
        return {"available": False, "reason": "empty stream"}
    if any(b <= a for times in (left, right) for a, b in pairwise(times)):
        return {"available": False, "reason": "timestamps are not strictly increasing"}
    pairs = pair_frames([Frame(t, i) for i, t in enumerate(left)],
                        [Frame(t, i) for i, t in enumerate(right)], round(tolerance_ms * 1e6))
    other = np.asarray(right, dtype=np.int64)
    nearest = []
    signed = []
    for stamp in left:
        index = int(np.searchsorted(other, stamp))
        candidates = [int(other[i]) for i in (index - 1, index) if 0 <= i < len(other)]
        chosen = min(candidates, key=lambda t: abs(t - stamp))
        nearest.append(abs(chosen - stamp))
        signed.append(chosen - stamp)
    capture = [(a.ts_ns + b.ts_ns) // 2 for a, b in pairs]
    gaps = np.diff(np.asarray(capture, dtype=np.int64))
    return {
        "available": True, "tolerance_ms": tolerance_ms,
        "pairs": len(pairs), "left_frames": len(left), "right_frames": len(right),
        "left_retained_fraction": len(pairs) / len(left),
        "nearest_absolute_gap_ms": _summary_ns(nearest),
        "nearest_right_minus_left_ms": _summary_ns(signed),
        "accepted_absolute_gap_ms": _summary_ns([abs(a.ts_ns - b.ts_ns) for a, b in pairs]),
        "paired_midpoint_interval_ms": _summary_ns(gaps),
        "paired_midpoint_gaps_over100ms": int(np.count_nonzero(gaps > 100_000_000)),
    }


def analyze_session(session: Path, tolerance_ms: float = 15.0) -> dict[str, Any]:
    if not np.isfinite(tolerance_ms) or tolerance_ms < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    video = session / "video"
    streams = {}
    per_side = {}
    for side in ("left", "right"):
        host_path = video / f"{side}_timestamps.txt"
        csv_path = driver_log_path(host_path)
        if not csv_path.is_file():
            raise ValueError(f"missing {csv_path}; old recordings cannot recover driver metadata")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        host = [int(line) for line in host_path.read_text().split()]
        frames = count_frames(video / f"{side}.mkv")
        if not rows or len(rows) != len(host) or frames != len(rows):
            raise ValueError(f"{side}: CSV, timestamps and video frame counts disagree or are empty")
        if [int(r["frame_index"]) for r in rows] != list(range(len(rows))):
            raise ValueError(f"{side}: frame_index does not match video order")
        if [int(r["host_decode_complete_ns"]) for r in rows] != host:
            raise ValueError(f"{side}: legacy timestamps differ from diagnostic timestamps")
        driver = [int(r["driver_timestamp_ns"]) for r in rows]
        dequeue = [int(r["host_dequeue_monotonic_ns"]) for r in rows]
        decoded_mono = [int(r["host_decode_complete_monotonic_ns"]) for r in rows]
        clocks = Counter(r["driver_clock"] for r in rows)
        sources = Counter(r["driver_timestamp_source"] for r in rows)
        monotonic = set(clocks) == {"monotonic"} and all(t > 0 for t in driver)
        encoded_start = [int(r["encoder_write_start_monotonic_ns"]) for r in rows]
        encoded_end = [int(r["encoder_write_end_monotonic_ns"]) for r in rows]
        streams[side] = {"driver": driver, "host": host, "dequeue": dequeue,
                         "decode_monotonic": decoded_mono, "monotonic": monotonic,
                         "sources": set(sources)}
        offsets = np.asarray(host, dtype=np.int64) - np.asarray(decoded_mono, dtype=np.int64)
        per_side[side] = {
            "saved_frames": len(rows), "video_csv_host_indices_match": True,
            "driver_clock_counts": dict(clocks), "driver_source_counts": dict(sources),
            "driver_error_frames": sum(r["driver_error"] in ("True", "1") for r in rows),
            "sequence": _steps([int(r["driver_sequence"]) for r in rows]),
            "driver_interval_ms": _summary_ns(np.diff(driver)) if monotonic else None,
            "host_decode_interval_ms": _summary_ns(np.diff(host)),
            "driver_to_dequeue_ms": _summary_ns(np.asarray(dequeue) - driver) if monotonic else None,
            "driver_to_decode_ms": _summary_ns(np.asarray(decoded_mono) - driver) if monotonic else None,
            "dequeue_to_decode_ms": _summary_ns(np.asarray(decoded_mono) - dequeue),
            "decode_to_encoder_write_ms": _summary_ns(np.asarray(encoded_start) - decoded_mono),
            "encoder_write_ms": _summary_ns(np.asarray(encoded_end) - encoded_start),
            "host_clock_pair_span_ms": _summary_ns([int(r["host_clock_pair_span_ns"]) for r in rows]),
            "wall_minus_monotonic_offset_range_ms": float(np.ptp(offsets) / 1e6),
        }
    left, right = streams["left"], streams["right"]
    report = {
        "session": str(session.resolve()), "streams": per_side,
        "stereo_host_decode": _series_pairing(left["host"], right["host"], tolerance_ms),
        "stereo_host_dequeue": _series_pairing(left["dequeue"], right["dequeue"], tolerance_ms),
        "stereo_host_decode_monotonic": _series_pairing(
            left["decode_monotonic"], right["decode_monotonic"], tolerance_ms),
        "stereo_driver": {"available": False, "reason": "requires positive MONOTONIC timestamps and the same source flag on both streams"},
        "interpretation": [
            "Only the driver MONOTONIC clock is compared with host monotonic time; UNKNOWN/COPY are not guessed.",
            "Source flags are driver declarations, not proof of upstream sensor exposure synchronization.",
            "Driver disparity includes the source/capture-card/driver layers; this log alone cannot separate them.",
            "host timestamps used by convert are still the wall-clock time after decoding.",
            "Sequence gaps indicate unobserved driver frame numbers, not a diagnosis of hardware failure.",
        ],
    }
    comparable = (left["monotonic"] and right["monotonic"]
                  and left["sources"] in ({"start_of_exposure"}, {"end_of_frame"})
                  and left["sources"] == right["sources"])
    if comparable:
        report["stereo_driver"] = _series_pairing(left["driver"], right["driver"], tolerance_ms)
        if report["stereo_driver"]["available"]:
            pairs = pair_frames([Frame(t, i) for i, t in enumerate(left["driver"])],
                                [Frame(t, i) for i, t in enumerate(right["driver"])],
                                round(tolerance_ms * 1e6))
            differences = {"driver": [], "dequeue": [], "decode_monotonic": []}
            for lframe, rframe in pairs:
                for key, values in differences.items():
                    values.append(right[key][rframe.index] - left[key][lframe.index])
            report["same_driver_pairs"] = {
                "pairs": len(pairs),
                "right_minus_left_ms": {key: _summary_ns(values) for key, values in differences.items()},
                "right_minus_left_driver_to_decode_delay_ms": _summary_ns([
                    decoded - driver for decoded, driver in zip(
                        differences["decode_monotonic"], differences["driver"])]),
            }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="one raw episode_N directory")
    parser.add_argument("--tolerance-ms", type=float, default=15.0)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)
    try:
        report = analyze_session(args.session, args.tolerance_ms)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"stereo timing: {exc}\n")
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
