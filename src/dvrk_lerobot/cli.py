"""Command line entry points for recording, conversion, verification, and checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dvrk_lerobot.acquire.session import load_session
from dvrk_lerobot.align.sync import AlignOptions, align_session
from dvrk_lerobot.config import ConfigError, load_config
from dvrk_lerobot.dataset.writer import export_dataset
from dvrk_lerobot.tools.doctor import run_doctor
from dvrk_lerobot.tools.verify import verify_dataset


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch one subcommand."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            return _record(args)
        if args.command == "convert":
            return _convert(args)
        if args.command == "verify":
            return _verify(args)
        if args.command == "doctor":
            return _doctor(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.print_help()
    return 2


def _record(args: argparse.Namespace) -> int:
    from dvrk_lerobot.acquire.recorder import run_record

    config = load_config(args.config)
    paths = run_record(config, Path(args.root), task=args.task, notes=args.notes)
    print(f"recorded {len(paths)} episode(s)")
    for path in paths:
        print(f"raw session: {path}")
    return 0


def _convert(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    options = AlignOptions(
        fps=config.dataset.fps,
        stereo_tolerance_ms=config.alignment.stereo_tolerance_ms,
        robot_tolerance_ms=config.alignment.robot_tolerance_ms,
    )
    episodes = []
    for path, from_run in _session_paths(args.sessions):
        session = load_session(path)
        status = session.meta.get("status")
        if status != "ok" and not args.allow_incomplete:
            if from_run:
                print(f"[convert] skipping {path}: status {status!r}")
                continue
            raise ValueError(
                f"{path}: session status is {status!r}; use --allow-incomplete to convert it"
            )
        episode = align_session(session, config.arms, options)
        for note in episode.notes:
            print(f"[align] {path.parent.name}/{path.name}: {note}")
        episodes.append(episode)
    result = export_dataset(episodes, config, Path(args.output))
    print(f"dataset: {result.root}")
    print(f"episodes: {result.episodes}, frames: {result.frames}")
    return 0


def _session_paths(paths: list[str]) -> list[tuple[Path, bool]]:
    """Expand run directories into their episode_N folders in numeric order."""
    result: list[tuple[Path, bool]] = []
    for path in map(Path, paths):
        if (path / "meta.json").is_file():
            result.append((path, False))
            continue
        episodes = [item for item in path.glob("episode_*") if item.name[8:].isdigit()]
        if not episodes:
            raise ValueError(f"{path}: neither an episode nor a run directory")
        result.extend((item, True) for item in sorted(episodes, key=lambda p: int(p.name[8:])))
    return result


def _verify(args: argparse.Namespace) -> int:
    summary = verify_dataset(args.dataset)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("dataset structure: PASS")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    return 0 if run_doctor(config, topic_timeout=args.topics_timeout) else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dvrk-lerobot",
        description="Record dVRK ROS 2 data and convert it to LeRobot Dataset v2.1.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser(
        "record", help="record episodes interactively: s = start, e = end, q = quit"
    )
    _config_argument(record)
    record.add_argument("--root", default="data/raw", help="raw session root directory")
    record.add_argument("--task", required=True, help="task description stored in metadata")
    record.add_argument("--notes", default="", help="optional free-form notes")

    convert = subparsers.add_parser(
        "convert", help="align raw episodes in time and convert them to LeRobot v2.1"
    )
    _config_argument(convert)
    convert.add_argument(
        "--sessions", nargs="+", required=True, help="run directories and/or episode directories"
    )
    convert.add_argument("--output", required=True, help="new dataset directory")
    convert.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="convert sessions whose meta.json status is not 'ok'",
    )

    verify = subparsers.add_parser("verify", help="check a converted dataset")
    verify.add_argument("--dataset", required=True, help="dataset root directory")

    doctor = subparsers.add_parser("doctor", help="check tools, cameras, and topics")
    _config_argument(doctor)
    doctor.add_argument(
        "--topics-timeout", type=float, default=5.0, help="seconds to listen for each topic set"
    )
    return parser


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c", "--config", default="configs/record.yaml", help="YAML configuration path"
    )


if __name__ == "__main__":
    raise SystemExit(main())
