"""Command line entry points for recording, export, verification, and checks."""

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
        if args.command == "export":
            return _export(args)
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

    if args.duration < 0:
        raise ValueError("--duration must be zero or positive")
    config = load_config(args.config)
    path = run_record(
        config,
        Path(args.root),
        task=args.task,
        operator=args.operator,
        notes=args.notes,
        duration=args.duration,
    )
    print(f"raw session: {path}")
    return 0


def _export(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    options = AlignOptions(
        fps=config.dataset.fps,
        stereo_tolerance_ms=config.alignment.stereo_tolerance_ms,
        robot_tolerance_ms=config.alignment.robot_tolerance_ms,
    )
    episodes = []
    for path in args.sessions:
        session = load_session(path)
        status = session.meta.get("status")
        if status != "ok" and not args.allow_incomplete:
            raise ValueError(
                f"{path}: session status is {status!r}; use --allow-incomplete to export it"
            )
        episode = align_session(session, config.arms, options)
        for note in episode.notes:
            print(f"[align] {Path(path).name}: {note}")
        episodes.append(episode)
    result = export_dataset(episodes, config, Path(args.output))
    print(f"dataset: {result.root}")
    print(f"episodes: {result.episodes}, frames: {result.frames}")
    return 0


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
        description="Record dVRK ROS 2 data and export LeRobot Dataset v2.1.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="record one raw session")
    _config_argument(record)
    record.add_argument("--root", default="data/raw", help="raw session root directory")
    record.add_argument("--task", required=True, help="task description stored in metadata")
    record.add_argument("--operator", default="", help="optional operator name")
    record.add_argument("--notes", default="", help="optional free-form notes")
    record.add_argument(
        "--duration", type=float, default=0.0, help="seconds to record; 0 means until Ctrl+C"
    )

    export = subparsers.add_parser("export", help="export raw sessions to LeRobot v2.1")
    _config_argument(export)
    export.add_argument("--sessions", nargs="+", required=True, help="raw session directories")
    export.add_argument("--output", required=True, help="new dataset directory")
    export.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="export sessions whose meta.json status is not 'ok'",
    )

    verify = subparsers.add_parser("verify", help="check an exported dataset")
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
