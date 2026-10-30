"""Command line entry point for the Vision Pro stereo viewer server."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """Parse options and serve the stereo stream until Ctrl+C."""
    parser = argparse.ArgumentParser(prog="dvrk-avp", description=__doc__)
    parser.add_argument("--left", default="/dev/video2", help="left-eye capture device")
    parser.add_argument("--right", default="/dev/video0", help="right-eye capture device")
    parser.add_argument("--cam-width", type=int, default=1920)
    parser.add_argument("--cam-height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=0, help="streamed width per eye, 0 = native")
    parser.add_argument("--quality", type=int, default=75, help="JPEG quality 1-100")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--cert", default=".certs/avp.crt")
    parser.add_argument("--key", default=".certs/avp.key")
    args = parser.parse_args(argv)

    from dvrk_avp.server import serve

    serve(args)
    return 0
