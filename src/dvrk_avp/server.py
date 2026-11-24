"""HTTPS + WebSocket server that streams the stereo pair to the WebXR viewer."""

from __future__ import annotations

import asyncio
import ssl
import struct
from pathlib import Path

import cv2
import numpy as np
from aiohttp import WSMsgType, web

from dvrk_avp.capture import LatestFrameCamera

VIEWER = Path(__file__).with_name("viewer.html")


def encode(frame: np.ndarray, width: int, quality: int) -> bytes:
    """Optionally downscale to `width` (0 keeps size) and JPEG-encode one eye."""
    if width and frame.shape[1] != width:
        height = round(frame.shape[0] * width / frame.shape[1])
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def pack(left: np.ndarray, right: np.ndarray, width: int, quality: int) -> bytes:
    """Encode both eyes as: uint32 little-endian left length, left JPEG, right JPEG."""
    left_jpg, right_jpg = encode(left, width, quality), encode(right, width, quality)
    return struct.pack("<I", len(left_jpg)) + left_jpg + right_jpg


def build_app(left: LatestFrameCamera, right: LatestFrameCamera, width: int, quality: int):
    """Create the aiohttp app serving the viewer page and the /ws frame stream."""

    async def index(_: web.Request) -> web.FileResponse:
        return web.FileResponse(VIEWER)

    async def stream(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        print(f"viewer connected: {request.remote}")
        loop = asyncio.get_running_loop()
        last = 0.0
        try:
            while not ws.closed:
                # Wait for a new left frame, then pair it with the newest right frame.
                (lf, lt), (rf, _) = left.latest(), right.latest()
                if lf is None or rf is None or lt <= last:
                    await asyncio.sleep(0.002)
                    continue
                last = lt
                msg = await loop.run_in_executor(None, pack, lf, rf, width, quality)
                await ws.send_bytes(msg)
                # Flow control: the viewer acks after decoding, so frames never queue up.
                ack = await ws.receive()
                if ack.type != WSMsgType.TEXT:
                    break
        except ConnectionResetError:
            pass
        print(f"viewer disconnected: {request.remote}")
        return ws

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", stream)
    return app


def serve(args) -> None:
    """Open both cameras and run the HTTPS server until Ctrl+C."""
    left = LatestFrameCamera(args.left, args.cam_width, args.cam_height, args.fps)
    right = LatestFrameCamera(args.right, args.cam_width, args.cam_height, args.fps)
    tls = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    tls.load_cert_chain(args.cert, args.key)
    try:
        app = build_app(left, right, args.width, args.quality)
        web.run_app(app, host=args.host, port=args.port, ssl_context=tls)
    finally:
        left.close()
        right.close()
