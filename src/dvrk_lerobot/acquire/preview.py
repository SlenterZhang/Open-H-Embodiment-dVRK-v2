"""Live stereo preview window with a recording-status overlay."""

from __future__ import annotations

import cv2
import numpy as np

WINDOW = "dVRK recorder  [s] start  [e] end  [q] quit"
PREVIEW_HEIGHT = 480
PREVIEW_MAX_WIDTH = 1800


def show(
    left: np.ndarray, right: np.ndarray, status: str, recording: bool,
    external: np.ndarray | None = None,
) -> str | None:
    """Draw all active color views and return a key pressed in the window."""
    images = (left, right) if external is None else (left, right, external)
    canvas = np.hstack([
        cv2.resize(image, (round(image.shape[1] * PREVIEW_HEIGHT / image.shape[0]), PREVIEW_HEIGHT))
        for image in images
    ])
    if canvas.shape[1] > PREVIEW_MAX_WIDTH:
        canvas = cv2.resize(
            canvas,
            (PREVIEW_MAX_WIDTH, round(canvas.shape[0] * PREVIEW_MAX_WIDTH / canvas.shape[1])),
        )
    color = (0, 0, 255) if recording else (0, 200, 0)
    cv2.putText(canvas, status, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow(WINDOW, canvas)
    return poll_key()


def poll_key() -> str | None:
    """Process window events for a moment and return a pressed key in lower case."""
    key = cv2.waitKey(15) & 0xFF
    return None if key == 0xFF else chr(key).lower()


def close() -> None:
    """Destroy the preview window."""
    cv2.destroyAllWindows()
