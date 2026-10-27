"""Live stereo preview window with a recording-status overlay."""

from __future__ import annotations

import cv2
import numpy as np

WINDOW = "dVRK recorder  [s] start  [e] end  [q] quit"
PREVIEW_HEIGHT = 480


def show(left: np.ndarray, right: np.ndarray, status: str, recording: bool) -> str | None:
    """Draw both eyes side by side and return a key pressed in the window, if any."""
    width = round(left.shape[1] * PREVIEW_HEIGHT / left.shape[0])
    canvas = np.hstack([cv2.resize(image, (width, PREVIEW_HEIGHT)) for image in (left, right)])
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
