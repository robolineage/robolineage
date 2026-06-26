from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class VideoSource(Protocol):
    """Minimal interface for a frame provider (file or live camera)."""
    def read(self) -> Optional[np.ndarray]: ...
    def release(self) -> None: ...
    def is_live(self) -> bool: ...


class SyntheticVideoSource:
    """Returns identical solid-colour frames. Use in tests — no file I/O or camera."""

    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        color: tuple[int, int, int] = (80, 80, 80),
    ) -> None:
        self._frame = np.full((height, width, 3), color, dtype=np.uint8)

    def read(self) -> Optional[np.ndarray]:
        return self._frame.copy()

    def release(self) -> None:
        pass

    def is_live(self) -> bool:
        return True


class LatestFrameVideoSource:
    """Reads the latest frame from a callback, with an optional fallback source."""

    def __init__(
        self,
        frame_provider: Callable[[], Optional[np.ndarray]],
        *,
        fallback: Optional[VideoSource] = None,
    ) -> None:
        self._frame_provider = frame_provider
        self._fallback = fallback

    def read(self) -> Optional[np.ndarray]:
        frame = self._frame_provider()
        if frame is not None:
            return frame.copy()
        if self._fallback is not None:
            return self._fallback.read()
        return None

    def release(self) -> None:
        if self._fallback is not None:
            self._fallback.release()

    def is_live(self) -> bool:
        return True


class FileVideoSource:
    """Reads frames from an MP4 file via OpenCV; loops back to the start when done.

    Looping is intentional for C_rollout replay scenarios where a pre-recorded
    demonstration is looped as the background video during policy dry-run.
    """

    def __init__(self, path: Path) -> None:
        self._path = str(path)
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self._path}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        if not ok:
            # EOF — loop back to beginning
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def is_live(self) -> bool:
        return False

    def __del__(self) -> None:
        self.release()


class LiveCameraSource:
    """Reads live frames from a V4L2/USB camera via OpenCV.

    device_index: 0 = first USB camera, 1 = second, etc.
    On the industrial PC, the head camera is typically /dev/video0 → index 0.
    """

    def __init__(self, device_index: int = 0) -> None:
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {device_index}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def is_live(self) -> bool:
        return True

    def __del__(self) -> None:
        self.release()
