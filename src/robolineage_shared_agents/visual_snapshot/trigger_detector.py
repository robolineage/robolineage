"""
Implementation note.

Implementation note.
    Implementation note.
    Implementation note.
    Implementation note.

Implementation note.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .types import VisualObservationWindow

logger = logging.getLogger(__name__)


class TriggerDetector:
    """
    Implementation note.

    Args:
        Implementation note.
        Implementation note.
    """

    def __init__(
        self,
        rgb_diff_threshold: float = 0.05,
        watchdog_interval: float = 8.0,
    ):
        self.rgb_diff_threshold = rgb_diff_threshold
        self.watchdog_interval = watchdog_interval

        self._last_trigger_ts: float = -watchdog_interval  # Internal implementation note.
        self._last_color: Optional[np.ndarray] = None

    def should_trigger(self, window: VisualObservationWindow) -> bool:
        """Implementation note."""
        current_ts = window.end_timestamp

        # Internal implementation note.
        if current_ts - self._last_trigger_ts >= self.watchdog_interval:
            logger.debug(f"Watchdog triggered at t={current_ts:.2f}s")
            return True

        # Internal implementation note.
        current_color = self._latest_valid_frame(window.color_frames)
        if current_color is not None and self._last_color is not None:
            diff = float(
                np.mean(np.abs(current_color.astype(np.float32) - self._last_color.astype(np.float32)))
            ) / 255.0
            if diff > self.rgb_diff_threshold:
                logger.debug(f"RGB diff {diff:.4f} > threshold {self.rgb_diff_threshold}, triggered at t={current_ts:.2f}s")
                return True

        return False

    def mark_triggered(self, window: VisualObservationWindow) -> None:
        """Implementation note."""
        self._last_trigger_ts = window.end_timestamp
        current_color = self._latest_valid_frame(window.color_frames)
        if current_color is not None:
            self._last_color = current_color.copy()

    def reset(self) -> None:
        """Implementation note."""
        self._last_trigger_ts = -self.watchdog_interval
        self._last_color = None

    @staticmethod
    def _latest_valid_frame(frames: list) -> Optional[np.ndarray]:
        for f in reversed(frames):
            if f is not None:
                return f
        return None
