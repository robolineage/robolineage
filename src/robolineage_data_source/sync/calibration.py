"""Affine clock calibration: host_mono_ns = a * device_hw_ns + b.

Given a sequence of (device_hw_ns, host_mono_ns) pairs captured during an
adapter's warmup window, fit a least-squares line so we can later convert any
device timestamp into host-monotonic time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineCalibration:
    """A fitted host↔device clock affine mapping.

    Attributes:
        a: slope (host_mono delta per device_hw delta; close to 1.0 for well-synced clocks)
        b: offset in ns (host_mono when device_hw is 0)
        residual_std_ns: standard deviation of residuals after fit; a diagnostic
            for "is this clock stable enough to trust?"
        n_samples: number of calibration points used
    """
    a: float
    b: float
    residual_std_ns: float = 0.0
    n_samples: int = 0

    def apply(self, device_hw_ns: int) -> int:
        """Convert a device timestamp to effective host-monotonic time.

        Uses round-half-to-even rather than truncation to avoid systematic
        sub-ns bias when slope ≠ 1.0 exactly (common for real fits).
        """
        return int(round(self.a * device_hw_ns + self.b))


def fit_affine(device_hw_ns: np.ndarray, host_mono_ns: np.ndarray) -> AffineCalibration:
    """Fit `host_mono_ns ~= a * device_hw_ns + b` by ordinary least squares.

    Requires at least 2 points. Arrays must have equal length.
    """
    device_hw_ns = np.asarray(device_hw_ns)
    host_mono_ns = np.asarray(host_mono_ns)

    if device_hw_ns.shape != host_mono_ns.shape:
        raise ValueError(
            f"device_hw_ns and host_mono_ns must have the same length; "
            f"got {device_hw_ns.shape} and {host_mono_ns.shape}"
        )
    if device_hw_ns.size < 2:
        raise ValueError("need at least 2 calibration points to fit an affine")

    # Fit on centered values. Raw monotonic timestamps are large enough that
    # a direct polyfit can produce unstable intercepts even when the true
    # device-host offset is constant.
    x = device_hw_ns.astype(np.longdouble)
    y = host_mono_ns.astype(np.longdouble)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    dx = x - x_mean
    dy = y - y_mean
    denom = np.sum(dx * dx)
    if denom == 0:
        raise ValueError("device_hw_ns samples must not all be equal")
    slope = np.sum(dx * dy) / denom
    intercept = y_mean - slope * x_mean
    predicted = slope * x + intercept
    residuals = y - predicted
    return AffineCalibration(
        a=float(slope),
        b=float(intercept),
        residual_std_ns=float(np.std(residuals)),
        n_samples=int(device_hw_ns.size),
    )
