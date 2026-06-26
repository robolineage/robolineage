"""Tests for affine clock calibration."""
import numpy as np
import pytest

from robolineage_data_source.sync.calibration import (
    AffineCalibration,
    fit_affine,
)


def test_fit_affine_perfect_identity():
    device_hw = np.array([1_000, 2_000, 3_000, 4_000], dtype=np.int64)
    host_mono = device_hw.copy()
    calib = fit_affine(device_hw, host_mono)
    assert calib.a == pytest.approx(1.0, abs=1e-9)
    assert calib.b == pytest.approx(0.0, abs=1e-6)


def test_fit_affine_with_offset():
    device_hw = np.array([0, 1_000, 2_000, 3_000], dtype=np.int64)
    host_mono = device_hw + 1_234_567
    calib = fit_affine(device_hw, host_mono)
    assert calib.a == pytest.approx(1.0, abs=1e-9)
    assert calib.b == pytest.approx(1_234_567.0, abs=1.0)


def test_fit_affine_with_drift():
    # Clocks drift at slightly different rates
    device_hw = np.linspace(0, 1_000_000, 50).astype(np.int64)
    host_mono = (device_hw * 1.000_010 + 42.0).astype(np.int64)
    calib = fit_affine(device_hw, host_mono)
    assert calib.a == pytest.approx(1.000_010, abs=1e-6)
    assert calib.b == pytest.approx(42.0, abs=100.0)  # noisy due to int truncation


def test_fit_affine_requires_at_least_two_points():
    with pytest.raises(ValueError, match="at least 2"):
        fit_affine(np.array([1]), np.array([1]))


def test_fit_affine_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        fit_affine(np.array([1, 2]), np.array([1, 2, 3]))


def test_affine_calibration_apply():
    calib = AffineCalibration(a=1.0, b=1_000_000)
    assert calib.apply(500) == 1_000_500


def test_affine_calibration_apply_with_slope():
    calib = AffineCalibration(a=2.0, b=100)
    assert calib.apply(50) == 200  # 2*50 + 100


def test_residual_std_is_zero_on_exact_line():
    device_hw = np.array([0, 100, 200, 300], dtype=np.int64)
    host_mono = device_hw * 2 + 7
    calib = fit_affine(device_hw, host_mono)
    assert calib.residual_std_ns == pytest.approx(0.0, abs=1.0)


def test_residual_std_nonzero_with_noise():
    rng = np.random.default_rng(42)
    device_hw = np.arange(0, 1_000_000, 10_000, dtype=np.int64)
    noise = rng.normal(0, 1000, size=device_hw.size).astype(np.int64)
    host_mono = device_hw + 500_000 + noise
    calib = fit_affine(device_hw, host_mono)
    assert calib.residual_std_ns > 100  # noise is ~1000 ns
    assert calib.residual_std_ns < 2000
