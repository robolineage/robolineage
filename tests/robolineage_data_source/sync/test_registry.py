"""Tests for DeviceRegistry."""
import pytest

from robolineage_data_source.sync.calibration import AffineCalibration
from robolineage_data_source.sync.registry import DeviceRegistry


def test_registry_starts_empty():
    reg = DeviceRegistry()
    assert reg.get("realsense_global") is None


def test_registry_set_and_get():
    reg = DeviceRegistry()
    calib = AffineCalibration(a=1.0, b=1_234_567)
    reg.set("realsense_global", calib)
    assert reg.get("realsense_global") is calib


def test_registry_calibrate_uses_stored():
    reg = DeviceRegistry()
    reg.set("realsense_global", AffineCalibration(a=1.0, b=1_000_000))
    assert reg.calibrate("realsense_global", 500) == 1_000_500


def test_registry_calibrate_unknown_domain_returns_none():
    reg = DeviceRegistry()
    assert reg.calibrate("unknown_domain", 500) is None


def test_registry_overwrite():
    reg = DeviceRegistry()
    reg.set("d1", AffineCalibration(a=1.0, b=1))
    reg.set("d1", AffineCalibration(a=1.0, b=2))
    assert reg.calibrate("d1", 0) == 2


def test_registry_list_domains():
    reg = DeviceRegistry()
    reg.set("d1", AffineCalibration(a=1.0, b=0))
    reg.set("d2", AffineCalibration(a=1.0, b=0))
    assert set(reg.list_domains()) == {"d1", "d2"}


def test_registry_is_threadsafe():
    import threading
    reg = DeviceRegistry()
    def writer(k):
        for i in range(100):
            reg.set(f"d{k}", AffineCalibration(a=1.0, b=i))
    threads = [threading.Thread(target=writer, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(reg.list_domains()) == {"d0", "d1", "d2", "d3"}


def test_registry_is_threadsafe_same_key():
    """Writers mutating a single key must never produce a None or partial
    read for subsequent get() calls (matches module docstring claim about
    SyncManager-writes-and-consumers-reads pattern)."""
    import threading
    import time
    reg = DeviceRegistry()
    reg.set("d", AffineCalibration(a=1.0, b=0))  # seed

    stop = threading.Event()
    errors: list[BaseException] = []

    def writer():
        i = 0
        while not stop.is_set():
            reg.set("d", AffineCalibration(a=1.0, b=i))
            i += 1

    def reader():
        try:
            while not stop.is_set():
                c = reg.get("d")
                assert c is not None
                assert c.a == 1.0  # fully constructed
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer)] + [
        threading.Thread(target=reader) for _ in range(3)
    ]
    for t in threads:
        t.start()
    time.sleep(0.05)
    stop.set()
    for t in threads:
        t.join()

    assert not errors, f"concurrent readers saw inconsistent state: {errors}"
