"""DeviceRegistry — stores one AffineCalibration per device_hw_domain.

Used by local replay and future aligned_observation code to convert
`Sample.device_hw_ns` into effective host-monotonic time. Thread-safe:
SyncManager writes during calibration, consumers read at query time.
"""
from __future__ import annotations

import threading
from typing import Optional

from robolineage_data_source.sync.calibration import AffineCalibration


class DeviceRegistry:
    def __init__(self) -> None:
        self._calibs: dict[str, AffineCalibration] = {}
        self._lock = threading.Lock()

    def set(self, domain: str, calib: AffineCalibration) -> None:
        with self._lock:
            self._calibs[domain] = calib

    def get(self, domain: str) -> Optional[AffineCalibration]:
        with self._lock:
            return self._calibs.get(domain)

    def calibrate(self, domain: str, device_hw_ns: int) -> Optional[int]:
        """Convert a device timestamp to effective host-monotonic time.
        Returns None if the domain has no calibration.
        """
        calib = self.get(domain)
        if calib is None:
            return None
        return calib.apply(device_hw_ns)

    def list_domains(self) -> list[str]:
        with self._lock:
            return list(self._calibs.keys())
