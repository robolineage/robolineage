"""DeviceAdapter: base class for source supervision integrations.

Each adapter owns one source-side runtime such as a ROS2 profile monitor or a
hardware sync controller. Raw capture and online consumers subscribe to ROS2
topics directly; adapters no longer forward samples through an in-process
transport.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from robolineage_data_source.sample import HealthStatus


class UnsupportedSyncError(Exception):
    """Raised when hardware sync is requested on an adapter that does not support it."""


class DeviceAdapter(ABC):
    """Abstract base for every device adapter.

    Lifecycle contract:
        adapter = MyAdapter(...)               # cheap, no I/O
        adapter.configure_sync("master")       # optional, before start()
        adapter.start()                        # allocate SDK, spawn threads
        adapter.stop()                         # clean shutdown
    """

    @abstractmethod
    def start(self) -> None:
        """Initialize source-side resources and start background threads."""

    @abstractmethod
    def stop(self) -> None:
        """Signal threads to terminate and release hardware. Idempotent."""

    @abstractmethod
    def health(self) -> HealthStatus:
        """Return current health. Must not block on hardware."""

    def supports_hw_sync(self) -> bool:
        """Override to advertise hardware trigger / inter-camera sync capability."""
        return False

    def configure_sync(self, role: str) -> None:
        """Configure hardware sync role before start().

        `role` is typically one of "master", "slave", "none". Adapters that
        advertise `supports_hw_sync() is True` must override this method and
        apply the configuration. The default implementation rejects the call.
        """
        if not self.supports_hw_sync():
            raise UnsupportedSyncError(
                f"{type(self).__name__} does not support hardware sync "
                f"(requested role={role!r})"
            )
