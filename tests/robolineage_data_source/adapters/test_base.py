"""Tests for DeviceAdapter ABC contract."""
import pytest

from robolineage_data_source.adapters.base import DeviceAdapter, UnsupportedSyncError
from robolineage_data_source.sample import HealthState, HealthStatus


class _StubAdapter(DeviceAdapter):
    """Minimal concrete subclass for ABC contract testing."""
    def __init__(self):
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def health(self):
        return HealthStatus(
            state=HealthState.OK if self.started else HealthState.NOT_STARTED
        )


class _SyncCapableAdapter(_StubAdapter):
    def supports_hw_sync(self) -> bool:
        return True

    def configure_sync(self, role: str) -> None:
        self.sync_role = role


def test_cannot_instantiate_abstract_adapter():
    with pytest.raises(TypeError):
        DeviceAdapter()  # type: ignore[abstract]


def test_subclass_must_implement_start_stop_health():
    class Broken(DeviceAdapter):
        pass

    with pytest.raises(TypeError):
        Broken()  # type: ignore[abstract]


def test_stub_adapter_lifecycle():
    a = _StubAdapter()
    assert a.health().state is HealthState.NOT_STARTED
    a.start()
    assert a.health().state is HealthState.OK
    a.stop()
    assert a.health().state is HealthState.NOT_STARTED


def test_default_supports_hw_sync_is_false():
    a = _StubAdapter()
    assert a.supports_hw_sync() is False


def test_default_configure_sync_raises_unsupported():
    a = _StubAdapter()
    with pytest.raises(UnsupportedSyncError):
        a.configure_sync("master")


def test_configure_sync_accepts_when_supported():
    a = _SyncCapableAdapter()
    a.configure_sync("master")
    assert a.sync_role == "master"
