"""Tests for SyncManager — uses stub adapters, no hardware."""
import logging
import threading

import pytest

from robolineage_data_source.adapters.base import DeviceAdapter
from robolineage_data_source.config.schema import SyncGroupConfig
from robolineage_data_source.sample import HealthState, HealthStatus
from robolineage_data_source.sync.manager import SyncManager
from robolineage_data_source.sync.registry import DeviceRegistry


class _RecordingAdapter(DeviceAdapter):
    """Adapter that records the order of configure_sync / start / stop calls."""
    _call_order: list[str] = []
    _order_lock = threading.Lock()

    def __init__(self, name: str, supports_sync: bool = True):
        self.name = name
        self.sync_role: str | None = None
        self._supports_sync = supports_sync
        self._started = False

    def supports_hw_sync(self) -> bool:
        return self._supports_sync

    def configure_sync(self, role: str) -> None:
        if not self._supports_sync:
            from robolineage_data_source.adapters.base import UnsupportedSyncError
            raise UnsupportedSyncError(self.name)
        self.sync_role = role
        with _RecordingAdapter._order_lock:
            _RecordingAdapter._call_order.append(f"{self.name}.configure_sync({role})")

    def start(self):
        self._started = True
        with _RecordingAdapter._order_lock:
            _RecordingAdapter._call_order.append(f"{self.name}.start")

    def stop(self):
        self._started = False
        with _RecordingAdapter._order_lock:
            _RecordingAdapter._call_order.append(f"{self.name}.stop")

    def health(self):
        return HealthStatus(state=HealthState.OK if self._started else HealthState.NOT_STARTED)

@pytest.fixture(autouse=True)
def clear_call_order():
    _RecordingAdapter._call_order.clear()
    yield
    _RecordingAdapter._call_order.clear()


def test_sync_manager_starts_slaves_before_master():
    registry = DeviceRegistry()
    master = _RecordingAdapter("master")
    slave1 = _RecordingAdapter("slave1")
    slave2 = _RecordingAdapter("slave2")

    group = SyncGroupConfig(
        name="main",
        backend="realsense_inter_cam",
        master="master",
        slaves=["slave1", "slave2"],
    )

    mgr = SyncManager(
        registry=registry,
        adapters={"master": master, "slave1": slave1, "slave2": slave2},
        groups=[group],
    )
    mgr.start()

    # Assert: slaves configured + started before master
    order = _RecordingAdapter._call_order
    assert order.index("slave1.configure_sync(slave)") < order.index("master.configure_sync(master)")
    assert order.index("slave2.configure_sync(slave)") < order.index("master.configure_sync(master)")
    assert order.index("slave1.start") < order.index("master.start")
    assert order.index("slave2.start") < order.index("master.start")

    mgr.stop()


def test_sync_manager_starts_non_sync_adapters_too():
    """Adapters not listed in any group still get started."""
    registry = DeviceRegistry()
    robot = _RecordingAdapter("robot", supports_sync=False)

    mgr = SyncManager(
        registry=registry,
        adapters={"robot": robot},
        groups=[],  # no sync groups
    )
    mgr.start()

    assert "robot.start" in _RecordingAdapter._call_order
    mgr.stop()
    assert "robot.stop" in _RecordingAdapter._call_order


def test_sync_manager_wait_until_calibrated_true_without_collector():
    registry = DeviceRegistry()
    adapter = _RecordingAdapter("cam", supports_sync=False)
    mgr = SyncManager(
        registry=registry,
        adapters={"cam": adapter},
        groups=[],
    )
    mgr.start()

    assert mgr.wait_until_calibrated(timeout=1.0) is True
    assert registry.list_domains() == []

    mgr.stop()


def test_sync_manager_stop_reverses_start_order():
    """stop() tears down in reverse start order so slaves survive master long enough."""
    registry = DeviceRegistry()
    master = _RecordingAdapter("master")
    slave1 = _RecordingAdapter("slave1")
    mgr = SyncManager(
        registry=registry,
        adapters={"master": master, "slave1": slave1},
        groups=[SyncGroupConfig(name="main", backend="realsense_inter_cam",
                                master="master", slaves=["slave1"])],
    )
    mgr.start()
    _RecordingAdapter._call_order.clear()
    mgr.stop()
    order = _RecordingAdapter._call_order
    assert order.index("master.stop") < order.index("slave1.stop")


class _FailingStartAdapter(_RecordingAdapter):
    """Adapter whose start() raises — used to exercise rollback."""
    def start(self):
        with _RecordingAdapter._order_lock:
            _RecordingAdapter._call_order.append(f"{self.name}.start_attempt")
        raise RuntimeError(f"{self.name} boom")


class _FailingStopAdapter(_RecordingAdapter):
    """Adapter whose stop() raises — used to exercise stop's logged-swallow."""
    def stop(self):
        with _RecordingAdapter._order_lock:
            _RecordingAdapter._call_order.append(f"{self.name}.stop_attempt")
        raise RuntimeError(f"{self.name} stop boom")


def test_sync_manager_start_rolls_back_on_failure():
    """If any adapter fails to start, previously-started adapters are stopped in
    reverse order and the exception is re-raised; _started stays False and no
    calibration thread is launched."""
    registry = DeviceRegistry()
    slave1 = _RecordingAdapter("slave1")
    slave2 = _RecordingAdapter("slave2")
    # master's start raises — slaves should be rolled back.
    master = _FailingStartAdapter("master")

    mgr = SyncManager(
        registry=registry,
        adapters={"master": master, "slave1": slave1, "slave2": slave2},
        groups=[SyncGroupConfig(
            name="main", backend="realsense_inter_cam",
            master="master", slaves=["slave1", "slave2"],
        )],
    )

    with pytest.raises(RuntimeError, match="master boom"):
        mgr.start()

    # _started must still be False after a failed start.
    assert mgr._started is False

    order = _RecordingAdapter._call_order
    # Slaves started (slave1, slave2), master failed, then rollback stops in
    # reverse order of started_so_far (slave2 before slave1).
    assert "slave1.start" in order
    assert "slave2.start" in order
    assert "master.start_attempt" in order
    assert order.index("slave2.stop") < order.index("slave1.stop")
    assert order.index("slave1.start") < order.index("slave2.stop")

def test_sync_manager_stop_logs_but_does_not_raise(caplog):
    """A failing adapter.stop() must be logged and swallowed; other adapters
    still get their stop() called; stop() itself does not raise."""
    registry = DeviceRegistry()
    master = _FailingStopAdapter("master")
    slave1 = _RecordingAdapter("slave1")

    mgr = SyncManager(
        registry=registry,
        adapters={"master": master, "slave1": slave1},
        groups=[SyncGroupConfig(
            name="main", backend="realsense_inter_cam",
            master="master", slaves=["slave1"],
        )],
    )
    mgr.start()

    with caplog.at_level(logging.ERROR, logger="robolineage_data_source.sync.manager"):
        # Must not raise even though master.stop() throws.
        mgr.stop()

    order = _RecordingAdapter._call_order
    # Both adapters had their stop invoked (master in reverse order first).
    assert "master.stop_attempt" in order
    assert "slave1.stop" in order
    assert order.index("master.stop_attempt") < order.index("slave1.stop")

    # Failure was surfaced via logs rather than swallowed silently.
    assert any("master" in rec.getMessage() and "stop failed" in rec.getMessage()
               for rec in caplog.records)


def test_sync_manager_rejects_duplicate_membership():
    adapters = {
        "a": _RecordingAdapter("a", supports_sync=True),
        "b": _RecordingAdapter("b", supports_sync=True),
        "c": _RecordingAdapter("c", supports_sync=True),
    }
    groups = [
        SyncGroupConfig(name="g1", backend="x", master="a", slaves=["b"]),
        SyncGroupConfig(name="g2", backend="x", master="b", slaves=["c"]),
    ]
    registry = DeviceRegistry()
    with pytest.raises(ValueError, match="multiple sync groups"):
        SyncManager(registry=registry, adapters=adapters, groups=groups)


def test_sync_manager_rejects_group_referencing_unknown_adapter():
    registry = DeviceRegistry()
    with pytest.raises(ValueError, match="unknown"):
        SyncManager(
            registry=registry,
            adapters={"only_one": _RecordingAdapter("only_one")},
            groups=[SyncGroupConfig(name="g", backend="realsense_inter_cam",
                                    master="missing", slaves=[])],
        )
