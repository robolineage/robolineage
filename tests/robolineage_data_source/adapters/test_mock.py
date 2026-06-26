"""Tests for MockAdapter."""
import time

from robolineage_data_source.adapters.mock import MockAdapter
from robolineage_data_source.sample import HealthState


def test_mock_emits_liveness_ticks_at_expected_rate():
    adapter = MockAdapter(topic="mock/test", rate_hz=100.0)
    adapter.start()

    deadline = time.monotonic() + 0.3
    last = None
    while time.monotonic() < deadline:
        health = adapter.health()
        if health.last_sample_mono_ns is not None:
            last = health
        time.sleep(0.02)

    adapter.stop()

    assert last is not None
    assert last.meta["topic"] == "mock/test"
    assert last.meta["last_payload"]["seq"] >= 1


def test_mock_health_transitions():
    adapter = MockAdapter(topic="mock/test", rate_hz=50.0)

    assert adapter.health().state is HealthState.NOT_STARTED

    adapter.start()
    # Give the thread a moment to emit at least one tick.
    time.sleep(0.1)
    assert adapter.health().state is HealthState.OK
    assert adapter.health().last_sample_mono_ns is not None

    adapter.stop()
    assert adapter.health().state is HealthState.NOT_STARTED


def test_mock_custom_payload_factory():
    adapter = MockAdapter(
        topic="mock/test",
        rate_hz=100.0,
        payload_factory=lambda seq: {"seq": seq, "doubled": seq * 2},
    )
    adapter.start()
    time.sleep(0.1)
    health = adapter.health()

    adapter.stop()

    payload = health.meta["last_payload"]
    assert payload["doubled"] == payload["seq"] * 2


def test_mock_cannot_double_start():
    import pytest
    adapter = MockAdapter(topic="mock/test", rate_hz=50.0)
    adapter.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            adapter.start()
    finally:
        adapter.stop()


def test_mock_stop_is_idempotent():
    adapter = MockAdapter(topic="mock/test", rate_hz=50.0)
    adapter.start()
    adapter.stop()
    adapter.stop()  # must not raise
