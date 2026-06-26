"""Tests for Sample dataclass and HealthStatus."""
from dataclasses import asdict, FrozenInstanceError

import pytest

from robolineage_data_source.sample import HealthState, HealthStatus, Sample


def test_sample_minimal_construction():
    s = Sample(topic="cam/h", host_mono_ns=1_000_000_000, payload={"frame": 0})
    assert s.topic == "cam/h"
    assert s.host_mono_ns == 1_000_000_000
    assert s.payload == {"frame": 0}
    assert s.device_hw_ns is None
    assert s.device_hw_domain is None
    assert s.meta == {}


def test_sample_full_construction():
    s = Sample(
        topic="cam/h",
        host_mono_ns=1_000_000_000,
        payload=b"\x00\x01\x02",
        device_hw_ns=999_000_000,
        device_hw_domain="realsense_global",
        meta={"exposure_us": 8000},
    )
    assert s.device_hw_ns == 999_000_000
    assert s.device_hw_domain == "realsense_global"
    assert s.meta["exposure_us"] == 8000


def test_sample_is_frozen():
    s = Sample(topic="x", host_mono_ns=0, payload=None)
    with pytest.raises(FrozenInstanceError):
        s.topic = "y"  # type: ignore[misc]


def test_sample_meta_is_immutable():
    s = Sample(topic="a", host_mono_ns=0, payload=None, meta={"k": "v"})
    assert s.meta["k"] == "v"
    with pytest.raises(TypeError):
        s.meta["k"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        s.meta["new"] = "x"  # type: ignore[index]


def test_health_state_values():
    assert HealthState.OK.value == "ok"
    assert HealthState.DEGRADED.value == "degraded"
    assert HealthState.FAILED.value == "failed"
    assert HealthState.NOT_STARTED.value == "not_started"


def test_health_status_defaults():
    h = HealthStatus(state=HealthState.OK)
    assert h.state is HealthState.OK
    assert h.message == ""
    assert h.last_sample_mono_ns is None


def test_health_status_is_frozen():
    h = HealthStatus(state=HealthState.OK)
    with pytest.raises(FrozenInstanceError):
        h.message = "x"  # type: ignore[misc]


def test_health_status_dict_roundtrip():
    import json
    h = HealthStatus(state=HealthState.DEGRADED, message="lag", last_sample_mono_ns=42)
    d = asdict(h)
    assert d["state"] == "degraded"  # str-Enum serializes as its string value
    assert d["message"] == "lag"
    assert d["last_sample_mono_ns"] == 42
    # Must be JSON-serializable end-to-end
    encoded = json.dumps(d)
    assert json.loads(encoded) == {
        "state": "degraded",
        "message": "lag",
        "last_sample_mono_ns": 42,
        "meta": {},
    }
