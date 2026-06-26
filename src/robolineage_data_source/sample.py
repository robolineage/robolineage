"""Core data types used throughout the data-source layer.

`Sample` is the lightweight observation unit still used by local mocks,
offline replay, and hardware sync utilities. Production live/raw capture reads
ROS2 topics directly and stores rosbag2; `HealthStatus` reports the runtime
state of an adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class HealthState(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"
    NOT_STARTED = "not_started"


@dataclass(frozen=True)
class HealthStatus:
    """Runtime health snapshot of one DeviceAdapter.

    Attributes:
        state: coarse status (NOT_STARTED / OK / DEGRADED / FAILED).
        message: human-readable detail (one short line; ASCII or UTF-8).
        last_sample_mono_ns: host monotonic ns of the most recently published
            sample (None if no sample yet). Used for liveness probes.
        meta: optional adapter-specific telemetry dict (Phase 5+). Copied in
            __post_init__ so later caller-side mutations do not leak into the
            health snapshot. Kept as a plain dict because health payloads are
            frequently passed through dataclasses.asdict() and JSON endpoints.
            Typical entries: {"jpeg_decode_failures": {topic: count}, ...}.
    """
    state: HealthState
    message: str = ""
    last_sample_mono_ns: int | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The dataclass itself is frozen; copy nested metadata for snapshot
        # isolation while keeping it JSON/asdict friendly.
        object.__setattr__(self, "meta", dict(self.meta))


@dataclass(frozen=True)
class Sample:
    """A single observation produced by a device adapter.

    Attributes:
        topic: Canonical topic string, e.g. "camera/primary/color", "robot/active/state".
        host_mono_ns: Host monotonic timestamp (time.monotonic_ns()) captured
            at the moment the orchestrator first saw this sample. Main
            time-base for all cross-device alignment.
        payload: Opaque data. May be a numpy ndarray (frame), dict (pose/imu),
            bytes (encoded), or any other type the consumer understands.
        device_hw_ns: Optional SDK-reported hardware timestamp, interpretable
            only within `device_hw_domain`. None when the device has no
            hardware clock.
        device_hw_domain: Identifier of the hardware clock domain, e.g.
            "realsense_global", "zed", or None. Used by SyncManager to apply
            the correct affine calibration.
        meta: Free-form dict for adapter-specific fields (exposure, gain, etc.).
    """
    topic: str
    host_mono_ns: int
    payload: Any
    device_hw_ns: int | None = None
    device_hw_domain: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Freeze meta so the "frozen=True" contract holds end-to-end.
        # Accept either a dict (convert) or an existing Mapping (pass through).
        if isinstance(self.meta, MappingProxyType):
            return
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))
