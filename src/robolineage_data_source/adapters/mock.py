"""MockAdapter — emits synthetic liveness ticks at a fixed rate.

Used for smoke tests and as a reference implementation of the adapter
lifecycle. Does not touch any hardware.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

from robolineage_data_source.adapters.base import DeviceAdapter
from robolineage_data_source.sample import HealthState, HealthStatus


_DefaultPayload = Callable[[int], Any]


def _default_payload_factory(seq: int) -> dict[str, int]:
    return {"seq": seq}


class MockAdapter(DeviceAdapter):
    """Runs a synthetic source loop every 1/`rate_hz` seconds.

    Args:
        topic: Topic string to publish under (e.g. "mock/test").
        rate_hz: Publishing rate in Hz. Must be > 0.
        payload_factory: Callable(seq_int) -> payload. Defaults to {"seq": seq}.
    """

    def __init__(
        self,
        topic: str,
        rate_hz: float = 30.0,
        payload_factory: _DefaultPayload | None = None,
    ) -> None:
        if rate_hz <= 0:
            raise ValueError(f"rate_hz must be > 0, got {rate_hz}")
        self._topic = topic
        self._period_ns = int(1e9 / rate_hz)
        self._factory: _DefaultPayload = payload_factory or _default_payload_factory
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_mono: int | None = None
        self._started = False
        self._last_payload: Any | None = None

    def start(self) -> None:
        if self._started:
            raise RuntimeError("MockAdapter already started")
        self._stop_event.clear()
        self._last_mono = None
        self._started = True
        self._thread = threading.Thread(
            target=self._run,
            name=f"MockAdapter[{self._topic}]",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._started = False

    def health(self) -> HealthStatus:
        if not self._started:
            return HealthStatus(state=HealthState.NOT_STARTED)
        if self._last_mono is None:
            return HealthStatus(
                state=HealthState.DEGRADED,
                message="no ticks emitted yet",
            )
        return HealthStatus(
            state=HealthState.OK,
            last_sample_mono_ns=self._last_mono,
            meta={"topic": self._topic, "last_payload": self._last_payload},
        )

    def _run(self) -> None:
        seq = 0
        next_tick_ns = time.monotonic_ns()
        while not self._stop_event.is_set():
            now_ns = time.monotonic_ns()
            wait_ns = next_tick_ns - now_ns
            if wait_ns > 0:
                # Sleep in short chunks so stop() responds promptly
                if self._stop_event.wait(timeout=wait_ns / 1e9):
                    return
                now_ns = time.monotonic_ns()
            self._last_payload = self._factory(seq)
            self._last_mono = now_ns
            seq += 1
            next_tick_ns += self._period_ns
