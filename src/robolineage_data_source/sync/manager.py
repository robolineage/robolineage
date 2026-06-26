"""SyncManager — orchestrates source-adapter startup order.

Responsibilities:
  1. Validate sync-group references (master/slaves must be known adapters).
  2. On start(): for each group, configure + start all slaves first (they sit
     waiting for the first trigger), then configure + start master. Any
     adapters not in any group start at the end, in insertion order.
  3. On stop(): reverse order — masters first, then slaves, then non-grouped.

SyncManager owns no frame threads; it only drives lifecycle.
"""
from __future__ import annotations

import logging
from typing import Iterable

from robolineage_data_source.adapters.base import DeviceAdapter
from robolineage_data_source.config.schema import SyncGroupConfig
from robolineage_data_source.sync.registry import DeviceRegistry

_LOG = logging.getLogger(__name__)


class SyncManager:
    def __init__(
        self,
        registry: DeviceRegistry,
        adapters: dict[str, DeviceAdapter],
        groups: Iterable[SyncGroupConfig],
    ) -> None:
        self._registry = registry
        self._adapters = dict(adapters)
        self._groups = list(groups)

        self._validate_groups()

        # Ordering: slaves before master within a group; non-grouped adapters last.
        self._grouped_adapter_names: set[str] = set()
        for g in self._groups:
            self._grouped_adapter_names.add(g.master)
            self._grouped_adapter_names.update(g.slaves)

        self._start_order: list[str] = []
        for g in self._groups:
            self._start_order.extend(g.slaves)
            self._start_order.append(g.master)
        for name in self._adapters:
            if name not in self._grouped_adapter_names:
                self._start_order.append(name)

        self._started = False

    def _validate_groups(self) -> None:
        known = set(self._adapters)
        seen: set[str] = set()
        for g in self._groups:
            if g.master not in known:
                raise ValueError(f"sync group {g.name!r}: unknown master {g.master!r}")
            if g.master in seen:
                raise ValueError(
                    f"sync group {g.name!r}: adapter {g.master!r} appears in multiple sync groups"
                )
            seen.add(g.master)
            for s in g.slaves:
                if s not in known:
                    raise ValueError(f"sync group {g.name!r}: unknown slave {s!r}")
                if s in seen:
                    raise ValueError(
                        f"sync group {g.name!r}: adapter {s!r} appears in multiple sync groups"
                    )
                seen.add(s)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("SyncManager already started")
        # Configure sync roles first (all before any start — order is consistent).
        for g in self._groups:
            for s_name in g.slaves:
                adapter = self._adapters[s_name]
                if adapter.supports_hw_sync():
                    adapter.configure_sync("slave")
            master = self._adapters[g.master]
            if master.supports_hw_sync():
                master.configure_sync("master")

        # Start in order: slaves → master → non-grouped. If any adapter fails to
        # start, best-effort stop everything already started (reverse order) and
        # re-raise so the caller sees the real error.
        started_so_far: list[str] = []
        try:
            for name in self._start_order:
                self._adapters[name].start()
                started_so_far.append(name)
        except Exception:
            for name in reversed(started_so_far):
                try:
                    self._adapters[name].stop()
                except Exception:
                    _LOG.exception("rollback stop of %s failed", name)
            raise

        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        # Reverse order. One adapter's failure must not block the others —
        # log it but keep going.
        for name in reversed(self._start_order):
            try:
                self._adapters[name].stop()
            except Exception:
                _LOG.exception("adapter %s stop failed", name)
        self._started = False

    def wait_until_calibrated(self, timeout: float | None = None) -> bool:
        """Direct ROS2 recording does not collect host/device calibration here."""
        return True
