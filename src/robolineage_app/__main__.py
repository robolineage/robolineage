"""``python -m robolineage_app --config <yaml>`` - production RoboLineage runtime entry.

Composes Phase 5 UnifiedRuntime + serves the session FastAPI on the operator
port and the health FastAPI on a separate ops port. SIGINT / SIGTERM trigger
graceful shutdown of all sub-runners.

Typical systemd unit (see docs/deployment/systemd/RoboLineage.service):
    [Service]
    EnvironmentFile=/etc/robolineage/.env
    ExecStart=/opt/robolineage/.venv/bin/python -m robolineage_app --config /etc/robolineage/ROBOLINEAGE_default.yaml
    Restart=on-failure
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from robolineage_data_source.config.loader import load_config

from .runtime import UnifiedRuntime

_LOG = logging.getLogger("robolineage_app")

_FILE_LOG_SUPPRESSED = frozenset(
    [
        "Ros2TopicConsumer arm queue full; dropping pose sample",
    ]
)

_SYSTEM_LOGGERS = frozenset(["robolineage_app", "uvicorn", "uvicorn.access", "uvicorn.error", "httpx", "httpcore"])

_LOG_FMT = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")


class _SuppressFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in _FILE_LOG_SUPPRESSED


class _AgentOnlyFilter(logging.Filter):
    """Accept only records whose logger is NOT a system logger."""
    def filter(self, record: logging.LogRecord) -> bool:
        top = record.name.split(".")[0]
        return top not in _SYSTEM_LOGGERS and record.getMessage() not in _FILE_LOG_SUPPRESSED


class _SystemOnlyFilter(logging.Filter):
    """Accept only records whose logger IS a system logger."""
    def filter(self, record: logging.LogRecord) -> bool:
        top = record.name.split(".")[0]
        return top in _SYSTEM_LOGGERS


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    task_dir = os.environ.get("ROBOLINEAGE_TASK_DIR")
    if task_dir:
        logs_dir = Path(task_dir) / "logs"
        agent_log = logs_dir / "agent.log"
        system_log = logs_dir / "system.log"
    else:
        logs_dir = Path(__file__).resolve().parents[2] / "logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_log = logs_dir / f"agent_{timestamp}.log"
        system_log = logs_dir / f"system_{timestamp}.log"
    logs_dir.mkdir(parents=True, exist_ok=True)

    agent_handler = logging.FileHandler(agent_log, encoding="utf-8")
    agent_handler.setLevel(logging.DEBUG)
    agent_handler.setFormatter(_LOG_FMT)
    agent_handler.addFilter(_AgentOnlyFilter())

    system_handler = logging.FileHandler(system_log, encoding="utf-8")
    system_handler.setLevel(logging.DEBUG)
    system_handler.setFormatter(_LOG_FMT)
    system_handler.addFilter(_SystemOnlyFilter())

    root = logging.getLogger()
    root.addHandler(agent_handler)
    root.addHandler(system_handler)


def _make_session_server(runtime: UnifiedRuntime, port: int, host: str = "0.0.0.0"):
    """Build a uvicorn server for the session app (None if session disabled)."""
    if runtime.session_app is None:
        return None
    import uvicorn

    config = uvicorn.Config(
        runtime.session_app,
        host=host,
        port=port,
        log_level="info",
    )
    return uvicorn.Server(config)


def _make_health_server(runtime: UnifiedRuntime):
    """Build a uvicorn server for the /health app (None if health disabled)."""
    services = runtime.config.services
    if services is not None and not services.health_check:
        return None
    if runtime.config.services is None and False:  # default-on
        return None
    import uvicorn

    from .health import create_health_app

    health_app = create_health_app(runtime)
    health_cfg = runtime.config.health
    host = health_cfg.bind if health_cfg else "0.0.0.0"
    port = health_cfg.port if health_cfg else 8081

    config = uvicorn.Config(
        health_app, host=host, port=port, log_level="warning"
    )
    return uvicorn.Server(config)


async def _serve(servers: list) -> None:
    """Run all uvicorn servers concurrently until any one exits."""
    coros = [s.serve() for s in servers if s is not None]
    if not coros:
        # Nothing to serve; block on a never-resolving event so SIGINT still
        # triggers shutdown.
        await asyncio.Event().wait()
        return
    await asyncio.gather(*coros)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robolineage_app")
    parser.add_argument("--config", required=True, help="Path to RoboLineage YAML")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--session-host", default="0.0.0.0")
    parser.add_argument("--session-port", type=int, default=8080)
    parser.add_argument("--no-vsa", action="store_true",
                        help="Force services.vsa = false at runtime")
    parser.add_argument("--no-session", action="store_true",
                        help="Force services.session = false at runtime")
    parser.add_argument("--no-health", action="store_true",
                        help="Force services.health_check = false at runtime")
    parser.add_argument("--recorder-output-dir", default=None,
                        help="Override recorder.output_dir from yaml (used by run.sh)")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    cfg = load_config(args.config)
    if args.recorder_output_dir and cfg.recorder is not None:
        cfg.recorder.output_dir = args.recorder_output_dir

    # DDS / ROS2 startup diagnostics.
    _dds_ip = "n/a"
    _cyclone_uri = os.environ.get("CYCLONEDDS_URI", "")
    if _cyclone_uri:
        try:
            import xml.etree.ElementTree as _ET
            _tree = _ET.parse(_cyclone_uri)
            _addr = _tree.find(".//{*}NetworkInterfaceAddress")
            if _addr is not None and _addr.text:
                _dds_ip = _addr.text.strip()
        except Exception:
            _dds_ip = f"(parse error: {_cyclone_uri})"
    _LOG.info(
        "[robolineage_app] ROS_DOMAIN_ID=%s  DDS_IP=%s  CYCLONEDDS_URI=%s",
        os.environ.get("ROS_DOMAIN_ID", "not set"),
        _dds_ip,
        _cyclone_uri or "not set",
    )

    # CLI flags override yaml services toggles
    if args.no_vsa or args.no_session or args.no_health:
        from robolineage_data_source.config.schema import ServicesToggle
        services = cfg.services or ServicesToggle()
        cfg.services = ServicesToggle(
            data_source=services.data_source,
            session=services.session and not args.no_session,
            vsa=services.vsa and not args.no_vsa,
            post_review=services.post_review,
            health_check=services.health_check and not args.no_health,
        )

    runtime = UnifiedRuntime(cfg)

    stop_event = asyncio.Event()

    def _signal_handler(*_: object) -> None:
        _LOG.info("[robolineage_app] received signal; initiating shutdown")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    runtime.start()

    session_server = _make_session_server(
        runtime, port=args.session_port, host=args.session_host
    )
    health_server = _make_health_server(runtime)

    async def _run() -> None:
        servers = [s for s in (session_server, health_server) if s is not None]
        serve_task = asyncio.create_task(_serve(servers))
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            {serve_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        # Trigger uvicorn shutdown
        for s in servers:
            s.should_exit = True
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    try:
        asyncio.run(_run())
    finally:
        runtime.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
