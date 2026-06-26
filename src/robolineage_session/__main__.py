"""Run the RoboLineage session service."""
from __future__ import annotations

import argparse

import uvicorn

from robolineage_session.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m robolineage_session")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
