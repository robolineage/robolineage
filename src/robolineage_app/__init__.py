"""RoboLineage unified launcher.

`python -m robolineage_app --config configs/ROBOLINEAGE_default.yaml` is the generic production
entrypoint. It composes the operator-facing services while raw data capture
and online VSA subscribe to ROS2 topics directly:

    ROS2 topics ──► rosbag2 raw recorder
          │
          └──────► VSA realtime ROS2 consumer

    Session FastAPI + AR + health endpoint run in the same operator process.
"""

ROBOLINEAGE_APP_VERSION = "0.2.0"

__all__ = ["ROBOLINEAGE_APP_VERSION"]
