"""Single source of truth for the robolineage_contracts package version.

Bump policy (SemVer):
  - **major** — breaking change to any exported dataclass / enum field
    (rename, remove, type change). Notify affected runtime-domain owners before merge.
  - **minor** — new model, new optional field, new enum value, or new JSON
    schema added. Backward compatible. Also: marking a sub-package
    deprecated (its types stay importable + functional, only emit a
    DeprecationWarning) is a minor bump. Pre-1.0 legacy removals are tracked
    as minor bumps and called out in the changelog.
  - **patch** — internal refactor with identical external shape.

Historical sub-release roadmap:
  v0.1.0 — H0+H1+H2 (skeleton + core + agents.snapshot/validation)
  v0.1.1 — H3 (stream)
  v0.1.2 — pipeline + compatibility agent contracts
  v0.1.3 — H5 (session)
  v0.1.4 — H6 (check scripts + doc reverse-index + final fixture pack)
  v0.1.5 — H patch (legacy frame envelope memoryview packing on Python 3.9)
  v0.2.0 — ROS2 refactor (RoboLineage realtime path moved to direct rclpy integration
            profile-driven ROS2 adapter)
  v0.3.0 — Remove legacy ZMQ/msgpack stream contract
  v0.4.0 — Remove compatibility-only success-risk/strategy/health/master
            and trajectory/report contracts — current
  v1.0.0 — final, after stabilization
"""

CONTRACTS_VERSION = "0.4.0"
