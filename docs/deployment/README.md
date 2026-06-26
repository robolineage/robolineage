# Deployment Notes

RoboLineage is normally launched on the Ubuntu robot workstation.

1. Install Python and frontend dependencies.
2. Copy `.env.example` to `.env` and fill in model routes.
3. Select `ROBOLINEAGE_CONFIG`.
4. Start `./run.sh`.
5. Open the frontend console at `http://localhost:5173`.

For systemd deployment, point `EnvironmentFile` at `/etc/robolineage/.env`, set `WorkingDirectory` to the repository root, and execute `./run.sh`. The service should run under the same Linux user that owns the ROS2 workspace and robot access permissions.
