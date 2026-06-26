"""Phase 5 — /health endpoint for the unified RoboLineage launcher.

Returns a JSON snapshot of every adapter's HealthStatus. systemd's
``Restart=on-failure`` policy can curl this and decide
whether to restart based on the HTTP status: 200 (everything OK / DEGRADED
acceptable) or 503 (one or more adapters FAILED).

Status mapping:
    NOT_STARTED → 200 (still warming up)
    OK          → 200
    DEGRADED    → 200 (warn but don't restart; meta surfaces details)
    FAILED      → 503 (systemd restart trigger)
"""
from __future__ import annotations

import dataclasses
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

_VSA_DASHBOARD_HTML = """\
<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VSA Phase Monitor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; padding: 24px; }
  h1 { font-size: 1.2rem; font-weight: 500; color: #8b949e; margin-bottom: 20px; letter-spacing: .05em; }
  #status { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 28px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 18px 20px; }
  .card-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: #8b949e; margin-bottom: 8px; }
  .card-value { font-size: 1.6rem; font-weight: 700; }
  #phase-value { font-size: 2rem; color: #58a6ff; }
  .risk-low    { color: #3fb950; }
  .risk-medium { color: #d29922; }
  .risk-high   { color: #f85149; }
  .risk-unknown{ color: #8b949e; }
  .prog-advancing  { color: #3fb950; }
  .prog-stalled    { color: #d29922; }
  .prog-regressing { color: #f85149; }
  .prog-unknown    { color: #8b949e; }
  #failure-banner { display: none; background: #f85149; color: #fff; border-radius: 8px; padding: 14px 20px; margin-bottom: 20px; font-weight: 700; font-size: 1.05rem; }
  #conf-bar-wrap { margin-top: 8px; background: #21262d; border-radius: 4px; height: 8px; overflow: hidden; }
  #conf-bar { height: 100%; background: #58a6ff; border-radius: 4px; transition: width .4s; width: 0%; }
  #history { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 18px 20px; }
  #history h2 { font-size: .85rem; color: #8b949e; margin-bottom: 14px; font-weight: 500; }
  table { width: 100%; border-collapse: collapse; font-size: .82rem; }
  th { text-align: left; color: #8b949e; font-weight: 500; padding: 0 8px 10px; border-bottom: 1px solid #21262d; }
  td { padding: 8px 8px; border-bottom: 1px solid #21262d; }
  tr:last-child td { border-bottom: none; }
  #no-data { color: #8b949e; font-size: .9rem; padding: 12px 0; }
  #ts { font-size: .8rem; color: #8b949e; margin-top: 10px; }
</style>
</head>
<body>
<h1>VSA PHASE MONITOR</h1>
<div id="failure-banner">⚠ Imminent failure</div>
<div id="status">
  <div class="card">
    <div class="card-label">Current Phase</div>
    <div class="card-value" id="phase-value">—</div>
  </div>
  <div class="card">
    <div class="card-label">Progress</div>
    <div class="card-value" id="progress-value">—</div>
  </div>
  <div class="card">
    <div class="card-label">Risk</div>
    <div class="card-value" id="risk-value">—</div>
  </div>
  <div class="card">
    <div class="card-label">Confidence</div>
    <div class="card-value" id="conf-value">—</div>
    <div id="conf-bar-wrap"><div id="conf-bar"></div></div>
  </div>
  <div class="card">
    <div class="card-label">Trigger</div>
    <div class="card-value" style="font-size:1.1rem;color:#8b949e" id="trigger-value">—</div>
  </div>
</div>
<div id="history">
  <h2>Recent Snapshots</h2>
  <div id="history-body"><div id="no-data">Waiting for data...</div></div>
</div>
<div id="ts"></div>
<script>
const RISK_CLASS = {low:'risk-low',medium:'risk-medium',high:'risk-high',unknown:'risk-unknown'};
const PROG_CLASS = {advancing:'prog-advancing',stalled:'prog-stalled',regressing:'prog-regressing',unknown:'prog-unknown'};
function setClass(el, map, val) {
  el.className = 'card-value ' + (map[val] || '');
}
async function refresh() {
  try {
    const [latest, hist] = await Promise.all([
      fetch('/vsa/latest').then(r=>r.json()),
      fetch('/vsa/history?n=20').then(r=>r.json()),
    ]);
    const pv = document.getElementById('phase-value');
    pv.textContent = latest.phase ?? '—';

    const rv = document.getElementById('risk-value');
    rv.textContent = latest.risk_level ?? '—';
    setClass(rv, RISK_CLASS, latest.risk_level);

    const pgv = document.getElementById('progress-value');
    pgv.textContent = latest.progress ?? '—';
    setClass(pgv, PROG_CLASS, latest.progress);

    const pct = latest.confidence != null ? Math.round(latest.confidence * 100) : null;
    document.getElementById('conf-value').textContent = pct != null ? pct + '%' : '—';
    document.getElementById('conf-bar').style.width = (pct ?? 0) + '%';

    document.getElementById('trigger-value').textContent = latest.trigger ?? '—';

    const banner = document.getElementById('failure-banner');
    banner.style.display = latest.imminent_failure ? 'block' : 'none';

    const hb = document.getElementById('history-body');
    if (!hist.length) {
      hb.innerHTML = '<div id="no-data">Waiting for data...</div>';
    } else {
      const rows = hist.slice().reverse().map(s => {
        const rc = RISK_CLASS[s.risk_level] || '';
        const pc = PROG_CLASS[s.progress] || '';
        const ts = new Date(s.timestamp * 1000).toISOString().substr(11,8);
        return `<tr>
          <td>${ts}</td>
          <td style="color:#58a6ff;font-weight:600">${s.phase}</td>
          <td class="${pc}">${s.progress}</td>
          <td class="${rc}">${s.risk_level}</td>
          <td>${Math.round(s.confidence*100)}%</td>
          <td style="color:#8b949e;font-size:.78rem">${s.trigger??''}</td>
        </tr>`;
      }).join('');
      hb.innerHTML = '<table><thead><tr><th>Time</th><th>Phase</th><th>Progress</th><th>Risk</th><th>Conf</th><th>Trigger</th></tr></thead><tbody>' + rows + '</tbody></table>';
    }
    document.getElementById('ts').textContent = 'Last update: ' + new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById('ts').textContent = 'Connection failed: ' + e.message;
  }
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def create_health_app(runtime: Any) -> FastAPI:
    """Build a tiny FastAPI app exposing GET /health for the given runtime."""
    app = FastAPI(title="RoboLineage Health", version="0.2.0")

    @app.get("/health")
    def get_health() -> JSONResponse:
        adapters: dict[str, dict[str, Any]] = {}
        worst_state = "ok"

        if runtime.orchestrator is not None:
            for name, adapter in runtime.orchestrator._adapters.items():
                try:
                    status = adapter.health()
                except Exception as exc:
                    adapters[name] = {
                        "state": "failed",
                        "message": f"health() raised: {exc!r}",
                        "meta": {},
                    }
                    worst_state = "failed"
                    continue
                state_value = status.state.value if hasattr(status.state, "value") else str(status.state)
                adapters[name] = {
                    "state": state_value,
                    "message": status.message,
                    "last_sample_mono_ns": status.last_sample_mono_ns,
                    "meta": dict(status.meta),
                }
                if state_value == "failed":
                    worst_state = "failed"
                elif state_value == "degraded" and worst_state != "failed":
                    worst_state = "degraded"

        body = {
            "status": worst_state,
            "adapters": adapters,
            "session_app_attached": runtime.session_app is not None,
            "vsa_thread_alive": (
                runtime._vsa_thread is not None and runtime._vsa_thread.is_alive()
            ),
            "vsa": (
                runtime.vsa_status()
                if hasattr(runtime, "vsa_status")
                else {"active": runtime._vsa_thread is not None and runtime._vsa_thread.is_alive()}
            ),
            "ai_routes": (
                runtime.ai_routes_status()
                if hasattr(runtime, "ai_routes_status")
                else {"routes": {}, "configured_count": 0}
            ),
            "data_flow": {
                "raw_capture": "rosbag2",
                "online_vsa": "ros2_topics",
            },
        }
        http_status = 503 if worst_state == "failed" else 200
        return JSONResponse(status_code=http_status, content=body)

    @app.get("/vsa", response_class=HTMLResponse)
    def vsa_dashboard():
        return HTMLResponse(content=_VSA_DASHBOARD_HTML)

    @app.get("/vsa/latest")
    def vsa_latest():
        snapshots = runtime.latest_snapshots(n=1)
        if not snapshots:
            return JSONResponse({"phase": None})
        s = snapshots[-1]
        return JSONResponse(_snapshot_to_dict(s))

    @app.get("/vsa/history")
    def vsa_history(n: int = Query(default=20, ge=1, le=50)):
        snapshots = runtime.latest_snapshots(n=n)
        return JSONResponse([_snapshot_to_dict(s) for s in snapshots])

    return app


def _snapshot_to_dict(s: Any) -> dict:
    d = dataclasses.asdict(s) if dataclasses.is_dataclass(s) else dict(s.__dict__)
    if "trigger" in d and d["trigger"] is not None:
        d["trigger"] = str(d["trigger"].value) if hasattr(d["trigger"], "value") else str(d["trigger"])
    return d
