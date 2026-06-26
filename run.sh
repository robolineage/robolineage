#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv-ros"
ROBOLINEAGE_CONFIG="${ROBOLINEAGE_CONFIG:-$SCRIPT_DIR/configs/robolineage_default.yaml}"

# 0. load nvm so the correct Node.js version is on PATH
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
# shellcheck source=/dev/null
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" --no-use
if command -v nvm >/dev/null 2>&1; then
    nvm use --silent 2>/dev/null || nvm use default --silent 2>/dev/null || true
fi

# 1. venv bootstrap
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[run.sh] .venv-ros not found, creating..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    echo "[run.sh] installing dependencies..."
    pip install -U pip -q
    pip install -e "$SCRIPT_DIR/.[dev,contracts,data-source,vsa,session,agents,dataset,train]" -q
    echo "[run.sh] dependencies installed."
else
    source "$VENV/bin/activate"
fi

# 2. kill stale robolineage_app processes
pids=$(pgrep -f "python.*robolineage_app" 2>/dev/null || true)
if [ -n "$pids" ]; then
    echo "[run.sh] killing stale robolineage_app pids: $pids"
    kill -9 $pids 2>/dev/null || true
fi

# 3. load env
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
else
    echo "[run.sh] WARN: .env not found; using environment variables already set by the shell."
fi
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export CYCLONEDDS_URI="$SCRIPT_DIR/cyclonedds.xml"

# 3b. Prepare console runtime scratch space.
# Business task directories are created only after Task Setup explicitly generates one.
export ROBOLINEAGE_TASKS_ROOT="${ROBOLINEAGE_TASKS_ROOT:-$SCRIPT_DIR/tasks}"
TASK_DIR="${ROBOLINEAGE_TASK_DIR:-$SCRIPT_DIR/.runtime/console}"
mkdir -p "$TASK_DIR/rollouts" "$TASK_DIR/logs" "$ROBOLINEAGE_TASKS_ROOT"
export ROBOLINEAGE_TASK_DIR="$TASK_DIR"
echo "[run.sh] config: $ROBOLINEAGE_CONFIG"
echo "[run.sh] runtime dir: $TASK_DIR"
echo "[run.sh] tasks root: $ROBOLINEAGE_TASKS_ROOT"

# 4. kill stale frontend dev servers (match vite regardless of path)
fe_pids=$(pgrep -f "vite" 2>/dev/null || true)
if [ -n "$fe_pids" ]; then
    echo "[run.sh] killing stale vite pids: $fe_pids"
    kill -9 $fe_pids 2>/dev/null || true
fi

# 5. start backend in background
python -m robolineage_app \
    --config "$ROBOLINEAGE_CONFIG" \
    --recorder-output-dir "$TASK_DIR/rollouts" \
    "$@" &
BACKEND_PID=$!
echo "[run.sh] backend started pid=$BACKEND_PID (session=:8080  health=:8081)"

# 6. start frontend dev server (skip with ROBOLINEAGE_NO_FRONTEND=1)
FRONTEND_PID=""
FRONTEND_PGID=""
if [ "${ROBOLINEAGE_NO_FRONTEND:-0}" != "1" ] && [ -f "$SCRIPT_DIR/frontend/package.json" ]; then
    if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
        echo "[run.sh] WARN: node/npm not found in PATH; skipping frontend dev server."
        echo "[run.sh]       install Node.js >=18 (https://nodejs.org) or set ROBOLINEAGE_NO_FRONTEND=1 to silence."
        echo "[run.sh]       backend (:8080 / :8081) will keep running."
    else
        node_major=$(node -p "process.versions.node.split('.')[0]" 2>/dev/null || echo 0)
        if [ "$node_major" -lt 18 ]; then
            echo "[run.sh] WARN: node $(node -v) is too old (need >=18); skipping frontend."
        else
            if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
                echo "[run.sh] installing frontend dependencies..."
                (cd "$SCRIPT_DIR/frontend" && npm install)
            fi
            (cd "$SCRIPT_DIR/frontend" && npm run dev) &
            FRONTEND_PID=$!
            FRONTEND_PGID=$(ps -o pgid= -p $FRONTEND_PID 2>/dev/null | tr -d ' ' || echo "")
            echo "[run.sh] frontend started pid=$FRONTEND_PID  (http://localhost:5173)"
        fi
    fi
fi

# 7. propagate signals so both processes shut down together.
#    Disable set -e from here on; wait/-n returns the child's exit status,
#    and a normal Ctrl-C / child crash must not prevent cleanup.
set +e

cleanup() {
    echo "[run.sh] shutting down..."
    if [ -n "$FRONTEND_PGID" ]; then
        kill -9 -- "-$FRONTEND_PGID" 2>/dev/null
    elif [ -n "$FRONTEND_PID" ]; then
        # fallback: kill npm + its children
        pkill -KILL -P "$FRONTEND_PID" 2>/dev/null
        kill -9 "$FRONTEND_PID" 2>/dev/null
    fi
    [ -n "$BACKEND_PID" ]  && kill "$BACKEND_PID"  2>/dev/null
    wait 2>/dev/null
}
trap cleanup INT TERM

# Exit as soon as either process dies, then tear the other one down.
if [ -n "$FRONTEND_PID" ]; then
    wait -n "$BACKEND_PID" "$FRONTEND_PID"
    rc=$?
else
    wait "$BACKEND_PID"
    rc=$?
fi
echo "[run.sh] a child exited (rc=$rc); tearing the rest down"
cleanup
exit "$rc"
