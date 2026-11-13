#!/usr/bin/env bash
# Start the Zyqora AI dashboard + API on the Raspberry Pi (offline, LAN-reachable).
#   bash scripts/start_pi_demo.sh                 # waste_mobilenetv2 on port 8000
#   PORT=8080 PROFILE=imagenet_mobilenetv2 bash scripts/start_pi_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"
PROFILE="${PROFILE:-waste_mobilenetv2}"
PORT="${PORT:-8000}"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "WARNING: no venv at $VENV; using $(command -v python)"
fi
PTE="models/mobilenetv2_waste_fp32_xnnpack.pte"
if [ "$PROFILE" = "waste_mobilenetv2" ] && [ ! -f "$PTE" ]; then
  echo "WARNING: $PTE not found: the dashboard will show MODEL UNAVAILABLE (MODEL_NOT_FOUND)."
fi
if [ ! -f frontend/dist/index.html ]; then
  echo "WARNING: frontend/dist missing: API only. Build it: (cd frontend && npm ci && npm run build)"
fi
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "Open on a laptop in the same network: http://${IP:-<pi-ip>}:${PORT}"
exec python serve_zyqora.py --host 0.0.0.0 --port "$PORT" --profile "$PROFILE" ${EXTRA_ARGS:-}
