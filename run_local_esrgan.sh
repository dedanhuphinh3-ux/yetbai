#!/usr/bin/env bash
set -euo pipefail
ESRGAN_PY="/Users/kharua/aicoworker/openclaw/workspace-main/.venv-esrgan/bin/python"
APP_DIR="/Users/kharua/Desktop/KLMWeb"
cd "$APP_DIR"
exec "$ESRGAN_PY" app.py
