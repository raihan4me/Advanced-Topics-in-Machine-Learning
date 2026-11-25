#!/usr/bin/env bash
set -euo pipefail
LAB="$HOME/ml-lab"
source "$LAB/.venv/bin/activate"
if curl -fs http://127.0.0.1:5000/health >/dev/null 2>&1; then
  echo "MLflow server already running"; exit 0
fi
mkdir -p "$LAB/mlartifacts"
nohup mlflow server \
  --backend-store-uri "sqlite:///$LAB/mlflow.db" \
  --artifacts-destination "$LAB/mlartifacts" \
  --host 127.0.0.1 --port 5000 --workers 1 \
  > "$LAB/mlflow-server.out" 2>&1 &
echo "MLflow server starting; log: $LAB/mlflow-server.out"
