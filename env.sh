# Usage: source ~/ml-lab/env.sh   (run this in every new SSH session)
export LAB="$HOME/ml-lab"
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export GIT_PYTHON_REFRESH=quiet
source "$LAB/.venv/bin/activate"

# Query runs of the "fraud-detection" experiment via the MLflow REST API.
# Usage: mlq "<mlflow filter string>"     e.g.  mlq "tags.task = 'baseline'"
mlq() {
  local exp_id
  exp_id=$(curl -s "$MLFLOW_TRACKING_URI/api/2.0/mlflow/experiments/get-by-name?experiment_name=fraud-detection" \
           | jq -r '.experiment.experiment_id')
  jq -n --arg id "$exp_id" --arg f "$1" '{experiment_ids: [$id], filter: $f, max_results: 200}' \
    | curl -s -X POST "$MLFLOW_TRACKING_URI/api/2.0/mlflow/runs/search" \
           -H 'Content-Type: application/json' -d @-
}
