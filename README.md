# Advanced Topics in Machine Learning — Fraud Detection Lab

End-to-end MLOps lab: synthetic data → baselines → imbalanced learning →
ensembles → Optuna tuning → model registry → explainability & calibration →
Dockerized FastAPI serving → drift monitoring.
## Layout
ml-lab/
├── src/ # training, tuning, explain, monitor scripts
├── serving/ # FastAPI app, Dockerfile, requirements
├── requirements.txt # pinned training dependencies
├── env.sh # source in every session
└── start_mlflow.sh # launches MLflow server (SQLite backend)

## Tasks completed

| # | Task | Key script |
|---|------|-----------|
| 1 | Data sim + MLflow server | `src/fraudsim.py`, `src/make_data.py`, `start_mlflow.sh` |
| 2 | Baseline models | `src/baseline.py` |
| 3 | Imbalance + cost thresholds | `src/imbalance.py`, `src/costs.py`, `src/threshold.py` |
| 4 | Ensembles (RF / HGB / XGB / stacking) | `src/ensembles.py` |
| 5 | Optuna tuning + registry (alias `champion`) | `src/tune.py` |
| 6 | Explainability + calibration | `src/explain_utils.py`, `src/explain.py` |
| 7 | Dockerized FastAPI serving | `serving/app.py`, `serving/Dockerfile` |
| 8 | Drift monitoring (PSI + KS + AP drop) | `src/drift_utils.py`, `src/monitor.py` |

## Quick start

```bash
# 1. environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. MLflow
./start_mlflow.sh
curl -fs http://127.0.0.1:5000/health     # -> OK

# 3. data
python src/make_data.py

# 4. baselines / imbalance / ensembles
python src/baseline.py
python src/imbalance.py
python src/threshold.py
python src/ensembles.py

# 5. tune + register champion
python src/tune.py 30

# 6. explain + calibrate
python src/explain.py

# 7. serve
python src/export_champion.py
docker build -t fraud-api:1.0 -f serving/Dockerfile .
docker run -d --name fraud-api -p 127.0.0.1:8000:8000 --memory 1g fraud-api:1.0
curl -s http://127.0.0.1:8000/health | jq

# 8. monitor
python src/monitor.py control
python src/monitor.py drift
