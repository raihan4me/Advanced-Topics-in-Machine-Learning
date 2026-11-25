#!/usr/bin/env bash
# Final validation for the Advanced Topics in Machine Learning lab
source "$HOME/ml-lab/env.sh"
cd "$LAB" || exit 1
PASS=0; FAIL=0
check() {
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "PASS  $desc"; PASS=$((PASS+1))
  else echo "FAIL  $desc"; FAIL=$((FAIL+1)); fi
}
runs_ge() { local n; n=$(mlq "tags.task = '$1'" | jq '.runs | length'); [ "${n:-0}" -ge "$2" ]; }
pyok() { python -c "$1"; }
container_healthy() { [ "$(docker inspect -f '{{.State.Health.Status}}' fraud-api 2>/dev/null)" = "healthy" ]; }
api_predicts() {
  curl -fs -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
       -d @serving/sample.json | jq -e '.predictions | length == 2'
}
api_rejects_bad() {
  [ "$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8000/predict \
       -H 'Content-Type: application/json' \
       -d '{"transactions":[{"amount":10,"hour":27,"channel":"pos","account_age_days":1,"txn_count_24h":1,"avg_amount_30d":10}]}')" = "422" ]
}

check "T1 data split is 24000/6000" pyok "import pandas as pd; assert len(pd.read_csv('data/train.csv'))==24000 and len(pd.read_csv('data/test.csv'))==6000"
check "T1 MLflow server healthy" bash -c "curl -fs http://127.0.0.1:5000/health | grep -q OK"
check "T1 dataset run logged" runs_ge data 1
check "T2 baseline runs (>=2)" runs_ge baseline 2
check "T3 imbalance runs (>=4)" runs_ge imbalance 4
check "T3 costs.py implemented" pyok "import sys; sys.path.insert(0,'src'); import numpy as np; from costs import expected_cost, best_threshold; y=np.array([1,0,1,0]); p=np.array([.9,.6,.2,.1]); assert expected_cost(y,p,0.5)==205; t,c=best_threshold(y,p); assert round(t,2)==0.11 and c==5"
check "T3 threshold run logged" runs_ge threshold 1
check "T4 ensemble runs (>=5)" runs_ge ensembles 5
check "T5 Optuna study has >=20 trials" pyok "import optuna; s=optuna.load_study(study_name='xgb-fraud', storage='sqlite:///optuna.db'); assert len(s.trials)>=20"
check "T5 tuning run logged" runs_ge tuning 1
check "T5 champion alias registered" bash -c "curl -fs 'http://127.0.0.1:5000/api/2.0/mlflow/registered-models/alias?name=fraud-detector&alias=champion' | jq -e '.model_version.version'"
check "T6 explain_utils implemented" pyok "import sys; sys.path.insert(0,'src'); import numpy as np; from explain_utils import aggregate_contributions as a; s=a(np.array([[1.0,-0.5,0.2],[-1.0,0.5,0.0]]),['num__amount','cat__channel_web','cat__channel_pos'],['merchant_category','channel']); assert s.round(3).to_dict()=={'amount':1.0,'channel':0.4}"
check "T6 SHAP additivity verified" jq -e '.additivity_check == true' reports/explain_report.json
check "T7 model + metadata exported" test -s models/fraud_model.joblib -a -s models/model_meta.json
check "T7 container healthy" container_healthy
check "T7 /predict returns 2 predictions" api_predicts
check "T7 invalid input rejected with 422" api_rejects_bad
check "T7 training/serving parity" bash -c "python src/compare_api.py | grep -qx MATCH"
check "T8 drift_utils implemented" pyok "import sys; sys.path.insert(0,'src'); from drift_utils import psi_categorical as p; assert abs(p(['a']*50+['b']*50, ['a']*80+['b']*20)-0.416)<0.001"
check "T8 control batch decision OK" jq -e '.decision == "OK"' reports/drift_control.json
check "T8 drift batch decision RETRAIN" jq -e '.decision == "RETRAIN"' reports/drift_drift.json
check "T8 monitoring runs (>=2)" runs_ge monitoring 2
echo "--------------------------------"
echo "PASSED: $PASS   FAILED: $FAIL"
[ "$FAIL" -eq 0 ]
