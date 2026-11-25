import sys
import time

import numpy as np
import optuna
from mlflow.tracking import MlflowClient
from optuna.trial import TrialState
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from common import LAB, MODEL_NAME, SEED, eval_metrics, init_mlflow, load_split, make_pipeline
from costs import COST_FN, COST_FP, best_threshold, expected_cost

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
X, y = load_split("train")
X_test, y_test = load_split("test")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))


def build(params):
    return make_pipeline(XGBClassifier(**params, tree_method="hist", eval_metric="aucpr",
                                       n_jobs=2, random_state=SEED), scale=False)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 30.0, log=True),
    }
    scores = []
    for step, (tr, va) in enumerate(folds):
        pipe = build(params)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        scores.append(average_precision_score(y.iloc[va], pipe.predict_proba(X.iloc[va])[:, 1]))
        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    study_name="xgb-fraud", storage=f"sqlite:///{LAB / 'optuna.db'}", direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    load_if_exists=True)

t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS)
done = len(study.get_trials(states=(TrialState.COMPLETE,)))
pruned = len(study.get_trials(states=(TrialState.PRUNED,)))
print(f"Finished {N_TRIALS} trials in {time.time()-t0:.0f}s "
      f"(study total: complete={done} pruned={pruned})")
print(f"Best CV PR-AUC : {study.best_value:.4f}")
print(f"Best params    : {study.best_params}")

best = study.best_params
oof = cross_val_predict(build(best), X, y, cv=skf, method="predict_proba")[:, 1]
thr, oof_cost = best_threshold(y.values, oof, COST_FN, COST_FP)

pipe = build(best)
pipe.fit(X, y)
proba = pipe.predict_proba(X_test)[:, 1]
m = eval_metrics(y_test, proba, threshold=thr)
test_cost = expected_cost(y_test.values, proba, thr)
test_cost_05 = expected_cost(y_test.values, proba, 0.5)
print(f"Cost-optimal threshold (OOF) : {thr:.2f}")
print(f"Test PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
      f"precision={m['precision']:.3f} recall={m['recall']:.3f}")
print(f"Test cost at 0.50={test_cost_05:,.0f}  at {thr:.2f}={test_cost:,.0f}")

mlflow = init_mlflow()
with mlflow.start_run(run_name="xgb_optuna_best") as run:
    mlflow.set_tag("task", "tuning")
    mlflow.log_params(best)
    mlflow.log_param("study_trials_total", len(study.trials))
    mlflow.log_metric("cv_pr_auc", study.best_value)
    mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
    mlflow.log_metrics({"decision_threshold": thr, "test_cost": test_cost,
                        "test_cost_at_0_5": test_cost_05})
    mlflow.sklearn.log_model(pipe, artifact_path="model", registered_model_name=MODEL_NAME,
                             input_example=X.dropna().head(5))
    run_id = run.info.run_id

client = MlflowClient()
version = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'")
              if v.run_id == run_id)
client.set_registered_model_alias(MODEL_NAME, "champion", str(version))
print(f"Registered {MODEL_NAME} version {version} with alias 'champion'")
