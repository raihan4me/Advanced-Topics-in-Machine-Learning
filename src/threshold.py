from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import SEED, init_mlflow, load_split, make_pipeline
from costs import COST_FN, COST_FP, best_threshold, expected_cost

X, y = load_split("train")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pipe = make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
oof = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]

c05 = expected_cost(y.values, oof, 0.5)
t, c = best_threshold(y.values, oof)
print(f"Out-of-fold predictions : {len(oof)} rows")
print(f"Cost at threshold 0.50  : {c05:,.0f}")
print(f"Best threshold          : {t:.2f}")
print(f"Cost at best threshold  : {c:,.0f}  (saving {100*(c05-c)/c05:.1f}%)")
print(f"Bayes threshold for calibrated probs would be {COST_FP/(COST_FP+COST_FN):.3f}")

mlflow = init_mlflow()
with mlflow.start_run(run_name="lr_balanced_cost_threshold"):
    mlflow.set_tag("task", "threshold")
    mlflow.log_params({"cost_fn": COST_FN, "cost_fp": COST_FP})
    mlflow.log_metrics({"oof_cost_at_0_5": c05, "oof_best_cost": c, "best_threshold": t})
