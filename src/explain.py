import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss

from common import CAT_FEATURES, MODEL_NAME, REPORTS, SEED, init_mlflow, load_split
from explain_utils import aggregate_contributions

mlflow = init_mlflow()
pipe = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
X_train, y_train = load_split("train")
X_test, y_test = load_split("test")
REPORTS.mkdir(exist_ok=True)

# 1. Model-agnostic global importance: drop in PR-AUC when a raw feature is shuffled
pi = permutation_importance(pipe, X_test, y_test, scoring="average_precision",
                            n_repeats=5, random_state=SEED)
perm = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False)

# 2. Exact TreeSHAP contributions from XGBoost (log-odds space)
prep = pipe.named_steps["prep"]
booster = pipe.named_steps["model"].get_booster()
Xt = prep.transform(X_test)
names = list(prep.get_feature_names_out())
dm = xgb.DMatrix(Xt)
contribs = booster.predict(dm, pred_contribs=True)      # last column = bias
margin = booster.predict(dm, output_margin=True)
additive_ok = bool(np.allclose(contribs.sum(axis=1), margin, atol=1e-3))
shap_global = aggregate_contributions(contribs[:, :-1], names, CAT_FEATURES)

print("Global importance (permutation, drop in PR-AUC) vs mean |SHAP| (log-odds):")
print(f"{'feature':<24}{'perm_imp':>10}{'mean|shap|':>12}")
for f in shap_global.index:
    print(f"{f:<24}{perm[f]:>10.4f}{shap_global[f]:>12.4f}")
print(f"SHAP additivity check (sum of contributions == margin): {additive_ok}")

# 3. Local explanation for the highest-risk test transaction
proba = pipe.predict_proba(X_test)[:, 1]
i = int(np.argmax(proba))
local = pd.Series(contribs[i, :-1], index=names)
top = local.reindex(local.abs().sort_values(ascending=False).index).head(6)
print(f"\nHighest-risk test transaction (row {i}), p={proba[i]:.3f}:")
print(X_test.iloc[i].to_string())
print(f"bias (expected log-odds) = {contribs[i, -1]:+.3f}")
for n, v in top.items():
    print(f"  {n:<40}{v:+.3f}")

# 4. Calibration
brier_raw = brier_score_loss(y_test, proba)
cal = CalibratedClassifierCV(estimator=clone(pipe), method="isotonic", cv=3)
cal.fit(X_train, y_train)
proba_cal = cal.predict_proba(X_test)[:, 1]
brier_cal = brier_score_loss(y_test, proba_cal)
frac_raw, mean_raw = calibration_curve(y_test, proba, n_bins=5, strategy="quantile")
frac_cal, mean_cal = calibration_curve(y_test, proba_cal, n_bins=5, strategy="quantile")
print("\nReliability (quantile bins): mean predicted -> observed fraud rate")
print("  raw       :", " ".join(f"{p:.3f}->{o:.3f}" for p, o in zip(mean_raw, frac_raw)))
print("  isotonic  :", " ".join(f"{p:.3f}->{o:.3f}" for p, o in zip(mean_cal, frac_cal)))
print(f"Observed fraud rate={y_test.mean():.4f}  mean p raw={proba.mean():.4f}  "
      f"mean p calibrated={proba_cal.mean():.4f}")
print(f"Brier raw={brier_raw:.5f}  calibrated={brier_cal:.5f}")
print(f"PR-AUC raw={average_precision_score(y_test, proba):.4f}  "
      f"calibrated={average_precision_score(y_test, proba_cal):.4f}")

report = {
    "permutation_importance": perm.round(5).to_dict(),
    "shap_global": shap_global.round(5).to_dict(),
    "additivity_check": additive_ok,
    "brier_raw": float(brier_raw), "brier_calibrated": float(brier_cal),
    "mean_pred_raw": float(proba.mean()), "mean_pred_calibrated": float(proba_cal.mean()),
    "observed_rate": float(y_test.mean()),
}
(REPORTS / "explain_report.json").write_text(json.dumps(report, indent=2))
with mlflow.start_run(run_name="champion_explain_calibration"):
    mlflow.set_tag("task", "explain")
    mlflow.log_metrics({"brier_raw": brier_raw, "brier_calibrated": brier_cal,
                        "additivity_ok": float(additive_ok)})
    mlflow.log_artifact(str(REPORTS / "explain_report.json"))
