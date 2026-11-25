import time

from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from common import SEED, eval_metrics, init_mlflow, load_split, make_pipeline

X, y = load_split("train")
X_test, y_test = load_split("test")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)


def lr():
    return make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))


def rf():   # bagging + random feature subsets
    return make_pipeline(RandomForestClassifier(
        n_estimators=150, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), scale=False)


def hgb():  # gradient boosting, histogram based (LightGBM-style) in scikit-learn
    return make_pipeline(HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, random_state=SEED), scale=False)


def xgbm():  # regularised gradient boosting
    return make_pipeline(XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
        n_jobs=2, random_state=SEED), scale=False)


candidates = {
    "logreg_balanced": lr(),
    "random_forest": rf(),
    "hist_gradient_boosting": hgb(),
    "xgboost": xgbm(),
    # Stacking: base-model OOF probabilities -> logistic-regression meta-learner
    "stacking_lr_rf_xgb": StackingClassifier(
        estimators=[("lr", lr()), ("rf", rf()), ("xgb", xgbm())],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba", cv=3, n_jobs=1),
}

mlflow = init_mlflow()
board = []
for name, est in candidates.items():
    t0 = time.time()
    cv = cross_val_score(est, X, y, cv=skf, scoring="average_precision", n_jobs=1)
    est.fit(X, y)
    m = eval_metrics(y_test, est.predict_proba(X_test)[:, 1])
    secs = time.time() - t0
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("task", "ensembles")
        mlflow.log_metrics({"cv_pr_auc_mean": float(cv.mean()), "cv_pr_auc_std": float(cv.std()),
                            "test_pr_auc": m["pr_auc"], "test_roc_auc": m["roc_auc"],
                            "train_seconds": secs})
    board.append((name, cv.mean(), cv.std(), m["pr_auc"], m["roc_auc"], secs))
    print(f"done: {name:<24} ({secs:.0f}s)", flush=True)

print(f"\n{'model':<24}{'cv_pr_auc':>16}{'test_pr_auc':>13}{'test_roc_auc':>14}")
for name, mu, sd, ap, auc, _ in sorted(board, key=lambda r: -r[1]):
    print(f"{name:<24}{mu:>9.4f}+/-{sd:.3f}{ap:>13.4f}{auc:>14.4f}")
