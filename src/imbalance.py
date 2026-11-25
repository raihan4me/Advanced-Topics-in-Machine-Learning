import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from common import SEED, build_preprocessor, init_mlflow, load_split, make_pipeline

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X, y = load_split("train")

candidates = {
    "lr_plain": make_pipeline(LogisticRegression(max_iter=1000)),
    "lr_balanced": make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
    # Correct: SMOTE is a pipeline step, so it only sees the training fold.
    "lr_smote": ImbPipeline([("prep", build_preprocessor()),
                             ("smote", SMOTE(random_state=SEED)),
                             ("model", LogisticRegression(max_iter=1000))]),
}

results = {}
for name, est in candidates.items():
    cv = cross_validate(est, X, y, cv=skf, scoring=SCORING, n_jobs=1)
    results[name] = {k: float(np.mean(cv[f"test_{k}"])) for k in SCORING}

# WRONG ON PURPOSE: preprocess + oversample the whole training set, then cross-validate.
Xp = build_preprocessor().fit_transform(X)
Xs, ys = SMOTE(random_state=SEED).fit_resample(Xp, y)
cv = cross_validate(LogisticRegression(max_iter=1000), Xs, ys, cv=skf, scoring=SCORING)
results["lr_smote_LEAKY"] = {k: float(np.mean(cv[f"test_{k}"])) for k in SCORING}

mlflow = init_mlflow()
print(f"{'strategy':<16}{'cv_pr_auc':>10}{'cv_roc_auc':>11}{'cv_precision':>13}{'cv_recall':>10}")
for name, r in results.items():
    print(f"{name:<16}{r['pr_auc']:>10.4f}{r['roc_auc']:>11.4f}{r['precision']:>13.4f}{r['recall']:>10.4f}")
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("task", "imbalance")
        mlflow.set_tag("leaky", str(name.endswith("LEAKY")))
        mlflow.log_metrics({f"cv_{k}": v for k, v in r.items()})
