from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from common import eval_metrics, init_mlflow, load_split, make_pipeline

X_train, y_train = load_split("train")
X_test, y_test = load_split("test")
mlflow = init_mlflow()

models = {
    "dummy_majority": make_pipeline(DummyClassifier(strategy="most_frequent")),
    "logreg_baseline": make_pipeline(LogisticRegression(max_iter=1000)),
}

header = f"{'model':<18}{'accuracy':>9}{'roc_auc':>9}{'pr_auc':>8}{'precision':>10}{'recall':>8}"
rows = []
for name, pipe in models.items():
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("task", "baseline")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        m = eval_metrics(y_test, proba, threshold=0.5)
        mlflow.log_param("estimator", type(pipe.named_steps["model"]).__name__)
        mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
        rows.append(f"{name:<18}{m['accuracy']:>9.4f}{m['roc_auc']:>9.4f}{m['pr_auc']:>8.4f}"
                    f"{m['precision']:>10.4f}{m['recall']:>8.4f}")
print(header)
print("\n".join(rows))
