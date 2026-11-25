import json
from datetime import datetime, timezone

import joblib
from mlflow.tracking import MlflowClient

from common import FEATURES, MODEL_NAME, MODELS, init_mlflow

mlflow = init_mlflow()
client = MlflowClient()
mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
run = client.get_run(mv.run_id)
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

MODELS.mkdir(exist_ok=True)
joblib.dump(model, MODELS / "fraud_model.joblib")
meta = {
    "model_name": MODEL_NAME,
    "model_version": mv.version,
    "run_id": mv.run_id,
    "threshold": run.data.metrics["decision_threshold"],
    "test_pr_auc": run.data.metrics["test_pr_auc"],
    "features": FEATURES,
    "exported_at": datetime.now(timezone.utc).isoformat(),
}
(MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
