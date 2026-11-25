import json
import urllib.request

import joblib
import numpy as np

from common import MODELS, load_split

X_test, _ = load_split("test")
sample = X_test.head(20)
local = joblib.load(MODELS / "fraud_model.joblib").predict_proba(sample)[:, 1]

records = sample.astype(object).where(sample.notna(), None).to_dict(orient="records")
body = json.dumps({"transactions": records}, default=lambda o: o.item()).encode()
req = urllib.request.Request("http://127.0.0.1:8000/predict", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    remote = np.array([p["fraud_probability"] for p in json.loads(r.read())["predictions"]])

diff = float(np.abs(local - remote).max())
print(f"rows compared: {len(sample)}")
print(f"max |local - api| = {diff:.2e}")
print("MATCH" if diff < 1e-6 else "MISMATCH")
