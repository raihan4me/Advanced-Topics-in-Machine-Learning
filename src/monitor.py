import json
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, precision_score, recall_score

from common import (CAT_FEATURES, DATA, FEATURES, MODELS, NUM_FEATURES, REPORTS, TARGET,
                    init_mlflow)
from drift_utils import psi_categorical, psi_numeric
from fraudsim import generate

API = "http://127.0.0.1:8000/predict"
BATCHES = {"control": {"seed": 8, "drift": False}, "drift": {"seed": 7, "drift": True}}
PSI_ALERT = 0.25        # significant population shift
AP_DROP_ALERT = 0.15    # absolute PR-AUC drop versus the champion's test score

name = sys.argv[1] if len(sys.argv) > 1 else "control"
path = DATA / f"batch_{name}.csv"
if not path.exists():
    generate(n=5000, **BATCHES[name]).to_csv(path, index=False)
batch = pd.read_csv(path)
ref = pd.read_csv(DATA / "train.csv")
meta = json.loads((MODELS / "model_meta.json").read_text())

drift = {}
for col in NUM_FEATURES:
    drift[col] = {"psi": psi_numeric(ref[col].values, batch[col].values),
                  "ks_pvalue": float(ks_2samp(ref[col].dropna(), batch[col].dropna()).pvalue)}
for col in CAT_FEATURES:
    drift[col] = {"psi": psi_categorical(ref[col].values, batch[col].values), "ks_pvalue": None}


def score(df, chunk=500):
    records = df[FEATURES].astype(object).where(df[FEATURES].notna(), None).to_dict("records")
    probs, flags = [], []
    for i in range(0, len(records), chunk):
        body = json.dumps({"transactions": records[i:i + chunk]},
                          default=lambda o: o.item()).encode()
        req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())["predictions"]
        probs += [p["fraud_probability"] for p in out]
        flags += [int(p["is_fraud"]) for p in out]
    return np.array(probs), np.array(flags)


probs, flags = score(batch)
y = batch[TARGET].values
ap = float(average_precision_score(y, probs))
ap_drop = float(meta["test_pr_auc"] - ap)
recall = float(recall_score(y, flags, zero_division=0))
precision = float(precision_score(y, flags, zero_division=0))

reasons = [f"PSI({c})={d['psi']:.3f} >= {PSI_ALERT}" for c, d in drift.items() if d["psi"] >= PSI_ALERT]
if ap_drop >= AP_DROP_ALERT:
    reasons.append(f"PR-AUC drop {ap_drop:.3f} >= {AP_DROP_ALERT}")
decision = "RETRAIN" if reasons else "OK"

print(f"Batch: {name} ({len(batch)} rows, fraud rate {y.mean():.4f})")
print(f"{'feature':<24}{'psi':>8}{'ks_pvalue':>12}")
for c, d in drift.items():
    ks = "-" if d["ks_pvalue"] is None else f"{d['ks_pvalue']:.4f}"
    print(f"{c:<24}{d['psi']:>8.3f}{ks:>12}")
print(f"Model on batch: pr_auc={ap:.4f} (reference {meta['test_pr_auc']:.4f}, drop {ap_drop:+.4f}) "
      f"recall={recall:.3f} precision={precision:.3f}")
print(f"Decision: {decision}")
for r in reasons:
    print(f"  reason: {r}")

REPORTS.mkdir(exist_ok=True)
report = {"batch": name, "rows": len(batch), "fraud_rate": float(y.mean()), "drift": drift,
          "pr_auc": ap, "reference_pr_auc": meta["test_pr_auc"], "pr_auc_drop": ap_drop,
          "recall": recall, "precision": precision, "decision": decision, "reasons": reasons,
          "model_version": meta["model_version"]}
(REPORTS / f"drift_{name}.json").write_text(json.dumps(report, indent=2))

mlflow = init_mlflow()
with mlflow.start_run(run_name=f"monitor_{name}"):
    mlflow.set_tags({"task": "monitoring", "decision": decision, "batch": name})
    mlflow.log_metrics({f"psi_{c}": d["psi"] for c, d in drift.items()})
    mlflow.log_metrics({"batch_pr_auc": ap, "pr_auc_drop": ap_drop,
                        "batch_recall": recall, "batch_precision": precision})
    mlflow.log_artifact(str(REPORTS / f"drift_{name}.json"))
