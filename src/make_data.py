import json

from sklearn.model_selection import train_test_split

from common import DATA, FEATURES, REPORTS, SEED, TARGET, init_mlflow
from fraudsim import generate

DATA.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

df = generate(n=30000, seed=SEED)
train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
train.to_csv(DATA / "train.csv", index=False)
test.to_csv(DATA / "test.csv", index=False)

profile = {
    "rows": len(df), "train_rows": len(train), "test_rows": len(test),
    "fraud_rate": float(df[TARGET].mean()),
    "missing": {c: int(v) for c, v in df[FEATURES].isna().sum().items()},
}
(REPORTS / "data_profile.json").write_text(json.dumps(profile, indent=2))

print(f"rows={len(df)} fraud_rate={df[TARGET].mean():.4f}")
print(f"train={len(train)} test={len(test)}")
print(f"train_fraud={int(train[TARGET].sum())} test_fraud={int(test[TARGET].sum())}")
print("missing values per column:")
print(df[FEATURES].isna().sum().to_string())

mlflow = init_mlflow()
with mlflow.start_run(run_name="dataset_v1"):
    mlflow.set_tag("task", "data")
    mlflow.log_params({"generator_seed": SEED, "rows": len(df), "test_size": 0.2})
    mlflow.log_metric("fraud_rate", profile["fraud_rate"])
    mlflow.log_artifact(str(REPORTS / "data_profile.json"))
