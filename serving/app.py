import json
import os
from pathlib import Path
from typing import Annotated, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

NUM_FEATURES = ["amount", "hour", "account_age_days", "txn_count_24h",
                "distance_from_home_km", "avg_amount_30d"]
CAT_FEATURES = ["merchant_category", "channel"]
FEATURES = NUM_FEATURES + CAT_FEATURES

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
model = joblib.load(MODEL_DIR / "fraud_model.joblib")
meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", meta["threshold"]))

app = FastAPI(title="fraud-detector", version=str(meta["model_version"]))


class Transaction(BaseModel):
    amount: float = Field(gt=0, allow_inf_nan=False)
    hour: int = Field(ge=0, le=23)
    merchant_category: Optional[str] = None
    channel: Literal["pos", "web", "mobile"]
    account_age_days: int = Field(ge=0)
    txn_count_24h: int = Field(ge=0)
    distance_from_home_km: Optional[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = None
    avg_amount_30d: float = Field(gt=0, allow_inf_nan=False)


class Batch(BaseModel):
    transactions: List[Transaction] = Field(min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok", "model_name": meta["model_name"],
            "model_version": meta["model_version"]}


@app.get("/metadata")
def metadata():
    return {**meta, "active_threshold": THRESHOLD}


@app.post("/predict")
def predict(batch: Batch):
    df = pd.DataFrame([t.model_dump() for t in batch.transactions])[FEATURES]
    df[NUM_FEATURES] = df[NUM_FEATURES].astype(float)                       # None -> NaN
    df[CAT_FEATURES] = df[CAT_FEATURES].where(df[CAT_FEATURES].notna(), np.nan)
    proba = model.predict_proba(df)[:, 1]
    return {
        "model_version": meta["model_version"],
        "threshold": THRESHOLD,
        "predictions": [{"fraud_probability": float(p), "is_fraud": bool(p >= THRESHOLD)}
                        for p in proba],
    }
