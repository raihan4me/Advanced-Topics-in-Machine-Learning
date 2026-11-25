"""Shared configuration and helpers for the fraud-detection lab."""
import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LAB = Path(os.environ.get("LAB", str(Path.home() / "ml-lab")))
DATA = LAB / "data"
MODELS = LAB / "models"
REPORTS = LAB / "reports"
SEED = 42
EXPERIMENT = "fraud-detection"
MODEL_NAME = "fraud-detector"

NUM_FEATURES = ["amount", "hour", "account_age_days", "txn_count_24h",
                "distance_from_home_km", "avg_amount_30d"]
CAT_FEATURES = ["merchant_category", "channel"]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "is_fraud"


def load_split(name):
    df = pd.read_csv(DATA / f"{name}.csv")
    return df[FEATURES], df[TARGET].astype(int)


def build_preprocessor(scale=True):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [("impute", SimpleImputer(strategy="most_frequent")),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_FEATURES),
        ("cat", Pipeline(cat_steps), CAT_FEATURES),
    ])


def make_pipeline(model, scale=True):
    return Pipeline([("prep", build_preprocessor(scale=scale)), ("model", model)])


def eval_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def init_mlflow():
    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)
    return mlflow
