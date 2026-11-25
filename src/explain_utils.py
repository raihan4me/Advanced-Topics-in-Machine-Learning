"""Helpers to express SHAP contributions in terms of the original (raw) features."""
import numpy as np
import pandas as pd


def raw_feature_name(transformed_name, cat_features):
    prefix, _, rest = transformed_name.partition("__")
    if prefix == "cat":
        for cat in sorted(cat_features, key=len, reverse=True):
            if rest.startswith(cat + "_"):
                return cat
    return rest


def aggregate_contributions(contribs, names, cat_features):
    raw = [raw_feature_name(n, cat_features) for n in names]
    df = pd.DataFrame(contribs, columns=names)
    per_row = df.T.groupby(raw).sum().T
    return per_row.abs().mean().sort_values(ascending=False)
