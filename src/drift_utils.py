"""Population Stability Index (PSI) for numeric and categorical features.

PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i), over bins i, using proportions.
Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, >= 0.25 significant shift.
"""
import numpy as np
import pandas as pd

EPS = 1e-4


def _psi_from_props(ref_p, cur_p):
    ref_p = np.clip(np.asarray(ref_p, dtype=float), EPS, None)
    cur_p = np.clip(np.asarray(cur_p, dtype=float), EPS, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def psi_numeric(ref, cur, bins=10):
    ref = np.asarray(ref, dtype=float); cur = np.asarray(cur, dtype=float)
    ref = ref[~np.isnan(ref)]; cur = cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    inner = edges[1:-1]
    k = len(inner) + 1
    ref_p = np.bincount(np.digitize(ref, inner), minlength=k) / len(ref)
    cur_p = np.bincount(np.digitize(cur, inner), minlength=k) / len(cur)
    return _psi_from_props(ref_p, cur_p)


def psi_categorical(ref, cur):
    r = pd.Series(ref, dtype=object).fillna("__MISSING__").astype(str)
    c = pd.Series(cur, dtype=object).fillna("__MISSING__").astype(str)
    cats = sorted(set(r) | set(c))
    ref_p = r.value_counts(normalize=True).reindex(cats, fill_value=0).values
    cur_p = c.value_counts(normalize=True).reindex(cats, fill_value=0).values
    return _psi_from_props(ref_p, cur_p)
