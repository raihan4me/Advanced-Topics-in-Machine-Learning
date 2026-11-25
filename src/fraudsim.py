"""Deterministic synthetic card-transaction generator used throughout the lab."""
import numpy as np
import pandas as pd

CATEGORIES = ["grocery", "electronics", "travel", "fuel", "online_gaming", "restaurant"]
CHANNELS = ["pos", "web", "mobile"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _intercept_for_rate(score, target_rate):
    """Bisection search for the intercept that yields the requested mean fraud probability."""
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _sigmoid(score + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def generate(n=30000, seed=42, drift=False, target_rate=None):
    rng = np.random.default_rng(seed)
    if target_rate is None:
        target_rate = 0.045 if drift else 0.03
    channel_p = [0.30, 0.25, 0.45] if drift else [0.50, 0.30, 0.20]   # covariate drift
    ratio_mu = 0.6 if drift else 0.0                                  # amounts inflate
    txn_lambda = 4.0 if drift else 3.0

    avg_amount_30d = rng.lognormal(mean=3.5, sigma=0.6, size=n)
    amount = avg_amount_30d * rng.lognormal(mean=ratio_mu, sigma=0.7, size=n)
    hour = rng.integers(0, 24, size=n)
    merchant = rng.choice(CATEGORIES, size=n, p=[0.30, 0.12, 0.08, 0.20, 0.05, 0.25])
    channel = rng.choice(CHANNELS, size=n, p=channel_p)
    account_age_days = rng.integers(1, 3650, size=n)
    txn_count_24h = rng.poisson(txn_lambda, size=n)
    distance = rng.exponential(15.0, size=n)

    night = (hour <= 5).astype(float)
    web = (channel == "web").astype(float)
    risky = np.isin(merchant, ["electronics", "online_gaming"]).astype(float)
    log_ratio = np.log(amount / avg_amount_30d)

    score = (1.1 * risky + 0.8 * web + 0.9 * night + 0.7 * night * web
             + 1.2 * np.clip(log_ratio, 0, None)
             + 0.03 * np.minimum(distance, 100.0)
             + 0.20 * (txn_count_24h - 3)
             - 0.0004 * account_age_days)
    if drift:  # concept drift: fraudsters move to mobile and fuel stations
        score += 1.0 * (channel == "mobile") + 0.9 * (merchant == "fuel") - 0.5 * risky

    b = _intercept_for_rate(score, target_rate)
    is_fraud = (rng.random(n) < _sigmoid(score + b)).astype(int)

    # Realistic data-quality issues: values missing at random
    merchant = merchant.astype(object)
    merchant[rng.random(n) < 0.02] = np.nan
    distance = np.where(rng.random(n) < 0.03, np.nan, distance)

    return pd.DataFrame({
        "transaction_id": [f"T{seed:03d}-{i:06d}" for i in range(n)],
        "amount": np.round(amount, 2),
        "hour": hour,
        "merchant_category": merchant,
        "channel": channel,
        "account_age_days": account_age_days,
        "txn_count_24h": txn_count_24h,
        "distance_from_home_km": np.round(distance, 2),
        "avg_amount_30d": np.round(avg_amount_30d, 2),
        "is_fraud": is_fraud,
    })
