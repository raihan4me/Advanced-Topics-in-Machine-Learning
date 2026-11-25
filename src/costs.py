"""Business cost model for fraud decisions."""
import numpy as np

COST_FN = 200.0   # missed fraud
COST_FP = 5.0     # unnecessary manual review / customer friction


def expected_cost(y_true, y_prob, threshold, cost_fn=COST_FN, cost_fp=COST_FP):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pred = y_prob >= threshold
    fn = np.sum((y_true == 1) & ~pred)
    fp = np.sum((y_true == 0) & pred)
    return float(cost_fn * fn + cost_fp * fp)


def best_threshold(y_true, y_prob, cost_fn=COST_FN, cost_fp=COST_FP, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    costs = [expected_cost(y_true, y_prob, t, cost_fn, cost_fp) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])
