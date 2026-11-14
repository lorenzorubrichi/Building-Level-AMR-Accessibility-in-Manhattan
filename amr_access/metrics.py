"""Evaluation metrics for probabilistic binary predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration: logit(P(y=1)) = a + b * logit(p). Ideal: a=0, b=1."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def evaluate(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = {"n": int(len(y)), "prevalence": float(y.mean())}
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    out["brier"] = float(brier_score_loss(y, p))
    out["brier_reference"] = float(y.mean() * (1 - y.mean()))  # constant-prevalence forecast
    out["brier_skill"] = 1 - out["brier"] / out["brier_reference"] if out["brier_reference"] > 0 else np.nan
    out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    out["accuracy"] = float(accuracy_score(y, p >= threshold))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, p >= threshold)) if len(np.unique(y)) == 2 else np.nan
    out["mean_predicted"] = float(p.mean())
    if len(np.unique(y)) == 2:
        a, b = calibration_slope_intercept(y, p)
        out["calibration_intercept"], out["calibration_slope"] = a, b
    out["ece"] = expected_calibration_error(y, p)
    return out


def bootstrap_ci(y, p, metric: str = "roc_auc", n: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(evaluate(y[i], p[i])[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def weighted_summary(rows: pd.DataFrame, metrics: list[str]) -> dict:
    w = rows["n"].to_numpy(dtype=float)
    return {m: float(np.nansum(rows[m] * w) / w[~rows[m].isna()].sum()) for m in metrics}
