"""
metrics.py — Regression metrics and inference timing for model evaluation.

All metrics are computed from actual predictions vs. actual ground truth.
No values are hardcoded or pre-filled.

Functions
---------
mae(y_true, y_pred)      — Mean Absolute Error
rmse(y_true, y_pred)     — Root Mean Squared Error
mape(y_true, y_pred)     — Mean Absolute Percentage Error (guarded against /0)
r2(y_true, y_pred)       — R-squared (coefficient of determination)
inference_time(fn, *args) — Wall-clock seconds for one fn(*args) call
compute_all(y_true, y_pred) — Returns dict of all metrics above
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """Mean Absolute Percentage Error.

    Guarded against division by zero: rows where |y_true| < epsilon are excluded
    from the mean.  If all rows are excluded, returns NaN rather than an invented value.
    """
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    mask = np.abs(y_true) >= epsilon
    if mask.sum() == 0:
        return float("nan")  # cannot compute MAPE — do not fabricate a number
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination).

    Returns NaN if the total variance of y_true is zero (constant target).
    """
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def inference_time(fn: Callable, *args: Any, **kwargs: Any) -> float:
    """Measure wall-clock time (seconds) for one call to fn(*args, **kwargs)."""
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Composite metrics dict
# ---------------------------------------------------------------------------

def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute and return all metrics as a dict.

    Returns
    -------
    {"mae": float, "rmse": float, "mape": float, "r2": float}
    All values are computed from actual data — never pre-filled.
    """
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(x: Any) -> np.ndarray:
    """Convert input to 1-D float64 numpy array."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr
