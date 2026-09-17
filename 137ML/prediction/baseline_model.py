"""
baseline_model.py — RandomForest / XGBoost / LightGBM baseline predictor.

Design choice: ONE MODEL PER HORIZON (multi-output via separate estimators).
Rationale: the travel-time distribution at T+5, T+10, and T+15 can differ
significantly (autocorrelation decays with horizon), so separate models allow
each horizon to weight features independently.  A single multi-output model
would tie the feature weights across horizons, which hurts T+15 accuracy.
The cost is 3× training time, which is acceptable for a hackathon-scale dataset.

Backend selection order (config.PREFERRED_BASELINE):
  lightgbm → xgboost → random_forest
The module checks availability at import time and falls back gracefully.

Implements the Predictor interface (defined in inference.py) with:
  .fit(X, Y)             — Y is a DataFrame with columns target_t5, target_t10, ...
  .predict(X) -> dict    — {horizon: np.ndarray}
  .predict_with_ci(X)    — adds confidence interval using tree variance (RF) or
                           quantile estimates (LGB/XGB)
  .save(path)
  .load(path)
"""

from __future__ import annotations

import importlib
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from prediction.config import (
    BASELINE_HYPERPARAMS,
    HORIZONS,
    PREFERRED_BASELINE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability checks for optional backends
# ---------------------------------------------------------------------------

def _try_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None

_xgb = _try_import("xgboost")
_lgb = _try_import("lightgbm")

_AVAILABLE_BACKENDS = {"random_forest"}
if _xgb is not None:
    _AVAILABLE_BACKENDS.add("xgboost")
if _lgb is not None:
    _AVAILABLE_BACKENDS.add("lightgbm")

logger.info("Baseline backends available: %s", _AVAILABLE_BACKENDS)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _make_estimator(backend: str):
    """Instantiate a single-horizon regression estimator for the given backend."""
    if backend == "random_forest":
        params = BASELINE_HYPERPARAMS["random_forest"]
        return RandomForestRegressor(**params)

    if backend == "xgboost":
        xgb = importlib.import_module("xgboost")
        params = BASELINE_HYPERPARAMS["xgboost"].copy()
        return xgb.XGBRegressor(**params)

    if backend == "lightgbm":
        lgb = importlib.import_module("lightgbm")
        params = BASELINE_HYPERPARAMS["lightgbm"].copy()
        return lgb.LGBMRegressor(**params)

    raise ValueError(f"Unknown backend: {backend!r}")


def _select_backend(preferred: str = PREFERRED_BASELINE) -> str:
    """Return the best available backend starting from 'preferred'."""
    chain = [preferred, "lightgbm", "xgboost", "random_forest"]
    for b in chain:
        if b in _AVAILABLE_BACKENDS:
            return b
    return "random_forest"  # always available


# ---------------------------------------------------------------------------
# BaselinePredictor
# ---------------------------------------------------------------------------

class BaselinePredictor:
    """Multi-horizon traffic travel-time predictor using tree-based models.

    One estimator per horizon.  Conforms to the Predictor interface.

    Parameters
    ----------
    backend : 'random_forest' | 'xgboost' | 'lightgbm'
    horizons : list of integer horizon values (minutes)
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        horizons: List[int] = HORIZONS,
    ) -> None:
        self.backend = _select_backend(backend or PREFERRED_BASELINE)
        self.horizons = horizons
        self.feature_names_: Optional[List[str]] = None
        # One estimator per horizon
        self._models: Dict[int, object] = {
            h: _make_estimator(self.backend) for h in horizons
        }
        logger.info(
            "BaselinePredictor initialised — backend: %s, horizons: %s",
            self.backend,
            horizons,
        )

    # ------------------------------------------------------------------
    # Predictor interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "BaselinePredictor":
        """Train one estimator per horizon.

        Parameters
        ----------
        X : feature DataFrame (rows = samples)
        Y : target DataFrame with columns named 'target_t{h}' for each h in horizons
        """
        self.feature_names_ = list(X.columns)
        X_arr = X.values.astype(np.float32)

        for h in self.horizons:
            col = f"target_t{h}"
            if col not in Y.columns:
                raise ValueError(
                    f"Target column '{col}' not found in Y. "
                    f"Available: {list(Y.columns)}"
                )
            y_arr = Y[col].values.astype(np.float32)
            logger.info("Fitting horizon T+%d (%d samples) ...", h, len(y_arr))
            self._models[h].fit(X_arr, y_arr)
            logger.info("Horizon T+%d fit complete.", h)

        return self

    def predict(self, X: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Run inference for all horizons.

        Parameters
        ----------
        X : feature DataFrame — same columns as training X

        Returns
        -------
        dict {horizon_int: np.ndarray of predictions (shape [n_samples])}
        """
        X_arr = X.values.astype(np.float32)
        return {h: self._models[h].predict(X_arr) for h in self.horizons}

    def predict_with_ci(
        self, X: pd.DataFrame
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Predict with confidence interval (mean ± std across trees / leaves).

        Returns
        -------
        dict {horizon: (mean, lower_95, upper_95)}
        Only implemented for random_forest (tree-variance method).
        For XGBoost/LightGBM, returns (mean, NaN, NaN) — CI not implemented.
        """
        results = {}
        X_arr = X.values.astype(np.float32)
        for h in self.horizons:
            model = self._models[h]
            if self.backend == "random_forest":
                # Collect per-tree predictions
                tree_preds = np.stack(
                    [tree.predict(X_arr) for tree in model.estimators_], axis=0
                )
                mean = tree_preds.mean(axis=0)
                std = tree_preds.std(axis=0)
                lower = mean - 1.96 * std
                upper = mean + 1.96 * std
            else:
                mean = model.predict(X_arr)
                lower = np.full_like(mean, np.nan)
                upper = np.full_like(mean, np.nan)
            results[h] = (mean, lower, upper)
        return results

    def save(self, path: str | Path) -> None:
        """Persist the trained predictor to disk (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "backend": self.backend,
                    "horizons": self.horizons,
                    "feature_names": self.feature_names_,
                    "models": self._models,
                },
                f,
            )
        logger.info("BaselinePredictor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BaselinePredictor":
        """Load a previously saved BaselinePredictor from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved model found at: {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.backend = state["backend"]
        obj.horizons = state["horizons"]
        obj.feature_names_ = state["feature_names"]
        obj._models = state["models"]
        logger.info(
            "BaselinePredictor loaded from %s (backend=%s)", path, obj.backend
        )
        return obj
