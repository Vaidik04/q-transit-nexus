"""
inference.py — Shared Predictor interface (ABC) + predict_future_costs() entry point.

This module is THE merge point between the ML prediction layer and the QPSO/
AT-DQPSO route optimizer.  The optimizer team should call:

    from prediction.inference import predict_future_costs
    results = predict_future_costs(edges, horizons=[5, 10, 15])

or POST to the HTTP endpoint in api.py.

IMPORTANT FOR OPTIMIZER TEAMMATES
-----------------------------------
Input type  : list[EdgeState]   — see EdgeState dataclass below
Output type : list[EdgePrediction] — see EdgePrediction dataclass below

Do NOT depend on anything else inside this package.  The predict_future_costs()
signature and the EdgeState / EdgePrediction types constitute the module boundary.
If you need to change the interface, raise it with both teams before merging.

Loading behaviour
-----------------
The first call to predict_future_costs() (or any PredictorModel.load_default())
loads the baseline model from config.BASELINE_MODEL_PATH.  If that file does not
exist, it triggers a full training run via evaluator.run_evaluation() automatically.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from prediction import config
from prediction.anomaly_detector import AnomalyDetector
from prediction.feature_engineering import build_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data contracts — DO NOT change without coordinating with optimizer team
# ---------------------------------------------------------------------------

@dataclass
class EdgeState:
    """Current state of a single road edge at time t.

    Attributes
    ----------
    edge_id   : unique road edge identifier (matches SUMO edge ID)
    speed     : current observed speed (km/h)
    flow      : vehicle count / flow (vehicles per time step)
    timestamp : Unix timestamp (seconds) of the observation
    occupancy : optional lane occupancy [0, 1]
    """
    edge_id: str
    speed: float
    flow: float
    timestamp: int
    occupancy: float = 0.0


@dataclass
class EdgePrediction:
    """Multi-horizon travel-time prediction for a single road edge.

    Attributes
    ----------
    edge_id  : road edge identifier
    t5, t10, t15 : predicted travel time (seconds) at T+5, T+10, T+15 minutes
    congestion   : current congestion score in [0, 1] (computed, not predicted)
    """
    edge_id: str
    t5: float
    t10: float
    t15: float
    congestion: float = 0.0


# ---------------------------------------------------------------------------
# Abstract Predictor base class
# ---------------------------------------------------------------------------

class PredictorModel(ABC):
    """Abstract base class that both BaselinePredictor and STGNNPredictor implement.

    This ensures the optimizer can swap models transparently.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "PredictorModel":
        """Train the model on feature matrix X and target matrix Y."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Return per-horizon predictions as {horizon_int: ndarray}."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "PredictorModel":
        """Load a model from disk."""
        ...


# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------

_model_cache: Optional[object] = None
_anomaly_cache: Optional[AnomalyDetector] = None


def _load_or_train() -> object:
    """Return the cached model, loading or training it if necessary."""
    global _model_cache, _anomaly_cache

    if _model_cache is not None:
        return _model_cache

    from prediction.baseline_model import BaselinePredictor

    model_path = config.BASELINE_MODEL_PATH
    anomaly_path = config.MODEL_DIR / "anomaly_detector.pkl"

    if model_path.exists():
        logger.info("Loading trained baseline from %s", model_path)
        _model_cache = BaselinePredictor.load(model_path)
    else:
        logger.warning(
            "No trained model found at %s — running full training pipeline.",
            model_path,
        )
        from prediction.evaluator import run_evaluation
        run_evaluation()
        _model_cache = BaselinePredictor.load(model_path)

    if anomaly_path.exists():
        _anomaly_cache = AnomalyDetector.load(anomaly_path)
    else:
        logger.warning("No anomaly detector found — anomaly detection disabled.")

    return _model_cache


# ---------------------------------------------------------------------------
# Main public function — THE merge point with the optimizer team
# ---------------------------------------------------------------------------

def predict_future_costs(
    edges: List[EdgeState],
    horizons: List[int] = None,
) -> List[EdgePrediction]:
    """Predict multi-horizon travel-time costs for a list of road edges.

    This function is the PRIMARY integration point with the QPSO optimizer.
    Call it in-process (no HTTP overhead) or via POST /prediction.

    Parameters
    ----------
    edges    : list of EdgeState objects (current traffic observations)
    horizons : list of integer horizons in minutes (default: config.HORIZONS)

    Returns
    -------
    list of EdgePrediction, one per input edge, in the same order.

    Example
    -------
    >>> from prediction.inference import predict_future_costs, EdgeState
    >>> edges = [EdgeState(edge_id="E14", speed=31.2, flow=742, timestamp=600)]
    >>> preds = predict_future_costs(edges, horizons=[5, 10, 15])
    >>> preds[0]
    EdgePrediction(edge_id='E14', t5=..., t10=..., t15=..., congestion=...)
    """
    if horizons is None:
        horizons = config.HORIZONS

    model = _load_or_train()

    # Build a minimal DataFrame from the edge snapshot
    records = []
    for e in edges:
        records.append({
            "timestamp": e.timestamp,
            "edge_id": e.edge_id,
            "vehicle_count": e.flow,
            "speed": e.speed,
            "occupancy": e.occupancy,
            "travel_time": 0.0,  # placeholder — not used for prediction features
        })

    df_snap = pd.DataFrame(records)

    # Build features — NOTE: for a live snapshot (single row per edge), lag/rolling
    # features will be NaN because there's no history.  We fill NaN with 0 here.
    # TODO: maintain a rolling edge-state buffer in a future version to compute proper
    #       lag features for live inference.
    df_feat = _build_live_features(df_snap, horizons)

    feature_cols = _get_feature_cols(df_feat)

    # Align columns to match the exact feature order the model was trained on.
    # If the model has feature_names_ set, use that order.
    if hasattr(model, "feature_names_") and model.feature_names_ is not None:
        trained_cols = model.feature_names_
        # Add any missing columns as 0 (e.g., optional cols absent in live data)
        for col in trained_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
        X = df_feat[trained_cols].fillna(0.0)
    else:
        X = df_feat[feature_cols].fillna(0.0)

    # Predict
    preds_dict = model.predict(X)  # {horizon: ndarray}

    # Compute congestion scores
    freeflow = config.DEFAULT_FREEFLOW_SPEED
    congestion_vals = [
        float(np.clip(1.0 - e.speed / freeflow, 0.0, 1.0)) for e in edges
    ]

    # Assemble results (one EdgePrediction per edge)
    output = []
    for i, edge in enumerate(edges):
        h_vals = {}
        for h in [5, 10, 15]:
            if h in preds_dict and i < len(preds_dict[h]):
                h_vals[h] = float(preds_dict[h][i])
            else:
                h_vals[h] = float("nan")

        output.append(
            EdgePrediction(
                edge_id=edge.edge_id,
                t5=h_vals.get(5, float("nan")),
                t10=h_vals.get(10, float("nan")),
                t15=h_vals.get(15, float("nan")),
                congestion=congestion_vals[i],
            )
        )

    return output


def detect_anomalies(edges: List[EdgeState]) -> List[dict]:
    """Run anomaly detection on the current edge snapshot.

    Returns list of {"edge_id", "is_anomaly", "severity"} dicts.
    """
    _load_or_train()  # ensures anomaly cache is populated

    if _anomaly_cache is None:
        logger.warning("Anomaly detector not available; returning empty results.")
        return [{"edge_id": e.edge_id, "is_anomaly": False, "severity": 0.0} for e in edges]

    edge_dicts = [
        {
            "edge_id": e.edge_id,
            "speed": e.speed,
            "timestamp": e.timestamp,
        }
        for e in edges
    ]
    return _anomaly_cache.detect(edge_dicts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_live_features(df_snap: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """Build features for a live snapshot (no historical lag available)."""
    from prediction.feature_engineering import _add_congestion, _add_time_encoding

    df = df_snap.copy()
    df = _add_congestion(df)
    df = _add_time_encoding(df)

    # Lag and rolling features will be NaN for single-row snapshots
    # This is acceptable for demo; a stateful buffer is a TODO for production.
    for col in ["speed", "vehicle_count", "occupancy", "congestion"]:
        for lag in config.LAG_WINDOWS:
            df[f"{col}_lag{lag}"] = np.nan
    for col in ["speed", "vehicle_count", "congestion"]:
        df[f"{col}_roll_mean{config.ROLLING_WINDOW}"] = np.nan
        df[f"{col}_roll_std{config.ROLLING_WINDOW}"] = np.nan

    return df


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return feature column names (exclude metadata and targets)."""
    exclude = {"timestamp", "edge_id", "travel_time", "freeflow_speed"}
    exclude |= {f"target_t{h}" for h in config.HORIZONS}
    return [c for c in df.columns if c not in exclude]
