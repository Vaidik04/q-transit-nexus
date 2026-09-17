"""
anomaly_detector.py — Speed-deviation-based anomaly detection per road edge.

Method
------
During fitting, the detector computes per-edge, per-hour-of-day mean and standard
deviation of speed from the training dataset.  At inference time it compares the
observed speed against the expected (historical) mean for that edge and hour, and
returns a z-score-based severity.

Output format (per edge):
    {"edge_id": str, "is_anomaly": bool, "severity": float}

severity is a REAL computed value:
    severity = clip(|z_score| / ANOMALY_Z_THRESHOLD, 0, 1)
    — 0.0 means no deviation, 1.0 means z-score ≥ threshold (severe anomaly).

is_anomaly = True when |z_score| >= ANOMALY_Z_THRESHOLD.

If an edge/hour combination has no historical data (std is NaN or 0), the detector
gracefully returns is_anomaly=False and severity=0.0 for that edge.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from prediction.config import ANOMALY_SEVERITY_CLIP, ANOMALY_Z_THRESHOLD

logger = logging.getLogger(__name__)

# Type alias for the per-edge anomaly result dict
AnomalyResult = dict  # {"edge_id": str, "is_anomaly": bool, "severity": float}


class AnomalyDetector:
    """Per-edge, time-of-day speed anomaly detector.

    Usage
    -----
    detector = AnomalyDetector()
    detector.fit(train_df)                     # compute historical baselines
    results = detector.detect(current_edges)   # list[AnomalyResult]
    """

    def __init__(
        self,
        z_threshold: float = ANOMALY_Z_THRESHOLD,
        severity_clip: float = ANOMALY_SEVERITY_CLIP,
    ) -> None:
        self.z_threshold = z_threshold
        self.severity_clip = severity_clip
        # Baseline statistics: {(edge_id, hour): (mean_speed, std_speed)}
        self._baseline: Dict[tuple, tuple] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """Compute per-edge, per-hour speed statistics from training data.

        Parameters
        ----------
        df : training DataFrame with 'edge_id', 'speed', and 'timestamp' columns.
             'hour' column is used if present; otherwise derived from 'timestamp'.
        """
        df = df.copy()
        if "hour" not in df.columns:
            df["hour"] = pd.to_datetime(df["timestamp"], unit="s").dt.hour

        stats = (
            df.groupby(["edge_id", "hour"])["speed"]
            .agg(["mean", "std"])
            .reset_index()
        )

        self._baseline = {}
        for _, row in stats.iterrows():
            key = (str(row["edge_id"]), int(row["hour"]))
            mean_val = float(row["mean"])
            std_val = float(row["std"]) if not np.isnan(row["std"]) else 0.0
            self._baseline[key] = (mean_val, std_val)

        self._fitted = True
        logger.info(
            "AnomalyDetector fitted on %d (edge, hour) combinations.", len(self._baseline)
        )
        return self

    # ------------------------------------------------------------------
    # Detect
    # ------------------------------------------------------------------

    def detect(
        self,
        edges: List[dict],
        timestamp: Optional[int] = None,
    ) -> List[AnomalyResult]:
        """Detect anomalies in the current edge state snapshot.

        Parameters
        ----------
        edges : list of dicts, each with at least 'edge_id' and 'speed'.
                Optionally 'timestamp' (Unix seconds) or 'hour' (int 0-23).
        timestamp : fallback timestamp (Unix seconds) if not in edge dict.

        Returns
        -------
        list of AnomalyResult dicts: [{"edge_id", "is_anomaly", "severity"}, ...]
        """
        if not self._fitted:
            raise RuntimeError(
                "AnomalyDetector.detect() called before fit(). Call fit(train_df) first."
            )

        results = []
        for edge in edges:
            edge_id = str(edge["edge_id"])
            speed = float(edge.get("speed", np.nan))

            # Determine hour
            ts = edge.get("timestamp", timestamp)
            if ts is not None:
                hour = int(pd.Timestamp(ts, unit="s").hour)
            elif "hour" in edge:
                hour = int(edge["hour"])
            else:
                hour = -1  # unknown hour

            key = (edge_id, hour)
            if key not in self._baseline or np.isnan(speed):
                # No historical data for this edge/hour — cannot assess anomaly
                results.append(
                    {"edge_id": edge_id, "is_anomaly": False, "severity": 0.0}
                )
                continue

            mean_speed, std_speed = self._baseline[key]

            if std_speed <= 0.0:
                # Zero variance — cannot compute z-score
                results.append(
                    {"edge_id": edge_id, "is_anomaly": False, "severity": 0.0}
                )
                continue

            z = (mean_speed - speed) / std_speed  # positive = speed below historical
            severity = float(
                np.clip(abs(z) / self.z_threshold, 0.0, self.severity_clip)
            )
            is_anomaly = abs(z) >= self.z_threshold

            results.append(
                {
                    "edge_id": edge_id,
                    "is_anomaly": bool(is_anomaly),
                    "severity": round(severity, 4),
                }
            )

        return results

    def detect_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies from a DataFrame snapshot (convenience wrapper)."""
        edge_dicts = df.to_dict(orient="records")
        results = self.detect(edge_dicts)
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "baseline": self._baseline,
                    "z_threshold": self.z_threshold,
                    "severity_clip": self.severity_clip,
                    "fitted": self._fitted,
                },
                f,
            )
        logger.info("AnomalyDetector saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved anomaly detector at: {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            z_threshold=state["z_threshold"],
            severity_clip=state["severity_clip"],
        )
        obj._baseline = state["baseline"]
        obj._fitted = state["fitted"]
        logger.info("AnomalyDetector loaded from %s", path)
        return obj
