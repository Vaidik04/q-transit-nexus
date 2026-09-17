"""
feature_engineering.py — Lag features, rolling stats, congestion score, time encoding.

LEAKAGE SAFETY GUARANTEE
-------------------------
All lag and rolling features are computed with pandas .shift(k) where k > 0, meaning
feature values at row index i only ever reference rows i-k, i-k-1, ... (the past).
shift() introduces NaN at the top of each edge's time series; these NaN rows are
dropped BEFORE training so the model never sees them.

Chronological ordering per edge is enforced by sorting on (edge_id, timestamp) before
any shift is applied.  If the DataFrame is not sorted, shift() would silently produce
wrong (data-leaking) results.

CONGESTION FORMULA
-------------------
    congestion = clip(1 - v_current / v_freeflow, 0, 1)
v_freeflow source priority:
  1. 'freeflow_speed' column in the DataFrame (per-edge, from SUMO or generator).
  2. config.DEFAULT_FREEFLOW_SPEED (global fallback, documented in config.py).
  3. v_freeflow == 0 → treated as missing → falls back to DEFAULT_FREEFLOW_SPEED.

MULTI-HORIZON TARGETS
----------------------
For each horizon h in config.HORIZONS, the target column `target_t{h}` is created as
travel_time shifted BACKWARD by h time-steps (i.e., the value h steps in the future
relative to the current row).  This is equivalent to what the model must predict.

Note: target creation uses shift(-h) which references FUTURE rows — this is intentional
because it defines what we want to predict.  The target columns are NEVER used as input
features.  The leakage unit test (tests/test_no_leakage.py) asserts this explicitly.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from prediction.config import (
    DEFAULT_FREEFLOW_SPEED,
    HORIZONS,
    LAG_WINDOWS,
    ROLLING_WINDOW,
)

logger = logging.getLogger(__name__)

# Columns to generate lag and rolling features for
_LAG_FEATURE_COLS = ["speed", "vehicle_count", "occupancy", "congestion"]
# Rolling stats columns
_ROLLING_COLS = ["speed", "vehicle_count", "congestion"]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    lag_windows: List[int] = LAG_WINDOWS,
    rolling_window: int = ROLLING_WINDOW,
    horizons: List[int] = HORIZONS,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Build the complete feature matrix from a preprocessed traffic DataFrame.

    Parameters
    ----------
    df : preprocessed DataFrame (sorted by edge_id, timestamp — enforced internally)
    lag_windows : list of integer lag offsets (in time-steps)
    rolling_window : window size for rolling mean/std
    horizons : prediction horizons (defines target columns)
    drop_na : drop rows whose features or targets contain NaN (default True)

    Returns
    -------
    DataFrame with original columns + lag features + rolling features +
    time-encoding columns + congestion column + target columns.
    """
    df = df.copy()

    # Enforce chronological order per edge before any shift operation
    df = df.sort_values(["edge_id", "timestamp"]).reset_index(drop=True)

    # Step 1: Compute congestion score (current time-step, no leakage)
    df = _add_congestion(df)

    # Step 2: Lag features — all computed within each edge group via shift(k>0)
    df = _add_lag_features(df, lag_windows)

    # Step 3: Rolling statistics — computed within each edge group via shift(1) + rolling
    df = _add_rolling_features(df, rolling_window)

    # Step 4: Time encoding (cyclic hour, one-hot day-of-week)
    df = _add_time_encoding(df)

    # Step 5: Multi-horizon target columns — shift(-h) to get future travel_time
    df = _add_horizon_targets(df, horizons)

    if drop_na:
        before = len(df)
        # Drop rows where any feature or target is NaN
        target_cols = [f"target_t{h}" for h in horizons]
        feature_cols = [
            c for c in df.columns
            if c not in {"timestamp", "edge_id"} and not c.startswith("target_")
        ]
        df = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
        logger.info(
            "Dropped %d rows with NaN after feature engineering (%d remaining).",
            before - len(df),
            len(df),
        )

    return df


def get_feature_names(
    lag_windows: List[int] = LAG_WINDOWS,
    rolling_window: int = ROLLING_WINDOW,
    df_columns: Optional[List[str]] = None,
) -> List[str]:
    """Return the expected list of feature column names (excludes targets/metadata)."""
    names = []

    # Lag feature names
    for col in _LAG_FEATURE_COLS:
        for lag in lag_windows:
            names.append(f"{col}_lag{lag}")

    # Rolling feature names
    for col in _ROLLING_COLS:
        names.append(f"{col}_roll_mean{rolling_window}")
        names.append(f"{col}_roll_std{rolling_window}")

    # Time encoding
    names += ["hour_sin", "hour_cos"]
    names += [f"dow_{i}" for i in range(7)]

    # Raw features still in the frame (not targets, not metadata)
    base = ["speed", "vehicle_count", "occupancy", "congestion"]
    if df_columns is not None:
        base = [c for c in base if c in df_columns]
    names = base + names

    return names


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """Compute congestion = clip(1 - speed / v_freeflow, 0, 1) per row."""
    if "freeflow_speed" in df.columns:
        v_ff = df["freeflow_speed"].copy()
        # Replace 0 and NaN with the default fallback
        v_ff = v_ff.replace(0.0, np.nan).fillna(DEFAULT_FREEFLOW_SPEED)
    else:
        v_ff = pd.Series(DEFAULT_FREEFLOW_SPEED, index=df.index)
        logger.debug(
            "No 'freeflow_speed' column found; using DEFAULT_FREEFLOW_SPEED=%.1f for congestion.",
            DEFAULT_FREEFLOW_SPEED,
        )

    raw_congestion = 1.0 - (df["speed"] / v_ff)
    df["congestion"] = raw_congestion.clip(0.0, 1.0)
    return df


def _add_lag_features(df: pd.DataFrame, lag_windows: List[int]) -> pd.DataFrame:
    """Add lag features for each column × lag window, computed per edge group.

    Uses pandas .groupby(...)[col].shift(k) — shift(k) with k > 0 means we access
    value k steps in the PAST for the current row.  NaN is inserted at the top of
    each group (first k rows), which are later dropped by build_features().
    """
    for col in _LAG_FEATURE_COLS:
        if col not in df.columns:
            logger.debug("Lag feature source column '%s' not in DataFrame; skipping.", col)
            continue
        for lag in lag_windows:
            feat_name = f"{col}_lag{lag}"
            # Per-edge shift so lag doesn't bleed across edge boundaries
            df[feat_name] = df.groupby("edge_id")[col].shift(lag)
    return df


def _add_rolling_features(df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """Add rolling mean and std features per edge group.

    Rolling is computed after shift(1) so that the window at time t does NOT
    include the current value t — it looks at [t-rolling_window, ..., t-1].
    This ensures no current-time information leaks into the rolling stat.
    """
    for col in _ROLLING_COLS:
        if col not in df.columns:
            logger.debug("Rolling feature source column '%s' not in DataFrame; skipping.", col)
            continue

        mean_name = f"{col}_roll_mean{rolling_window}"
        std_name = f"{col}_roll_std{rolling_window}"

        # shift(1) per group first (avoids bleeding across edges),
        # then compute rolling stats per group.
        # The shift(1) ensures the window at row t covers [t-window, ..., t-1],
        # i.e., current value t is excluded from the rolling window.
        def _roll_mean(s: pd.Series) -> pd.Series:
            return s.shift(1).rolling(rolling_window, min_periods=1).mean()

        def _roll_std(s: pd.Series) -> pd.Series:
            return s.shift(1).rolling(rolling_window, min_periods=1).std()

        df[mean_name] = df.groupby("edge_id")[col].transform(_roll_mean)
        df[std_name] = df.groupby("edge_id")[col].transform(_roll_std)
    return df


def _add_time_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic hour encoding and one-hot day-of-week."""
    # Derive hour and day_of_week from timestamp (Unix seconds) if not already present
    if "hour" not in df.columns:
        dt = pd.to_datetime(df["timestamp"], unit="s")
        df["hour"] = dt.dt.hour
    if "day_of_week" not in df.columns:
        dt = pd.to_datetime(df["timestamp"], unit="s")
        df["day_of_week"] = dt.dt.dayofweek

    # Cyclic encoding for hour (preserves 23→0 continuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # One-hot encode day of week (0=Monday, 6=Sunday) → 7 binary columns
    for i in range(7):
        df[f"dow_{i}"] = (df["day_of_week"] == i).astype(np.uint8)

    return df


def _add_horizon_targets(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """Create target columns by shifting travel_time backward (into the future).

    target_t{h} at row i = travel_time at row i+h (the value the model must predict).
    shift(-h) with negative h references FUTURE rows — this is the label, not a feature.
    NaN is inserted at the tail of each group for the last h rows (no future available).
    These NaN rows are dropped in build_features() before training.
    """
    for h in horizons:
        target_col = f"target_t{h}"
        # Per-edge shift so future values don't bleed across edge boundaries
        df[target_col] = df.groupby("edge_id")["travel_time"].shift(-h)
    return df


def compute_congestion_series(
    speed: pd.Series,
    freeflow_speed: float = DEFAULT_FREEFLOW_SPEED,
) -> pd.Series:
    """Convenience function: compute congestion for a speed Series.

    Parameters
    ----------
    speed : pd.Series of speed values (same units as freeflow_speed)
    freeflow_speed : scalar free-flow reference speed

    Returns
    -------
    pd.Series of congestion values in [0, 1]
    """
    if freeflow_speed == 0:
        logger.warning(
            "freeflow_speed=0 passed to compute_congestion_series; "
            "using DEFAULT_FREEFLOW_SPEED=%.1f",
            DEFAULT_FREEFLOW_SPEED,
        )
        freeflow_speed = DEFAULT_FREEFLOW_SPEED
    return (1.0 - speed / freeflow_speed).clip(0.0, 1.0)
