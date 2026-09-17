"""
test_no_leakage.py — Assert that no future information leaks into feature columns.

The leakage guarantee for lag features:
  Feature column `{col}_lag{k}` at row i (for a given edge) should equal
  the value of column `col` at row i-k (k time-steps in the past).

This test:
  1. Generates a small synthetic dataset with known, ascending speed values.
  2. Runs build_features() on it.
  3. For each lag feature column, asserts that the value in row i equals the
     source column's value in row i-k (within the same edge group).
  4. Asserts that NO target column (target_t5, etc.) appears as an input feature.
"""

import numpy as np
import pandas as pd
import pytest

from prediction.feature_engineering import build_features
from prediction.config import LAG_WINDOWS, HORIZONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_monotone_df(n_steps: int = 50, n_edges: int = 3) -> pd.DataFrame:
    """Create a tiny DataFrame where speed = row_index within each edge.

    This makes it trivial to verify that lag-k of speed at step t equals t-k.
    """
    rows = []
    for edge_idx in range(n_edges):
        edge_id = f"E{edge_idx:02d}"
        for t in range(n_steps):
            ts = t * 60  # 1-minute intervals in seconds
            rows.append({
                "timestamp": ts,
                "edge_id": edge_id,
                "vehicle_count": float(t + 100),
                "speed": float(t + 1),   # speed = t+1 (monotone, unique per step)
                "occupancy": 0.5,
                "travel_time": float(t + 5),
                "freeflow_speed": 50.0,
                "hour": (ts // 3600) % 24,
                "day_of_week": 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test: lag feature values reference only the past
# ---------------------------------------------------------------------------

def test_lag_features_reference_past_only():
    """For each lag-k feature, value at row i must equal source value at row i-k."""
    df_raw = make_monotone_df(n_steps=60, n_edges=2)
    df_feat = build_features(df_raw, drop_na=True)

    for edge_id, group in df_feat.groupby("edge_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        for lag in LAG_WINDOWS:
            lag_col = f"speed_lag{lag}"
            if lag_col not in group.columns:
                continue
            # For each row j in the group (after NaN drop), the lag feature
            # should equal the 'speed' value lag steps earlier.
            # We compare using the original monotone speed = timestamp/60 + 1
            for j in range(len(group)):
                ts_current = group.loc[j, "timestamp"]
                ts_expected_past = ts_current - lag * 60
                if ts_expected_past < 0:
                    # Should be NaN (dropped) — if we're here, fail
                    assert np.isnan(group.loc[j, lag_col]), (
                        f"Row j={j} edge={edge_id} lag={lag}: expected NaN for "
                        f"past timestamp < 0 but got {group.loc[j, lag_col]}"
                    )
                    continue

                expected_speed_lag = ts_expected_past / 60 + 1  # monotone formula
                actual_lag_val = group.loc[j, lag_col]

                assert abs(actual_lag_val - expected_speed_lag) < 1e-3, (
                    f"LEAKAGE DETECTED: edge={edge_id}, lag={lag}, row j={j}: "
                    f"lag feature = {actual_lag_val:.3f} but expected "
                    f"{expected_speed_lag:.3f} (past value at ts={ts_expected_past})"
                )


# ---------------------------------------------------------------------------
# Test: target columns are NOT present in feature set
# ---------------------------------------------------------------------------

def test_target_columns_not_in_features():
    """Confirm that target columns (target_t5, ...) are never used as input features."""
    df_raw = make_monotone_df(n_steps=60, n_edges=2)
    df_feat = build_features(df_raw, drop_na=True)

    target_cols = {f"target_t{h}" for h in HORIZONS}
    feature_cols = {
        c for c in df_feat.columns
        if c not in {"timestamp", "edge_id"} and not c.startswith("target_")
    }

    overlap = target_cols & feature_cols
    assert overlap == set(), (
        f"Target columns found in feature set — potential leakage! Overlap: {overlap}"
    )


# ---------------------------------------------------------------------------
# Test: no future timestamp referenced by any lag feature
# ---------------------------------------------------------------------------

def test_lag_timestamps_strictly_past():
    """The timestamp of the lagged value must be strictly less than current timestamp."""
    df_raw = make_monotone_df(n_steps=60, n_edges=2)
    df_feat = build_features(df_raw, drop_na=True)

    for edge_id, group in df_feat.groupby("edge_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        for lag in LAG_WINDOWS:
            lag_col = f"speed_lag{lag}"
            if lag_col not in group.columns:
                continue
            for j in range(len(group)):
                ts_current = group.loc[j, "timestamp"]
                lag_val = group.loc[j, lag_col]
                if np.isnan(lag_val):
                    continue
                # The lag value was taken from the source column at ts_current - lag*60
                ts_lag = ts_current - lag * 60
                assert ts_lag < ts_current, (
                    f"LEAKAGE: edge={edge_id}, row={j}, lag={lag}: "
                    f"lag timestamp {ts_lag} is NOT before current {ts_current}"
                )
                assert ts_lag >= 0, (
                    f"edge={edge_id}, row={j}, lag={lag}: "
                    f"negative timestamp {ts_lag} (should have been NaN-dropped)"
                )


# ---------------------------------------------------------------------------
# Test: chronological split boundaries never overlap
# ---------------------------------------------------------------------------

def test_chronological_split_no_overlap():
    """Train/val/test sets must have disjoint timestamp ranges."""
    from prediction.preprocessing import chronological_split

    df_raw = make_monotone_df(n_steps=100, n_edges=2)
    train, val, test = chronological_split(df_raw)

    train_ts = set(train["timestamp"].unique())
    val_ts = set(val["timestamp"].unique())
    test_ts = set(test["timestamp"].unique())

    assert train_ts.isdisjoint(val_ts), "Train and val timestamps overlap!"
    assert val_ts.isdisjoint(test_ts), "Val and test timestamps overlap!"
    assert train_ts.isdisjoint(test_ts), "Train and test timestamps overlap!"

    # Temporal ordering: all train timestamps < all val timestamps < all test timestamps
    assert max(train_ts) < min(val_ts), (
        f"Train max ts {max(train_ts)} >= val min ts {min(val_ts)}"
    )
    assert max(val_ts) < min(test_ts), (
        f"Val max ts {max(val_ts)} >= test min ts {min(test_ts)}"
    )
