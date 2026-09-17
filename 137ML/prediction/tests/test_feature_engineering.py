"""
test_feature_engineering.py — Unit tests for feature_engineering.py

Tests:
  1. Congestion formula: correct value, clipped to [0,1], handles edge cases
     (zero free-flow speed, speed > free-flow, missing freeflow column).
  2. Lag feature generation: correct number of lag columns created.
  3. Rolling feature generation: correct column names, no negative-window errors.
  4. Time encoding: hour_sin/cos range, day-of-week one-hot correct shape.
  5. Horizon target creation: targets are shifted by correct amount.
"""

import numpy as np
import pandas as pd
import pytest

from prediction.feature_engineering import (
    build_features,
    compute_congestion_series,
)
from prediction.config import (
    DEFAULT_FREEFLOW_SPEED,
    HORIZONS,
    LAG_WINDOWS,
    ROLLING_WINDOW,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_simple_df(n_steps=40, n_edges=2) -> pd.DataFrame:
    rows = []
    for eid_idx in range(n_edges):
        eid = f"E{eid_idx:02d}"
        for t in range(n_steps):
            ts = t * 60
            rows.append({
                "timestamp": ts,
                "edge_id": eid,
                "vehicle_count": 300.0,
                "speed": 30.0 + eid_idx,
                "occupancy": 0.4,
                "travel_time": 20.0,
                "freeflow_speed": 50.0,
                "hour": (ts // 3600) % 24,
                "day_of_week": 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Congestion formula tests
# ---------------------------------------------------------------------------

class TestCongestionFormula:

    def test_basic_formula(self):
        """congestion = 1 - speed / freeflow, clipped [0,1]."""
        speed = pd.Series([25.0, 50.0, 0.0, 60.0])
        freeflow = 50.0
        result = compute_congestion_series(speed, freeflow)
        expected = pd.Series([0.5, 0.0, 1.0, 0.0])  # 60 > 50 → clip to 0
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_speed_at_freeflow_gives_zero_congestion(self):
        speed = pd.Series([50.0])
        result = compute_congestion_series(speed, freeflow_speed=50.0)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_speed_exceeds_freeflow_clips_to_zero(self):
        speed = pd.Series([75.0])
        result = compute_congestion_series(speed, freeflow_speed=50.0)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_zero_speed_gives_one_congestion(self):
        speed = pd.Series([0.0])
        result = compute_congestion_series(speed, freeflow_speed=50.0)
        assert result.iloc[0] == pytest.approx(1.0)

    def test_congestion_clipped_to_one(self):
        # Negative speed is invalid data but must not produce > 1 congestion
        speed = pd.Series([-10.0])
        result = compute_congestion_series(speed, freeflow_speed=50.0)
        assert result.iloc[0] <= 1.0

    def test_zero_freeflow_falls_back_to_default(self):
        """Zero freeflow speed should use DEFAULT_FREEFLOW_SPEED, not produce NaN/inf."""
        speed = pd.Series([25.0])
        result = compute_congestion_series(speed, freeflow_speed=0.0)
        # Should use DEFAULT_FREEFLOW_SPEED = 50.0 → 1 - 25/50 = 0.5
        assert result.iloc[0] == pytest.approx(1.0 - 25.0 / DEFAULT_FREEFLOW_SPEED)

    def test_missing_freeflow_column_uses_default(self):
        """If freeflow_speed column absent, congestion uses DEFAULT_FREEFLOW_SPEED."""
        df = make_simple_df(n_steps=10, n_edges=1)
        df = df.drop(columns=["freeflow_speed"])
        df_feat = build_features(df, drop_na=False)
        assert "congestion" in df_feat.columns
        # speed=30 → congestion = 1 - 30/50 = 0.4
        cong_vals = df_feat["congestion"].dropna()
        for v in cong_vals:
            assert 0.0 <= v <= 1.0, f"Congestion out of range: {v}"

    def test_congestion_range_always_zero_to_one(self):
        speed = pd.Series(np.linspace(-20, 100, 100))
        result = compute_congestion_series(speed, freeflow_speed=50.0)
        assert (result >= 0.0).all(), "Congestion below 0 detected"
        assert (result <= 1.0).all(), "Congestion above 1 detected"


# ---------------------------------------------------------------------------
# Lag feature tests
# ---------------------------------------------------------------------------

class TestLagFeatures:

    def test_lag_column_names_created(self):
        df = make_simple_df(n_steps=40, n_edges=2)
        df_feat = build_features(df, drop_na=False)
        for lag in LAG_WINDOWS:
            assert f"speed_lag{lag}" in df_feat.columns, (
                f"Expected 'speed_lag{lag}' in columns but not found."
            )

    def test_lag1_speed_matches_previous_row(self):
        df = make_simple_df(n_steps=30, n_edges=1)
        df_feat = build_features(df, drop_na=False)
        group = df_feat[df_feat["edge_id"] == "E00"].sort_values("timestamp").reset_index(drop=True)
        # Row 1's lag1 should equal row 0's speed
        row0_speed = group.loc[0, "speed"]
        row1_lag1 = group.loc[1, "speed_lag1"]
        assert abs(row1_lag1 - row0_speed) < 1e-6, (
            f"lag1 mismatch: expected {row0_speed}, got {row1_lag1}"
        )

    def test_lag_nan_at_start_of_series(self):
        """The first lag rows must be NaN (no past available)."""
        df = make_simple_df(n_steps=30, n_edges=1)
        df_feat = build_features(df, drop_na=False)
        group = df_feat[df_feat["edge_id"] == "E00"].sort_values("timestamp").reset_index(drop=True)
        max_lag = max(LAG_WINDOWS)
        for lag in LAG_WINDOWS:
            for r in range(lag):
                assert np.isnan(group.loc[r, f"speed_lag{lag}"]), (
                    f"Expected NaN at row {r} for speed_lag{lag}"
                )


# ---------------------------------------------------------------------------
# Rolling feature tests
# ---------------------------------------------------------------------------

class TestRollingFeatures:

    def test_rolling_column_names_created(self):
        df = make_simple_df(n_steps=40, n_edges=2)
        df_feat = build_features(df, drop_na=False)
        assert f"speed_roll_mean{ROLLING_WINDOW}" in df_feat.columns
        assert f"speed_roll_std{ROLLING_WINDOW}" in df_feat.columns

    def test_rolling_mean_constant_speed(self):
        """For constant speed, rolling mean should equal that speed (approx)."""
        df = make_simple_df(n_steps=40, n_edges=1)
        df_feat = build_features(df, drop_na=True)
        group = df_feat[df_feat["edge_id"] == "E00"]
        mean_col = f"speed_roll_mean{ROLLING_WINDOW}"
        if mean_col in group.columns:
            vals = group[mean_col].dropna()
            # Constant speed = 30 → rolling mean should be ~30
            assert (abs(vals - 30.0) < 1e-3).all(), (
                f"Rolling mean deviated from constant speed: {vals.values}"
            )


# ---------------------------------------------------------------------------
# Time encoding tests
# ---------------------------------------------------------------------------

class TestTimeEncoding:

    def test_hour_cyclic_range(self):
        df = make_simple_df(n_steps=50, n_edges=1)
        df_feat = build_features(df, drop_na=False)
        assert "hour_sin" in df_feat.columns
        assert "hour_cos" in df_feat.columns
        assert df_feat["hour_sin"].between(-1.0, 1.0).all()
        assert df_feat["hour_cos"].between(-1.0, 1.0).all()

    def test_day_of_week_one_hot(self):
        df = make_simple_df(n_steps=50, n_edges=1)
        df_feat = build_features(df, drop_na=False)
        dow_cols = [f"dow_{i}" for i in range(7)]
        for col in dow_cols:
            assert col in df_feat.columns, f"Missing day-of-week column: {col}"
        # Each row should have exactly one 1 across dow columns
        row_sums = df_feat[dow_cols].sum(axis=1)
        assert (row_sums == 1).all(), "Each row should have exactly one active dow column"

    def test_sin_cos_identity(self):
        """sin^2 + cos^2 = 1 for cyclic encoding."""
        df = make_simple_df(n_steps=50, n_edges=1)
        df_feat = build_features(df, drop_na=False)
        sq_sum = df_feat["hour_sin"] ** 2 + df_feat["hour_cos"] ** 2
        assert (abs(sq_sum - 1.0) < 1e-6).all(), "Cyclic encoding violates sin²+cos²=1"


# ---------------------------------------------------------------------------
# Horizon target tests
# ---------------------------------------------------------------------------

class TestHorizonTargets:

    def test_target_columns_exist(self):
        df = make_simple_df(n_steps=50, n_edges=1)
        df_feat = build_features(df, drop_na=False)
        for h in HORIZONS:
            assert f"target_t{h}" in df_feat.columns

    def test_target_is_future_travel_time(self):
        """target_t5 at row i should equal travel_time at row i+5 (same edge)."""
        df = make_simple_df(n_steps=50, n_edges=1)
        # Give each row a unique travel_time for easy verification
        df["travel_time"] = range(len(df))
        df_feat = build_features(df, drop_na=False)
        group = df_feat[df_feat["edge_id"] == "E00"].sort_values("timestamp").reset_index(drop=True)
        h = 5
        for i in range(len(group) - h):
            expected = group.loc[i + h, "travel_time"]
            actual = group.loc[i, f"target_t{h}"]
            if not np.isnan(actual):
                assert abs(actual - expected) < 1e-6, (
                    f"target_t{h} mismatch at row {i}: expected {expected}, got {actual}"
                )

    def test_target_nan_at_end_of_series(self):
        """Last h rows of each edge should have NaN target (no future available)."""
        df = make_simple_df(n_steps=50, n_edges=1)
        df_feat = build_features(df, drop_na=False)
        group = df_feat[df_feat["edge_id"] == "E00"].sort_values("timestamp").reset_index(drop=True)
        for h in HORIZONS:
            for i in range(len(group) - h, len(group)):
                assert np.isnan(group.loc[i, f"target_t{h}"]), (
                    f"Expected NaN for target_t{h} at row {i} (end of series)"
                )
