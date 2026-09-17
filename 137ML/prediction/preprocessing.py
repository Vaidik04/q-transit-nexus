"""
preprocessing.py — Clean, normalize, and split traffic data for model training.

Steps performed (in order):
  1. Handle missing values (forward-fill per edge, then drop any remaining NaN rows).
  2. Normalise continuous features using StandardScaler fitted ONLY on the training
     portion — prevents leakage of validation/test statistics into training.
  3. Sort by (timestamp, edge_id) to guarantee chronological ordering.
  4. Chronological train / val / test split by global time boundary.
     Rationale: random shuffle would allow past rows to appear in test and future rows
     in train, inflating model scores.  We split on wall-clock time so that the test
     set is strictly the most-recent portion of history.

The fitted scaler is returned alongside split DataFrames so that inference.py can
apply the same normalisation to live data without re-fitting.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from prediction.config import TRAIN_FRAC, VAL_FRAC

logger = logging.getLogger(__name__)

# Continuous columns that will be standardised
_SCALE_COLS = ["vehicle_count", "speed", "occupancy", "travel_time"]
# Optional continuous cols to scale if present
_OPTIONAL_SCALE_COLS = ["freeflow_speed"]


def preprocess(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Clean and normalise a traffic DataFrame.

    Parameters
    ----------
    df : raw DataFrame from data_loader.load_data()
    scaler : pre-fitted StandardScaler to apply (use during inference/val/test).
             If None and fit_scaler=True, a new scaler is fitted on df.
    fit_scaler : if True (default) fit scaler on df before transforming.
                 Set to False + pass a fitted scaler for val/test/inference.

    Returns
    -------
    (cleaned_df, fitted_scaler)
    """
    df = df.copy()

    # 1. Sort chronologically per edge
    df = df.sort_values(["edge_id", "timestamp"]).reset_index(drop=True)

    # 2. Forward-fill missing values within each edge (time-ordered)
    #    This uses only past values — safe against leakage.
    fill_cols = [c for c in _SCALE_COLS + _OPTIONAL_SCALE_COLS if c in df.columns]
    df[fill_cols] = (
        df.groupby("edge_id")[fill_cols]
        .transform(lambda s: s.ffill())
    )

    # 3. Drop rows that still have NaN in required numeric columns
    required_numeric = [c for c in _SCALE_COLS if c in df.columns]
    before = len(df)
    df = df.dropna(subset=required_numeric).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.info("Dropped %d rows with unresolvable NaN values.", dropped)

    # 4. Scale continuous columns
    cols_to_scale = [c for c in _SCALE_COLS + _OPTIONAL_SCALE_COLS if c in df.columns]
    if fit_scaler:
        if scaler is not None:
            logger.warning(
                "fit_scaler=True but a scaler was also provided — fitting a NEW scaler."
            )
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale].values)
    else:
        if scaler is None:
            raise ValueError(
                "fit_scaler=False requires a pre-fitted scaler to be passed in."
            )
        df[cols_to_scale] = scaler.transform(df[cols_to_scale].values)

    logger.info(
        "Preprocessing complete: %d rows, cols scaled: %s", len(df), cols_to_scale
    )
    return df, scaler


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train / val / test sets by chronological timestamp boundary.

    This is a STRICT temporal split: the test set contains ONLY rows whose
    timestamps are later than all rows in the training set.  This mirrors real
    deployment conditions and prevents any form of temporal data leakage.

    Parameters
    ----------
    df : sorted DataFrame (expected to be sorted by timestamp already)
    train_frac, val_frac : fractions of unique timestamps to assign to train/val.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    # Use unique timestamps so every edge's readings at a given time-step land
    # in the same split (no edge-level mixing across split boundaries).
    unique_ts = np.sort(df["timestamp"].unique())
    n = len(unique_ts)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_ts = set(unique_ts[:train_end])
    val_ts = set(unique_ts[train_end:val_end])
    test_ts = set(unique_ts[val_end:])

    train_df = df[df["timestamp"].isin(train_ts)].reset_index(drop=True)
    val_df = df[df["timestamp"].isin(val_ts)].reset_index(drop=True)
    test_df = df[df["timestamp"].isin(test_ts)].reset_index(drop=True)

    logger.info(
        "Chronological split → train: %d rows (ts ≤ %s), val: %d rows, test: %d rows",
        len(train_df),
        unique_ts[train_end - 1] if train_end > 0 else "N/A",
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of feature column names (everything except non-feature cols)."""
    non_feature = {"timestamp", "edge_id", "travel_time"}
    return [c for c in df.columns if c not in non_feature]
