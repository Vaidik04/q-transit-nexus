"""
data_loader.py — Load traffic data from CSV or a list of dicts (live SUMO feed).

Supports two input modes:
  1. CSV path: pd.read_csv(path) — for historical logs or SUMO-exported files.
  2. List of dicts: construct DataFrame directly — for live SUMO state pushes.

A synthetic data generator is also provided so the pipeline is runnable end-to-end
immediately without a real dataset. The generator is clearly labeled as synthetic
and MUST be replaced with real SUMO output before production evaluation.

Usage:
    from prediction.data_loader import load_data, generate_synthetic_data
    df = load_data("path/to/traffic.csv")          # from file
    df = load_data(records_list)                    # from list of dicts
    df = generate_synthetic_data()                 # demo only
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from prediction.config import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SYNTHETIC_DATA_PATH,
    SYNTHETIC_DURATION_HOURS,
    SYNTHETIC_FREQ_MINUTES,
    SYNTHETIC_NUM_EDGES,
    SYNTHETIC_RANDOM_SEED,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(source: Union[str, Path, List[dict]]) -> pd.DataFrame:
    """Load traffic data from a CSV file path or a list of dicts.

    Parameters
    ----------
    source:
        * str / Path → read CSV
        * list[dict]  → construct DataFrame (e.g., live SUMO state)

    Returns
    -------
    pd.DataFrame with at least REQUIRED_COLUMNS.  Optional columns are included
    if present; missing optional columns are silently skipped.

    Raises
    ------
    ValueError  if required columns are missing from the loaded data.
    FileNotFoundError  if a CSV path does not exist.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Traffic data file not found: {path}")
        df = pd.read_csv(path)
        logger.info("Loaded %d rows from %s", len(df), path)
    elif isinstance(source, list):
        if not source:
            raise ValueError("Empty list passed to load_data(); no records to load.")
        df = pd.DataFrame(source)
        logger.info("Constructed DataFrame from %d dict records", len(df))
    else:
        raise TypeError(
            f"source must be a path (str/Path) or list of dicts, got {type(source)}"
        )

    df = _validate_and_coerce(df)
    return df


def generate_synthetic_data(
    save_path: Union[str, Path, None] = SYNTHETIC_DATA_PATH,
) -> pd.DataFrame:
    """Generate a synthetic traffic dataset for demo/testing purposes.

    *** SYNTHETIC DATA — NOT REAL MEASUREMENTS ***
    Replace this with actual SUMO-exported CSV before production evaluation.

    The generator produces realistic-looking patterns:
      * Rush-hour speed dips at 08:00-09:30 and 17:30-19:00.
      * Random incidents (speed drops ~50%) injected with 1% probability per row.
      * Gaussian noise on all continuous fields.
      * Per-edge free-flow speeds drawn uniformly from [40, 80] km/h.

    Parameters
    ----------
    save_path : path to save the CSV, or None to skip saving.

    Returns
    -------
    pd.DataFrame with REQUIRED_COLUMNS + ['weather', 'incident_flag'].
    """
    logger.warning(
        "generate_synthetic_data() called — output is SYNTHETIC and not real traffic data."
    )

    rng = np.random.default_rng(SYNTHETIC_RANDOM_SEED)
    random.seed(SYNTHETIC_RANDOM_SEED)

    edge_ids = [f"E{i:02d}" for i in range(SYNTHETIC_NUM_EDGES)]
    # Per-edge free-flow speed (km/h); stored so congestion can be computed consistently
    freeflow_speeds = {eid: rng.uniform(40.0, 80.0) for eid in edge_ids}
    # Per-edge road length (metres) for travel-time calculation
    edge_lengths = {eid: rng.uniform(200.0, 1500.0) for eid in edge_ids}

    total_minutes = SYNTHETIC_DURATION_HOURS * 60
    timestamps = list(range(0, total_minutes * 60, SYNTHETIC_FREQ_MINUTES * 60))
    base_dt = pd.Timestamp("2024-01-15 00:00:00")
    datetimes = [
        base_dt + pd.Timedelta(seconds=int(ts)) for ts in timestamps
    ]

    rows = []
    for dt, ts in zip(datetimes, timestamps):
        hour = dt.hour
        minute = dt.minute
        dow = dt.dayofweek  # 0=Monday

        # Rush-hour factor: speed reduced during peak hours
        rush = 0.0
        if (7 * 60 + 30) <= (hour * 60 + minute) <= (9 * 60 + 30):
            rush = 0.45  # morning peak
        elif (17 * 60 + 0) <= (hour * 60 + minute) <= (19 * 60 + 0):
            rush = 0.40  # evening peak
        elif 22 <= hour or hour <= 5:
            rush = -0.15  # night: slightly faster than free-flow

        # Weekend modifier
        weekend_factor = 0.8 if dow >= 5 else 1.0

        for eid in edge_ids:
            v_ff = freeflow_speeds[eid]
            length_m = edge_lengths[eid]

            # Incident injection (1% chance)
            incident = int(rng.random() < 0.01)

            # Speed with rush, weekend, noise, and incident factors
            base_speed = v_ff * (1.0 - rush * weekend_factor)
            if incident:
                base_speed *= 0.45  # severe slowdown
            speed = float(np.clip(
                base_speed + rng.normal(0, v_ff * 0.05),
                1.0,
                v_ff * 1.05,
            ))

            # Vehicle count: higher during rush
            base_flow = rng.integers(200, 600)
            if rush > 0:
                base_flow = int(base_flow * (1.0 + rush))
            vehicle_count = int(np.clip(base_flow + rng.integers(-50, 50), 0, 2000))

            # Occupancy roughly proportional to flow/capacity
            occupancy = float(np.clip(vehicle_count / 2000.0 + rng.normal(0, 0.02), 0.0, 1.0))

            # Travel time (seconds): length / speed, speed in m/s
            speed_ms = max(speed * 1000 / 3600, 0.5)  # km/h to m/s
            travel_time = float(length_m / speed_ms)

            # Simple weather code: 0=clear, 1=rain, 2=fog (random, correlated to hour)
            weather = int(rng.choice([0, 0, 0, 1, 1, 2], p=[0.6, 0.0, 0.0, 0.25, 0.0, 0.15]))

            rows.append({
                "timestamp": ts,
                "edge_id": eid,
                "vehicle_count": vehicle_count,
                "speed": round(speed, 3),
                "occupancy": round(occupancy, 4),
                "travel_time": round(travel_time, 3),
                "day_of_week": dow,
                "hour": hour,
                "weather": weather,
                "incident_flag": incident,
                "freeflow_speed": round(v_ff, 3),  # extra col for congestion calc
            })

    df = pd.DataFrame(rows)
    logger.info(
        "Generated synthetic dataset: %d rows, %d edges, %d time-steps",
        len(df),
        SYNTHETIC_NUM_EDGES,
        len(timestamps),
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Synthetic data saved to %s", save_path)

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_and_coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns exist and coerce dtypes."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Data is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Coerce timestamp to numeric seconds if it isn't already
    if not pd.api.types.is_numeric_dtype(df["timestamp"]):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(np.int64) // 10**9
        except Exception as exc:
            raise ValueError(
                "Could not parse 'timestamp' column as numeric or datetime."
            ) from exc

    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["edge_id"] = df["edge_id"].astype(str)
    df["vehicle_count"] = pd.to_numeric(df["vehicle_count"], errors="coerce")
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["occupancy"] = pd.to_numeric(df["occupancy"], errors="coerce")
    df["travel_time"] = pd.to_numeric(df["travel_time"], errors="coerce")

    # Silently include optional columns that happen to be present
    present_optional = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    if present_optional:
        logger.debug("Optional columns present: %s", present_optional)

    return df
