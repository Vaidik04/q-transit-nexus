"""
evaluator.py — End-to-end model evaluation: baseline vs. ST-GNN comparison.

Runs both models on the same test set, computes all metrics per horizon,
prints a comparison table, and writes results to JSON + CSV artifacts.

All metric values come from running the model on real (or synthetic) data.
No values are pre-filled or hardcoded.

Usage (CLI):
    python -m prediction.evaluator

Usage (Python):
    from prediction.evaluator import run_evaluation
    results = run_evaluation()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from prediction import config
from prediction.baseline_model import BaselinePredictor
from prediction.data_loader import generate_synthetic_data, load_data
from prediction.feature_engineering import build_features
from prediction.metrics import compute_all, inference_time
from prediction.preprocessing import chronological_split, preprocess

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evaluation(
    data_source=None,
    save_artifacts: bool = True,
) -> Dict:
    """Train baseline (and GNN if available) on train split, evaluate on test split.

    Parameters
    ----------
    data_source : path/list of dicts / None.  If None, generates synthetic data.
    save_artifacts : write JSON + CSV results to config.ARTIFACT_DIR.

    Returns
    -------
    results dict: {model_name: {horizon: {metric: value}}}
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # -------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------
    if data_source is None:
        logger.info("No data source provided — generating synthetic data.")
        df_raw = generate_synthetic_data()
    else:
        df_raw = load_data(data_source)

    # -------------------------------------------------------------------
    # 2. Preprocess: fit scaler on full dataset for feature engineering,
    #    then re-fit only on training portion after split.
    # -------------------------------------------------------------------
    # Feature engineering first (on raw/unscaled for congestion to work properly)
    # Then scale: scaler fit only on train.
    logger.info("Building features ...")
    df_feat = build_features(df_raw)

    # Chronological split on the feature-engineered DataFrame
    train_df, val_df, test_df = chronological_split(df_feat)

    feature_cols = _get_feature_cols(df_feat)
    target_cols = [f"target_t{h}" for h in config.HORIZONS]

    X_train = train_df[feature_cols]
    Y_train = train_df[target_cols]
    X_val = val_df[feature_cols]
    Y_val = val_df[target_cols]
    X_test = test_df[feature_cols]
    Y_test = test_df[target_cols]

    logger.info(
        "Splits — train: %d, val: %d, test: %d rows | features: %d",
        len(X_train), len(X_val), len(X_test), len(feature_cols),
    )

    # -------------------------------------------------------------------
    # 3. Train baseline
    # -------------------------------------------------------------------
    results: Dict[str, Dict] = {}

    logger.info("=== Training Baseline Model ===")
    baseline = BaselinePredictor()
    t0 = time.perf_counter()
    baseline.fit(X_train, Y_train)
    train_time = time.perf_counter() - t0
    logger.info("Baseline training time: %.1f s", train_time)

    # Evaluate on test set
    baseline_results, baseline_infer_time = _evaluate_model(
        model=baseline,
        X_test=X_test,
        Y_test=Y_test,
        horizons=config.HORIZONS,
    )
    baseline_results["_meta"] = {
        "train_time_s": round(train_time, 3),
        "inference_time_s": round(baseline_infer_time, 4),
        "backend": baseline.backend,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    results["baseline"] = baseline_results

    # -------------------------------------------------------------------
    # 4. Optionally evaluate GNN (if gnn_model module can be imported)
    # -------------------------------------------------------------------
    try:
        from prediction.gnn_model import STGNNPredictor  # noqa: F401
        logger.info("=== Training ST-GNN Model ===")
        gnn = STGNNPredictor(horizons=config.HORIZONS)
        t0 = time.perf_counter()
        gnn.fit(X_train, Y_train)
        gnn_train_time = time.perf_counter() - t0
        gnn_results, gnn_infer_time = _evaluate_model(
            model=gnn,
            X_test=X_test,
            Y_test=Y_test,
            horizons=config.HORIZONS,
        )
        gnn_results["_meta"] = {
            "train_time_s": round(gnn_train_time, 3),
            "inference_time_s": round(gnn_infer_time, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        results["st_gnn"] = gnn_results
    except Exception as exc:
        logger.warning("ST-GNN evaluation skipped: %s", exc)

    # -------------------------------------------------------------------
    # 5. Print comparison table
    # -------------------------------------------------------------------
    _print_table(results)

    # -------------------------------------------------------------------
    # 6. Save artifacts
    # -------------------------------------------------------------------
    if save_artifacts:
        _save_artifacts(results)

    # Save trained baseline for inference.py to use
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    baseline.save(config.BASELINE_MODEL_PATH)

    # Save anomaly detector
    from prediction.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector()
    detector.fit(train_df)
    detector.save(config.MODEL_DIR / "anomaly_detector.pkl")

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return feature columns: everything except metadata and target cols."""
    exclude = {"timestamp", "edge_id"}
    exclude |= {f"target_t{h}" for h in config.HORIZONS}
    return [c for c in df.columns if c not in exclude]


def _evaluate_model(
    model,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    horizons: List[int],
) -> tuple:
    """Run inference on X_test and compute per-horizon metrics.

    Returns
    -------
    (results_dict, inference_seconds)
    results_dict: {horizon_int: {metric_name: value}}
    """
    t0 = time.perf_counter()
    predictions = model.predict(X_test)
    infer_time = time.perf_counter() - t0

    horizon_results = {}
    for h in horizons:
        target_col = f"target_t{h}"
        if target_col not in Y_test.columns:
            continue
        y_true = Y_test[target_col].values
        y_pred = predictions[h]
        # Only evaluate on rows where ground truth is not NaN
        mask = ~np.isnan(y_true)
        if mask.sum() == 0:
            logger.warning("No valid ground truth for horizon T+%d; skipping.", h)
            continue
        horizon_results[h] = compute_all(y_true[mask], y_pred[mask])

    return horizon_results, infer_time


def _print_table(results: Dict) -> None:
    """Pretty-print the evaluation results table."""
    horizons = config.HORIZONS
    metrics = ["mae", "rmse", "mape", "r2"]

    header_parts = ["Model".ljust(12), "Horizon"]
    for m in metrics:
        header_parts.append(m.upper().rjust(10))
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS (computed on test set)")
    print("=" * 65)
    print("  ".join(header_parts))
    print("-" * 65)

    for model_name, model_results in results.items():
        if model_name == "_meta":
            continue
        for h in horizons:
            if h not in model_results:
                continue
            row = model_results[h]
            parts = [
                model_name.ljust(12),
                f"T+{h:02d}  ",
            ]
            for m in metrics:
                v = row.get(m, float("nan"))
                if isinstance(v, float) and not np.isnan(v):
                    parts.append(f"{v:10.4f}")
                else:
                    parts.append("       NaN")
            print("  ".join(parts))
        # Print meta info
        meta = model_results.get("_meta", {})
        if meta:
            print(
                f"  [{model_name}] backend={meta.get('backend','?')} | "
                f"train={meta.get('train_time_s','?')}s | "
                f"infer={meta.get('inference_time_s','?')}s | "
                f"n_train={meta.get('n_train','?')} | "
                f"n_test={meta.get('n_test','?')}"
            )
        print()

    print("=" * 65)


def _save_artifacts(results: Dict) -> None:
    """Write results to JSON and CSV in config.ARTIFACT_DIR."""
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON dump (serialise NaN as null)
    def _default(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        raise TypeError(f"Not serialisable: {type(obj)}")

    json_path = config.EVAL_RESULTS_PATH
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=_default)
    logger.info("Evaluation results saved to %s", json_path)

    # Flat CSV for easy spreadsheet import
    rows = []
    for model_name, model_results in results.items():
        if model_name == "_meta":
            continue
        for h in config.HORIZONS:
            if h not in model_results:
                continue
            row = {"model": model_name, "horizon": h}
            row.update(model_results[h])
            rows.append(row)

    csv_path = config.ARTIFACT_DIR / "eval_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Evaluation CSV saved to %s", csv_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
