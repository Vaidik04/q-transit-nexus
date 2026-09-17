# `prediction/` — Traffic Prediction / ML Engine

**PS-137 — Quantum-Inspired Intelligent Traffic Route Optimization**  
Smart India Hackathon 2024

---

## 1. Module Role in the System

```
SUMO Simulation  ──(traffic state)──►  prediction/   ──(EdgePrediction)──►  QPSO Optimizer
                                          │
                                          ├─ POST /prediction   (HTTP)
                                          └─ predict_future_costs()  (in-process import)
```

This module owns **only** the prediction layer.  
It does **not** own the SUMO simulation or the QPSO/AT-DQPSO optimizer.

---

## 2. How the Optimizer Team Should Call This Module

### Option A — In-process Python import (lower latency, recommended for co-located services)

```python
from prediction.inference import predict_future_costs, EdgeState

edges = [
    EdgeState(edge_id="E14", speed=31.2, flow=742, timestamp=600),
    EdgeState(edge_id="E07", speed=12.0, flow=1100, timestamp=600),
]
predictions = predict_future_costs(edges, horizons=[5, 10, 15])

for p in predictions:
    print(p.edge_id, p.t5, p.t10, p.t15, p.congestion)
```

`predict_future_costs()` auto-loads the trained model on first call.  
If no model is found, it triggers a training run automatically.

### Option B — HTTP endpoint (for microservice / separate process deployment)

Start the API server:
```bash
uvicorn prediction.api:app --reload --port 8000
```

POST request:
```bash
curl -X POST http://localhost:8000/prediction \
  -H "Content-Type: application/json" \
  -d '{
    "edges": [{"edge_id": "E14", "speed": 31.2, "flow": 742, "timestamp": 600}],
    "horizons": [5, 10, 15]
  }'
```

Expected response:
```json
{
  "predictions": [
    {"edge_id": "E14", "t5": 5.1, "t10": 8.7, "t15": 7.3, "congestion": 0.376}
  ]
}
```

### Interface Contract (DO NOT break without coordinating with both teams)

| Item | Type | Notes |
|------|------|-------|
| `EdgeState.edge_id` | `str` | Must match SUMO edge IDs |
| `EdgeState.speed` | `float` | km/h |
| `EdgeState.flow` | `float` | vehicles per time-step |
| `EdgeState.timestamp` | `int` | Unix seconds |
| `EdgePrediction.t5/t10/t15` | `float` | predicted travel time (seconds) |
| `EdgePrediction.congestion` | `float` | `[0, 1]` current congestion score |

---

## 3. Running the Full Pipeline

### Install dependencies
```bash
cd prediction/
pip install -r requirements.txt
```

### Generate synthetic data + train baseline + evaluate
```bash
# From the project root (one level above prediction/)
python -m prediction.evaluator
```

This will:
1. Generate a 3-day synthetic traffic dataset (20 edges, 1-min resolution).
2. Run feature engineering (lag/rolling features, congestion, time encoding).
3. Train the baseline (LightGBM if available, else RandomForest).
4. Evaluate on the held-out test set and print a metrics table.
5. Save the trained model to `prediction/saved_models/baseline_model.pkl`.
6. Save evaluation results to `prediction/artifacts/eval_results.json`.

### Run unit tests
```bash
pytest prediction/tests/ -v
```

### Start the API server
```bash
uvicorn prediction.api:app --reload --port 8000
# Browse API docs: http://localhost:8000/docs
```

---

## 4. Congestion Formula

```
congestion = clip(1 - v_current / v_freeflow, 0.0, 1.0)
```

- **0.0** = no congestion (traffic moving at or above free-flow speed)  
- **1.0** = severe congestion (near standstill)

### Edge-case handling

| Situation | Behaviour |
|-----------|-----------|
| `v_freeflow == 0` | Falls back to `config.DEFAULT_FREEFLOW_SPEED` (50 km/h) |
| `freeflow_speed` column missing from data | Uses `DEFAULT_FREEFLOW_SPEED` globally |
| `v_current > v_freeflow` | Clips to 0.0 (valid during low-traffic periods) |
| `v_current < 0` | Clips to 1.0 (invalid sensor reading treated as maximum congestion) |

> ⚠️ The fallback to `DEFAULT_FREEFLOW_SPEED` must be noted in any evaluation report
> that uses congestion scores, as it affects edges without per-edge SUMO calibration data.

---

## 5. Leakage-Prevention Approach

### Temporal split
```
─────────────────────────────────────────────────────────▶  time
│◄─────── 70% train ──────►│◄─ 15% val ─►│◄─ 15% test ─►│
```
The split is on **unique timestamps**, so every edge's reading at time T lands in
the same partition. Random shuffle is explicitly **not used** — it would allow
future rows in train and past rows in test, artificially inflating metrics.

### Lag features
```python
df[f"{col}_lag{k}"] = df.groupby("edge_id")[col].shift(k)
```
`shift(k)` with `k > 0` references **strictly past** rows within each edge group.
NaN rows at the start of each group (no past available) are dropped before training.

### Rolling features
```python
shifted = df.groupby("edge_id")[col].shift(1)
rolling_mean = shifted.rolling(window).mean()
```
The `shift(1)` before rolling ensures the window `[t-W, ..., t-1]` excludes the
current time-step `t`.

### Target columns
`target_t{h}` uses `shift(-h)` (negative = future) and is treated as the label,
never as an input feature.  The leakage unit test explicitly asserts this.

---

## 6. Project Structure

```
prediction/
├── config.py               ← all knobs (paths, hyperparams, formula constants)
├── data_loader.py          ← CSV / list-of-dicts loader + synthetic generator
├── preprocessing.py        ← missing values, scaling (fit on train only), splitting
├── feature_engineering.py  ← lag/rolling features, congestion, time encoding, targets
├── baseline_model.py       ← RF/XGBoost/LightGBM, common Predictor interface
├── gnn_model.py            ← ST-GNN (GCN + GRU), same interface as baseline
├── anomaly_detector.py     ← per-edge z-score anomaly detection
├── metrics.py              ← MAE, RMSE, MAPE, R², inference time
├── evaluator.py            ← end-to-end train + eval + artifact output
├── inference.py            ← EdgeState, EdgePrediction, predict_future_costs()
├── api.py                  ← FastAPI: POST /prediction, POST /anomaly, GET /health
├── tests/
│   ├── test_no_leakage.py         ← lag reference past only, split disjoint
│   ├── test_feature_engineering.py← congestion formula, lag/rolling, time encoding
│   └── test_api.py                ← HTTP endpoint request/response shapes
└── requirements.txt
```

---

## 7. ST-GNN Notes

The GNN model (`gnn_model.py`) is built **after** the baseline is validated end-to-end.

Architecture: **GCN (spatial) + GRU (temporal) + linear head**

- Graph nodes = road edge segments
- Graph edges = spatial adjacency (ring topology for demo; replace with SUMO network adjacency)
- Sequence length = 12 past time-steps (configurable via `config.GNN_HYPERPARAMS`)

**Requires**: `torch` + `torch-geometric` (commented out in `requirements.txt`).  
Install with:
```bash
pip install torch torch-geometric
```

The evaluator will automatically include GNN results in the comparison table if
PyTorch/PyG are installed.

---

## 8. Anomaly Detection

Endpoint: `POST /anomaly`  
Detector: per-edge, per-hour-of-day z-score against historical speed distribution.

```json
[
  {"edge_id": "E01", "is_anomaly": true,  "severity": 0.87},
  {"edge_id": "E02", "is_anomaly": false, "severity": 0.12}
]
```

`severity` is a real z-score-based deviation ratio clipped to `[0, 1]`.  
It is **never hardcoded** — computed from the trained historical baseline.

---

## 9. Replacing Synthetic Data with Real SUMO Output

1. Export SUMO edge data to CSV with columns: `timestamp, edge_id, vehicle_count, speed, occupancy, travel_time`.
2. Pass the CSV path to the evaluator:
   ```python
   from prediction.evaluator import run_evaluation
   run_evaluation(data_source="path/to/sumo_output.csv")
   ```
3. Or pass a list of dicts directly (for a live feed):
   ```python
   from prediction.data_loader import load_data
   df = load_data(sumo_state_list)  # list[dict]
   ```
