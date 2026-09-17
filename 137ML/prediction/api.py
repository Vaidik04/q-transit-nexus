"""
api.py — FastAPI HTTP layer for the Traffic Prediction module.

Exposes:
    POST /prediction  — multi-horizon travel-time predictions per edge
    POST /anomaly     — anomaly detection per edge
    GET  /health      — liveness check

The HTTP endpoints delegate to inference.predict_future_costs() and
inference.detect_anomalies() — the same functions the optimizer can call
in-process.  Both integration paths (HTTP and direct Python import) are
fully supported.

Run locally:
    uvicorn prediction.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prediction.inference import (
    EdgePrediction,
    EdgeState,
    detect_anomalies,
    predict_future_costs,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Traffic Prediction API",
    description=(
        "Multi-horizon traffic travel-time prediction and anomaly detection. "
        "Part of PS-137 Smart India Hackathon — Quantum-Inspired Intelligent "
        "Traffic Route Optimization."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class EdgeInput(BaseModel):
    """Single road-edge state snapshot."""
    edge_id: str = Field(..., description="Unique road edge ID", json_schema_extra={"example": "E14"})
    speed: float = Field(..., ge=0.0, description="Current speed (km/h)", json_schema_extra={"example": 31.2})
    flow: float = Field(..., ge=0.0, description="Vehicle count / flow", json_schema_extra={"example": 742.0})
    timestamp: int = Field(..., description="Unix timestamp (seconds)", json_schema_extra={"example": 600})
    occupancy: float = Field(0.0, ge=0.0, le=1.0, description="Lane occupancy [0,1]")


class PredictionRequest(BaseModel):
    edges: List[EdgeInput] = Field(..., min_length=1)
    horizons: List[int] = Field(
        default=[5, 10, 15],
        description="Prediction horizons in minutes (e.g. [5, 10, 15])",
    )


class EdgePredictionOut(BaseModel):
    edge_id: str
    t5: float
    t10: float
    t15: float
    congestion: float = Field(..., description="Current congestion score [0, 1]")


class PredictionResponse(BaseModel):
    predictions: List[EdgePredictionOut]


class AnomalyRequest(BaseModel):
    edges: List[EdgeInput] = Field(..., min_length=1)


class AnomalyResult(BaseModel):
    edge_id: str
    is_anomaly: bool
    severity: float = Field(..., description="Normalized deviation severity [0, 1]")


class AnomalyResponse(BaseModel):
    anomalies: List[AnomalyResult]


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health_check() -> HealthResponse:
    """Liveness check — returns 200 OK when the service is running."""
    return HealthResponse(status="ok", version=app.version)


@app.post("/prediction", response_model=PredictionResponse, tags=["Prediction"])
def prediction_endpoint(request: PredictionRequest) -> PredictionResponse:
    """Predict multi-horizon travel-time costs for a batch of road edges.

    Request body:
        edges    : list of edge state snapshots (edge_id, speed, flow, timestamp)
        horizons : list of integer horizons in minutes (default [5, 10, 15])

    Response:
        predictions : list of per-edge predictions (t5, t10, t15, congestion)
    """
    try:
        edge_states = [
            EdgeState(
                edge_id=e.edge_id,
                speed=e.speed,
                flow=e.flow,
                timestamp=e.timestamp,
                occupancy=e.occupancy,
            )
            for e in request.edges
        ]
        preds: List[EdgePrediction] = predict_future_costs(
            edge_states, horizons=request.horizons
        )
        return PredictionResponse(
            predictions=[
                EdgePredictionOut(
                    edge_id=p.edge_id,
                    t5=p.t5,
                    t10=p.t10,
                    t15=p.t15,
                    congestion=p.congestion,
                )
                for p in preds
            ]
        )
    except Exception as exc:
        logger.exception("Error in /prediction endpoint")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/anomaly", response_model=AnomalyResponse, tags=["Anomaly"])
def anomaly_endpoint(request: AnomalyRequest) -> AnomalyResponse:
    """Detect anomalies in the current edge state snapshot.

    Returns per-edge anomaly flag and severity score.
    """
    try:
        edge_states = [
            EdgeState(
                edge_id=e.edge_id,
                speed=e.speed,
                flow=e.flow,
                timestamp=e.timestamp,
                occupancy=e.occupancy,
            )
            for e in request.edges
        ]
        results = detect_anomalies(edge_states)
        return AnomalyResponse(
            anomalies=[
                AnomalyResult(
                    edge_id=r["edge_id"],
                    is_anomaly=r["is_anomaly"],
                    severity=r["severity"],
                )
                for r in results
            ]
        )
    except Exception as exc:
        logger.exception("Error in /anomaly endpoint")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
