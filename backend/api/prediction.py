from fastapi import APIRouter, HTTPException
import time
from backend.orchestrator import orchestrator
from backend.schemas.prediction import PredictionResult, PredictionRequest

router = APIRouter(prefix="/prediction", tags=["prediction"])

@router.post("", response_model=PredictionResult)
def get_or_generate_predictions(req: PredictionRequest = PredictionRequest()):
    preds = orchestrator.latest_predictions
    if not preds:
        preds = orchestrator.predictor.predict_network()
        orchestrator.latest_predictions = preds

    return PredictionResult(
        timestamp=time.time(),
        horizon=req.horizon_minutes,
        predictions=preds
    )

@router.get("/edge/{edge_id}")
def get_edge_prediction(edge_id: str):
    if not orchestrator.latest_predictions:
        orchestrator.latest_predictions = orchestrator.predictor.predict_network()
    
    pred = orchestrator.latest_predictions.get(edge_id)
    if not pred:
        raise HTTPException(status_code=404, detail=f"Edge {edge_id} not found")
    return pred

@router.get("/anomalies")
def get_detected_anomalies():
    anomalies = orchestrator.predictor.detect_traffic_anomalies()
    return {"anomalies": anomalies, "count": len(anomalies)}
