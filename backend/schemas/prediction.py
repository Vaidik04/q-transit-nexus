from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class Prediction(BaseModel):
    edge_id: str
    t5: float = Field(..., description="Predicted travel time at T+5 min in seconds")
    t10: float = Field(..., description="Predicted travel time at T+10 min in seconds")
    t15: float = Field(..., description="Predicted travel time at T+15 min in seconds")
    speed_t5: Optional[float] = None
    speed_t10: Optional[float] = None
    speed_t15: Optional[float] = None
    congestion_t5: Optional[float] = None
    congestion_t10: Optional[float] = None
    congestion_t15: Optional[float] = None
    p95_travel_time: Optional[float] = None
    confidence: float = 0.92

class PredictionRequest(BaseModel):
    horizon_minutes: int = 15
    include_uncertainty: bool = True

class PredictionResult(BaseModel):
    timestamp: float
    horizon: int
    predictions: Dict[str, Prediction]
