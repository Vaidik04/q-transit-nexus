from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class IncidentType(str, Enum):
    ACCIDENT = "accident"
    ROAD_CLOSURE = "road_closure"
    SPEED_COLLAPSE = "speed_collapse"
    WEATHER = "weather"
    SPECIAL_EVENT = "special_event"

class IncidentCreate(BaseModel):
    edge_id: str
    incident_type: IncidentType = IncidentType.ACCIDENT
    severity: float = Field(0.9, ge=0.0, le=1.0, description="Severity: 1.0 is full blockage")
    description: Optional[str] = "Traffic accident blocking lane"
    duration_sec: float = 300.0

class IncidentImpact(BaseModel):
    incident_id: str
    edge_id: str
    affected_vehicles_count: int
    affected_buses_count: int
    affected_passengers_count: int
    predicted_downstream_congestion: float
    recommended_action: str

class IncidentState(BaseModel):
    incident_id: str
    edge_id: str
    incident_type: IncidentType
    severity: float
    description: str
    start_time: float
    duration_sec: float
    is_active: bool = True
    impact: Optional[IncidentImpact] = None
