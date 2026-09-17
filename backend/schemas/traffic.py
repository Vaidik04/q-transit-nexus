from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class EdgeState(BaseModel):
    edge_id: str
    from_node: str
    to_node: str
    length: float = Field(..., description="Length in meters")
    speed: float = Field(..., description="Current speed in km/h")
    flow: float = Field(..., description="Vehicles per hour")
    travel_time: float = Field(..., description="Estimated travel time in seconds")
    congestion: float = Field(..., description="Congestion index between 0.0 and 1.0")
    free_flow_speed: float = 50.0
    capacity: float = 1200.0

class NetworkState(BaseModel):
    timestamp: float
    edges: Dict[str, EdgeState]
    total_vehicles: int
    avg_speed: float
    avg_congestion: float
    active_incidents: int
