from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class OptimizationMode(BaseModel):
    name: str = "balanced"  # balanced, emergency, green, transit
    w_time: float = 1.0
    w_dist: float = 0.5
    w_cong: float = 1.2
    w_pass: float = 2.0
    w_emis: float = 0.8
    w_reroute: float = 0.4
    w_viol: float = 10.0

class OptimizationRequest(BaseModel):
    vehicle_ids: Optional[List[str]] = None
    mode: str = "balanced"
    use_prediction: bool = True
    prediction_horizon: int = 10
    max_iterations: int = 50
    swarm_size: int = 30

class SingleRouteResult(BaseModel):
    vehicle_id: str
    route: List[str]
    fitness: float
    travel_time: float
    distance: float
    emissions_co2_kg: float = 0.0
    rerouted: bool = False
    old_route: Optional[List[str]] = None
    reason: Optional[str] = None
    expected_savings_min: Optional[float] = None

class OptimizationResult(BaseModel):
    timestamp: float
    algorithm: str = "AT-DQPSO"
    iterations: int
    runtime_sec: float
    best_fitness: float
    routes: List[SingleRouteResult]
    routes_changed: int
    mode_applied: str

class ParetoPoint(BaseModel):
    solution_id: str
    travel_time_sec: float
    emissions_kg: float
    passenger_delay_sec: float
    reliability_index: float
    description: str
    weights: Dict[str, float]
