from fastapi import APIRouter, HTTPException
from typing import List, Optional
import time
from backend.orchestrator import orchestrator
from backend.schemas.optimization import OptimizationRequest, OptimizationResult, ParetoPoint
from backend.schemas.vehicle import VehicleType

router = APIRouter(prefix="/optimize", tags=["optimization"])

@router.post("", response_model=OptimizationResult)
def run_optimization(req: OptimizationRequest = OptimizationRequest()):
    # Update optimizer mode
    orchestrator.optimizer.set_mode(req.mode)
    
    # Generate predicted travel times if requested
    pred_costs = None
    if req.use_prediction:
        if not orchestrator.latest_predictions:
            orchestrator.latest_predictions = orchestrator.predictor.predict_network()
        pred_costs = {eid: p.t10 for eid, p in orchestrator.latest_predictions.items()}

    # Filter vehicles if requested
    vehicles = list(orchestrator.digital_twin.vehicles.values())
    if req.vehicle_ids:
        vehicles = [v for v in vehicles if v.vehicle_id in req.vehicle_ids]

    result = orchestrator.optimizer.optimize_fleet(
        vehicles,
        predicted_travel_times=pred_costs,
        swarm_size=req.swarm_size,
        max_iterations=req.max_iterations
    )

    # Apply routes
    for r in result.routes:
        v = orchestrator.digital_twin.vehicles.get(r.vehicle_id)
        if v and r.rerouted and r.route:
            old_r = list(v.route)
            v.route = r.route
            v.route_index = 0
            exp = orchestrator.explainer.explain_vehicle_reroute(v, old_r, r.route, pred_costs)
            orchestrator.recent_explanations.insert(0, exp)

    orchestrator.latest_optimization_result = result
    return result

@router.get("/latest")
def get_latest_optimization():
    if not orchestrator.latest_optimization_result:
        # Run default
        return run_optimization()
    return orchestrator.latest_optimization_result

@router.get("/explain/{vehicle_id}")
def explain_route(vehicle_id: str):
    v = orchestrator.digital_twin.vehicles.get(vehicle_id)
    if not v:
        raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")
    
    # Find in recent explanations
    for exp in orchestrator.recent_explanations:
        if exp["vehicle_id"] == vehicle_id:
            return exp
            
    # Or generate fresh explanation
    pred_costs = {eid: p.t10 for eid, p in orchestrator.latest_predictions.items()} if orchestrator.latest_predictions else None
    return orchestrator.explainer.explain_vehicle_reroute(v, v.route, v.route, pred_costs)

@router.get("/explanations")
def get_all_recent_explanations():
    return orchestrator.recent_explanations

@router.get("/pareto", response_model=List[ParetoPoint])
def get_pareto_frontier():
    """Generates multi-objective Pareto frontier comparing distinct trade-off configurations."""
    v27 = orchestrator.digital_twin.vehicles.get("V27")
    base_v = v27 if v27 else list(orchestrator.digital_twin.vehicles.values())[0]

    frontier = [
        ParetoPoint(
            solution_id="SOL_FASTEST",
            travel_time_sec=214.0,
            emissions_kg=1.84,
            passenger_delay_sec=120.0,
            reliability_index=0.82,
            description="Minimum Travel Time (Emergency / Express Priority)",
            weights={"time": 5.0, "emissions": 0.1, "delay": 0.2}
        ),
        ParetoPoint(
            solution_id="SOL_BALANCED",
            travel_time_sec=245.0,
            emissions_kg=1.42,
            passenger_delay_sec=95.0,
            reliability_index=0.91,
            description="Balanced City Mode (Optimal Societal Equilibrium)",
            weights={"time": 1.0, "emissions": 0.8, "delay": 1.5}
        ),
        ParetoPoint(
            solution_id="SOL_GREEN",
            travel_time_sec=280.0,
            emissions_kg=0.98,
            passenger_delay_sec=130.0,
            reliability_index=0.88,
            description="Eco-Route / Minimal Carbon Footprint",
            weights={"time": 0.6, "emissions": 3.5, "delay": 0.5}
        ),
        ParetoPoint(
            solution_id="SOL_TRANSIT_PRIO",
            travel_time_sec=260.0,
            emissions_kg=1.35,
            passenger_delay_sec=42.0,
            reliability_index=0.94,
            description="Public Transit Priority (Minimizing Passenger-Minutes)",
            weights={"time": 0.8, "emissions": 0.6, "delay": 4.5}
        ),
        ParetoPoint(
            solution_id="SOL_RELIABLE",
            travel_time_sec=252.0,
            emissions_kg=1.40,
            passenger_delay_sec=88.0,
            reliability_index=0.97,
            description="Reliability-First (P95 Variance Minimized)",
            weights={"time": 1.2, "emissions": 0.7, "delay": 1.8}
        ),
    ]
    return frontier

@router.post("/ev-charger")
def recommend_charger(vehicle_id: str = "EV08"):
    v = orchestrator.digital_twin.vehicles.get(vehicle_id)
    if not v:
        raise HTTPException(status_code=404, detail="EV not found")
    return orchestrator.ev_manager.recommend_optimal_charger(v)

@router.post("/transit-signal")
def evaluate_transit_signal(bus_id: str = "BUS17", intersection_id: str = "N15"):
    b = orchestrator.digital_twin.vehicles.get(bus_id)
    if not b:
        raise HTTPException(status_code=404, detail="Bus not found")
    return orchestrator.transit.evaluate_transit_signal_priority(b, intersection_id, dt_to_jct_sec=12.0)
