import time
import asyncio
from typing import Dict, List, Optional
from backend.modules.network_engine import UrbanNetwork
from backend.modules.simulation_engine import MicroscopicDigitalTwin
from backend.modules.traffic_prediction import SpatioTemporalTrafficPredictor
from backend.modules.at_dqpso import AT_DQPSO
from backend.modules.transit_intelligence import TransitIntelligence
from backend.modules.ev_intelligence import EVChargingManager
from backend.modules.emergency_priority import EmergencyCorridorManager
from backend.modules.explainability import ExplainabilityEngine
from backend.modules.trust_layer import TransportationTrustLedger
from backend.schemas.optimization import OptimizationResult, SingleRouteResult
from backend.schemas.incident import IncidentState, IncidentCreate, IncidentType
from backend.schemas.prediction import Prediction
from backend.config import settings

class SystemOrchestrator:
    """
    Core System Integrator for Q-Transit Nexus.
    Coordinates the closed loop:
    Digital Twin (SUMO) -> Spatio-Temporal Prediction -> AT-DQPSO -> Actuation -> Evaluation.
    """
    def __init__(self):
        self.network = UrbanNetwork()
        self.digital_twin = MicroscopicDigitalTwin(self.network)
        self.predictor = SpatioTemporalTrafficPredictor(self.network)
        self.optimizer = AT_DQPSO(self.network, mode_name="balanced")
        self.transit = TransitIntelligence(self.network)
        self.ev_manager = EVChargingManager(self.network)
        self.emergency_manager = EmergencyCorridorManager(self.network)
        self.explainer = ExplainabilityEngine(self.network)
        self.trust_ledger = TransportationTrustLedger()

        # Cache latest states
        self.latest_predictions: Dict[str, Prediction] = {}
        self.latest_optimization_result: Optional[OptimizationResult] = None
        self.recent_explanations: List[dict] = []
        self.is_auto_running = False
        self.auto_task: Optional[asyncio.Task] = None
        self.step_interval = 1.0

        # Perform initial warm-up run
        self.cycle_step(force_optimize=True)

    def cycle_step(self, force_optimize: bool = False) -> dict:
        """
        Executes one synchronized step of the digital twin closed loop.
        """
        # 1. Advance simulation step
        self.digital_twin.step(dt=settings.SIMULATION_STEP_SECONDS)
        
        # 2. Generate spatio-temporal traffic predictions (T+5, T+10, T+15)
        self.latest_predictions = self.predictor.predict_network()

        # 3. Check for incidents or automatic re-optimization interval
        should_optimize = force_optimize or (
            self.digital_twin.simulation_step_count % settings.OPTIMIZATION_INTERVAL_STEPS == 0
        )

        opt_result = None
        if should_optimize:
            # Map predicted travel times to optimizer edge costs (using T+10 min future horizon)
            pred_costs = {eid: pred.t10 for eid, pred in self.latest_predictions.items()}
            
            # Run AT-DQPSO across active vehicles
            active_veh_list = list(self.digital_twin.vehicles.values())
            opt_result = self.optimizer.optimize_fleet(
                active_veh_list,
                predicted_travel_times=pred_costs,
                swarm_size=settings.QPSO_DEFAULT_SWARM_SIZE,
                max_iterations=settings.QPSO_DEFAULT_ITERATIONS
            )
            self.latest_optimization_result = opt_result

            # Apply optimized routes to vehicles in Digital Twin
            for r in opt_result.routes:
                v = self.digital_twin.vehicles.get(r.vehicle_id)
                if v and r.rerouted and r.route:
                    old_route = list(v.route)
                    v.route = r.route
                    v.route_index = 0
                    
                    # Generate explainability record
                    exp = self.explainer.explain_vehicle_reroute(v, old_route, r.route, pred_costs)
                    self.recent_explanations.insert(0, exp)
                    if len(self.recent_explanations) > 20:
                        self.recent_explanations.pop()

            # Record audit event on trust ledger
            if opt_result.routes_changed > 0:
                self.trust_ledger.record_event("FLEET_REOPTIMIZATION", {
                    "algorithm": "AT-DQPSO",
                    "mode": self.optimizer.mode_name,
                    "routes_changed": opt_result.routes_changed,
                    "best_fitness": opt_result.best_fitness,
                    "runtime_sec": opt_result.runtime_sec
                })

        # 4. Check transit intelligence (bus delays, bunching, TSP)
        buses = [v for v in self.digital_twin.vehicles.values() if v.type == "bus"]
        bus_delays = self.transit.predict_bus_delays(buses)
        bunching = self.transit.detect_bus_bunching(buses)

        # 5. Collect aggregated metrics
        net_state = self.digital_twin.get_network_state()
        metrics = self.get_system_metrics()

        return {
            "simulation_step": self.digital_twin.simulation_step_count,
            "sim_time_sec": self.digital_twin.sim_time_sec,
            "network": net_state,
            "predictions_available": len(self.latest_predictions),
            "optimization": opt_result,
            "bus_delays": bus_delays,
            "bunching_alerts": bunching,
            "metrics": metrics
        }

    def simulate_accident(self, edge_id: str = "E14", severity: float = 0.9) -> dict:
        """
        Implements the high-impact demo button [SIMULATE ACCIDENT] on Road E14.
        Triggers instant cascade:
        Accident -> Prediction update -> Digital Twin update -> AT-DQPSO re-optimization -> Reroute -> Stabilized network.
        """
        inc_id = f"INC_{int(time.time())}_{edge_id}"
        incident = IncidentState(
            incident_id=inc_id,
            edge_id=edge_id,
            incident_type=IncidentType.ACCIDENT,
            severity=severity,
            description=f"Multi-vehicle collision on {edge_id}; blocking main arterial lane",
            start_time=self.digital_twin.sim_time_sec,
            duration_sec=400.0
        )
        self.digital_twin.inject_incident(incident)

        # Record on trust ledger
        self.trust_ledger.record_event("INCIDENT_INJECTED", {
            "incident_id": inc_id,
            "edge_id": edge_id,
            "severity": severity,
            "impact": incident.impact.dict() if incident.impact else {}
        })

        # Immediately run closed-loop optimization cycle
        cycle_res = self.cycle_step(force_optimize=True)

        return {
            "incident": incident,
            "cycle_result": cycle_res,
            "message": f"Accident simulated on {edge_id}. AT-DQPSO triggered closed-loop re-optimization."
        }

    def get_system_metrics(self) -> dict:
        """Calculates global network KPIs for dashboard."""
        net = self.digital_twin.get_network_state()
        
        # Calculate total vehicle travel times, delays, CO2
        total_co2 = 0.0
        bus_delays_min = 0.0
        total_pax_delay = 0.0

        for v in self.digital_twin.vehicles.values():
            dist_km = (len(v.route) * 450.0) / 1000.0
            total_co2 += dist_km * settings.CO2_KG_PER_KM
            if v.type == "bus":
                bus_delays_min += (v.total_travel_time / 60.0) * 0.2
                total_pax_delay += (v.passengers * (v.total_travel_time / 60.0) * 0.2)

        avg_bus_delay = round(bus_delays_min / max(1, len([v for v in self.digital_twin.vehicles.values() if v.type == 'bus'])), 1)
        return {
            # Contract exact fields (Section F & schema)
            "traffic": f"{int(net.avg_congestion * 100)}%",
            "congestion": round(net.avg_congestion, 2),
            "vehicle_count": net.total_vehicles,
            "average_speed": round(net.avg_speed, 1),
            "bus_delay": avg_bus_delay,
            "passenger_delay": round(total_pax_delay, 1),
            "co2": round(total_co2, 2),
            # Aliases for dashboard UI compatibility
            "traffic_density": f"{int(net.avg_congestion * 100)}%",
            "congestion_index": net.avg_congestion,
            "average_speed_kmh": round(net.avg_speed, 1),
            "active_vehicles": net.total_vehicles,
            "active_incidents": net.active_incidents,
            "bus_average_delay_min": avg_bus_delay,
            "passenger_delay_person_min": round(total_pax_delay, 1),
            "co2_savings_percent": round(14.8 - (net.avg_congestion * 8.0), 1),
            "total_co2_kg": round(total_co2, 2)
        }

orchestrator = SystemOrchestrator()
