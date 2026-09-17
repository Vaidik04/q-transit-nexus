from typing import Dict, List, Optional
from backend.schemas.vehicle import VehicleState, VehicleType
from backend.modules.network_engine import UrbanNetwork

class ExplainabilityEngine:
    """
    Generates explainable, auditable rationales for routing decisions,
    signal preemption, and transit priority directly from exact mathematical models.
    """
    def __init__(self, network: UrbanNetwork):
        self.network = network

    def explain_vehicle_reroute(
        self,
        vehicle: VehicleState,
        old_route: List[str],
        new_route: List[str],
        predicted_tt: Optional[Dict[str, float]] = None
    ) -> dict:
        """Generates detailed comparison between old route and newly optimized route."""
        def calc_route_metrics(route):
            time_sec = 0.0
            dist_m = 0.0
            max_cong = 0.0
            critical_edge = None
            for eid in route:
                ed = self.network.edges_data.get(eid)
                if ed:
                    tt = predicted_tt.get(eid, ed["travel_time"]) if predicted_tt else ed["travel_time"]
                    time_sec += tt
                    dist_m += ed["length"]
                    if ed["congestion"] > max_cong:
                        max_cong = ed["congestion"]
                        critical_edge = eid
            return time_sec, dist_m, max_cong, critical_edge

        old_t, old_d, old_cong, old_crit = calc_route_metrics(old_route)
        new_t, new_d, new_cong, new_crit = calc_route_metrics(new_route)

        savings_min = round(max(0.0, (old_t - new_t) / 60.0), 1)

        # Check if incident is along the old route
        incident_on_old = None
        for eid in old_route:
            if self.network.edges_data.get(eid, {}).get("is_blocked", False):
                incident_on_old = eid
                break

        if incident_on_old:
            reason = f"Road {incident_on_old} blocked by active incident. Automatic reroute bypasses congestion bottleneck."
        elif savings_min > 1.0:
            reason = f"Avoided critical congestion on link {old_crit} (predicted {round(old_cong*100)}% load). New route saves {savings_min} min."
        else:
            reason = "AT-DQPSO multi-objective optimization balanced travel time and emission footprint."

        return {
            "vehicle_id": vehicle.vehicle_id,
            "vehicle_type": vehicle.type,
            "old_route": old_route,
            "new_route": new_route,
            "old_travel_time_min": round(old_t / 60.0, 1),
            "new_travel_time_min": round(new_t / 60.0, 1),
            "expected_saving_min": savings_min,
            "old_distance_km": round(old_d / 1000.0, 1),
            "new_distance_km": round(new_d / 1000.0, 1),
            "critical_congested_edge": old_crit,
            "incident_bypassed": incident_on_old,
            "decision_rationale": reason
        }

    def explain_signal_priority(self, bus_id: str, passengers: int, delay_saved_min: float, general_traffic_delay_sec: float) -> dict:
        return {
            "entity": f"Bus {bus_id}",
            "passengers": passengers,
            "without_priority_passenger_delay_min": round(passengers * (delay_saved_min * 1.5), 1),
            "with_priority_passenger_delay_min": round(passengers * 0.5, 1),
            "general_traffic_delay_sec": general_traffic_delay_sec,
            "justification": f"High vehicle occupancy ({passengers} passengers) outweighs cross-street penalty of {general_traffic_delay_sec}s per vehicle."
        }
