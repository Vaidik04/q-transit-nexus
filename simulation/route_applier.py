"""
Module: simulation.route_applier
Description:
    Applies externally computed route decisions (from AT-DQPSO optimizer, RL controller,
    or incident manager) to vehicles in the live SUMO simulation via TraCI.
    Validates route connectivity, handles intersection internal-lane delays,
    and logs rerouting stability metrics.

What it reads:
    - Optimizer route payloads: {"vehicle_id": "car_PV01", "route": ["E_01_11", "E_11_12", ...]}
    - Network adjacency topology

What it writes:
    - TraCI route adjustments (traci.vehicle.setRoute)
    - Reroute audit log and stability score tracking

Dependencies:
    - Depends on: traci_controller.py
    - Depended on by: AT-DQPSO optimizer interface, scenario_manager.py
"""

from typing import List, Dict, Any, Optional

# Adjacency map for 4x3 prototype city grid to validate route continuity
VALID_TRANSITIONS: Dict[str, List[str]] = {
    # Row 0
    "E_00_10": ["E_10_20", "E_10_11"],
    "E_10_20": ["E_20_30", "E_20_21"],
    "E_20_30": ["E_30_31"],
    "E_30_20": ["E_20_10", "E_20_21"],
    "E_20_10": ["E_10_00", "E_10_11"],
    "E_10_00": ["E_00_01"],
    # Row 1 (Central Arterial)
    "E_01_11": ["E_11_21", "E_11_12", "E_11_10"],
    "E_11_21": ["E_21_31", "E_21_22", "E_21_20"],
    "E_21_31": ["E_31_32", "E_31_30"],
    "E_31_21": ["E_21_11", "E_21_22", "E_21_20"],
    "E_21_11": ["E_11_01", "E_11_12", "E_11_10"],
    "E_11_01": ["E_01_02", "E_01_00"],
    # Row 2
    "E_02_12": ["E_12_22", "E_12_11"],
    "E_12_22": ["E_22_32", "E_22_21"],
    "E_22_32": ["E_32_31"],
    "E_32_22": ["E_22_12", "E_22_21"],
    "E_22_12": ["E_12_02", "E_12_11"],
    "E_12_02": ["E_02_01"],
    # Col 0
    "E_00_01": ["E_01_02", "E_01_11"],
    "E_01_00": ["E_00_10"],
    "E_01_02": ["E_02_12"],
    "E_02_01": ["E_01_00", "E_01_11"],
    # Col 1
    "E_10_11": ["E_11_12", "E_11_21", "E_11_01"],
    "E_11_10": ["E_10_20", "E_10_00"],
    "E_11_12": ["E_12_22", "E_12_02"],
    "E_12_11": ["E_11_10", "E_11_01", "E_11_21"],
    # Col 2
    "E_20_21": ["E_21_22", "E_21_31", "E_21_11"],
    "E_21_20": ["E_20_30", "E_20_10"],
    "E_21_22": ["E_22_32", "E_22_12"],
    "E_22_21": ["E_21_20", "E_21_11", "E_21_31"],
    # Col 3
    "E_30_31": ["E_31_32", "E_31_21"],
    "E_31_30": ["E_30_20"],
    "E_31_32": ["E_32_22"],
    "E_32_31": ["E_31_30", "E_31_21"]
}

class RouteApplier:
    """
    Validates and dynamically injects new routes into running vehicles via TraCI.
    Tracks route churn and stability to safeguard against optimizer oscillation.
    """

    def __init__(self, controller):
        self.controller = controller
        self.reroute_history: List[Dict[str, Any]] = []
        self.reroute_counts: Dict[str, int] = {}

    def validate_route_continuity(self, route: List[str]) -> bool:
        """Verifies that consecutive edges in the proposed route are connected in the network."""
        if len(route) <= 1:
            return True
        for i in range(len(route) - 1):
            curr_e = route[i]
            next_e = route[i + 1]
            allowed = VALID_TRANSITIONS.get(curr_e, [])
            if next_e not in allowed:
                return False
        return True

    def apply_route(self, vehicle_id: str, new_edges: List[str], force: bool = False) -> Dict[str, Any]:
        """
        Applies a new route to an active vehicle.
        Ensures vehicle is currently on the network and validates continuity.
        """
        traci = self.controller.traci
        sim_time = self.controller.get_time()

        if not self.controller.is_connected or not traci:
            return {"success": False, "reason": "TraCI not connected"}

        try:
            active_vehs = traci.vehicle.getIDList()
            if vehicle_id not in active_vehs:
                return {"success": False, "reason": f"Vehicle {vehicle_id} not active in simulation"}

            current_edge = traci.vehicle.getRoadID(vehicle_id)
            if current_edge.startswith(":"):
                # Vehicle is currently inside junction intersection; defer or prepend target
                return {"success": False, "reason": f"Vehicle {vehicle_id} is inside junction {current_edge}"}

            # Align current edge with route beginning
            if new_edges and new_edges[0] != current_edge:
                if current_edge in new_edges:
                    idx = new_edges.index(current_edge)
                    new_edges = new_edges[idx:]
                else:
                    new_edges = [current_edge] + new_edges

            # Validate route continuity
            if not force and not self.validate_route_continuity(new_edges):
                return {"success": False, "reason": f"Route continuity violation: {new_edges}"}

            # Apply route to SUMO
            traci.vehicle.setRoute(vehicle_id, new_edges)
            traci.vehicle.setParameter(vehicle_id, "is_rerouted", "true")

            # Track stability / churn
            self.reroute_counts[vehicle_id] = self.reroute_counts.get(vehicle_id, 0) + 1
            record = {
                "time": sim_time,
                "vehicle_id": vehicle_id,
                "new_route": new_edges,
                "reroute_count": self.reroute_counts[vehicle_id],
                "status": "APPLIED"
            }
            self.reroute_history.append(record)
            return {"success": True, "vehicle_id": vehicle_id, "route": new_edges}

        except Exception as e:
            record = {
                "time": sim_time,
                "vehicle_id": vehicle_id,
                "new_route": new_edges,
                "status": f"FAILED: {e}"
            }
            self.reroute_history.append(record)
            return {"success": False, "reason": str(e)}

    def batch_apply_routes(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Applies a batch of route decisions returned by AT-DQPSO.
        Payload format: [{"vehicle_id": "car_PV01", "route": [...]}, ...]
        """
        results = {}
        success_count = 0
        for item in decisions:
            vid = item.get("vehicle_id")
            route = item.get("route")
            if vid and route:
                res = self.apply_route(vid, route)
                results[vid] = res
                if res.get("success"):
                    success_count += 1

        return {
            "total": len(decisions),
            "applied": success_count,
            "details": results
        }

    def get_rerouting_stability_score(self) -> float:
        """
        Computes a rerouting stability score (0.0 to 1.0).
        High churn (>3 reroutes per vehicle) reduces stability.
        """
        if not self.reroute_counts:
            return 1.0
        excessive = sum(1 for cnt in self.reroute_counts.values() if cnt > 2)
        penalty = excessive / len(self.reroute_counts)
        return max(0.0, 1.0 - penalty)
