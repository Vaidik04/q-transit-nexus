import time
from typing import Dict, List, Optional
from backend.schemas.vehicle import VehicleState, VehicleType
from backend.modules.network_engine import UrbanNetwork

class EVChargingManager:
    """
    Electric Vehicle & Charging Station Optimization Module:
    - Tracks Battery State of Charge (SOC)
    - Monitors fast-charging stations, current queues, and waiting times
    - Co-optimizes route + charging station selection: Charger A vs Charger B
    """
    def __init__(self, network: UrbanNetwork):
        self.network = network
        self.stations: Dict[str, dict] = {
            "CS_N21": {
                "station_id": "CS_N21",
                "node_id": "N21",
                "name": "South EV Fast-Charge SuperHub",
                "total_plugs": 8,
                "occupied_plugs": 7,
                "power_kw": 150,
                "current_queue": 3,
                "expected_queue_wait_min": 25.0, # Charger A in prompt
                "price_per_kwh": 0.28
            },
            "CS_N10": {
                "station_id": "CS_N10",
                "node_id": "N10",
                "name": "Civic Center Green Charge",
                "total_plugs": 6,
                "occupied_plugs": 2,
                "power_kw": 120,
                "current_queue": 0,
                "expected_queue_wait_min": 4.0,  # Charger B in prompt
                "price_per_kwh": 0.32
            },
            "CS_N18": {
                "station_id": "CS_N18",
                "node_id": "N18",
                "name": "East Freight Fast Charger",
                "total_plugs": 4,
                "occupied_plugs": 1,
                "power_kw": 200,
                "current_queue": 0,
                "expected_queue_wait_min": 2.0,
                "price_per_kwh": 0.30
            }
        }

    def recommend_optimal_charger(self, vehicle: VehicleState) -> dict:
        """
        Evaluates chargers based on (Travel Time to Charger + Expected Queue + Charging Time).
        Prevents naive 'nearest charger' selection when queues cause excessive delay.
        """
        options = []
        origin_node = vehicle.origin
        if vehicle.current_edge in self.network.edges_data:
            origin_node = self.network.edges_data[vehicle.current_edge]["to_node"]

        for sid, station in self.stations.items():
            dest_node = station["node_id"]
            path = self.network.find_shortest_path_edges(origin_node, dest_node)
            travel_time_sec = sum(self.network.edges_data[e]["travel_time"] for e in path if e in self.network.edges_data)
            travel_time_min = travel_time_sec / 60.0
            dist_km = sum(self.network.edges_data[e]["length"] for e in path if e in self.network.edges_data) / 1000.0

            # Battery recharge needed (e.g. from current SOC to 80%)
            cur_soc = vehicle.soc or 0.3
            cap_kwh = vehicle.battery_capacity_kwh or 60.0
            needed_kwh = max(0.0, (0.80 - cur_soc) * cap_kwh)
            charge_duration_min = (needed_kwh / station["power_kw"]) * 60.0

            queue_wait_min = station["expected_queue_wait_min"]
            total_duration_min = travel_time_min + queue_wait_min + charge_duration_min

            options.append({
                "station_id": sid,
                "name": station["name"],
                "node_id": dest_node,
                "distance_km": round(dist_km, 1),
                "travel_time_min": round(travel_time_min, 1),
                "expected_queue_wait_min": queue_wait_min,
                "charging_duration_min": round(charge_duration_min, 1),
                "total_time_min": round(total_duration_min, 1),
                "path": path
            })

        options.sort(key=lambda x: x["total_time_min"])
        best = options[0]
        runner_up = options[1] if len(options) > 1 else None

        reason = None
        if runner_up and best["distance_km"] > runner_up["distance_km"] and best["total_time_min"] < runner_up["total_time_min"]:
            reason = (f"Selected {best['station_id']} ({best['distance_km']} km away) over {runner_up['station_id']} "
                      f"({runner_up['distance_km']} km away) because queue delay is {runner_up['expected_queue_wait_min']} min vs {best['expected_queue_wait_min']} min.")

        return {
            "vehicle_id": vehicle.vehicle_id,
            "current_soc": vehicle.soc,
            "recommended_station": best,
            "comparison_options": options,
            "reason": reason or f"Optimal total time to charge: {best['total_time_min']} min"
        }
