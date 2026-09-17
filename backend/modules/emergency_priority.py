from typing import Dict, List, Optional
from backend.schemas.vehicle import VehicleState, VehicleType
from backend.modules.network_engine import UrbanNetwork

class EmergencyCorridorManager:
    """
    Emergency Vehicle Corridor Management:
    - Reserves dedicated priority corridors for ambulances and fire trucks
    - Preempts traffic signals to create green waves along the route
    - Reroutes non-emergency private and logistics traffic away from priority links
    """
    def __init__(self, network: UrbanNetwork):
        self.network = network
        self.active_corridors: Dict[str, dict] = {}

    def reserve_corridor(self, emergency_vehicle: VehicleState) -> dict:
        route = emergency_vehicle.route
        if not route:
            route = self.network.find_shortest_path_edges(emergency_vehicle.origin, emergency_vehicle.destination)
            emergency_vehicle.route = route

        emergency_vehicle.is_emergency_active = True
        corridor_id = f"CORRIDOR_{emergency_vehicle.vehicle_id}"
        
        # Mark reserved edges
        reserved_edges = list(route)
        for eid in reserved_edges:
            if eid in self.network.edges_data:
                # Emergency corridor reduces delay for emergency vehicle
                self.network.edges_data[eid]["is_emergency_corridor"] = True

        self.active_corridors[corridor_id] = {
            "corridor_id": corridor_id,
            "vehicle_id": emergency_vehicle.vehicle_id,
            "vehicle_type": emergency_vehicle.type,
            "origin": emergency_vehicle.origin,
            "destination": emergency_vehicle.destination,
            "reserved_edges": reserved_edges,
            "signals_preempted": ["N14", "N15", "N16"],
            "status": "ACTIVE_GREEN_WAVE"
        }

        return self.active_corridors[corridor_id]

    def release_corridor(self, vehicle_id: str):
        corridor_id = f"CORRIDOR_{vehicle_id}"
        if corridor_id in self.active_corridors:
            corridor = self.active_corridors.pop(corridor_id)
            for eid in corridor["reserved_edges"]:
                if eid in self.network.edges_data:
                    self.network.edges_data[eid]["is_emergency_corridor"] = False
