"""
route_decoder.py — Decodes particle coordinates into actionable vehicle routes.

Transforms continuous particle vectors X in [0, 1]^M or discrete indices into:
- Concrete sequence of edge IDs per vehicle
- Detailed routing decisions and empirical properties (distance, travel time, reroute flag)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import numpy as np

from optimization.graph_interface import TransportationGraph
from optimization.route_encoder import RouteEncoder, VehicleRoutingRequest

logger = logging.getLogger(__name__)


@dataclass
class DecodedVehicleRoute:
    """Detailed routing decision for a single vehicle."""
    vehicle_id: str
    vehicle_type: str
    priority: str
    route: List[str]                   # Ordered edge IDs
    total_length_m: float
    estimated_travel_time_sec: float
    is_rerouted: bool = False
    candidate_index: int = 0


class RouteDecoder:
    """Translates particle vectors into valid vehicle routes and network metrics."""

    def __init__(self, encoder: RouteEncoder, graph: TransportationGraph) -> None:
        self.encoder = encoder
        self.graph = graph

    def decode_particle(
        self,
        particle_vector: np.ndarray,
        future_horizon_min: int = 0,
    ) -> List[DecodedVehicleRoute]:
        """Decodes a continuous particle position vector X in [0, 1]^M into vehicle routes."""
        decoded_routes: List[DecodedVehicleRoute] = []

        for i, vehicle in enumerate(self.encoder.vehicles):
            val = float(np.clip(particle_vector[i], 0.0, 1.0))
            k_choices = self.encoder.get_candidate_count(i)
            
            # Map [0, 1] interval to discrete choice [0, k-1]
            idx = int(val * k_choices)
            if idx >= k_choices:
                idx = k_choices - 1

            candidate_paths = self.encoder.candidate_paths[vehicle.vehicle_id]
            selected_path = candidate_paths[idx] if candidate_paths else []

            # Compute route distance and travel time
            total_dist = 0.0
            total_time = 0.0
            for edge_id in selected_path:
                edge = self.graph.get_edge(edge_id)
                if edge:
                    total_dist += edge.length_m
                    total_time += edge.get_travel_time_for_horizon(future_horizon_min)

            # Check if this route represents a reroute from current route
            is_rerouted = False
            if vehicle.current_route is not None and selected_path != vehicle.current_route:
                is_rerouted = True

            decoded_routes.append(
                DecodedVehicleRoute(
                    vehicle_id=vehicle.vehicle_id,
                    vehicle_type=vehicle.vehicle_type,
                    priority=vehicle.priority,
                    route=list(selected_path),
                    total_length_m=total_dist,
                    estimated_travel_time_sec=total_time,
                    is_rerouted=is_rerouted,
                    candidate_index=idx,
                )
            )

        return decoded_routes

    def decode_discrete_indices(
        self,
        indices: Sequence[int],
        future_horizon_min: int = 0,
    ) -> List[DecodedVehicleRoute]:
        """Decodes discrete candidate index choices directly (used by discrete baselines like GA/ACO)."""
        particle = self.encoder.encode_discrete_indices(indices)
        return self.decode_particle(particle, future_horizon_min=future_horizon_min)

    def to_api_decisions(self, decoded_routes: Sequence[DecodedVehicleRoute]) -> List[Dict[str, Union[str, List[str]]]]:
        """Converts decoded routes to the standardized dynamic route applier API format."""
        return [
            {
                "vehicle_id": r.vehicle_id,
                "route": r.route,
            }
            for r in decoded_routes
        ]
