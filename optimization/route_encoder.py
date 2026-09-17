"""
route_encoder.py — Encodes multi-vehicle routing configurations into continuous/discrete particle space.

Maps a set of M vehicles with candidate K-paths into:
- Continuous particle position vectors X in [0, 1]^M
- Discrete particle route selection indices in {0, ..., K-1}^M

Supports:
- Initialization of swarm populations (uniform random, heuristic greedy bias)
- Encoding existing route baselines into particle positions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from optimization.graph_interface import TransportationGraph

logger = logging.getLogger(__name__)


@dataclass
class VehicleRoutingRequest:
    """Represents an active vehicle requiring routing or rerouting."""
    vehicle_id: str
    origin: str                        # Node ID or Edge ID
    destination: str                   # Node ID or Edge ID
    vehicle_type: str = "passenger"    # "passenger" | "bus" | "ambulance" | "ev" | "delivery"
    priority: str = "NORMAL"           # "NORMAL" | "HIGH" | "EMERGENCY"
    current_edge: Optional[str] = None
    passenger_load: int = 1            # For buses / passenger count
    battery_soc: float = 100.0         # Battery State-of-Charge (%) for EVs
    current_route: Optional[List[str]] = None # Existing route for stability tracking

    # Extended constraint parameters
    capacity: float = 4.0              # Maximum vehicle passenger / load capacity
    current_load: float = 1.0          # Current onboard load
    max_duration_sec: Optional[float] = None  # Max allowed route duration in seconds
    max_distance_m: Optional[float] = None    # Max allowed route distance in meters
    time_window_earliest_sec: Optional[float] = None # Earliest acceptable arrival time (s)
    time_window_latest_sec: Optional[float] = None   # Latest acceptable arrival deadline (s)
    departure_time_sec: float = 0.0    # Vehicle departure timestamp
    depot_node: Optional[str] = None   # Required terminal depot junction node ID
    is_available: bool = True          # Fleet vehicle availability status


class RouteEncoder:
    """Manages vehicle candidate paths and encodes decisions into swarm particle vectors."""

    def __init__(
        self,
        graph: TransportationGraph,
        vehicles: Sequence[VehicleRoutingRequest],
        k_paths: int = 5,
    ) -> None:
        self.graph = graph
        self.vehicles: List[VehicleRoutingRequest] = list(vehicles)
        self.k_paths = k_paths
        
        # Vehicle index mappings: vehicle_id -> index in particle vector
        self.vehicle_to_idx: Dict[str, int] = {v.vehicle_id: i for i, v in enumerate(self.vehicles)}
        self.dimension = len(self.vehicles)

        # Precompute candidate paths: vehicle_id -> List[List[edge_id]]
        self.candidate_paths: Dict[str, List[List[str]]] = {}
        self._generate_candidate_paths()

    def _generate_candidate_paths(self) -> None:
        """Finds up to k alternative paths for each vehicle."""
        for v in self.vehicles:
            start = v.current_edge if v.current_edge else v.origin
            end = v.destination
            
            # Static freeflow path (index 0 candidate)
            static_paths = self.graph.find_k_shortest_edge_paths(start, end, k=1, weight_attr="freeflow_time")
            # Dynamic alternative paths
            dynamic_paths = self.graph.find_k_shortest_edge_paths(start, end, k=self.k_paths, weight_attr="weight")
            
            paths: List[List[str]] = []
            for p in static_paths + dynamic_paths:
                if p and p not in paths:
                    paths.append(p)
            
            # Fallback: if no edge path found, try direct node path
            if not paths:
                node_paths = self.graph.find_k_shortest_node_paths(v.origin, v.destination, k=self.k_paths)
                for np in node_paths:
                    ep = self.graph.node_path_to_edge_path(np)
                    if ep and ep not in paths:
                        paths.append(ep)

            # Fallback guarantee: if graph is disconnected or start == end
            if not paths and v.current_route:
                paths = [list(v.current_route)]
            elif not paths:
                paths = [[start]]

            self.candidate_paths[v.vehicle_id] = paths[:self.k_paths]

    def get_candidate_count(self, vehicle_idx: int) -> int:
        """Returns number of candidate paths available for the vehicle at vehicle_idx."""
        vid = self.vehicles[vehicle_idx].vehicle_id
        return len(self.candidate_paths[vid])

    def encode_discrete_indices(self, indices: Sequence[int]) -> np.ndarray:
        """Encodes discrete candidate path indices into continuous particle coordinates in [0, 1]."""
        pos = np.zeros(self.dimension, dtype=np.float64)
        for i, idx in enumerate(indices):
            k = max(self.get_candidate_count(i), 1)
            # Center of the discrete bucket [idx/k, (idx+1)/k]
            pos[i] = (min(idx, k - 1) + 0.5) / k
        return pos

    def initialize_population(
        self,
        n_particles: int,
        seed: Optional[int] = None,
        include_greedy: bool = True,
    ) -> np.ndarray:
        """Initializes a swarm population of shape (n_particles, dimension) within [0, 1]."""
        rng = np.random.default_rng(seed)
        population = rng.uniform(0.0, 1.0, size=(n_particles, self.dimension))

        if include_greedy and n_particles > 0:
            # Particle 0: Greedy shortest path (index 0 for all vehicles)
            greedy_indices = [0] * self.dimension
            population[0] = self.encode_discrete_indices(greedy_indices)

        return population
