"""
fitness.py — Multi-objective fitness function for Q-Transit Nexus.

Evaluates:
min F = w1*T + w2*D + w3*C + w4*P + w5*E + w6*R + w7*V

Where:
- T: Travel Time (seconds, incorporating ML future horizon predicted costs)
- D: Travel Distance (normalized / meters)
- C: Network Congestion index along selected paths
- P: Transit Passenger Delay (bus delay * passenger load)
- E: CO2 emissions & EV energy consumption
- R: Rerouting Instability penalty (avoids unnecessary churning of vehicle paths)
- V: Constraint violations penalty from ConstraintManager
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from optimization.config import OptimizationConfig
from optimization.constraints import ConstraintManager
from optimization.graph_interface import TransportationGraph
from optimization.route_decoder import DecodedVehicleRoute
from optimization.route_encoder import RouteEncoder

logger = logging.getLogger(__name__)


@dataclass
class FitnessBreakdown:
    """Detailed components of the multi-objective fitness evaluation."""
    total_fitness: float = 0.0
    travel_time_sec: float = 0.0
    distance_m: float = 0.0
    congestion_score: float = 0.0
    passenger_delay_sec: float = 0.0
    co2_emissions_kg: float = 0.0
    rerouted_count: int = 0
    constraint_violations: int = 0
    constraint_penalty: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_fitness": self.total_fitness,
            "travel_time_sec": self.travel_time_sec,
            "distance_m": self.distance_m,
            "congestion_score": self.congestion_score,
            "passenger_delay_sec": self.passenger_delay_sec,
            "co2_emissions_kg": self.co2_emissions_kg,
            "rerouted_count": float(self.rerouted_count),
            "constraint_violations": float(self.constraint_violations),
            "constraint_penalty": self.constraint_penalty,
        }


class MultiObjectiveFitness:
    """Computes weighted multi-objective fitness and empirical metrics for route assignments."""

    def __init__(
        self,
        graph: TransportationGraph,
        config: OptimizationConfig,
        constraint_manager: Optional[ConstraintManager] = None,
    ) -> None:
        self.graph = graph
        self.config = config
        self.constraint_manager = constraint_manager or ConstraintManager(graph, config)

    def evaluate(
        self,
        routes: Sequence[DecodedVehicleRoute],
        encoder: RouteEncoder,
        future_horizon_min: int = 5,
    ) -> FitnessBreakdown:
        """Evaluates the multi-objective fitness of a candidate fleet route assignment."""
        w = self.config.weights
        n_veh = max(len(routes), 1)

        total_time = 0.0
        total_dist = 0.0
        total_congestion = 0.0
        total_passenger_delay = 0.0
        total_co2_kg = 0.0
        rerouted_count = 0

        # Aggregate vehicle loads per edge across the fleet candidate assignment
        edge_loads: Dict[str, int] = {}
        for r in routes:
            for edge_id in r.route:
                edge_loads[edge_id] = edge_loads.get(edge_id, 0) + 1

        for r in routes:
            v_req = next((v for v in encoder.vehicles if v.vehicle_id == r.vehicle_id), None)
            passenger_load = v_req.passenger_load if v_req else 1
            v_type = v_req.vehicle_type if v_req else "passenger"

            route_time = 0.0
            route_dist = 0.0
            route_cong = 0.0

            for edge_id in r.route:
                edge = self.graph.get_edge(edge_id)
                if edge:
                    route_dist += edge.length_m
                    # Use ML prediction horizon for base edge travel time
                    base_edge_time = edge.get_travel_time_for_horizon(future_horizon_min)

                    if self.config.enable_bpr_latency:
                        v_load = edge_loads.get(edge_id, 1)
                        window_min = max(self.config.bpr_capacity_window_sec / 60.0, 0.1)
                        # Practical vehicle capacity for the dispatch window
                        c_eff = max(2.0, (edge.capacity_vph / 60.0) * window_min)
                        # Standard BPR latency: t = t0 * [1 + alpha * (V / C)^beta]
                        bpr_factor = 1.0 + self.config.bpr_alpha * ((v_load / c_eff) ** self.config.bpr_beta)
                        edge_time = base_edge_time * bpr_factor
                        add_cong = min(0.6, (v_load / c_eff) * 0.15)
                        edge_cong = min(1.0, edge.congestion_score + add_cong)
                    else:
                        edge_time = base_edge_time
                        edge_cong = edge.congestion_score

                    route_time += edge_time
                    route_cong += edge_cong

            n_edges = max(len(r.route), 1)
            avg_cong = route_cong / n_edges

            total_time += route_time
            total_dist += route_dist
            total_congestion += avg_cong

            # Transit passenger delay: Bus delay is weighted by onboard passengers
            if v_type == "bus":
                freeflow_time = sum(
                    self.graph.get_edge(eid).length_m / (self.graph.get_edge(eid).freeflow_speed_kmh * 1000.0 / 3600.0)
                    for eid in r.route if self.graph.get_edge(eid)
                )
                delay = max(0.0, route_time - freeflow_time)
                total_passenger_delay += delay * passenger_load

            # Emissions calculation: ICE vehicles produce ~150g CO2/km; EVs produce 0 tailpipe emissions
            if v_type != "ev":
                dist_km = route_dist / 1000.0
                co2_kg = (dist_km * self.config.co2_grams_per_km) / 1000.0
                total_co2_kg += co2_kg

            if r.is_rerouted:
                rerouted_count += 1

        # Averages for scale-independent fitness weighting
        avg_time = total_time / n_veh
        avg_dist = total_dist / n_veh
        avg_cong = total_congestion / n_veh

        # Evaluate constraints
        c_res = self.constraint_manager.evaluate_constraints(routes, encoder)

        # Fitness objective: min F
        # Scale terms to comparable orders of magnitude
        f_time = w.w_time * avg_time
        f_dist = w.w_distance * (avg_dist / 100.0)
        f_cong = w.w_congestion * (avg_cong * 100.0)
        f_delay = w.w_passenger_delay * (total_passenger_delay / max(n_veh, 1))
        f_emissions = w.w_emissions * (total_co2_kg * 10.0)
        f_reroute = w.w_reroute_instability * float(rerouted_count)
        f_violations = w.w_violations * c_res.total_penalty

        total_fitness = (
            f_time
            + f_dist
            + f_cong
            + f_delay
            + f_emissions
            + f_reroute
            + f_violations
        )

        return FitnessBreakdown(
            total_fitness=total_fitness,
            travel_time_sec=avg_time,
            distance_m=avg_dist,
            congestion_score=avg_cong,
            passenger_delay_sec=total_passenger_delay,
            co2_emissions_kg=total_co2_kg,
            rerouted_count=rerouted_count,
            constraint_violations=c_res.violation_count,
            constraint_penalty=c_res.total_penalty,
        )
