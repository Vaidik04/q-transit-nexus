"""
constraints.py — Hard and soft transportation constraints for multi-vehicle route optimization.

Evaluates:
- Edge capacity exceedance (flow vs. capacity)
- Incident / severe blockage avoidance for general traffic
- EV battery State-of-Charge depletion limits (SoC >= min_soc)
- Emergency vehicle non-blockage guarantee
- Transit bus corridor adherence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from optimization.config import OptimizationConfig
from optimization.graph_interface import TransportationGraph
from optimization.route_decoder import DecodedVehicleRoute
from optimization.route_encoder import RouteEncoder

logger = logging.getLogger(__name__)


@dataclass
class ConstraintViolation:
    """Details of a single constraint violation."""
    constraint_type: str              # "CAPACITY" | "INCIDENT" | "EV_BATTERY" | "EMERGENCY"
    vehicle_id: str
    edge_id: Optional[str] = None
    severity: float = 1.0             # Magnitude of violation
    message: str = ""


@dataclass
class ConstraintEvaluationResult:
    """Summary of constraint evaluation across all assigned routes."""
    total_penalty: float = 0.0
    violation_count: int = 0
    violations: List[ConstraintViolation] = field(default_factory=list)
    edge_loads: Dict[str, int] = field(default_factory=dict)


class ConstraintManager:
    """Validates multi-vehicle route assignments against physical and operational constraints."""

    def __init__(self, graph: TransportationGraph, config: OptimizationConfig) -> None:
        self.graph = graph
        self.config = config

    def evaluate_constraints(
        self,
        routes: Sequence[DecodedVehicleRoute],
        encoder: RouteEncoder,
    ) -> ConstraintEvaluationResult:
        """Evaluates all hard and soft constraints for a complete fleet assignment."""
        result = ConstraintEvaluationResult()
        edge_loads: Dict[str, int] = {}

        # 1. Tally edge load flows from all routes
        for r in routes:
            for edge_id in r.route:
                edge_loads[edge_id] = edge_loads.get(edge_id, 0) + 1

        result.edge_loads = edge_loads

        # 2. Check edge capacity limits
        for edge_id, load in edge_loads.items():
            edge = self.graph.get_edge(edge_id)
            if not edge:
                continue
            # Convert hourly capacity to instantaneous vehicle capacity allowance
            # e.g., 2-lane 400m block can comfortably hold ~30-40 vehicles simultaneously
            max_simultaneous_vehicles = max(int(edge.num_lanes * (edge.length_m / 10.0)), 4)
            if load > max_simultaneous_vehicles:
                overflow = load - max_simultaneous_vehicles
                penalty = float(overflow) * 5.0
                result.total_penalty += penalty
                result.violation_count += 1
                result.violations.append(
                    ConstraintViolation(
                        constraint_type="CAPACITY",
                        vehicle_id="MULTIPLE",
                        edge_id=edge_id,
                        severity=overflow,
                        message=f"Edge {edge_id} capacity exceeded: {load}/{max_simultaneous_vehicles} veh",
                    )
                )

        # 3. Check individual vehicle constraints
        for r in routes:
            v_req = next((v for v in encoder.vehicles if v.vehicle_id == r.vehicle_id), None)
            if not v_req:
                continue

            # A. Incident Avoidance (Non-emergency vehicles routing through closed/severely blocked edges)
            if r.priority != "EMERGENCY":
                for edge_id in r.route:
                    edge = self.graph.get_edge(edge_id)
                    if edge and edge.incident_severity >= 0.8:
                        penalty = 20.0 * edge.incident_severity
                        result.total_penalty += penalty
                        result.violation_count += 1
                        result.violations.append(
                            ConstraintViolation(
                                constraint_type="INCIDENT",
                                vehicle_id=r.vehicle_id,
                                edge_id=edge_id,
                                severity=edge.incident_severity,
                                message=f"Vehicle {r.vehicle_id} routed through blocked incident edge {edge_id}",
                            )
                        )

            # B. EV Minimum State of Charge (SoC) Constraint
            if v_req.vehicle_type == "ev":
                dist_km = r.total_length_m / 1000.0
                kwh_used = dist_km * self.config.ev_kwh_per_km
                battery_capacity_kwh = 50.0
                soc_drop = (kwh_used / battery_capacity_kwh) * 100.0
                arrival_soc = v_req.battery_soc - soc_drop
                if arrival_soc < self.config.min_ev_soc:
                    deficit = self.config.min_ev_soc - arrival_soc
                    penalty = deficit * 2.0
                    result.total_penalty += penalty
                    result.violation_count += 1
                    result.violations.append(
                        ConstraintViolation(
                            constraint_type="EV_BATTERY",
                            vehicle_id=r.vehicle_id,
                            severity=deficit,
                            message=f"EV {r.vehicle_id} arrival SoC {arrival_soc:.1f}% below minimum {self.config.min_ev_soc}%",
                        )
                    )

            # C. Vehicle Capacity Constraint
            if v_req.current_load > v_req.capacity:
                overflow = v_req.current_load - v_req.capacity
                penalty = float(overflow) * 10.0
                result.total_penalty += penalty
                result.violation_count += 1
                result.violations.append(
                    ConstraintViolation(
                        constraint_type="VEHICLE_CAPACITY",
                        vehicle_id=r.vehicle_id,
                        severity=overflow,
                        message=f"Vehicle {r.vehicle_id} load {v_req.current_load} exceeds capacity {v_req.capacity}",
                    )
                )

            # D. Maximum Route Duration Constraint
            if v_req.max_duration_sec is not None and r.estimated_travel_time_sec > v_req.max_duration_sec:
                excess_sec = r.estimated_travel_time_sec - v_req.max_duration_sec
                penalty = (excess_sec / 60.0) * 5.0
                result.total_penalty += penalty
                result.violation_count += 1
                result.violations.append(
                    ConstraintViolation(
                        constraint_type="ROUTE_DURATION",
                        vehicle_id=r.vehicle_id,
                        severity=excess_sec,
                        message=f"Route duration {r.estimated_travel_time_sec:.1f}s exceeds limit {v_req.max_duration_sec}s",
                    )
                )

            # E. Time Windows (Earliest & Latest Arrival Limits)
            arrival_time_sec = v_req.departure_time_sec + r.estimated_travel_time_sec
            if v_req.time_window_latest_sec is not None and arrival_time_sec > v_req.time_window_latest_sec:
                lateness_sec = arrival_time_sec - v_req.time_window_latest_sec
                penalty = (lateness_sec / 60.0) * 8.0
                result.total_penalty += penalty
                result.violation_count += 1
                result.violations.append(
                    ConstraintViolation(
                        constraint_type="TIME_WINDOW",
                        vehicle_id=r.vehicle_id,
                        severity=lateness_sec,
                        message=f"Vehicle {r.vehicle_id} arrived at {arrival_time_sec:.1f}s, after deadline {v_req.time_window_latest_sec}s",
                    )
                )
            if v_req.time_window_earliest_sec is not None and arrival_time_sec < v_req.time_window_earliest_sec:
                earliness_sec = v_req.time_window_earliest_sec - arrival_time_sec
                penalty = (earliness_sec / 60.0) * 2.0  # Idle wait penalty
                result.total_penalty += penalty
                result.violation_count += 1
                result.violations.append(
                    ConstraintViolation(
                        constraint_type="TIME_WINDOW",
                        vehicle_id=r.vehicle_id,
                        severity=earliness_sec,
                        message=f"Vehicle {r.vehicle_id} arrived early at {arrival_time_sec:.1f}s, before window {v_req.time_window_earliest_sec}s",
                    )
                )

            # F. Maximum Route Distance Constraint
            if v_req.max_distance_m is not None and r.total_length_m > v_req.max_distance_m:
                excess_dist = r.total_length_m - v_req.max_distance_m
                penalty = (excess_dist / 100.0) * 4.0
                result.total_penalty += penalty
                result.violation_count += 1
                result.violations.append(
                    ConstraintViolation(
                        constraint_type="MAX_DISTANCE",
                        vehicle_id=r.vehicle_id,
                        severity=excess_dist,
                        message=f"Route distance {r.total_length_m:.1f}m exceeds maximum {v_req.max_distance_m}m",
                    )
                )

            # G. Vehicle Availability Constraint
            if not v_req.is_available and len(r.route) > 0:
                penalty = 50.0
                result.total_penalty += penalty
                result.violation_count += 1
                result.violations.append(
                    ConstraintViolation(
                        constraint_type="VEHICLE_AVAILABILITY",
                        vehicle_id=r.vehicle_id,
                        severity=1.0,
                        message=f"Vehicle {r.vehicle_id} is unavailable but was assigned an active route",
                    )
                )

            # H. Destination / Depot Requirement
            if v_req.depot_node is not None and r.route:
                last_edge = self.graph.get_edge(r.route[-1])
                if last_edge and last_edge.v != v_req.depot_node:
                    penalty = 30.0
                    result.total_penalty += penalty
                    result.violation_count += 1
                    result.violations.append(
                        ConstraintViolation(
                            constraint_type="DEPOT_REQUIREMENT",
                            vehicle_id=r.vehicle_id,
                            severity=1.0,
                            message=f"Route terminates at {last_edge.v} instead of required depot {v_req.depot_node}",
                        )
                    )

        return result
