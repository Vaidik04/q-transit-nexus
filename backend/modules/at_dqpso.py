import time
import math
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx

from backend.schemas.optimization import OptimizationResult, SingleRouteResult, OptimizationMode
from backend.schemas.vehicle import VehicleState, VehicleType
from backend.config import settings

class AT_DQPSO:
    """
    Adaptive Traffic-Aware Discrete Quantum Particle Swarm Optimization (AT-DQPSO).
    Solves multi-objective, multi-vehicle routing with dynamic traffic,
    passenger delay, emissions, EV range, and operational constraints.
    """
    def __init__(self, network, mode_name: str = "balanced"):
        self.network = network
        self.mode_name = mode_name
        self.weights = settings.MODE_WEIGHTS.get(mode_name, settings.MODE_WEIGHTS["balanced"])
        self.alpha_start = settings.QPSO_ALPHA_START
        self.alpha_end = settings.QPSO_ALPHA_END

    def set_mode(self, mode_name: str):
        self.mode_name = mode_name
        self.weights = settings.MODE_WEIGHTS.get(mode_name, settings.MODE_WEIGHTS["balanced"])

    def evaluate_route(
        self,
        vehicle: VehicleState,
        route_edges: List[str],
        predicted_travel_times: Optional[Dict[str, float]] = None,
        old_route: Optional[List[str]] = None
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculates the multi-objective fitness F = w1*T + w2*D + w3*C + w4*P + w5*E + w6*R + w7*V
        Returns: (F, total_time_sec, total_dist_m, emissions_co2_kg, passenger_delay_sec)
        """
        if not route_edges:
            return 999999.0, 99999.0, 99999.0, 999.0, 9999.0

        total_time = 0.0
        total_dist = 0.0
        congestion_sum = 0.0
        violations = 0.0

        # Check topological continuity
        for i in range(len(route_edges)):
            eid = route_edges[i]
            if eid not in self.network.edges_data:
                violations += 100.0
                continue
            
            edata = self.network.edges_data[eid]
            if edata.get("is_blocked", False):
                violations += 200.0  # Massive penalty for blocked road

            total_dist += edata["length"]
            
            # Use predicted travel time if provided, else network travel time
            if predicted_travel_times and eid in predicted_travel_times:
                tt = predicted_travel_times[eid]
            else:
                tt = edata["travel_time"]
            total_time += tt
            congestion_sum += edata["congestion"]

            # Topological check between successive edges
            if i < len(route_edges) - 1:
                next_eid = route_edges[i + 1]
                next_edata = self.network.edges_data.get(next_eid)
                if next_edata and edata["to_node"] != next_edata["from_node"]:
                    violations += 50.0  # Discontinuous edge sequence

        # Passenger Delay Component (crucial for buses)
        passenger_delay = 0.0
        if vehicle.type == VehicleType.BUS:
            # Baseline free-flow time for this route
            free_flow_time = sum(
                (self.network.edges_data[e]["length"] / 1000.0) / 
                (self.network.edges_data[e]["free_flow_speed"] / 3600.0)
                for e in route_edges if e in self.network.edges_data
            )
            delay_sec = max(0.0, total_time - free_flow_time)
            # P = sum(passengers * delay)
            pax = max(1, vehicle.passengers)
            passenger_delay = (pax * delay_sec) / 60.0  # passenger-minutes
        elif vehicle.type == VehicleType.EMERGENCY:
            # Emergency vehicles have extreme time penalty
            total_time *= 2.5

        # Emissions: standard fuel model (distance based + idle delay)
        # kg CO2 = (dist_km * 0.192) + (idle_sec * 0.0006)
        dist_km = total_dist / 1000.0
        idle_time = max(0.0, total_time - (dist_km / (50.0 / 3600.0)))
        
        if vehicle.type == VehicleType.EV:
            emissions = 0.0  # Zero tailpipe emissions
            # Check EV SOC constraint: energy consumption ~ 0.20 kWh / km
            kwh_used = dist_km * 0.20
            if vehicle.battery_capacity_kwh and vehicle.soc is not None:
                remaining_kwh = (vehicle.soc * vehicle.battery_capacity_kwh) - kwh_used
                if remaining_kwh < (0.15 * vehicle.battery_capacity_kwh):  # Minimum 15% SOC
                    violations += 80.0
        else:
            emissions = (dist_km * settings.CO2_KG_PER_KM) + (idle_time * settings.IDLE_CO2_KG_PER_SEC)

        # Rerouting instability penalty
        reroute_penalty = 0.0
        if old_route and old_route != route_edges:
            # Small penalty to prevent excessive flapping if routes are almost identical
            diff_edges = len(set(route_edges) ^ set(old_route))
            reroute_penalty = diff_edges * 1.5

        # Weighted objective
        w = self.weights
        F = (
            w["w_time"] * (total_time / 60.0) +
            w["w_dist"] * dist_km +
            w["w_cong"] * (congestion_sum * 2.0) +
            w["w_pass"] * passenger_delay +
            w["w_emis"] * (emissions * 10.0) +
            w["w_reroute"] * reroute_penalty +
            w["w_viol"] * violations
        )

        return F, total_time, total_dist, emissions, passenger_delay

    def get_candidate_paths(self, origin: str, destination: str, k: int = 6) -> List[List[str]]:
        """Finds up to k loop-free candidate paths using diverse edge weights."""
        candidates = []
        try:
            # 1. Shortest by free distance
            p_dist = nx.shortest_path(self.network.graph, source=origin, target=destination, weight="length")
            candidates.append(self._nodes_to_edges(p_dist))
        except nx.NetworkXNoPath:
            pass

        try:
            # 2. Shortest by current travel time
            p_time = nx.shortest_path(self.network.graph, source=origin, target=destination, weight="travel_time")
            e_time = self._nodes_to_edges(p_time)
            if e_time not in candidates:
                candidates.append(e_time)
        except nx.NetworkXNoPath:
            pass

        # 3. K-shortest paths with small randomized jitter to explore corridors
        try:
            generator = nx.shortest_simple_paths(self.network.graph, source=origin, target=destination, weight="travel_time")
            for _ in range(k * 2):
                p = next(generator)
                e_path = self._nodes_to_edges(p)
                if e_path and e_path not in candidates:
                    candidates.append(e_path)
                if len(candidates) >= k:
                    break
        except (nx.NetworkXNoPath, StopIteration):
            pass

        return [c for c in candidates if c]

    def _nodes_to_edges(self, nodes: List[str]) -> List[str]:
        edges = []
        for i in range(len(nodes) - 1):
            eid = self.network.edge_by_nodes.get((nodes[i], nodes[i + 1]))
            if eid:
                edges.append(eid)
        return edges

    def optimize_vehicle_route(
        self,
        vehicle: VehicleState,
        predicted_travel_times: Optional[Dict[str, float]] = None,
        swarm_size: int = 25,
        max_iterations: int = 35
    ) -> SingleRouteResult:
        """
        Runs AT-DQPSO for an individual vehicle, selecting the optimal path from candidates
        or discovering continuous permutations with heuristic repair.
        """
        start_time = time.time()
        
        # Origin is either the end node of vehicle's current edge or vehicle's origin
        if vehicle.current_edge and vehicle.current_edge in self.network.edges_data:
            current_edge_data = self.network.edges_data[vehicle.current_edge]
            from_node = current_edge_data["to_node"]
        else:
            from_node = vehicle.origin
            
        to_node = vehicle.destination
        
        if from_node == to_node:
            return SingleRouteResult(
                vehicle_id=vehicle.vehicle_id,
                route=[vehicle.current_edge] if vehicle.current_edge else [],
                fitness=0.0,
                travel_time=0.0,
                distance=0.0,
                emissions_co2_kg=0.0,
                rerouted=False
            )

        candidate_paths = self.get_candidate_paths(from_node, to_node, k=8)
        
        # If road blocked on candidate, filter or repair
        repaired_candidates = []
        for path in candidate_paths:
            # Check if blocked
            has_blocked = any(self.network.edges_data.get(eid, {}).get("is_blocked", False) for eid in path)
            if not has_blocked:
                # Prepend current edge if not already at head
                full_path = ([vehicle.current_edge] if vehicle.current_edge and vehicle.current_edge not in path else []) + path
                repaired_candidates.append(full_path)

        if not repaired_candidates:
            # Generate fallback via Dijkstra on unblocked subgraph
            temp_graph = self.network.graph.copy()
            for u, v, d in list(temp_graph.edges(data=True)):
                eid = d.get("edge_id")
                if self.network.edges_data.get(eid, {}).get("is_blocked", False):
                    temp_graph.remove_edge(u, v)
            try:
                p = nx.shortest_path(temp_graph, source=from_node, target=to_node, weight="travel_time")
                repaired_candidates.append(
                    ([vehicle.current_edge] if vehicle.current_edge else []) + self._nodes_to_edges(p)
                )
            except nx.NetworkXNoPath:
                # Retain original candidate with penalty
                if candidate_paths:
                    repaired_candidates.append(candidate_paths[0])
                else:
                    repaired_candidates.append([vehicle.current_edge] if vehicle.current_edge else [])

        num_candidates = len(repaired_candidates)
        dim = 1  # Index in candidate paths mapped continuously

        # Initialize Swarm Positions X and Personal Bests P_best
        # In QPSO, X represents continuous coordinates mapped to discrete route selection
        X = np.random.uniform(0.0, float(num_candidates), (swarm_size, dim))
        P_best = np.copy(X)
        P_best_fitness = np.full(swarm_size, float("inf"))
        
        G_best = X[0].copy()
        G_best_fitness = float("inf")

        # Initial fitness evaluation
        for i in range(swarm_size):
            idx = int(np.clip(X[i, 0], 0, num_candidates - 1))
            cand = repaired_candidates[idx]
            fit, _, _, _, _ = self.evaluate_route(vehicle, cand, predicted_travel_times, vehicle.route)
            P_best_fitness[i] = fit
            if fit < G_best_fitness:
                G_best_fitness = fit
                G_best = X[i].copy()

        # AT-DQPSO Iteration Loop
        for it in range(max_iterations):
            # Adaptive contraction-expansion coefficient alpha
            progress = it / float(max_iterations)
            alpha = self.alpha_start - progress * (self.alpha_start - self.alpha_end)
            
            # Swarm diversity metric
            swarm_variance = np.var(X)
            if swarm_variance < 0.05:
                # Add quantum mutation jitter if swarm stagnates
                alpha += 0.2

            # Compute mean best position (mbest)
            mbest = np.mean(P_best, axis=0)

            for i in range(swarm_size):
                phi = np.random.uniform(0.0, 1.0, dim)
                p = phi * P_best[i] + (1.0 - phi) * G_best
                
                u = np.random.uniform(0.0, 1.0, dim)
                # Quantum wave-function potential well update
                sign = np.where(np.random.uniform(0.0, 1.0, dim) > 0.5, 1.0, -1.0)
                X[i] = p + sign * alpha * np.abs(mbest - X[i]) * np.log(1.0 / np.maximum(u, 1e-7))
                
                # Wrap boundaries
                X[i] = np.clip(X[i], 0.0, float(num_candidates - 1e-4))
                
                # Decode to discrete route
                discrete_idx = int(np.floor(X[i, 0]))
                route_cand = repaired_candidates[discrete_idx]
                
                # Evaluate fitness
                f, _, _, _, _ = self.evaluate_route(vehicle, route_cand, predicted_travel_times, vehicle.route)
                
                # Personal best update
                if f < P_best_fitness[i]:
                    P_best_fitness[i] = f
                    P_best[i] = X[i].copy()
                    
                    # Global best update
                    if f < G_best_fitness:
                        G_best_fitness = f
                        G_best = X[i].copy()

        # Final best route
        best_idx = int(np.floor(G_best[0]))
        final_route = repaired_candidates[best_idx]
        final_fit, final_tt, final_dist, final_emis, final_pax = self.evaluate_route(
            vehicle, final_route, predicted_travel_times, vehicle.route
        )

        # Check if route changed
        rerouted = (vehicle.route != final_route) if vehicle.route else False
        
        # Build explainability rationale
        reason = None
        savings = None
        if rerouted and vehicle.route:
            old_fit, old_tt, _, _, _ = self.evaluate_route(vehicle, vehicle.route, predicted_travel_times)
            savings = round(max(0.0, (old_tt - final_tt) / 60.0), 1)
            
            # Check which road caused the reroute
            blocked_in_old = [e for e in vehicle.route if self.network.edges_data.get(e, {}).get("is_blocked", False)]
            if blocked_in_old:
                reason = f"Road {blocked_in_old[0]} blocked by incident; rerouted to unblocked corridor saving {savings} min"
            elif savings > 0.5:
                reason = f"Downstream congestion predicted; alternative corridor saves {savings} min travel time"
            else:
                reason = "Multi-objective optimization rebalanced network load"

        return SingleRouteResult(
            vehicle_id=vehicle.vehicle_id,
            route=final_route,
            fitness=round(final_fit, 2),
            travel_time=round(final_tt, 1),
            distance=round(final_dist, 1),
            emissions_co2_kg=round(final_emis, 3),
            rerouted=rerouted,
            old_route=vehicle.route if rerouted else None,
            reason=reason,
            expected_savings_min=savings
        )

    def optimize_fleet(
        self,
        vehicles: List[VehicleState],
        predicted_travel_times: Optional[Dict[str, float]] = None,
        swarm_size: int = 25,
        max_iterations: int = 30
    ) -> OptimizationResult:
        """Optimizes routes for all active vehicles."""
        t0 = time.time()
        results: List[SingleRouteResult] = []
        routes_changed = 0

        for v in vehicles:
            res = self.optimize_vehicle_route(v, predicted_travel_times, swarm_size, max_iterations)
            if res.rerouted:
                routes_changed += 1
            results.append(res)

        elapsed = time.time() - t0
        best_fit = sum(r.fitness for r in results) / max(1, len(results))

        return OptimizationResult(
            timestamp=time.time(),
            algorithm="AT-DQPSO",
            iterations=max_iterations,
            runtime_sec=round(elapsed, 3),
            best_fitness=round(best_fit, 2),
            routes=results,
            routes_changed=routes_changed,
            mode_applied=self.mode_name
        )
