import time
import math
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx

from backend.schemas.vehicle import VehicleState, VehicleType
from backend.schemas.optimization import SingleRouteResult
from backend.modules.network_engine import UrbanNetwork

class RoutingBaselines:
    """
    Implements benchmark routing algorithms for scientific comparison:
    1. Dijkstra (Greedy Single-Source Shortest Path)
    2. A* (Heuristic Search with Euclidean Distance)
    3. Genetic Algorithm (GA: selection, crossover, mutation)
    4. Ant Colony Optimization (ACO: pheromone deposit and evaporation)
    5. Standard PSO (Continuous velocity-position with discrete rounding)
    6. Standard QPSO (Quantum PSO without adaptive parameter control or prediction)
    """
    def __init__(self, network: UrbanNetwork):
        self.network = network

    def _nodes_to_edges(self, nodes: List[str]) -> List[str]:
        edges = []
        for i in range(len(nodes) - 1):
            eid = self.network.edge_by_nodes.get((nodes[i], nodes[i + 1]))
            if eid:
                edges.append(eid)
        return edges

    def run_dijkstra(self, vehicle: VehicleState) -> SingleRouteResult:
        t0 = time.time()
        start = vehicle.origin
        target = vehicle.destination
        
        # Dijkstra on current edge travel time
        try:
            p = nx.dijkstra_path(self.network.graph, source=start, target=target, weight="travel_time")
            route = self._nodes_to_edges(p)
        except nx.NetworkXNoPath:
            route = []

        elapsed = time.time() - t0
        tt, dist, emis = self._eval_path(route)
        return SingleRouteResult(
            vehicle_id=vehicle.vehicle_id,
            route=route,
            fitness=round(tt + dist * 0.1, 2),
            travel_time=round(tt, 1),
            distance=round(dist, 1),
            emissions_co2_kg=round(emis, 3),
            rerouted=False
        )

    def run_astar(self, vehicle: VehicleState) -> SingleRouteResult:
        t0 = time.time()
        start = vehicle.origin
        target = vehicle.destination

        def heuristic(u, v):
            nu = self.network.nodes_data[u]
            nv = self.network.nodes_data[v]
            dx = nu["x"] - nv["x"]
            dy = nu["y"] - nv["y"]
            # Euclidean distance / max speed (60 km/h = 16.6 m/s)
            dist_m = math.sqrt(dx*dx + dy*dy) * 4.0
            return dist_m / 16.6

        try:
            p = nx.astar_path(self.network.graph, source=start, target=target, heuristic=heuristic, weight="travel_time")
            route = self._nodes_to_edges(p)
        except nx.NetworkXNoPath:
            route = []

        elapsed = time.time() - t0
        tt, dist, emis = self._eval_path(route)
        return SingleRouteResult(
            vehicle_id=vehicle.vehicle_id,
            route=route,
            fitness=round(tt + dist * 0.1, 2),
            travel_time=round(tt, 1),
            distance=round(dist, 1),
            emissions_co2_kg=round(emis, 3),
            rerouted=False
        )

    def run_ga(self, vehicle: VehicleState, pop_size: int = 25, generations: int = 30) -> SingleRouteResult:
        t0 = time.time()
        candidates = self._get_paths(vehicle.origin, vehicle.destination, k=8)
        if not candidates:
            return self.run_dijkstra(vehicle)

        # Population initialized with candidate indices
        pop = [random.randint(0, len(candidates) - 1) for _ in range(pop_size)]
        best_cand = candidates[0]
        best_fit = float("inf")

        for gen in range(generations):
            fitnesses = []
            for idx in pop:
                c = candidates[idx]
                tt, dist, _ = self._eval_path(c)
                fit = tt + dist * 0.2
                fitnesses.append(fit)
                if fit < best_fit:
                    best_fit = fit
                    best_cand = c

            # Tournament selection and mutation
            new_pop = []
            for _ in range(pop_size):
                i1, i2 = random.sample(range(pop_size), 2)
                winner = pop[i1] if fitnesses[i1] < fitnesses[i2] else pop[i2]
                # Mutation
                if random.random() < 0.25:
                    winner = random.randint(0, len(candidates) - 1)
                new_pop.append(winner)
            pop = new_pop

        tt, dist, emis = self._eval_path(best_cand)
        return SingleRouteResult(
            vehicle_id=vehicle.vehicle_id,
            route=best_cand,
            fitness=round(best_fit, 2),
            travel_time=round(tt, 1),
            distance=round(dist, 1),
            emissions_co2_kg=round(emis, 3),
            rerouted=False
        )

    def run_aco(self, vehicle: VehicleState, n_ants: int = 20, iterations: int = 25) -> SingleRouteResult:
        t0 = time.time()
        candidates = self._get_paths(vehicle.origin, vehicle.destination, k=8)
        if not candidates:
            return self.run_dijkstra(vehicle)

        pheromones = np.ones(len(candidates)) * 1.0
        best_cand = candidates[0]
        best_fit = float("inf")

        for it in range(iterations):
            probs = pheromones / np.sum(pheromones)
            choices = np.random.choice(len(candidates), size=n_ants, p=probs)
            
            # Evaluate ants
            for c_idx in choices:
                c = candidates[c_idx]
                tt, dist, _ = self._eval_path(c)
                fit = tt + dist * 0.2
                if fit < best_fit:
                    best_fit = fit
                    best_cand = c
                # Pheromone deposit
                pheromones[c_idx] += (100.0 / max(1.0, fit))
            
            # Evaporation
            pheromones *= 0.85

        tt, dist, emis = self._eval_path(best_cand)
        return SingleRouteResult(
            vehicle_id=vehicle.vehicle_id,
            route=best_cand,
            fitness=round(best_fit, 2),
            travel_time=round(tt, 1),
            distance=round(dist, 1),
            emissions_co2_kg=round(emis, 3),
            rerouted=False
        )

    def run_pso(self, vehicle: VehicleState, swarm_size: int = 25, iterations: int = 30) -> SingleRouteResult:
        """Classical Particle Swarm Optimization with velocity vector and inertia weight."""
        t0 = time.time()
        candidates = self._get_paths(vehicle.origin, vehicle.destination, k=8)
        if not candidates:
            return self.run_dijkstra(vehicle)

        num_cands = len(candidates)
        X = np.random.uniform(0.0, float(num_cands), swarm_size)
        V = np.random.uniform(-1.0, 1.0, swarm_size)
        P_best = np.copy(X)
        P_best_fit = np.full(swarm_size, float("inf"))
        G_best = X[0]
        G_best_fit = float("inf")

        w, c1, c2 = 0.7, 1.5, 1.5
        for it in range(iterations):
            for i in range(swarm_size):
                idx = int(np.clip(X[i], 0, num_cands - 1))
                tt, dist, _ = self._eval_path(candidates[idx])
                fit = tt + dist * 0.2
                if fit < P_best_fit[i]:
                    P_best_fit[i] = fit
                    P_best[i] = X[i]
                    if fit < G_best_fit:
                        G_best_fit = fit
                        G_best = X[i]

            # Velocity and position update
            r1 = np.random.rand(swarm_size)
            r2 = np.random.rand(swarm_size)
            V = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (G_best - X)
            V = np.clip(V, -2.0, 2.0)
            X = np.clip(X + V, 0, num_cands - 1)

        best_cand = candidates[int(np.clip(G_best, 0, num_cands - 1))]
        tt, dist, emis = self._eval_path(best_cand)
        return SingleRouteResult(
            vehicle_id=vehicle.vehicle_id,
            route=best_cand,
            fitness=round(G_best_fit, 2),
            travel_time=round(tt, 1),
            distance=round(dist, 1),
            emissions_co2_kg=round(emis, 3),
            rerouted=False
        )

    def run_standard_qpso(self, vehicle: VehicleState, swarm_size: int = 25, iterations: int = 30) -> SingleRouteResult:
        """Standard QPSO: static alpha=0.75, no adaptive diversity, static edge weights."""
        t0 = time.time()
        candidates = self._get_paths(vehicle.origin, vehicle.destination, k=8)
        if not candidates:
            return self.run_dijkstra(vehicle)

        num_cands = len(candidates)
        X = np.random.uniform(0.0, float(num_cands), swarm_size)
        P_best = np.copy(X)
        P_best_fit = np.full(swarm_size, float("inf"))
        G_best = X[0]
        G_best_fit = float("inf")

        for it in range(iterations):
            for i in range(swarm_size):
                idx = int(np.clip(X[i], 0, num_cands - 1))
                tt, dist, _ = self._eval_path(candidates[idx])
                fit = tt + dist * 0.2
                if fit < P_best_fit[i]:
                    P_best_fit[i] = fit
                    P_best[i] = X[i]
                    if fit < G_best_fit:
                        G_best_fit = fit
                        G_best = X[i]

            mbest = np.mean(P_best)
            alpha = 0.75  # Fixed static coefficient
            for i in range(swarm_size):
                phi = random.random()
                p = phi * P_best[i] + (1 - phi) * G_best
                u = max(1e-6, random.random())
                sign = 1.0 if random.random() > 0.5 else -1.0
                X[i] = p + sign * alpha * abs(mbest - X[i]) * math.log(1.0 / u)
                X[i] = np.clip(X[i], 0, num_cands - 1)

        best_cand = candidates[int(np.clip(G_best, 0, num_cands - 1))]
        tt, dist, emis = self._eval_path(best_cand)
        return SingleRouteResult(
            vehicle_id=vehicle.vehicle_id,
            route=best_cand,
            fitness=round(G_best_fit, 2),
            travel_time=round(tt, 1),
            distance=round(dist, 1),
            emissions_co2_kg=round(emis, 3),
            rerouted=False
        )

    def _get_paths(self, origin: str, destination: str, k: int = 8) -> List[List[str]]:
        paths = []
        try:
            gen = nx.shortest_simple_paths(self.network.graph, source=origin, target=destination, weight="travel_time")
            for _ in range(k):
                p = next(gen)
                ep = self._nodes_to_edges(p)
                if ep and ep not in paths:
                    paths.append(ep)
        except (nx.NetworkXNoPath, StopIteration):
            pass
        return paths

    def _eval_path(self, route: List[str]) -> Tuple[float, float, float]:
        if not route:
            return 99999.0, 99999.0, 999.0
        tt = 0.0
        dist = 0.0
        for eid in route:
            ed = self.network.edges_data.get(eid)
            if ed:
                tt += ed["travel_time"]
                dist += ed["length"]
        dist_km = dist / 1000.0
        emis = dist_km * 0.192
        return tt, dist, emis
