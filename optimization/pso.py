"""
pso.py — Classical Continuous Particle Swarm Optimization (PSO) baseline.

Implements traditional velocity-position kinematic equations:
V_{id}^{t+1} = w * V_{id}^t + c1 * r1 * (P_{id} - X_{id}^t) + c2 * r2 * (G_d - X_{id}^t)
X_{id}^{t+1} = clip(X_{id}^t + V_{id}^{t+1}, 0, 1)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from optimization.config import OptimizationConfig
from optimization.convergence import ConvergenceTracker
from optimization.fitness import FitnessBreakdown, MultiObjectiveFitness
from optimization.graph_interface import TransportationGraph
from optimization.route_decoder import DecodedVehicleRoute, RouteDecoder
from optimization.route_encoder import RouteEncoder, VehicleRoutingRequest

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Standardized output result returned by all optimization algorithms."""
    algorithm_name: str
    best_fitness: float
    best_particle: np.ndarray
    best_routes: List[DecodedVehicleRoute]
    fitness_breakdown: FitnessBreakdown
    runtime_sec: float
    iterations_run: int
    convergence_history: List[float] = field(default_factory=list) # best fitness
    mean_fitness_history: List[float] = field(default_factory=list)
    worst_fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)

    def summary(self) -> Dict[str, Union[str, float, int]]:
        return {
            "algorithm": self.algorithm_name,
            "best_fitness": round(self.best_fitness, 4),
            "travel_time_sec": round(self.fitness_breakdown.travel_time_sec, 2),
            "distance_m": round(self.fitness_breakdown.distance_m, 2),
            "co2_kg": round(self.fitness_breakdown.co2_emissions_kg, 4),
            "runtime_sec": round(self.runtime_sec, 4),
            "violations": self.fitness_breakdown.constraint_violations,
            "routes_changed": self.fitness_breakdown.rerouted_count,
        }


class StandardPSO:
    """Classical continuous Particle Swarm Optimization for multi-vehicle routing."""

    def __init__(
        self,
        graph: TransportationGraph,
        config: Optional[OptimizationConfig] = None,
    ) -> None:
        self.graph = graph
        self.config = config or OptimizationConfig()
        self.fitness_evaluator = MultiObjectiveFitness(graph, self.config)

    def optimize(
        self,
        vehicles: List[VehicleRoutingRequest],
        future_horizon_min: int = 5,
        max_iter: Optional[int] = None,
    ) -> OptimizationResult:
        """Executes classical PSO optimization."""
        start_time = time.perf_counter()
        encoder = RouteEncoder(self.graph, vehicles, k_paths=self.config.k_paths)
        decoder = RouteDecoder(encoder, self.graph)
        tracker = ConvergenceTracker()

        n_particles = self.config.n_particles
        dimension = encoder.dimension
        iterations = max_iter or self.config.max_iter
        rng = np.random.default_rng(self.config.seed)

        # Initialize particles and velocities
        X = encoder.initialize_population(n_particles, seed=self.config.seed)
        V = rng.uniform(-0.1, 0.1, size=(n_particles, dimension))

        # Personal bests and global best
        P = np.copy(X)
        fitnesses = np.zeros(n_particles)
        pbest_fitnesses = np.full(n_particles, float("inf"))

        gbest_particle = np.copy(X[0])
        gbest_fitness = float("inf")
        gbest_breakdown: Optional[FitnessBreakdown] = None

        # Evaluate initial population
        for i in range(n_particles):
            routes = decoder.decode_particle(X[i], future_horizon_min=future_horizon_min)
            breakdown = self.fitness_evaluator.evaluate(routes, encoder, future_horizon_min)
            fitnesses[i] = breakdown.total_fitness
            pbest_fitnesses[i] = breakdown.total_fitness
            if breakdown.total_fitness < gbest_fitness:
                gbest_fitness = breakdown.total_fitness
                gbest_particle = np.copy(X[i])
                gbest_breakdown = breakdown

        tracker.update(0, X, fitnesses)

        # Swarm loop
        w = self.config.pso_w
        c1 = self.config.pso_c1
        c2 = self.config.pso_c2

        for t in range(1, iterations + 1):
            r1 = rng.uniform(0.0, 1.0, size=(n_particles, dimension))
            r2 = rng.uniform(0.0, 1.0, size=(n_particles, dimension))

            # Velocity update
            V = w * V + c1 * r1 * (P - X) + c2 * r2 * (gbest_particle - X)
            V = np.clip(V, -0.2, 0.2)

            # Position update
            X = np.clip(X + V, 0.0, 1.0)

            # Fitness evaluation
            for i in range(n_particles):
                routes = decoder.decode_particle(X[i], future_horizon_min=future_horizon_min)
                breakdown = self.fitness_evaluator.evaluate(routes, encoder, future_horizon_min)
                fitnesses[i] = breakdown.total_fitness

                if breakdown.total_fitness < pbest_fitnesses[i]:
                    pbest_fitnesses[i] = breakdown.total_fitness
                    P[i] = np.copy(X[i])

                if breakdown.total_fitness < gbest_fitness:
                    gbest_fitness = breakdown.total_fitness
                    gbest_particle = np.copy(X[i])
                    gbest_breakdown = breakdown

            tracker.update(t, X, fitnesses)

        runtime = time.perf_counter() - start_time
        final_routes = decoder.decode_particle(gbest_particle, future_horizon_min=future_horizon_min)
        if gbest_breakdown is None:
            gbest_breakdown = self.fitness_evaluator.evaluate(final_routes, encoder, future_horizon_min)

        return OptimizationResult(
            algorithm_name="PSO",
            best_fitness=gbest_fitness,
            best_particle=gbest_particle,
            best_routes=final_routes,
            fitness_breakdown=gbest_breakdown,
            runtime_sec=runtime,
            iterations_run=iterations,
            convergence_history=tracker.best_fitness_history,
            mean_fitness_history=tracker.mean_fitness_history,
            worst_fitness_history=tracker.worst_fitness_history,
            diversity_history=tracker.diversity_history,
        )
