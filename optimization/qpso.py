"""
qpso.py — Standard Quantum-behaved Particle Swarm Optimization (QPSO) baseline.

Formulation:
- Local attractor: p_{ij}^t = phi * P_{ij}^t + (1 - phi) * G_j^t, phi ~ U(0, 1)
- Mean best position (mbest): C_j^t = (1 / N) * sum_{i=1}^N P_{ij}^t
- Quantum update: X_{ij}^{t+1} = p_{ij}^t +- alpha * |C_j^t - X_{ij}^t| * ln(1 / u), u ~ U(0, 1)
  where alpha is a fixed contraction-expansion coefficient.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from optimization.config import OptimizationConfig
from optimization.convergence import ConvergenceTracker
from optimization.fitness import FitnessBreakdown, MultiObjectiveFitness
from optimization.graph_interface import TransportationGraph
from optimization.pso import OptimizationResult
from optimization.route_decoder import RouteDecoder
from optimization.route_encoder import RouteEncoder, VehicleRoutingRequest

logger = logging.getLogger(__name__)


class StandardQPSO:
    """Standard Quantum-behaved Particle Swarm Optimization with static alpha."""

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
        """Executes standard QPSO optimization."""
        start_time = time.perf_counter()
        encoder = RouteEncoder(self.graph, vehicles, k_paths=self.config.k_paths)
        decoder = RouteDecoder(encoder, self.graph)
        tracker = ConvergenceTracker()

        n_particles = self.config.n_particles
        dimension = encoder.dimension
        iterations = max_iter or self.config.max_iter
        alpha = self.config.alpha_fixed
        rng = np.random.default_rng(self.config.seed)

        # Initialize particles
        X = encoder.initialize_population(n_particles, seed=self.config.seed)
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

        tracker.update(0, X, fitnesses, alpha=alpha)

        # QPSO Main Loop
        for t in range(1, iterations + 1):
            # Compute mean best position (mbest)
            mbest = np.mean(P, axis=0)

            for i in range(n_particles):
                phi = rng.uniform(0.0, 1.0, size=dimension)
                u = rng.uniform(1e-7, 1.0, size=dimension)
                sign = rng.choice([-1.0, 1.0], size=dimension)

                # Local attractor
                p_local = phi * P[i] + (1.0 - phi) * gbest_particle

                # Quantum delta-potential well wave-function position update
                X[i] = p_local + sign * alpha * np.abs(mbest - X[i]) * np.log(1.0 / u)

            # Clamp coordinates to [0, 1]
            X = np.clip(X, 0.0, 1.0)

            # Evaluate new positions
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

            tracker.update(t, X, fitnesses, alpha=alpha)

        runtime = time.perf_counter() - start_time
        final_routes = decoder.decode_particle(gbest_particle, future_horizon_min=future_horizon_min)
        if gbest_breakdown is None:
            gbest_breakdown = self.fitness_evaluator.evaluate(final_routes, encoder, future_horizon_min)

        return OptimizationResult(
            algorithm_name="QPSO",
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
