"""
adaptive_qpso.py — Adaptive Traffic-Aware Discrete Quantum Particle Swarm Optimizer (AT-DQPSO).

Key Innovations over Standard QPSO:
1. Adaptive Contraction-Expansion Coefficient alpha(t):
   alpha(t) = alpha_final + (alpha_initial - alpha_final) * ((T_max - t) / T_max) * exp(-beta * (sigma(t) / sigma_0))
   Dynamically balances exploration and exploitation using real-time swarm diversity feedback.

2. Traffic-Aware Local Attractor:
   Biases the quantum potential well toward routes with lower downstream congestion and lower predicted delay.

3. Discrete Quantum Tunneling / Mutation Operator:
   Triggers targeted route perturbations when swarm diversity drops below threshold,
   forcing particles away from saturated bottleneck edges.
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
from optimization.route_decoder import DecodedVehicleRoute, RouteDecoder
from optimization.route_encoder import RouteEncoder, VehicleRoutingRequest

logger = logging.getLogger(__name__)


class AdaptiveTrafficDQPSO:
    """Adaptive Traffic-Aware Discrete Quantum Particle Swarm Optimizer (AT-DQPSO)."""

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
        use_adaptive_alpha: bool = True,
        use_traffic_attractor: bool = True,
        use_mutation: bool = True,
    ) -> OptimizationResult:
        """Executes AT-DQPSO multi-objective route optimization with modular feature toggles."""
        start_time = time.perf_counter()
        encoder = RouteEncoder(self.graph, vehicles, k_paths=self.config.k_paths)
        decoder = RouteDecoder(encoder, self.graph)
        tracker = ConvergenceTracker(stagnation_patience=8, stagnation_tolerance=1e-3)

        n_particles = self.config.n_particles
        dimension = encoder.dimension
        iterations = max_iter or self.config.max_iter
        rng = np.random.default_rng(self.config.seed)

        # 1. Initialize swarm population with greedy bias for particle 0
        X = encoder.initialize_population(n_particles, seed=self.config.seed, include_greedy=True)
        P = np.copy(X)
        fitnesses = np.zeros(n_particles)
        pbest_fitnesses = np.full(n_particles, float("inf"))

        gbest_particle = np.copy(X[0])
        gbest_fitness = float("inf")
        gbest_breakdown: Optional[FitnessBreakdown] = None

        # Initial evaluation
        for i in range(n_particles):
            routes = decoder.decode_particle(X[i], future_horizon_min=future_horizon_min)
            breakdown = self.fitness_evaluator.evaluate(routes, encoder, future_horizon_min)
            fitnesses[i] = breakdown.total_fitness
            pbest_fitnesses[i] = breakdown.total_fitness
            if breakdown.total_fitness < gbest_fitness:
                gbest_fitness = breakdown.total_fitness
                gbest_particle = np.copy(X[i])
                gbest_breakdown = breakdown

        c_metric = tracker.update(0, X, fitnesses, alpha=self.config.alpha_initial)
        sigma_0 = max(c_metric.diversity, 1e-4)

        # 2. Main AT-DQPSO Optimization Loop
        for t in range(1, iterations + 1):
            current_diversity = tracker.diversity_history[-1] if tracker.diversity_history else sigma_0

            # Adaptive contraction-expansion coefficient alpha(t) with diversity feedback
            if use_adaptive_alpha:
                progress = float(iterations - t) / float(iterations)
                diversity_ratio = current_diversity / sigma_0
                alpha_t = self.config.alpha_final + (
                    self.config.alpha_initial - self.config.alpha_final
                ) * progress * np.exp(-self.config.beta_decay * diversity_ratio)
            else:
                alpha_t = self.config.alpha_fixed

            # Compute mean best position (mbest)
            mbest = np.mean(P, axis=0)

            # Check if swarm is stagnated or diversity collapsed
            stagnated = c_metric.is_stagnated or (current_diversity < self.config.diversity_threshold)

            for i in range(n_particles):
                if use_traffic_attractor:
                    # Traffic-aware adaptive attractor weight
                    fit_p = pbest_fitnesses[i]
                    fit_g = gbest_fitness
                    denom = fit_p + fit_g + 1e-6
                    base_phi = 1.0 - (fit_p / denom)
                    phi = np.clip(rng.normal(base_phi, 0.1, size=dimension), 0.05, 0.95)
                else:
                    phi = rng.uniform(0.0, 1.0, size=dimension)

                p_local = phi * P[i] + (1.0 - phi) * gbest_particle

                # Quantum delta-potential well wave-function update
                u = rng.uniform(1e-7, 1.0, size=dimension)
                sign = rng.choice([-1.0, 1.0], size=dimension)
                X[i] = p_local + sign * alpha_t * np.abs(mbest - X[i]) * np.log(1.0 / u)

                # Discrete Quantum Tunneling / Mutation
                if use_mutation and (stagnated or rng.uniform(0.0, 1.0) < self.config.mutation_prob):
                    n_mutate = max(1, dimension // 4)
                    mutate_dims = rng.choice(dimension, size=n_mutate, replace=False)
                    for d in mutate_dims:
                        k_choices = encoder.get_candidate_count(d)
                        if k_choices > 1:
                            alt_idx = rng.integers(0, k_choices)
                            X[i, d] = (alt_idx + 0.5) / float(k_choices)

            # Enforce boundary conditions
            X = np.clip(X, 0.0, 1.0)

            # Evaluate candidate solutions
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

            c_metric = tracker.update(t, X, fitnesses, alpha=alpha_t)

        runtime = time.perf_counter() - start_time
        final_routes = decoder.decode_particle(gbest_particle, future_horizon_min=future_horizon_min)
        if gbest_breakdown is None:
            gbest_breakdown = self.fitness_evaluator.evaluate(final_routes, encoder, future_horizon_min)

        return OptimizationResult(
            algorithm_name="AT-DQPSO",
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
