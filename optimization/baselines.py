"""
baselines.py — Comparative baseline routing and optimization algorithms.

Implements:
1. DijkstraBaseline: Static greedy shortest path on free-flow weights
2. AStarBaseline: A* heuristic shortest path on current graph weights
3. GeneticAlgorithmBaseline: Multi-vehicle discrete GA with crossover & mutation
4. AntColonyBaseline: Ant Colony Optimization with pheromone trails
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from optimization.config import OptimizationConfig
from optimization.fitness import MultiObjectiveFitness
from optimization.graph_interface import TransportationGraph
from optimization.pso import OptimizationResult
from optimization.route_decoder import DecodedVehicleRoute, RouteDecoder
from optimization.route_encoder import RouteEncoder, VehicleRoutingRequest

logger = logging.getLogger(__name__)


class DijkstraBaseline:
    """Greedy static shortest path baseline using Dijkstra's algorithm on freeflow weights."""

    def __init__(self, graph: TransportationGraph, config: Optional[OptimizationConfig] = None) -> None:
        self.graph = graph
        self.config = config or OptimizationConfig()
        self.fitness_evaluator = MultiObjectiveFitness(graph, self.config)

    def optimize(
        self,
        vehicles: List[VehicleRoutingRequest],
        future_horizon_min: int = 0,
    ) -> OptimizationResult:
        start_time = time.perf_counter()
        encoder = RouteEncoder(self.graph, vehicles, k_paths=1)
        decoder = RouteDecoder(encoder, self.graph)

        # Candidate path index 0 is the static shortest path for all vehicles
        discrete_indices = [0] * encoder.dimension
        particle = encoder.encode_discrete_indices(discrete_indices)
        routes = decoder.decode_particle(particle, future_horizon_min=future_horizon_min)
        breakdown = self.fitness_evaluator.evaluate(routes, encoder, future_horizon_min)
        runtime = time.perf_counter() - start_time

        return OptimizationResult(
            algorithm_name="Dijkstra",
            best_fitness=breakdown.total_fitness,
            best_particle=particle,
            best_routes=routes,
            fitness_breakdown=breakdown,
            runtime_sec=runtime,
            iterations_run=1,
            convergence_history=[breakdown.total_fitness],
        )


class AStarBaseline:
    """Heuristic A* routing baseline using dynamic edge weights."""

    def __init__(self, graph: TransportationGraph, config: Optional[OptimizationConfig] = None) -> None:
        self.graph = graph
        self.config = config or OptimizationConfig()
        self.fitness_evaluator = MultiObjectiveFitness(graph, self.config)

    def optimize(
        self,
        vehicles: List[VehicleRoutingRequest],
        future_horizon_min: int = 0,
    ) -> OptimizationResult:
        start_time = time.perf_counter()
        encoder = RouteEncoder(self.graph, vehicles, k_paths=self.config.k_paths)
        decoder = RouteDecoder(encoder, self.graph)

        # Select the path with lowest current travel time for each vehicle individually
        chosen_indices = []
        for i, v in enumerate(vehicles):
            candidate_paths = encoder.candidate_paths[v.vehicle_id]
            best_p_idx = 0
            best_time = float("inf")
            for p_idx, p in enumerate(candidate_paths):
                p_time = sum(
                    self.graph.get_edge(eid).travel_time_sec
                    for eid in p if self.graph.get_edge(eid)
                )
                if p_time < best_time:
                    best_time = p_time
                    best_p_idx = p_idx
            chosen_indices.append(best_p_idx)

        particle = encoder.encode_discrete_indices(chosen_indices)
        routes = decoder.decode_particle(particle, future_horizon_min=future_horizon_min)
        breakdown = self.fitness_evaluator.evaluate(routes, encoder, future_horizon_min)
        runtime = time.perf_counter() - start_time

        return OptimizationResult(
            algorithm_name="A*",
            best_fitness=breakdown.total_fitness,
            best_particle=particle,
            best_routes=routes,
            fitness_breakdown=breakdown,
            runtime_sec=runtime,
            iterations_run=1,
            convergence_history=[breakdown.total_fitness],
        )


class GeneticAlgorithmBaseline:
    """Genetic Algorithm (GA) baseline for combinatorial multi-vehicle route optimization."""

    def __init__(self, graph: TransportationGraph, config: Optional[OptimizationConfig] = None) -> None:
        self.graph = graph
        self.config = config or OptimizationConfig()
        self.fitness_evaluator = MultiObjectiveFitness(graph, self.config)

    def optimize(
        self,
        vehicles: List[VehicleRoutingRequest],
        future_horizon_min: int = 5,
        generations: Optional[int] = None,
    ) -> OptimizationResult:
        start_time = time.perf_counter()
        encoder = RouteEncoder(self.graph, vehicles, k_paths=self.config.k_paths)
        decoder = RouteDecoder(encoder, self.graph)

        pop_size = self.config.ga_population_size
        n_gen = generations or self.config.ga_generations
        n_vars = encoder.dimension
        rng = np.random.default_rng(self.config.seed)

        # Discrete chromosomes: each gene in {0, ..., K-1}
        population = np.zeros((pop_size, n_vars), dtype=np.int32)
        for i in range(pop_size):
            for d in range(n_vars):
                k = encoder.get_candidate_count(d)
                population[i, d] = rng.integers(0, max(k, 1))

        fitness_history = []
        best_chromosome = population[0].copy()
        best_fitness = float("inf")
        best_breakdown = None

        for gen in range(n_gen):
            fitnesses = np.zeros(pop_size)
            for i in range(pop_size):
                particle = encoder.encode_discrete_indices(population[i])
                routes = decoder.decode_particle(particle, future_horizon_min=future_horizon_min)
                breakdown = self.fitness_evaluator.evaluate(routes, encoder, future_horizon_min)
                fitnesses[i] = breakdown.total_fitness
                if breakdown.total_fitness < best_fitness:
                    best_fitness = breakdown.total_fitness
                    best_chromosome = population[i].copy()
                    best_breakdown = breakdown

            fitness_history.append(best_fitness)

            # Tournament Selection & Elitism
            new_pop = [best_chromosome.copy()]
            while len(new_pop) < pop_size:
                # Tournament size 3
                t1 = rng.integers(0, pop_size, size=3)
                t2 = rng.integers(0, pop_size, size=3)
                p1 = population[t1[np.argmin(fitnesses[t1])]]
                p2 = population[t2[np.argmin(fitnesses[t2])]]

                # Crossover
                if rng.uniform() < self.config.ga_crossover_prob and n_vars > 1:
                    cx_point = rng.integers(1, n_vars)
                    child = np.concatenate([p1[:cx_point], p2[cx_point:]])
                else:
                    child = p1.copy()

                # Mutation
                for d in range(n_vars):
                    if rng.uniform() < self.config.ga_mutation_prob:
                        k = encoder.get_candidate_count(d)
                        child[d] = rng.integers(0, max(k, 1))

                new_pop.append(child)

            population = np.array(new_pop)

        runtime = time.perf_counter() - start_time
        final_particle = encoder.encode_discrete_indices(best_chromosome)
        final_routes = decoder.decode_particle(final_particle, future_horizon_min=future_horizon_min)
        if best_breakdown is None:
            best_breakdown = self.fitness_evaluator.evaluate(final_routes, encoder, future_horizon_min)

        return OptimizationResult(
            algorithm_name="GA",
            best_fitness=best_fitness,
            best_particle=final_particle,
            best_routes=final_routes,
            fitness_breakdown=best_breakdown,
            runtime_sec=runtime,
            iterations_run=n_gen,
            convergence_history=fitness_history,
        )


class AntColonyBaseline:
    """Ant Colony Optimization (ACO) baseline for multi-vehicle candidate path selection."""

    def __init__(self, graph: TransportationGraph, config: Optional[OptimizationConfig] = None) -> None:
        self.graph = graph
        self.config = config or OptimizationConfig()
        self.fitness_evaluator = MultiObjectiveFitness(graph, self.config)

    def optimize(
        self,
        vehicles: List[VehicleRoutingRequest],
        future_horizon_min: int = 5,
        iterations: Optional[int] = None,
    ) -> OptimizationResult:
        start_time = time.perf_counter()
        encoder = RouteEncoder(self.graph, vehicles, k_paths=self.config.k_paths)
        decoder = RouteDecoder(encoder, self.graph)

        n_ants = self.config.aco_n_ants
        n_iters = iterations or self.config.aco_iterations
        n_vars = encoder.dimension
        k_max = self.config.k_paths
        rng = np.random.default_rng(self.config.seed)

        # Pheromone matrix: tau[vehicle_idx, candidate_path_idx]
        tau = np.ones((n_vars, k_max), dtype=np.float64)

        # Heuristic visibility: eta = 1.0 / (freeflow_time + 1.0)
        eta = np.zeros((n_vars, k_max), dtype=np.float64)
        for d, v in enumerate(vehicles):
            candidates = encoder.candidate_paths[v.vehicle_id]
            for p_idx, path in enumerate(candidates):
                t_sum = sum(self.graph.get_edge(e).travel_time_sec for e in path if self.graph.get_edge(e))
                eta[d, p_idx] = 1.0 / max(t_sum, 1.0)

        best_ant = np.zeros(n_vars, dtype=np.int32)
        best_fitness = float("inf")
        best_breakdown = None
        fitness_history = []

        alpha = self.config.aco_alpha
        beta = self.config.aco_beta
        rho = self.config.aco_evaporation_rate
        q = self.config.aco_q

        for it in range(n_iters):
            ant_solutions = np.zeros((n_ants, n_vars), dtype=np.int32)
            ant_fitnesses = np.zeros(n_ants)

            for a in range(n_ants):
                solution = np.zeros(n_vars, dtype=np.int32)
                for d in range(n_vars):
                    k = encoder.get_candidate_count(d)
                    probs = (tau[d, :k] ** alpha) * (eta[d, :k] ** beta)
                    p_sum = np.sum(probs)
                    if p_sum > 0:
                        probs /= p_sum
                    else:
                        probs = np.full(k, 1.0 / k)
                    solution[d] = rng.choice(k, p=probs)

                ant_solutions[a] = solution
                particle = encoder.encode_discrete_indices(solution)
                routes = decoder.decode_particle(particle, future_horizon_min=future_horizon_min)
                breakdown = self.fitness_evaluator.evaluate(routes, encoder, future_horizon_min)
                ant_fitnesses[a] = breakdown.total_fitness

                if breakdown.total_fitness < best_fitness:
                    best_fitness = breakdown.total_fitness
                    best_ant = solution.copy()
                    best_breakdown = breakdown

            fitness_history.append(best_fitness)

            # Evaporation
            tau *= (1.0 - rho)

            # Pheromone deposit (elitist ant deposit)
            for d in range(n_vars):
                chosen_idx = best_ant[d]
                deposit = q / max(best_fitness, 1.0)
                tau[d, chosen_idx] += deposit

        runtime = time.perf_counter() - start_time
        final_particle = encoder.encode_discrete_indices(best_ant)
        final_routes = decoder.decode_particle(final_particle, future_horizon_min=future_horizon_min)
        if best_breakdown is None:
            best_breakdown = self.fitness_evaluator.evaluate(final_routes, encoder, future_horizon_min)

        return OptimizationResult(
            algorithm_name="ACO",
            best_fitness=best_fitness,
            best_particle=final_particle,
            best_routes=final_routes,
            fitness_breakdown=best_breakdown,
            runtime_sec=runtime,
            iterations_run=n_iters,
            convergence_history=fitness_history,
        )
