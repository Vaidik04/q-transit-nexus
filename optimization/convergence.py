"""
convergence.py — Convergence monitoring, stagnation detection, and swarm diversity tracking.

Implements:
- Population spatial diversity metric sigma(t)
- Fitness history recording (best fitness, mean fitness)
- Stagnation detector triggering quantum mutation / exploration operators
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    """Convergence metrics snapshot for an iteration."""
    iteration: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float
    alpha: float
    is_stagnated: bool = False


class ConvergenceTracker:
    """Monitors optimization progress, records trajectory, and assesses swarm diversity."""

    def __init__(
        self,
        stagnation_patience: int = 10,
        stagnation_tolerance: float = 1e-4,
    ) -> None:
        self.stagnation_patience = stagnation_patience
        self.stagnation_tolerance = stagnation_tolerance

        self.history: List[ConvergenceMetrics] = []
        self.best_fitness_history: List[float] = []
        self.mean_fitness_history: List[float] = []
        self.worst_fitness_history: List[float] = []
        self.diversity_history: List[float] = []

        self._best_ever_fitness: float = float("inf")
        self._stagnation_counter: int = 0
        self._initial_diversity: Optional[float] = None

    def calculate_diversity(self, population: np.ndarray) -> float:
        """Calculates normalized population diversity:
        sigma(t) = (1 / (N * L)) * sum_{i=1}^N sqrt(sum_{d=1}^D (X_{id} - X_bar_d)^2)
        """
        n_particles, dimension = population.shape
        if n_particles <= 1 or dimension == 0:
            return 0.0

        # Mean position vector across all particles
        mean_pos = np.mean(population, axis=0)

        # Distance of each particle to centroid
        diffs = population - mean_pos
        distances = np.linalg.norm(diffs, axis=1)

        # Normalization constant: maximum diagonal in [0, 1]^D is sqrt(D)
        max_dist = np.sqrt(float(dimension))
        diversity = float(np.mean(distances) / max(max_dist, 1e-6))

        if self._initial_diversity is None and diversity > 0:
            self._initial_diversity = diversity

        return diversity

    def update(
        self,
        iteration: int,
        population: np.ndarray,
        fitnesses: np.ndarray,
        alpha: float = 1.0,
    ) -> ConvergenceMetrics:
        """Records an iteration, computes diversity, and checks for stagnation."""
        best_fit = float(np.min(fitnesses))
        mean_fit = float(np.mean(fitnesses))
        worst_fit = float(np.max(fitnesses))
        diversity = self.calculate_diversity(population)

        # Check stagnation
        if best_fit < self._best_ever_fitness - self.stagnation_tolerance:
            self._best_ever_fitness = best_fit
            self._stagnation_counter = 0
            is_stagnated = False
        else:
            self._stagnation_counter += 1
            is_stagnated = self._stagnation_counter >= self.stagnation_patience

        metric = ConvergenceMetrics(
            iteration=iteration,
            best_fitness=best_fit,
            mean_fitness=mean_fit,
            worst_fitness=worst_fit,
            diversity=diversity,
            alpha=alpha,
            is_stagnated=is_stagnated,
        )

        self.history.append(metric)
        self.best_fitness_history.append(best_fit)
        self.mean_fitness_history.append(mean_fit)
        self.worst_fitness_history.append(worst_fit)
        self.diversity_history.append(diversity)

        return metric

    @property
    def initial_diversity(self) -> float:
        return self._initial_diversity if self._initial_diversity is not None else 1.0

    @property
    def current_stagnation_count(self) -> int:
        return self._stagnation_counter
