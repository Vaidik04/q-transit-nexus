"""
config.py — Central configuration and hyperparameter settings for Q-Transit Nexus Optimization Engine.

Defines:
- OptimizationMode (Balanced, Emergency Priority, Green Eco-Mode, Public Transit Priority)
- Objective weights (Travel Time, Distance, Congestion, Passenger Delay, Emissions, Reroute Instability, Constraint Violations)
- Swarm hyperparameters for Classical PSO, Standard QPSO, and AT-DQPSO
- Candidate path parameters and emission constants
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class OptimizationMode(str, Enum):
    BALANCED = "balanced"
    EMERGENCY = "emergency"
    FAST = "fast"
    GREEN = "green"
    ECO = "green"                          # Backward-compatible alias for GREEN
    PUBLIC_TRANSPORT_PRIORITY = "transit"
    TRANSIT = "transit"                    # Backward-compatible alias for PUBLIC_TRANSPORT_PRIORITY


@dataclass
class ObjectiveWeights:
    """Weights for the multi-objective fitness function:
    min F = w1*Time + w2*Dist + w3*Cong + w4*PassDelay + w5*Emissions + w6*Reroute + w7*Violations
    """
    w_time: float = 0.30           # w1: Travel Time (seconds)
    w_distance: float = 0.15       # w2: Distance (meters / normalized)
    w_congestion: float = 0.20     # w3: Network Congestion index
    w_passenger_delay: float = 0.15# w4: Bus passenger delay (passenger-seconds)
    w_emissions: float = 0.10      # w5: CO2 / energy emissions
    w_reroute_instability: float = 0.10 # w6: Penalty for changing routes
    w_violations: float = 10.0     # w7: Heavy penalty for constraint violations

    def to_dict(self) -> Dict[str, float]:
        return {
            "w_time": self.w_time,
            "w_distance": self.w_distance,
            "w_congestion": self.w_congestion,
            "w_passenger_delay": self.w_passenger_delay,
            "w_emissions": self.w_emissions,
            "w_reroute_instability": self.w_reroute_instability,
            "w_violations": self.w_violations,
        }


# Standard operational profiles
MODE_WEIGHTS: Dict[OptimizationMode, ObjectiveWeights] = {
    OptimizationMode.BALANCED: ObjectiveWeights(
        w_time=0.30,
        w_distance=0.15,
        w_congestion=0.20,
        w_passenger_delay=0.15,
        w_emissions=0.10,
        w_reroute_instability=0.10,
        w_violations=10.0,
    ),
    OptimizationMode.FAST: ObjectiveWeights(
        w_time=0.80,
        w_distance=0.05,
        w_congestion=0.15,
        w_passenger_delay=0.00,
        w_emissions=0.00,
        w_reroute_instability=0.00,
        w_violations=15.0,
    ),
    OptimizationMode.EMERGENCY: ObjectiveWeights(
        w_time=0.70,
        w_distance=0.00,
        w_congestion=0.30,
        w_passenger_delay=0.00,
        w_emissions=0.00,
        w_reroute_instability=0.00,
        w_violations=25.0,
    ),
    OptimizationMode.GREEN: ObjectiveWeights(
        w_time=0.15,
        w_distance=0.25,
        w_congestion=0.15,
        w_passenger_delay=0.00,
        w_emissions=0.45,
        w_reroute_instability=0.00,
        w_violations=10.0,
    ),
    OptimizationMode.PUBLIC_TRANSPORT_PRIORITY: ObjectiveWeights(
        w_time=0.25,
        w_distance=0.00,
        w_congestion=0.15,
        w_passenger_delay=0.50,
        w_emissions=0.00,
        w_reroute_instability=0.10,
        w_violations=15.0,
    ),
}


@dataclass
class OptimizationConfig:
    """Master configuration dataclass for the optimization algorithms."""
    # Swarm parameters
    n_particles: int = 30
    max_iter: int = 50
    seed: int = 42

    # Classical PSO parameters
    pso_w: float = 0.729           # Inertia weight
    pso_c1: float = 1.494          # Cognitive acceleration
    pso_c2: float = 1.494          # Social acceleration

    # Quantum PSO (QPSO) parameters
    alpha_fixed: float = 0.75      # Fixed contraction-expansion coefficient for standard QPSO

    # Adaptive Traffic-Aware Discrete QPSO (AT-DQPSO) parameters
    alpha_initial: float = 1.0     # Starting alpha (exploration)
    alpha_final: float = 0.5       # Final alpha (exploitation)
    beta_decay: float = 1.2        # Diversity feedback decay rate
    mutation_prob: float = 0.15    # Quantum mutation probability when stagnated
    diversity_threshold: float = 0.05 # Stagnation diversity threshold

    # Genetic Algorithm (GA) baseline parameters
    ga_population_size: int = 30
    ga_generations: int = 50
    ga_crossover_prob: float = 0.85
    ga_mutation_prob: float = 0.10

    # Ant Colony Optimization (ACO) baseline parameters
    aco_n_ants: int = 25
    aco_iterations: int = 40
    aco_alpha: float = 1.0         # Pheromone importance
    aco_beta: float = 2.5          # Heuristic importance
    aco_evaporation_rate: float = 0.15
    aco_q: float = 100.0

    # Graph routing & Candidate path parameters
    k_paths: int = 5               # Number of alternative K-shortest paths per O-D pair
    mode: OptimizationMode = OptimizationMode.BALANCED
    weights: ObjectiveWeights = field(default_factory=lambda: MODE_WEIGHTS[OptimizationMode.BALANCED])

    # Environmental & EV constants
    co2_grams_per_km: float = 150.0       # Average ICE vehicle CO2 emissions g/km
    fuel_liters_per_100km: float = 6.5    # Fuel consumption rate L/100km
    ev_kwh_per_km: float = 0.18           # EV energy consumption kWh/km
    min_ev_soc: float = 15.0              # Minimum allowable battery SoC (%)
    # Flow-dependent congestion parameters (BPR formula: t = t0 * [1 + alpha * (V/C)^beta])
    enable_bpr_latency: bool = True
    bpr_alpha: float = 0.15                # Standard Bureau of Public Roads coefficient
    bpr_beta: float = 2.0                  # Congestion curve exponent
    bpr_capacity_window_sec: float = 60.0  # Effective vehicle capacity time window in seconds

    def set_mode(self, mode: OptimizationMode | str) -> None:
        if isinstance(mode, str):
            mode = OptimizationMode(mode.lower())
        self.mode = mode
        self.weights = MODE_WEIGHTS[mode]
