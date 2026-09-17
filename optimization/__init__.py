"""
optimization — Adaptive QPSO & Transportation Optimization Engine for Q-Transit Nexus.

SIH PS-137: Quantum-Inspired Intelligent Traffic Route Optimization in Transportation Systems.
"""

from optimization.adaptive_qpso import AdaptiveTrafficDQPSO
from optimization.baselines import (
    AntColonyBaseline,
    AStarBaseline,
    DijkstraBaseline,
    GeneticAlgorithmBaseline,
)
from optimization.benchmarks import (
    AblationEngine,
    AblationResult,
    BenchmarkEngine,
    generate_benchmark_scenario,
)
from optimization.config import (
    MODE_WEIGHTS,
    ObjectiveWeights,
    OptimizationConfig,
    OptimizationMode,
)
from optimization.constraints import (
    ConstraintEvaluationResult,
    ConstraintManager,
    ConstraintViolation,
)
from optimization.convergence import ConvergenceMetrics, ConvergenceTracker
from optimization.fitness import FitnessBreakdown, MultiObjectiveFitness
from optimization.graph_interface import EdgeData, NodeData, TransportationGraph
from optimization.pso import OptimizationResult, StandardPSO
from optimization.qpso import StandardQPSO
from optimization.repair import RouteRepairer
from optimization.route_decoder import DecodedVehicleRoute, RouteDecoder
from optimization.route_encoder import RouteEncoder, VehicleRoutingRequest

__all__ = [
    "AblationEngine",
    "AblationResult",
    "AdaptiveTrafficDQPSO",
    "AntColonyBaseline",
    "AStarBaseline",
    "BenchmarkEngine",
    "ConstraintEvaluationResult",
    "ConstraintManager",
    "ConstraintViolation",
    "ConvergenceMetrics",
    "ConvergenceTracker",
    "DecodedVehicleRoute",
    "DijkstraBaseline",
    "EdgeData",
    "FitnessBreakdown",
    "GeneticAlgorithmBaseline",
    "MODE_WEIGHTS",
    "MultiObjectiveFitness",
    "NodeData",
    "ObjectiveWeights",
    "OptimizationConfig",
    "OptimizationMode",
    "OptimizationResult",
    "RouteDecoder",
    "RouteEncoder",
    "RouteRepairer",
    "StandardPSO",
    "StandardQPSO",
    "TransportationGraph",
    "VehicleRoutingRequest",
    "generate_benchmark_scenario",
]
