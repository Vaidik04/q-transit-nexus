"""
runner.py — Experiment runner connecting CLI arguments to the optimization benchmark engine.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from optimization.adaptive_qpso import AdaptiveTrafficDQPSO
from optimization.baselines import (
    AntColonyBaseline,
    AStarBaseline,
    DijkstraBaseline,
    GeneticAlgorithmBaseline,
)
from optimization.benchmarks import BenchmarkEngine, generate_benchmark_scenario
from optimization.config import OptimizationConfig
from optimization.graph_interface import TransportationGraph
from optimization.pso import StandardPSO
from optimization.qpso import StandardQPSO

logger = logging.getLogger(__name__)


def run_experiment(
    scenario: str = "accident_corridor",
    vehicle_count: int = 25,
    algorithm: str = "AT-DQPSO",
    seed: int = 42,
    simulation_duration_sec: int = 60,
    output_file: str = "experiments/results.csv",
) -> None:
    """Runs single or comparative optimization experiments."""
    print("=" * 80)
    print("Q-TRANSIT NEXUS: EXPERIMENT BENCHMARK RUNNER")
    print("=" * 80)
    print(f"Scenario            : {scenario}")
    print(f"Vehicle Fleet Size  : {vehicle_count}")
    print(f"Target Algorithm    : {algorithm}")
    print(f"Random Seed         : {seed}")
    print(f"Simulation Duration : {simulation_duration_sec}s")
    print(f"Output Destination  : {output_file}")
    print("-" * 80)

    config = OptimizationConfig(seed=seed)
    graph = TransportationGraph.create_prototype_network()
    vehicles = generate_benchmark_scenario(graph, scenario_name=scenario, vehicle_count=vehicle_count, seed=seed)
    engine = BenchmarkEngine(graph=graph, config=config)

    algo_upper = algorithm.upper()
    results = {}

    if algo_upper in ("ABLATION", "ABLATION_STUDY"):
        print("[INFO] Executing 5-Stage Component Ablation Study...")
        from optimization.benchmarks import AblationEngine
        ablation_engine = AblationEngine(graph=graph, config=config)
        ablation_results = ablation_engine.run_ablation(vehicles)
        table_str = ablation_engine.format_table(ablation_results)
        print("\n" + table_str + "\n")
        out_path = Path(output_file)
        ablation_engine.export_csv(ablation_results, out_path)
        json_path = out_path.with_suffix(".json")
        ablation_engine.export_json(ablation_results, json_path)
        print(f"[SUCCESS] Exported ablation findings to: {out_path} and {json_path}")
        print("=" * 80)
        return
    elif algo_upper in ("ALL", "SUITE", "BENCHMARK"):
        print("[INFO] Running comparative suite across all 6 algorithms...")
        results = engine.run_all(vehicles)
    elif algo_upper in ("AT-DQPSO", "AT_DQPSO", "ADAPTIVE_QPSO"):
        opt = AdaptiveTrafficDQPSO(graph, config)
        res = opt.optimize(vehicles)
        results["AT-DQPSO"] = res
        # Run Dijkstra for comparison baseline
        dijk = DijkstraBaseline(graph, config)
        results["Dijkstra"] = dijk.optimize(vehicles)
    elif algo_upper in ("QPSO", "STANDARD_QPSO"):
        opt = StandardQPSO(graph, config)
        results["Standard QPSO"] = opt.optimize(vehicles)
    elif algo_upper == "PSO":
        opt = StandardPSO(graph, config)
        results["PSO"] = opt.optimize(vehicles)
    elif algo_upper == "DIJKSTRA":
        opt = DijkstraBaseline(graph, config)
        results["Dijkstra"] = opt.optimize(vehicles)
    elif algo_upper in ("ASTAR", "A*"):
        opt = AStarBaseline(graph, config)
        results["A*"] = opt.optimize(vehicles)
    elif algo_upper == "GA":
        opt = GeneticAlgorithmBaseline(graph, config)
        results["GA"] = opt.optimize(vehicles)
    elif algo_upper == "ACO":
        opt = AntColonyBaseline(graph, config)
        results["ACO"] = opt.optimize(vehicles)
    else:
        print(f"[WARNING] Unknown algorithm '{algorithm}'. Defaulting to AT-DQPSO...")
        opt = AdaptiveTrafficDQPSO(graph, config)
        results["AT-DQPSO"] = opt.optimize(vehicles)

    # Print formatted comparative table
    table_str = engine.format_table(results)
    print("\n" + table_str + "\n")

    # Export CSV results
    out_path = Path(output_file)
    engine.export_csv(results, out_path)
    print(f"[SUCCESS] Exported benchmark findings to: {out_path}")

    # Also export JSON alongside if possible
    json_path = out_path.with_suffix(".json")
    engine.export_json(results, json_path)
    print(f"[SUCCESS] Exported benchmark telemetry to: {json_path}")
    print("=" * 80)
