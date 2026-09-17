"""
test_baselines.py — Unit tests for baseline algorithms and benchmark engine.
"""

from pathlib import Path

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


def test_all_baselines_run_successfully():
    tg = TransportationGraph.create_prototype_network()
    vehicles = generate_benchmark_scenario(tg, scenario_name="accident_corridor", vehicle_count=6, seed=42)
    cfg = OptimizationConfig(n_particles=10, max_iter=10, ga_population_size=10, ga_generations=10, aco_n_ants=10, aco_iterations=10)

    # 1. Dijkstra
    dijk = DijkstraBaseline(tg, cfg)
    res_dijk = dijk.optimize(vehicles)
    assert res_dijk.algorithm_name == "Dijkstra"
    assert res_dijk.best_fitness > 0.0

    # 2. A*
    astar = AStarBaseline(tg, cfg)
    res_astar = astar.optimize(vehicles)
    assert res_astar.algorithm_name == "A*"

    # 3. GA
    ga = GeneticAlgorithmBaseline(tg, cfg)
    res_ga = ga.optimize(vehicles)
    assert res_ga.algorithm_name == "GA"

    # 4. ACO
    aco = AntColonyBaseline(tg, cfg)
    res_aco = aco.optimize(vehicles)
    assert res_aco.algorithm_name == "ACO"

    # 5. PSO
    pso = StandardPSO(tg, cfg)
    res_pso = pso.optimize(vehicles)
    assert res_pso.algorithm_name == "PSO"

    # 6. QPSO
    qpso = StandardQPSO(tg, cfg)
    res_qpso = qpso.optimize(vehicles)
    assert res_qpso.algorithm_name == "QPSO"

    # 7. AT-DQPSO
    at_dqpso = AdaptiveTrafficDQPSO(tg, cfg)
    res_at = at_dqpso.optimize(vehicles)
    assert res_at.algorithm_name == "AT-DQPSO"

    # AT-DQPSO should beat or match Dijkstra on accident scenario due to bottleneck rerouting
    assert res_at.best_fitness <= res_dijk.best_fitness


def test_benchmark_engine_export(tmp_path):
    tg = TransportationGraph.create_prototype_network()
    vehicles = generate_benchmark_scenario(tg, vehicle_count=4, seed=42)
    cfg = OptimizationConfig(n_particles=8, max_iter=5)
    engine = BenchmarkEngine(tg, cfg)

    results = {
        "Dijkstra": DijkstraBaseline(tg, cfg).optimize(vehicles),
        "AT-DQPSO": AdaptiveTrafficDQPSO(tg, cfg).optimize(vehicles),
    }

    table_str = engine.format_table(results)
    assert "Dijkstra" in table_str
    assert "AT-DQPSO" in table_str

    csv_path = tmp_path / "test_benchmark.csv"
    engine.export_csv(results, csv_path)
    assert csv_path.exists()

    json_path = tmp_path / "test_benchmark.json"
    engine.export_json(results, json_path)
    assert json_path.exists()


def test_benchmark_engine_run_all():
    tg = TransportationGraph.create_prototype_network()
    vehicles = generate_benchmark_scenario(tg, vehicle_count=3, seed=42)
    cfg = OptimizationConfig(n_particles=6, max_iter=3, ga_population_size=6, ga_generations=3, aco_n_ants=6, aco_iterations=3)
    engine = BenchmarkEngine(tg, cfg)

    results = engine.run_all(vehicles, max_iter=3)
    assert "PSO" in results
    assert "Standard QPSO" in results
    assert "AT-DQPSO" in results
    assert "Dijkstra" in results
    assert "A*" in results
    assert "GA" in results
    assert "ACO" in results

    table_str = engine.format_table(results)
    assert "[UNMITIGATED BOTTLENECK BASELINE (Dijkstra) COMPARISON]" in table_str
    assert "[BEST HEURISTIC / METAHEURISTIC BASELINE COMPARISON]" in table_str


def test_ablation_engine(tmp_path):
    from optimization.benchmarks import AblationEngine

    tg = TransportationGraph.create_prototype_network()
    vehicles = generate_benchmark_scenario(tg, scenario_name="accident_corridor", vehicle_count=5, seed=42)
    cfg = OptimizationConfig(n_particles=8, max_iter=5)
    ablation = AblationEngine(tg, cfg)

    results = ablation.run_ablation(vehicles, max_iter=5)
    assert len(results) == 5
    assert "QPSO" in results
    assert "QPSO + adaptive mechanism" in results
    assert "QPSO + traffic awareness" in results
    assert "QPSO + prediction" in results
    assert "Full Adaptive QPSO + prediction" in results

    table_str = ablation.format_table(results)
    assert "Standard QPSO" in table_str
    assert "Full Adaptive QPSO + prediction" in table_str

    csv_path = tmp_path / "test_ablation.csv"
    ablation.export_csv(results, csv_path)
    assert csv_path.exists()

    json_path = tmp_path / "test_ablation.json"
    ablation.export_json(results, json_path)
    assert json_path.exists()
