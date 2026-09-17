"""
test_at_dqpso.py — Unit tests for AT-DQPSO optimizer.
"""

from optimization.adaptive_qpso import AdaptiveTrafficDQPSO
from optimization.benchmarks import generate_benchmark_scenario
from optimization.config import OptimizationConfig
from optimization.graph_interface import TransportationGraph
from optimization.repair import RouteRepairer


def test_at_dqpso_accident_rerouting():
    tg = TransportationGraph.create_prototype_network()
    vehicles = generate_benchmark_scenario(tg, scenario_name="accident_corridor", vehicle_count=10, seed=42)

    config = OptimizationConfig(n_particles=15, max_iter=20, seed=42)
    optimizer = AdaptiveTrafficDQPSO(tg, config)

    res = optimizer.optimize(vehicles, future_horizon_min=5)

    assert res.algorithm_name == "AT-DQPSO"
    assert res.best_fitness > 0.0
    assert len(res.best_routes) == 10
    assert len(res.convergence_history) == 21  # Iter 0..20
    assert len(res.diversity_history) == 21

    # Fitness should improve or stay non-increasing
    assert res.convergence_history[-1] <= res.convergence_history[0]

    # Verify worst and mean fitness histories are recorded
    assert len(res.worst_fitness_history) == 21
    assert len(res.mean_fitness_history) == 21
    for b, m, w in zip(res.convergence_history, res.mean_fitness_history, res.worst_fitness_history):
        assert b <= m + 1e-6
        assert m <= w + 1e-6

    # Verify all returned routes are valid connected paths
    repairer = RouteRepairer(tg)
    for r in res.best_routes:
        assert len(r.route) > 0
        assert repairer.is_valid_route(r.route) is True


def test_at_dqpso_summary_and_breakdown():
    tg = TransportationGraph.create_prototype_network()
    vehicles = generate_benchmark_scenario(tg, vehicle_count=5, seed=10)

    config = OptimizationConfig(n_particles=10, max_iter=10)
    optimizer = AdaptiveTrafficDQPSO(tg, config)
    res = optimizer.optimize(vehicles)

    summary = res.summary()
    assert summary["algorithm"] == "AT-DQPSO"
    assert "travel_time_sec" in summary
    assert "co2_kg" in summary
    assert "runtime_sec" in summary
