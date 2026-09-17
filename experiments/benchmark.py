import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.runner import run_experiment

def run_full_benchmark_suite(scenario: str = "accident_corridor", vehicle_count: int = 20, seed: int = 100):
    algorithms = ["Dijkstra", "A*", "PSO", "QPSO", "Adaptive QPSO"]
    csv_path = "experiments/benchmark_suite_results.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)

    print("\n" + "="*80)
    print("      Q-TRANSIT NEXUS: COMPARATIVE BENCHMARK EXPERIMENTAL SUITE")
    print("="*80)

    results = []
    for algo in algorithms:
        res = run_experiment(
            scenario=scenario,
            vehicle_count=vehicle_count,
            algorithm=algo,
            seed=seed,
            output_file=csv_path
        )
        results.append(res)

    print("\n" + "="*80)
    print(f"{'Algorithm':<16} | {'Runtime (s)':<12} | {'Travel Time (s)':<16} | {'Violations':<10} | {'Congestion':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['algorithm']:<16} | {r['runtime']:<12.4f} | {r['travel_time']:<16.1f} | {r['violations']:<10} | {r['congestion']:<10.3f}")
    print("="*80 + "\n")
    return results

if __name__ == "__main__":
    run_full_benchmark_suite()
