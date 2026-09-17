"""
Root CLI entry point for running experiments.
Matches Part 2, Section K specification:
python run_experiment.py --scenario accident_corridor --vehicles 25 --algorithm AT-DQPSO
"""
from experiments.runner import run_experiment
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Transit Nexus Experiment Runner")
    parser.add_argument("--scenario", type=str, default="accident_corridor")
    parser.add_argument("--vehicles", type=int, default=25)
    parser.add_argument("--algorithm", type=str, default="AT-DQPSO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--out", type=str, default="experiments/results.csv")
    args = parser.parse_args()

    run_experiment(
        scenario=args.scenario,
        vehicle_count=args.vehicles,
        algorithm=args.algorithm,
        seed=args.seed,
        simulation_duration_sec=args.duration,
        output_file=args.out
    )
