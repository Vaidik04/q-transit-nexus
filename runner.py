import argparse
import time
import random
import csv
import os
import sys
import numpy as np

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.modules.network_engine import UrbanNetwork
from backend.modules.simulation_engine import MicroscopicDigitalTwin
from backend.modules.at_dqpso import AT_DQPSO
from backend.modules.baselines import RoutingBaselines
from backend.modules.traffic_prediction import SpatioTemporalTrafficPredictor
from backend.schemas.vehicle import VehicleState, VehicleType

def run_experiment(
    scenario: str = "accident_corridor",
    vehicle_count: int = 25,
    algorithm: str = "AT-DQPSO",
    seed: int = 42,
    simulation_duration_sec: int = 60,
    output_file: str = "experiments/results.csv"
):
    random.seed(seed)
    np.random.seed(seed)

    network = UrbanNetwork()
    sim = MicroscopicDigitalTwin(network)
    predictor = SpatioTemporalTrafficPredictor(network)
    at_dqpso = AT_DQPSO(network, mode_name="balanced")
    baselines = RoutingBaselines(network)

    # Setup scenario
    if scenario == "accident_corridor":
        # Block Road E14
        network.edges_data["E14"]["is_blocked"] = True
        network.edges_data["E14"]["incident_penalty"] = 45.0
        network.edges_data["E14"]["current_speed"] = 3.0
    elif scenario == "heavy_peak":
        for eid in network.edges_data:
            network.edges_data[eid]["flow"] *= 1.8

    # Generate test vehicle population
    vehicles = []
    for i in range(vehicle_count):
        orig = f"N{random.randint(1, 10)}"
        dest = f"N{random.randint(15, 24)}"
        vtype = VehicleType.BUS if i % 5 == 0 else (VehicleType.EV if i % 4 == 0 else VehicleType.DELIVERY)
        v = VehicleState(
            vehicle_id=f"TEST_{i:03d}",
            type=vtype,
            current_edge=network.find_shortest_path_edges(orig, dest)[0] if network.find_shortest_path_edges(orig, dest) else "E1",
            origin=orig,
            destination=dest,
            capacity=25,
            passengers=60 if vtype == VehicleType.BUS else 0,
            soc=0.4 if vtype == VehicleType.EV else None,
            battery_capacity_kwh=65.0 if vtype == VehicleType.EV else None
        )
        vehicles.append(v)

    # Run algorithm
    t_start = time.time()
    total_travel_time = 0.0
    total_distance = 0.0
    total_emissions = 0.0
    violations = 0

    if algorithm in ["AT-DQPSO", "Adaptive QPSO", "Adaptive_QPSO"]:
        preds = predictor.predict_network()
        pred_costs = {eid: p.t10 for eid, p in preds.items()}
        res = at_dqpso.optimize_fleet(vehicles, predicted_travel_times=pred_costs, swarm_size=25, max_iterations=30)
        runtime = res.runtime_sec
        for r in res.routes:
            total_travel_time += r.travel_time
            total_distance += r.distance
            total_emissions += r.emissions_co2_kg
    elif algorithm == "Dijkstra":
        for v in vehicles:
            r = baselines.run_dijkstra(v)
            total_travel_time += r.travel_time
            total_distance += r.distance
            total_emissions += r.emissions_co2_kg
        runtime = time.time() - t_start
    elif algorithm == "A*":
        for v in vehicles:
            r = baselines.run_astar(v)
            total_travel_time += r.travel_time
            total_distance += r.distance
            total_emissions += r.emissions_co2_kg
        runtime = time.time() - t_start
    elif algorithm in ["PSO", "Standard_PSO"]:
        for v in vehicles:
            r = baselines.run_pso(v, swarm_size=20, iterations=20)
            total_travel_time += r.travel_time
            total_distance += r.distance
            total_emissions += r.emissions_co2_kg
        runtime = time.time() - t_start
    elif algorithm in ["Standard_QPSO", "QPSO"]:
        for v in vehicles:
            r = baselines.run_standard_qpso(v, swarm_size=20, iterations=20)
            total_travel_time += r.travel_time
            total_distance += r.distance
            total_emissions += r.emissions_co2_kg
        runtime = time.time() - t_start
    elif algorithm == "GA":
        for v in vehicles:
            r = baselines.run_ga(v, pop_size=15, generations=20)
            total_travel_time += r.travel_time
            total_distance += r.distance
            total_emissions += r.emissions_co2_kg
        runtime = time.time() - t_start
    elif algorithm == "ACO":
        for v in vehicles:
            r = baselines.run_aco(v, n_ants=12, iterations=15)
            total_travel_time += r.travel_time
            total_distance += r.distance
            total_emissions += r.emissions_co2_kg
        runtime = time.time() - t_start
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")

    avg_travel_time = round(total_travel_time / max(1, len(vehicles)), 2)
    avg_distance = round(total_distance / max(1, len(vehicles)), 2)
    avg_congestion = round(np.mean([ed["congestion"] for ed in network.edges_data.values()]), 3)

    result_row = {
        "algorithm": algorithm,
        "scenario": scenario,
        "vehicles": vehicle_count,
        "runtime": round(runtime, 4),
        "travel_time": avg_travel_time,
        "congestion": avg_congestion,
        "violations": violations,
        "seed": seed,
        "runtime_sec": round(runtime, 4),
        "avg_travel_time_sec": avg_travel_time,
        "avg_distance_m": avg_distance,
        "total_emissions_kg": round(total_emissions, 3),
        "avg_congestion": avg_congestion
    }

    # Ensure output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    file_exists = os.path.exists(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)

    print(f"[EXPERIMENT COMPLETED] Algorithm={algorithm} | Vehicles={vehicle_count} | Runtime={round(runtime, 3)}s | AvgTravelTime={avg_travel_time}s | Congestion={avg_congestion}")
    return result_row

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
