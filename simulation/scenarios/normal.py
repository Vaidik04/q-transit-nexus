"""
Scenario: simulation.scenarios.normal
Description:
    Runs the baseline unperturbed simulation under normal operating conditions.
    Establishes the gold-standard benchmark metrics for traffic throughput,
    travel times, transit bus adherence, and emissions.

Usage:
    python -m simulation.scenarios.normal
    python -m simulation.scenarios.normal --gui --steps 400
"""

import os
import sys
import argparse
from dataclasses import asdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.traci_controller import TraCIController
from simulation.state_manager import DigitalTwinStateManager
from simulation.route_applier import RouteApplier
from simulation.signal_controller import SignalController
from simulation.metrics import MetricsCollector
from simulation.ev_manager import EVManager
from simulation.scenario_manager import ScenarioManager

def run_normal_scenario(steps: int = 300, gui: bool = False):
    config_path = os.path.join(PROJECT_ROOT, "simulation", "sumo_config", "simulation.sumocfg")
    print("=" * 70)
    print("Q-TRANSIT NEXUS: Baseline Normal Operating Scenario")
    print("=" * 70)
    print(f"Duration   : {steps} seconds")
    print(f"Mode       : {'GUI' if gui else 'Headless'}")
    print("-" * 70)

    controller = TraCIController(config_path, gui=gui)
    ev_mgr = EVManager(controller)
    state_mgr = DigitalTwinStateManager(controller, ev_mgr=ev_mgr)
    route_applier = RouteApplier(controller)
    signal_ctrl = SignalController(controller)
    metrics_col = MetricsCollector(controller)

    scenario_mgr = ScenarioManager(
        controller, state_mgr, route_applier, signal_ctrl, metrics_col, ev_mgr
    )

    try:
        controller.start()
        print("[INFO] Simulation initialized. Stepping normal baseline...")

        # Run unperturbed simulation loop
        summary = scenario_mgr.run_simulation_loop(total_steps=steps, enable_evp=True, enable_ev=True)

        print("\n" + "=" * 70)
        print("BASELINE PERFORMANCE METRICS SUMMARY")
        print("=" * 70)
        print(f"Total Simulation Time    : {summary.simulation_time} s")
        print(f"Active Vehicles Remaining: {summary.active_vehicles}")
        print(f"Completed Trips          : {summary.completed_trips}")
        print(f"Average Network Speed    : {summary.average_speed_kmh} km/h")
        print(f"Average Travel Time      : {summary.average_travel_time_s} s")
        print(f"Total Waiting Time       : {summary.total_waiting_time_s} s")
        print(f"Vehicle Throughput       : {summary.throughput_veh_per_hr} veh/hr")
        print(f"Bus Average Delay        : {summary.bus_average_delay_s} s")
        print(f"Passenger Total Delay    : {summary.passenger_total_delay_s} s")
        print(f"Estimated CO2 Emissions  : {summary.estimated_co2_kg} kg")
        print(f"Estimated Fuel Consumed  : {summary.estimated_fuel_liters} L")
        print("=" * 70)

        # Export benchmark artifacts
        results_dir = os.path.join(PROJECT_ROOT, "benchmark_results")
        os.makedirs(results_dir, exist_ok=True)
        metrics_col.export_json(os.path.join(results_dir, "normal_baseline.json"))
        metrics_col.export_csv(os.path.join(results_dir, "normal_baseline.csv"))
        state_mgr.export_snapshot_json(os.path.join(results_dir, "canonical_state_v1.json"))
        print(f"[SUCCESS] Exported benchmark logs to {results_dir}")

        return summary

    finally:
        controller.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normal Baseline Scenario")
    parser.add_argument("--steps", type=int, default=300, help="Simulation duration (default: 300s)")
    parser.add_argument("--gui", action="store_true", help="Launch in SUMO GUI")
    args = parser.parse_args()

    run_normal_scenario(steps=args.steps, gui=args.gui)
