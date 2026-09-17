"""
Scenario: simulation.scenarios.recovery
Description:
    Demonstrates post-incident network resilience and queue dissipation:
    1. Disruption: Accident blocks Central Arterial E_11_21 at t=50s.
    2. Clearance: Roadway cleared and full speed limit restored at t=140s.
    3. Recovery: Monitors dissipation of accumulated queues, recovery of mean velocity,
       and stabilization of transit schedules.

Usage:
    python -m simulation.scenarios.recovery
    python -m simulation.scenarios.recovery --steps 350
"""

import os
import sys
import json
import argparse

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

def run_recovery_scenario(steps: int = 350, gui: bool = False):
    config_path = os.path.join(PROJECT_ROOT, "simulation", "sumo_config", "simulation.sumocfg")
    print("=" * 75)
    print("Q-TRANSIT NEXUS: Incident Recovery & Queue Dissipation Benchmark")
    print("=" * 75)
    print("Timeline:")
    print("  t = 50s  : Incident blocks Central Arterial (E_11_21)")
    print("  t = 140s : Incident CLEARED -> Nominal speeds restored")
    print("  t = 350s : Full recovery and queue dissipation monitoring")
    print("-" * 75)

    controller = TraCIController(config_path, gui=gui)
    ev_mgr = EVManager(controller)
    state_mgr = DigitalTwinStateManager(controller, ev_mgr=ev_mgr)
    route_applier = RouteApplier(controller)
    signal_ctrl = SignalController(controller)
    metrics_col = MetricsCollector(controller)
    scenario_mgr = ScenarioManager(controller, state_mgr, route_applier, signal_ctrl, metrics_col, ev_mgr)

    accident_on = False
    accident_cleared = False
    recovery_checkpoints = {}

    def step_hook(sim_time: float):
        nonlocal accident_on, accident_cleared

        # Trigger blockage at t=50s
        if sim_time >= 50.0 and not accident_on:
            scenario_mgr.trigger_accident("E_11_21", duration_s=90.0, severity=1.0)
            accident_on = True
            print(f"  -> [T={sim_time:.0f}s] INCIDENT OCCURRED: E_11_21 blocked.")

        # Checkpoint at peak disruption (t=130s)
        if sim_time == 130.0:
            recovery_checkpoints["peak_disruption"] = metrics_col.capture_snapshot()
            print(f"  -> [T={sim_time:.0f}s] CHECKPOINT: Peak Disruption captured (Speed: {recovery_checkpoints['peak_disruption'].average_speed_kmh} km/h)")

        # Clear accident at t=140s
        if sim_time >= 140.0 and not accident_cleared:
            scenario_mgr.clear_accident("E_11_21")
            accident_cleared = True
            print(f"  -> [T={sim_time:.0f}s] INCIDENT CLEARED: Full corridor capacity restored.")

        # Checkpoint during dissipation (t=220s)
        if sim_time == 220.0:
            recovery_checkpoints["mid_recovery"] = metrics_col.capture_snapshot()
            print(f"  -> [T={sim_time:.0f}s] CHECKPOINT: Mid Recovery (Speed: {recovery_checkpoints['mid_recovery'].average_speed_kmh} km/h)")

    controller.start()
    try:
        final_summary = scenario_mgr.run_simulation_loop(
            total_steps=steps,
            step_hook=step_hook,
            enable_evp=True,
            enable_ev=True
        )
        recovery_checkpoints["final_recovered"] = final_summary

        print("\n" + "=" * 80)
        print(f"{'Phase':<22} | {'Time (s)':<10} | {'Mean Speed':<14} | {'Wait Time (s)':<16} | {'Queue (veh)':<12}")
        print("-" * 80)
        for phase_name, snap in recovery_checkpoints.items():
            print(f"{phase_name:<22} | {snap.simulation_time:<10} | {str(snap.average_speed_kmh) + ' km/h':<14} | {snap.total_waiting_time_s:<16} | {snap.average_queue_length:<12}")
        print("=" * 80)

        results_dir = os.path.join(PROJECT_ROOT, "benchmark_results")
        os.makedirs(results_dir, exist_ok=True)
        out_file = os.path.join(results_dir, "recovery_benchmark.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({k: snap.__dict__ for k, snap in recovery_checkpoints.items()}, f, indent=2)
        print(f"[SUCCESS] Recovery trajectory exported to: {out_file}\n")

        return recovery_checkpoints

    finally:
        controller.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incident Recovery Scenario")
    parser.add_argument("--steps", type=int, default=350, help="Simulation duration (default: 350s)")
    parser.add_argument("--gui", action="store_true", help="Launch in SUMO GUI")
    args = parser.parse_args()

    run_recovery_scenario(steps=args.steps, gui=args.gui)
