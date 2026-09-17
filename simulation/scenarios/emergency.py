"""
Scenario: simulation.scenarios.emergency
Description:
    Evaluates Emergency Vehicle Preemption (EVP / Green Wave) for Priority-1 Ambulance:
    1. PART 1 (BEFORE): Ambulance navigates network under normal fixed-time signal cycles.
       Ambulance encounters red lights and delays behind stopped queues.
    2. PART 2 (AFTER): Emergency Preemption active! Approaching intersections (TL_J11, TL_J21)
       detect ambulance within 160m and preemptively switch to Green Wave.
    3. COMPARISON: Quantifies ambulance transit time savings and emergency response acceleration.

Usage:
    python -m simulation.scenarios.emergency
    python -m simulation.scenarios.emergency --steps 300
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
from simulation.metrics import MetricsCollector, compare_metrics
from simulation.ev_manager import EVManager
from simulation.scenario_manager import ScenarioManager

def run_emergency_pass(steps: int, enable_evp: bool, gui: bool = False):
    config_path = os.path.join(PROJECT_ROOT, "simulation", "sumo_config", "simulation.sumocfg")
    controller = TraCIController(config_path, gui=gui)
    ev_mgr = EVManager(controller)
    state_mgr = DigitalTwinStateManager(controller, ev_mgr=ev_mgr)
    route_applier = RouteApplier(controller)
    signal_ctrl = SignalController(controller)
    metrics_col = MetricsCollector(controller)
    scenario_mgr = ScenarioManager(controller, state_mgr, route_applier, signal_ctrl, metrics_col, ev_mgr)

    mode_label = "WITH EMERGENCY VEHICLE PREEMPTION (GREEN WAVE)" if enable_evp else "WITHOUT PREEMPTION (STANDARD SIGNALS)"
    print(f"\n[RUN] Starting pass: {mode_label}...")

    controller.start()

    try:
        summary = scenario_mgr.run_simulation_loop(
            total_steps=steps,
            step_hook=None,
            enable_evp=enable_evp,
            enable_ev=True
        )
        return summary
    finally:
        controller.close()

def run_emergency_benchmark(steps: int = 300, gui: bool = False):
    print("=" * 75)
    print("Q-TRANSIT NEXUS: Emergency Vehicle Preemption (EVP / Green Wave) Benchmark")
    print("=" * 75)
    print("Hypothesis: Preempting downstream traffic lights upon ambulance detection")
    print("clears intersection conflicts and dramatically reduces emergency response time")
    print("without causing systemic gridlock on cross-streets.")
    print("-" * 75)

    metrics_before = run_emergency_pass(steps=steps, enable_evp=False, gui=gui)
    metrics_after = run_emergency_pass(steps=steps, enable_evp=True, gui=gui)

    comparison = compare_metrics(
        before=metrics_before,
        after=metrics_after,
        label_before="BEFORE (No Signal Priority)",
        label_after="AFTER (Green-Wave Preemption)"
    )

    print("\n" + "=" * 115)
    print("EMERGENCY VEHICLE PREEMPTION (EVP) PERFORMANCE BENCHMARK")
    print("=" * 115)
    print(comparison["formatted_table"])
    print("=" * 115)

    results_dir = os.path.join(PROJECT_ROOT, "benchmark_results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "emergency_evp_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"[SUCCESS] Benchmark report written to: {out_file}\n")

    return comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emergency Vehicle Preemption Scenario")
    parser.add_argument("--steps", type=int, default=300, help="Simulation duration (default: 300s)")
    parser.add_argument("--gui", action="store_true", help="Launch in SUMO GUI")
    args = parser.parse_args()

    run_emergency_benchmark(steps=args.steps, gui=args.gui)
